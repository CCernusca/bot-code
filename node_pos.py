from robus_core.libs.lib_telemtrybroker import TelemetryBroker
import json
import math

# ── Field configuration ────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.82   # metres, X axis (across width)
FIELD_HEIGHT = 2.43   # metres, Y axis (along length)
ROBOT_RADIUS = 0.09   # metres — robot centre cannot be closer than this to any wall

# Tolerance around the valid region to absorb measurement noise.
# Must be strictly less than ROBOT_RADIUS, otherwise the radius constraint
# is fully cancelled and positions inside the wall become accepted.
_MARGIN = 0.05   # metres
# Maximum distance between two candidate positions to be considered "the same".
_OUTLIER_THRESHOLD = 0.15  # metres
# How far outside the field boundary a lidar point is allowed to land.
_LIDAR_FIELD_TOL = 0.05  # metres
# ──────────────────────────────────────────────────────────────────────────────

_FIELD_CORNERS = [
    (0,           0          ),
    (FIELD_WIDTH, 0          ),
    (0,           FIELD_HEIGHT),
    (FIELD_WIDTH, FIELD_HEIGHT),
]

mb = TelemetryBroker()

# State updated by broker callbacks
_imu_pitch = None  # degrees — from imu_pitch broker key; None = not yet received
_lidar     = {}    # {angle_deg (int): dist_mm (int)}  — sensor frame
_depth_corners      = []    # [(angle_deg, dist_mm), ...]  sensor polar frame
_wall_corners       = []    # [[x, y], ...]  robot-centred field-aligned metres
_wall_corners_hist  = []    # [[x, y], ...]  from histogram wall detection


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


def _compute_position():
    """
    Builds a unified list of (dx, dy, label, dist_mm) displacements — the
    vector from the robot to a detected corner expressed in the field frame.

    Depth corners  (sensor polar)  → convert via field_angle.
    Wall corners   (field Cartesian, robot-centred) → already dx, dy.

    For every displacement, all four field corners are tried; valid robot
    positions are scored by support (other candidates within _OUTLIER_THRESHOLD)
    and the best-supported one is returned.  Ties break on sensor distance.
    """
    if not _depth_corners and not _wall_corners and not _wall_corners_hist:
        return None

    eff_angle = _heading()
    fa_rad    = math.radians(eff_angle)
    print(f"[POS] heading={eff_angle:.1f}°"
          f"  depth={len(_depth_corners)}  wall={len(_wall_corners)}"
          f"  wall_hist={len(_wall_corners_hist)}")

    # ── Build unified displacement list ───────────────────────────────────────
    # Each entry: (dx, dy, label, dist_mm)
    displacements = []

    for i, (angle_deg, dist_mm) in enumerate(_depth_corners):
        d      = dist_mm / 1000.0
        abs_rad = math.radians(angle_deg) + fa_rad
        dx, dy = d * math.cos(abs_rad), d * math.sin(abs_rad)
        print(f"  depth #{i}  ({angle_deg:.0f}°, {dist_mm:.0f} mm)"
              f"  field_dir={math.degrees(abs_rad) % 360:.1f}°"
              f"  offset=({dx:.3f}, {dy:.3f}) m")
        displacements.append((dx, dy, f"depth#{i}", dist_mm))

    for i, (wx, wy) in enumerate(_wall_corners):
        dist_mm = math.hypot(wx, wy) * 1000
        print(f"  wall  #{i}  offset=({wx:.3f}, {wy:.3f}) m  dist={dist_mm:.0f} mm")
        displacements.append((wx, wy, f"wall#{i}", dist_mm))

    for i, (wx, wy) in enumerate(_wall_corners_hist):
        dist_mm = math.hypot(wx, wy) * 1000
        print(f"  wallH #{i}  offset=({wx:.3f}, {wy:.3f}) m  dist={dist_mm:.0f} mm")
        displacements.append((wx, wy, f"wallH#{i}", dist_mm))

    # ── Precompute lidar bounding box (field-aligned, robot-centred) ─────────
    # Shifting by (rx, ry) gives the field-frame AABB of all lidar points.
    lidar_min_x = lidar_max_x = lidar_min_y = lidar_max_y = 0.0
    if _lidar:
        offsets_x = [dist_mm / 1000.0 * math.cos(math.radians(a) + fa_rad)
                     for a, dist_mm in _lidar.items()]
        offsets_y = [dist_mm / 1000.0 * math.sin(math.radians(a) + fa_rad)
                     for a, dist_mm in _lidar.items()]
        lidar_min_x, lidar_max_x = min(offsets_x), max(offsets_x)
        lidar_min_y, lidar_max_y = min(offsets_y), max(offsets_y)

    # ── Generate candidates ───────────────────────────────────────────────────
    # (rx, ry, label, field_corner_xy, dist_mm)
    candidates = []
    for dx, dy, label, dist_mm in displacements:
        for cx, cy in _FIELD_CORNERS:
            rx, ry = cx - dx, cy - dy
            inside = (
                ROBOT_RADIUS - _MARGIN <= rx <= FIELD_WIDTH  - ROBOT_RADIUS + _MARGIN and
                ROBOT_RADIUS - _MARGIN <= ry <= FIELD_HEIGHT - ROBOT_RADIUS + _MARGIN
            )
            if inside and _lidar:
                inside = (
                    rx + lidar_min_x >= -_LIDAR_FIELD_TOL and
                    rx + lidar_max_x <= FIELD_WIDTH  + _LIDAR_FIELD_TOL and
                    ry + lidar_min_y >= -_LIDAR_FIELD_TOL and
                    ry + lidar_max_y <= FIELD_HEIGHT + _LIDAR_FIELD_TOL
                )
            print(f"    [{label}] vs ({cx}, {cy})  →  robot=({rx:.3f}, {ry:.3f})"
                  f"  {'✓' if inside else '✗'}")
            if inside:
                candidates.append((rx, ry, label, (cx, cy), dist_mm))

    if not candidates:
        print("  [POS] No valid candidates.")
        return None

    positions = [(rx, ry) for rx, ry, *_ in candidates]

    def _score(c):
        rx, ry, _, _, dist_mm = c
        support = sum(
            1 for ox, oy in positions
            if math.hypot(rx - ox, ry - oy) <= _OUTLIER_THRESHOLD
        ) - 1
        return support, -dist_mm

    best                             = max(candidates, key=_score)
    rx, ry, label, fc, dist_mm      = best
    support                          = _score(best)[0]

    # Recover displacement so the vis can locate the winning corner
    fc_x, fc_y = fc
    dx, dy     = fc_x - rx, fc_y - ry
    mb.set("positioning_corner", json.dumps({"dx": round(dx, 3), "dy": round(dy, 3)}))

    print(f"  [POS] Best: {label}  → field corner {fc}"
          f"  support={support}/{len(candidates) - 1}"
          f"  pos=({rx:.3f}, {ry:.3f}) m")

    return round(rx, 3), round(ry, 3)


def on_update(key, value):
    global _imu_pitch, _lidar, _depth_corners, _wall_corners, _wall_corners_hist

    if value is None:
        return

    if key == "lidar":
        try:
            raw = json.loads(value)
            _lidar = {int(k): int(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return   # lidar alone doesn't trigger repositioning

    if key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except (ValueError, TypeError):
            pass
        return

    if key == "depth_corners":
        try:
            _depth_corners = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return

    elif key == "wall_corners":
        try:
            _wall_corners = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return

    elif key == "wall_corners_hist":
        try:
            _wall_corners_hist = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return

    pos = _compute_position()
    if pos is not None:
        mb.set("robot_position", json.dumps({"x": pos[0], "y": pos[1]}))


if __name__ == "__main__":
    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass

    mb.setcallback(["lidar", "imu_pitch",
                    "depth_corners", "wall_corners", "wall_corners_hist"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping positioning node.")
        mb.close()
