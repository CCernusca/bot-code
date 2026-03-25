from robus_core.libs.lib_telemtrybroker import TelemetryBroker
import json
import math

# ── Field configuration ────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.0   # metres, X axis (across width)
FIELD_HEIGHT = 2.0   # metres, Y axis (along length)
ROBOT_RADIUS = 0.1   # metres — robot centre cannot be closer than this to any wall

# Tolerance around the valid region to absorb measurement noise.
_MARGIN = 0.1    # metres
# Maximum distance between two candidate positions to be considered "the same".
_OUTLIER_THRESHOLD = 0.15  # metres
# ──────────────────────────────────────────────────────────────────────────────

_FIELD_CORNERS = [
    (0,           0          ),
    (FIELD_WIDTH, 0          ),
    (0,           FIELD_HEIGHT),
    (FIELD_WIDTH, FIELD_HEIGHT),
]

mb = TelemetryBroker()

# State updated by broker callbacks
_field_angle   = None  # degrees — explicitly set field angle; None = not set
_sim_heading   = None  # degrees — angle_f from lidar simulation, used as fallback
_lidar_corners = []    # [(angle_deg, dist_mm), ...]


def _effective_field_angle():
    """Return field_angle if explicitly set, fall back to sim_heading, then 0."""
    if _field_angle is not None:
        return _field_angle
    if _sim_heading is not None:
        return _sim_heading
    return 0.0


def _reject_outliers(candidates):
    """
    If more than 2 candidates exist, find the one with the most neighbours
    within _OUTLIER_THRESHOLD.  If those neighbours form a strict majority,
    discard everything else.
    """
    if len(candidates) <= 2:
        return candidates

    best_center, best_count = None, 0
    for ax, ay in candidates:
        count = sum(
            1 for bx, by in candidates
            if math.hypot(ax - bx, ay - by) <= _OUTLIER_THRESHOLD
        )
        if count > best_count:
            best_count, best_center = count, (ax, ay)

    if best_count > len(candidates) / 2:
        cx, cy = best_center
        kept = [(x, y) for x, y in candidates if math.hypot(x - cx, y - cy) <= _OUTLIER_THRESHOLD]
        discarded = len(candidates) - len(kept)
        if discarded:
            print(f"  [POS] Outlier rejection: kept {len(kept)}, discarded {discarded}")
        return kept

    return candidates


def _compute_position():
    """
    For each detected corner, compute the robot position implied by each of the
    four field corners.  Return the average of all candidates that land inside
    (or within _MARGIN of) the field.  Returns None if no valid candidate exists.
    """
    if not _lidar_corners:
        return None

    candidates = []
    eff_angle = _effective_field_angle()
    fa_rad = math.radians(eff_angle)
    source = "field_angle" if _field_angle is not None else ("sim_heading" if _sim_heading is not None else "default")
    print(f"[POS] field_angle={eff_angle:.1f}° (from {source})  corners detected: {len(_lidar_corners)}")

    for corner_angle_deg, corner_dist_mm in _lidar_corners:
        d = corner_dist_mm / 1000.0   # mm → metres

        # Direction of this corner in the field coordinate frame
        abs_angle_rad = math.radians(corner_angle_deg) + fa_rad
        abs_angle_deg = math.degrees(abs_angle_rad) % 360

        # Displacement from robot to corner (field frame)
        dx = d * math.cos(abs_angle_rad)
        dy = d * math.sin(abs_angle_rad)

        sensor_heading = eff_angle % 360

        print(f"  corner  lidar=({corner_angle_deg:.1f}°, {corner_dist_mm:.0f} mm)"
              f"  field_dir={abs_angle_deg:.1f}°"
              f"  offset=({dx:.3f}, {dy:.3f}) m"
              f"  sensor_heading={sensor_heading:.1f}°")

        for cx, cy in _FIELD_CORNERS:
            rx = cx - dx
            ry = cy - dy
            inside = (
                ROBOT_RADIUS - _MARGIN <= rx <= FIELD_WIDTH  - ROBOT_RADIUS + _MARGIN and
                ROBOT_RADIUS - _MARGIN <= ry <= FIELD_HEIGHT - ROBOT_RADIUS + _MARGIN
            )
            print(f"    vs field corner ({cx}, {cy})  →  robot=({rx:.3f}, {ry:.3f})"
                  f"  {'✓ accepted' if inside else '✗ rejected'}")
            if inside:
                candidates.append((rx, ry))

    if not candidates:
        print("  [POS] No valid candidates.")
        return None

    candidates = _reject_outliers(candidates)

    avg_x = sum(p[0] for p in candidates) / len(candidates)
    avg_y = sum(p[1] for p in candidates) / len(candidates)
    print(f"  [POS] Position: ({avg_x:.3f}, {avg_y:.3f}) m  (avg of {len(candidates)} candidate(s))")
    return round(avg_x, 3), round(avg_y, 3)


def on_update(key, value):
    global _field_angle, _sim_heading, _lidar_corners

    if value is None:
        return

    if key == "field_angle":
        try:
            _field_angle = float(value)
        except (ValueError, TypeError):
            pass
        return   # position is recalculated when corners arrive

    if key == "sim_heading":
        try:
            _sim_heading = float(value)
        except (ValueError, TypeError):
            pass
        return

    if key == "lidar_corners":
        try:
            _lidar_corners = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return

        pos = _compute_position()
        if pos is not None:
            mb.set("robot_position", json.dumps({"x": pos[0], "y": pos[1]}))


if __name__ == "__main__":
    # Seed values from broker; leave as None if not present (fallback chain applies)
    try:
        val = mb.get("field_angle")
        if val is not None:
            _field_angle = float(val)
    except Exception:
        pass
    try:
        val = mb.get("sim_heading")
        if val is not None:
            _sim_heading = float(val)
    except Exception:
        pass

    mb.setcallback(["field_angle", "sim_heading", "lidar_corners"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping positioning node.")
        mb.close()
