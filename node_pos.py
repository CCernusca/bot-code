from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math

# ── Field configuration ────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.82   # metres, X axis
FIELD_HEIGHT = 2.43   # metres, Y axis
ROBOT_RADIUS = 0.09   # metres

# Tolerance for the valid-position bounds check.
_MARGIN            = 0.05   # metres
# Candidates within this distance count as mutual support.
_OUTLIER_THRESHOLD = 0.15   # metres
# How far outside the field boundary a lidar point may land.
_LIDAR_FIELD_TOL   = 0.05   # metres
# ──────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_pos", broker=mb)

_imu_pitch   = None   # degrees — from imu_pitch broker key
_lidar       = {}     # {angle_deg: dist_mm}
_lidar_walls = []     # [{"gradient": 0|null, "offset": float}, ...]


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


def _compute_position():
    """
    Derive robot position from detected wall offsets.

    Each wall offset can correspond to either of the two parallel field
    boundaries.  Both candidates are generated, filtered to positions that
    keep the robot inside the field (with margin), and optionally tested
    against the lidar bounding box.  The most-supported candidate along
    each axis is returned.

    Vertical walls   (gradient=null)  constrain x.
    Horizontal walls (gradient=0)     constrain y.
    """
    if not _lidar_walls:
        return None

    eff_angle = _heading()
    fa_rad    = math.radians(eff_angle)
    print(f"[POS] heading={eff_angle:.1f}°  walls={len(_lidar_walls)}")

    # ── Lidar bounding box (field-aligned, robot-centred) ─────────────────────
    lidar_min_x = lidar_max_x = lidar_min_y = lidar_max_y = 0.0
    if _lidar:
        offsets_x = [d / 1000.0 * math.cos(math.radians(a) + fa_rad)
                     for a, d in _lidar.items()]
        offsets_y = [d / 1000.0 * math.sin(math.radians(a) + fa_rad)
                     for a, d in _lidar.items()]
        lidar_min_x, lidar_max_x = min(offsets_x), max(offsets_x)
        lidar_min_y, lidar_max_y = min(offsets_y), max(offsets_y)

    # ── Generate axis candidates ───────────────────────────────────────────────
    x_candidates = []
    y_candidates = []

    for wall in _lidar_walls:
        gradient = wall.get("gradient")
        offset   = float(wall.get("offset", 0.0))

        if gradient is None:  # vertical wall → constrains x
            for rx in (-offset, FIELD_WIDTH - offset):
                if not (ROBOT_RADIUS - _MARGIN <= rx <= FIELD_WIDTH - ROBOT_RADIUS + _MARGIN):
                    continue
                if _lidar and not (
                    rx + lidar_min_x >= -_LIDAR_FIELD_TOL and
                    rx + lidar_max_x <= FIELD_WIDTH + _LIDAR_FIELD_TOL
                ):
                    continue
                print(f"  [POS] x={rx:.3f}")
                x_candidates.append(rx)

        else:  # horizontal wall → constrains y
            for ry in (-offset, FIELD_HEIGHT - offset):
                if not (ROBOT_RADIUS - _MARGIN <= ry <= FIELD_HEIGHT - ROBOT_RADIUS + _MARGIN):
                    continue
                if _lidar and not (
                    ry + lidar_min_y >= -_LIDAR_FIELD_TOL and
                    ry + lidar_max_y <= FIELD_HEIGHT + _LIDAR_FIELD_TOL
                ):
                    continue
                print(f"  [POS] y={ry:.3f}")
                y_candidates.append(ry)

    if not x_candidates or not y_candidates:
        print("  [POS] No valid candidates.")
        return None

    def _best(candidates):
        return max(candidates, key=lambda c: sum(
            1 for o in candidates if abs(c - o) <= _OUTLIER_THRESHOLD
        ))

    rx = _best(x_candidates)
    ry = _best(y_candidates)
    print(f"  [POS] pos=({rx:.3f}, {ry:.3f}) m")
    return round(rx, 3), round(ry, 3)


def on_update(key, value):
    global _imu_pitch, _lidar, _lidar_walls

    if value is None:
        return

    if key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except (ValueError, TypeError):
            pass
        return

    if key == "lidar":
        try:
            raw = json.loads(value)
            _lidar = {int(k): int(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return  # lidar alone doesn't trigger repositioning

    if key == "lidar_walls":
        try:
            _lidar_walls = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return

        with _perf.measure("lidar_walls"):
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

    mb.setcallback(["lidar", "imu_pitch", "lidar_walls"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping positioning node.")
        mb.close()
