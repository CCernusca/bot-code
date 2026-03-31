from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import numpy as np

# ── Field & detection configuration ──────────────────────────────────────────
FIELD_WIDTH  = 1.82   # metres, X axis
FIELD_HEIGHT = 2.43   # metres, Y axis
ROBOT_RADIUS = 0.09   # metres — assumed radius of all robots
ROBOT_DIAMETER = ROBOT_RADIUS * 2

# Cluster centres within this distance of any field boundary are rejected.
WALL_MARGIN = 0.08   # metres

# Max Cartesian gap between consecutive (angle-sorted) points in a cluster.
CLUSTER_THRESHOLD = 0.08   # metres

# Clusters smaller than this are discarded as noise.
MIN_CLUSTER_POINTS = 3

# Detected diameter may differ from ROBOT_DIAMETER by at most this much.
SIZE_TOLERANCE = 0.05   # metres

# ── Confidence & detection limits ─────────────────────────────────────────────
MAX_ROBOTS   = 3
OVERLAP_DIST = ROBOT_RADIUS * 2   # metres

DEBUG = False   # set True to print per-scan detection results

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_pos_robots", broker=mb)

_lidar     = {}   # {angle_deg (int): dist_mm (int)}
_robot_pos = None # (x, y) metres, in field frame
_imu_pitch = None # degrees — from imu_pitch broker key


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


# ── Detection ─────────────────────────────────────────────────────────────────

def _lidar_points(angles, distances, lidar_pos):
    """Convert polar scan to absolute field-frame Cartesian numpy array."""
    x = lidar_pos[0] + distances * np.cos(angles)
    y = lidar_pos[1] + distances * np.sin(angles)
    return np.column_stack((x, y))


def _detect_clusters(points, threshold=CLUSTER_THRESHOLD):
    """
    Group angle-sorted Cartesian points into clusters by consecutive distance.
    Returns a list of (N, 2) numpy arrays.
    """
    if len(points) == 0:
        return []
    diffs  = np.diff(points, axis=0)
    dists  = np.hypot(diffs[:, 0], diffs[:, 1])
    splits = np.where(dists >= threshold)[0] + 1
    return np.split(points, splits)


def _is_near_wall(center):
    x, y = center
    return (
        x < WALL_MARGIN or x > FIELD_WIDTH  - WALL_MARGIN or
        y < WALL_MARGIN or y > FIELD_HEIGHT - WALL_MARGIN
    )


# ── Overlap filtering ─────────────────────────────────────────────────────────

def _filter_overlapping(robots):
    kept = []
    for r in robots:
        if not any(math.hypot(r["x"] - k["x"], r["y"] - k["y"]) < OVERLAP_DIST
                   for k in kept):
            kept.append(r)
    return kept


# ── Main detection ────────────────────────────────────────────────────────────

def _detect_robots():
    if not _lidar or _robot_pos is None:
        return [], None

    rx, ry = _robot_pos
    fa_rad = math.radians(_heading())

    # Build angle-sorted absolute field-frame point array
    sorted_items  = sorted(_lidar.items())
    angles        = np.radians([a for a, _ in sorted_items]) + fa_rad
    distances     = np.array([d for _, d in sorted_items]) / 1000.0
    pts           = _lidar_points(angles, distances, (rx, ry))

    robot_pos_arr = np.array([rx, ry])
    clusters      = _detect_clusters(pts)

    robots = []
    for cluster in clusters:
        if len(cluster) < MIN_CLUSTER_POINTS:
            continue

        center = np.mean(cluster, axis=0)

        # Arc centroid sits on the near surface; push outward by one radius
        # along the vector from the observing robot to get the true centre.
        direction = center - robot_pos_arr
        d = np.hypot(direction[0], direction[1])
        if d > 1e-9:
            center = center + (direction / d) * ROBOT_RADIUS

        if _is_near_wall(center):
            continue

        dists = np.linalg.norm(cluster - center, axis=1)
        
        # Standard deviation of distances should be small for a perfect circle, big for line-like features
        if np.std(dists) > 0.03:
            continue

        cx, cy = float(center[0]), float(center[1])
        robots.append({
            "x": round(cx, 3), "y": round(cy, 3),
            "pts": len(cluster), "method": "cluster",
            "confidence": float(len(cluster)),
        })

    robots.sort(key=lambda r: r["confidence"], reverse=True)
    robots = _filter_overlapping(robots)[:MAX_ROBOTS]

    if DEBUG:
        for r in robots:
            print(f"  [ROBOTS] {r['pts']:2d} pts  ({r['x']:.3f}, {r['y']:.3f})"
                  f"  conf={r['confidence']:.2f}")

    origin = {"x": round(rx, 4), "y": round(ry, 4), "heading": round(math.degrees(fa_rad), 3)}
    return robots, origin


# ── Broker interface ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _lidar, _robot_pos, _imu_pitch

    if value is None:
        return

    if key == "lidar":
        try:
            raw = json.loads(value)
            _lidar = {int(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError):
            return

    elif key == "robot_position":
        try:
            pos = json.loads(value)
            _robot_pos = (float(pos["x"]), float(pos["y"]))
        except Exception:
            return

    elif key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except (ValueError, TypeError):
            return

    if key == "lidar":
        with _perf.measure("lidar"):
            robots, origin = _detect_robots()
            mb.set("other_robots_raw", json.dumps({"origin": origin, "robots": robots}))


if __name__ == "__main__":
    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass
    try:
        raw = mb.get("robot_position")
        if raw:
            pos = json.loads(raw)
            _robot_pos = (float(pos["x"]), float(pos["y"]))
    except Exception:
        pass

    mb.setcallback(["lidar", "robot_position", "imu_pitch"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping robot detection.")
        mb.close()
