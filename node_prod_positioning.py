from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import time
import threading
import numpy as np

# ── Field configuration ───────────────────────────────────────────────────────
FIELD_WIDTH  = 1.82   # metres
FIELD_HEIGHT = 2.43
ROBOT_RADIUS = 0.09

# ── Wall detection ────────────────────────────────────────────────────────────
WALL_TOL        = 0.05   # metres
MIN_WALL_POINTS = 8

# ── Position estimation ───────────────────────────────────────────────────────
_POS_MARGIN        = 0.05   # metres
_OUTLIER_THRESHOLD = 0.15
_LIDAR_FIELD_TOL   = 0.05

# ── Robot detection & tracking ────────────────────────────────────────────────
WALL_MARGIN        = 0.08
CLUSTER_THRESHOLD  = 0.08
MIN_CLUSTER_POINTS = 3
MAX_ROBOTS         = 3
OVERLAP_DIST       = ROBOT_RADIUS * 2
VEL_MIN_DT         = 0.05
VEL_HISTORY_N      = 10
VEL_HISTORY_MIN    = 3
MAX_ROBOT_SPEED    = 2.0
_MAX_PRED_DT       = 0.5

# ── Time / history ────────────────────────────────────────────────────────────
TIME_PUBLISH_HZ   = 10
POSITION_WINDOW_S = 3.0
POSITION_SAMPLE_S = 0.5
BALL_SAMPLE_S     = 0.1

DEBUG = False

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_positioning", broker=mb)

# ── Shared sensor state ───────────────────────────────────────────────────────
_imu_pitch = None   # degrees
_lidar     = {}     # angle_deg (int) → dist_mm (int)

# ── Wall detection state ──────────────────────────────────────────────────────
_lidar_walls = []   # [{"gradient": 0|None, "offset": float}, ...]

# ── Robot detection & tracking state ─────────────────────────────────────────
_robot_pos = None   # (x, y) metres — updated after each position computation
_tracked   = {}     # id → {"x","y","vx","vy","t","history"}
_next_id   = 1

# ── Time node state ───────────────────────────────────────────────────────────
_time_start    = time.monotonic()
_pos_lock      = threading.Lock()
_pos_history   = []
_pos_last_t    = -999.0
_robots_lock   = threading.Lock()
_robots_history = []
_robots_last_t = -999.0
_ball_lock     = threading.Lock()
_ball_history  = []
_ball_last_t   = -999.0


def _heading():
    return _imu_pitch if _imu_pitch is not None else 0.0


def _elapsed():
    return time.monotonic() - _time_start


def _prune_list(lst, window=POSITION_WINDOW_S):
    cutoff = _elapsed() - window
    while lst and lst[0]["t"] < cutoff:
        lst.pop(0)


def _time_loop():
    interval = 1.0 / TIME_PUBLISH_HZ
    while True:
        mb.set("system_time", str(round(_elapsed(), 3)))
        time.sleep(interval)


# ── Wall detection ────────────────────────────────────────────────────────────

def _lidar_to_cartesian():
    if not _lidar:
        return None
    keys, vals = zip(*_lidar.items())
    angles    = np.radians(np.array(keys, dtype=float)) + np.radians(_heading())
    distances = np.array(vals, dtype=float) / 1000.0
    return np.column_stack((distances * np.cos(angles), distances * np.sin(angles)))


def _detect_walls(pts):
    walls = []
    specs = [
        (0, None, "Left",   "Right"),
        (1,    0, "Bottom", "Top"),
    ]
    for axis, gradient, label_lo, label_hi in specs:
        vals  = pts[:, axis]
        v_min = float(np.min(vals))
        v_max = float(np.max(vals))

        lo_group = vals[vals <= v_min + WALL_TOL]
        if len(lo_group) >= MIN_WALL_POINTS:
            offset = round(float(np.median(lo_group)), 3)
            if DEBUG:
                print(f"  [WALLS] {label_lo:6s}  offset={offset:+.3f} m  pts={len(lo_group)}")
            walls.append({"gradient": gradient, "offset": offset})

        hi_group = vals[vals >= v_max - WALL_TOL]
        if len(hi_group) >= MIN_WALL_POINTS:
            offset = round(float(np.median(hi_group)), 3)
            if DEBUG:
                print(f"  [WALLS] {label_hi:6s}  offset={offset:+.3f} m  pts={len(hi_group)}")
            walls.append({"gradient": gradient, "offset": offset})

    return walls


# ── Position estimation ───────────────────────────────────────────────────────

def _compute_position():
    if not _lidar_walls:
        return None

    fa_rad = math.radians(_heading())

    lidar_min_x = lidar_max_x = lidar_min_y = lidar_max_y = 0.0
    if _lidar:
        keys, vals = zip(*_lidar.items())
        a_arr = np.radians(np.array(keys, dtype=float)) + fa_rad
        d_arr = np.array(vals, dtype=float) / 1000.0
        ox = d_arr * np.cos(a_arr)
        oy = d_arr * np.sin(a_arr)
        lidar_min_x, lidar_max_x = float(ox.min()), float(ox.max())
        lidar_min_y, lidar_max_y = float(oy.min()), float(oy.max())

    x_candidates = []
    y_candidates = []

    for wall in _lidar_walls:
        gradient = wall.get("gradient")
        offset   = float(wall.get("offset", 0.0))

        if gradient is None:
            for rx in (-offset, FIELD_WIDTH - offset):
                if not (ROBOT_RADIUS - _POS_MARGIN <= rx <= FIELD_WIDTH - ROBOT_RADIUS + _POS_MARGIN):
                    continue
                if _lidar and not (
                    rx + lidar_min_x >= -_LIDAR_FIELD_TOL and
                    rx + lidar_max_x <= FIELD_WIDTH + _LIDAR_FIELD_TOL
                ):
                    continue
                x_candidates.append(rx)
        else:
            for ry in (-offset, FIELD_HEIGHT - offset):
                if not (ROBOT_RADIUS - _POS_MARGIN <= ry <= FIELD_HEIGHT - ROBOT_RADIUS + _POS_MARGIN):
                    continue
                if _lidar and not (
                    ry + lidar_min_y >= -_LIDAR_FIELD_TOL and
                    ry + lidar_max_y <= FIELD_HEIGHT + _LIDAR_FIELD_TOL
                ):
                    continue
                y_candidates.append(ry)

    if not x_candidates or not y_candidates:
        return None

    def _best(candidates):
        return max(candidates, key=lambda c: sum(
            1 for o in candidates if abs(c - o) <= _OUTLIER_THRESHOLD
        ))

    return round(_best(x_candidates), 3), round(_best(y_candidates), 3)


# ── Robot detection & tracking ────────────────────────────────────────────────

def _lidar_to_field_points():
    if not _lidar or _robot_pos is None:
        return None, None
    rx, ry = _robot_pos
    fa_rad = math.radians(_heading())
    sorted_items = sorted(_lidar.items())
    angles       = np.radians([a for a, _ in sorted_items]) + fa_rad
    distances    = np.array([d for _, d in sorted_items]) / 1000.0
    x = rx + distances * np.cos(angles)
    y = ry + distances * np.sin(angles)
    return np.column_stack((x, y)), np.array([rx, ry])


def _detect_clusters(points):
    if len(points) == 0:
        return []
    diffs  = np.diff(points, axis=0)
    dists  = np.hypot(diffs[:, 0], diffs[:, 1])
    splits = np.where(dists >= CLUSTER_THRESHOLD)[0] + 1
    return np.split(points, splits)


def _is_near_wall(cx, cy):
    return (
        cx < WALL_MARGIN or cx > FIELD_WIDTH  - WALL_MARGIN or
        cy < WALL_MARGIN or cy > FIELD_HEIGHT - WALL_MARGIN
    )


def _filter_overlapping(robots):
    kept = []
    for r in robots:
        if not any(math.hypot(r["x"] - k["x"], r["y"] - k["y"]) < OVERLAP_DIST
                   for k in kept):
            kept.append(r)
    return kept


def _predict_pos(x, y, vx, vy, dt):
    dt   = min(dt, _MAX_PRED_DT)
    n    = max(1, int(dt / 0.02) + 1)
    step = dt / n
    for _ in range(n):
        x += vx * step;  y += vy * step
        if   x < ROBOT_RADIUS:               x = ROBOT_RADIUS;               vx =  abs(vx)
        elif x > FIELD_WIDTH - ROBOT_RADIUS:  x = FIELD_WIDTH - ROBOT_RADIUS; vx = -abs(vx)
        if   y < ROBOT_RADIUS:               y = ROBOT_RADIUS;               vy =  abs(vy)
        elif y > FIELD_HEIGHT - ROBOT_RADIUS: y = FIELD_HEIGHT - ROBOT_RADIUS; vy = -abs(vy)
    return x, y


def _fit_velocity(history):
    if len(history) < 2:
        return 0.0, 0.0
    arr = np.array(history, dtype=float)
    ts  = arr[:, 0] - arr[0, 0]
    if ts[-1] < 1e-9:
        return 0.0, 0.0
    coeffs = np.polyfit(ts, arr[:, 1:3], 1)
    return float(coeffs[0, 0]), float(coeffs[0, 1])


def _match_and_track(detections, now):
    global _tracked, _next_id

    predictions = {
        tid: _predict_pos(tr["x"], tr["y"], tr["vx"], tr["vy"], now - tr["t"])
        for tid, tr in _tracked.items()
    }

    matched_det   = [None] * len(detections)
    matched_track = set()

    if detections and predictions:
        pred_ids = list(predictions.keys())
        det_xy   = np.array([[d["x"], d["y"]] for d in detections])
        pred_xy  = np.array([predictions[tid] for tid in pred_ids])
        dist_mat = np.hypot(det_xy[:, 0:1] - pred_xy[:, 0],
                            det_xy[:, 1:2] - pred_xy[:, 1])
        for _, di, tid in sorted(
            (dist_mat[di, ti], di, pred_ids[ti])
            for di in range(len(detections))
            for ti in range(len(pred_ids))
        ):
            if matched_det[di] is None and tid not in matched_track:
                matched_det[di] = tid
                matched_track.add(tid)

    new_tracked = {}

    for di, det in enumerate(detections):
        tid = matched_det[di]
        if tid is not None:
            old     = _tracked[tid]
            history = old.get("history", [])
            if now - old["t"] >= VEL_MIN_DT:
                history = (history + [(now, det["x"], det["y"])])[-VEL_HISTORY_N:]
            new_vx, new_vy = _fit_velocity(history) if len(history) >= VEL_HISTORY_MIN \
                             else (old["vx"], old["vy"])
            spd = math.hypot(new_vx, new_vy)
            if spd > MAX_ROBOT_SPEED:
                new_vx *= MAX_ROBOT_SPEED / spd
                new_vy *= MAX_ROBOT_SPEED / spd
            new_tracked[tid] = {"x": det["x"], "y": det["y"], "t": now,
                                 "vx": new_vx, "vy": new_vy, "history": history}
        else:
            if len(new_tracked) >= MAX_ROBOTS:
                continue
            tid = _next_id;  _next_id += 1
            new_tracked[tid] = {"x": det["x"], "y": det["y"], "t": now,
                                 "vx": 0.0, "vy": 0.0,
                                 "history": [(now, det["x"], det["y"])]}
        det["id"]  = tid
        det["vx"]  = round(new_tracked[tid]["vx"], 3)
        det["vy"]  = round(new_tracked[tid]["vy"], 3)

    for tid, tr in _tracked.items():
        if tid not in matched_track:
            new_tracked[tid] = tr

    _tracked = new_tracked
    return list(detections)


def _detect_and_track_robots(now):
    pts, robot_pos_arr = _lidar_to_field_points()
    if pts is None:
        return None, None

    rx, ry = _robot_pos
    fa_rad = math.radians(_heading())
    clusters = _detect_clusters(pts)
    robots = []

    for cluster in clusters:
        if len(cluster) < MIN_CLUSTER_POINTS:
            continue
        center    = np.mean(cluster, axis=0)
        direction = center - robot_pos_arr
        d = np.hypot(direction[0], direction[1])
        if d > 1e-9:
            center = center + (direction / d) * ROBOT_RADIUS
        if _is_near_wall(center[0], center[1]):
            continue
        dists = np.linalg.norm(cluster - center, axis=1)
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
    robots = _match_and_track(robots, now)

    origin = {"x": round(rx, 4), "y": round(ry, 4),
              "heading": round(math.degrees(fa_rad), 3)}
    return robots, origin


# ── Broker callback ───────────────────────────────────────────────────────────

def on_update(key, value):
    global _imu_pitch, _lidar, _lidar_walls, _robot_pos
    global _pos_last_t, _robots_last_t, _ball_last_t

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
            raw    = json.loads(value)
            _lidar = {int(k): int(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            return

        now = time.monotonic()

        with _perf.measure("walls"):
            pts = _lidar_to_cartesian()
            if pts is not None:
                _lidar_walls = _detect_walls(pts)
                mb.set("lidar_walls", json.dumps(_lidar_walls))

        with _perf.measure("pos"):
            pos = _compute_position()
            if pos is not None:
                _robot_pos = pos
                mb.set("robot_position", json.dumps({"x": pos[0], "y": pos[1]}))

        with _perf.measure("robots"):
            robots, origin = _detect_and_track_robots(now)
            if robots is not None:
                mb.set("other_robots_detected",
                       json.dumps({"origin": origin, "robots": robots, "t": now}))
        return

    if key == "robot_position":
        now = time.monotonic()
        if now - _pos_last_t < POSITION_SAMPLE_S:
            return
        try:
            pos   = json.loads(value)
            entry = {"x": float(pos["x"]), "y": float(pos["y"]),
                     "t": round(_elapsed(), 3)}
        except Exception:
            return
        _pos_last_t = now
        with _pos_lock:
            _pos_history.append(entry)
            _prune_list(_pos_history)
            snapshot = list(_pos_history)
        mb.set("position_history", json.dumps(snapshot))
        return

    if key == "other_robots":
        now = time.monotonic()
        if now - _robots_last_t < POSITION_SAMPLE_S:
            return
        try:
            payload    = json.loads(value)
            robot_list = payload.get("robots", payload) if isinstance(payload, dict) else payload
            robots     = [{"x": float(r["x"]), "y": float(r["y"]),
                           "id": int(r.get("id", 0))} for r in robot_list]
        except Exception:
            return
        _robots_last_t = now
        entry = {"t": round(_elapsed(), 3), "robots": robots}
        with _robots_lock:
            _robots_history.append(entry)
            _prune_list(_robots_history)
            snapshot = list(_robots_history)
        mb.set("other_robots_history", json.dumps(snapshot))
        return

    if key == "ball":
        now = time.monotonic()
        if now - _ball_last_t < BALL_SAMPLE_S:
            return
        try:
            pos = json.loads(value).get("global_pos")
            if pos is None:
                return
            entry = {"x": float(pos["x"]), "y": float(pos["y"]),
                     "t": round(_elapsed(), 3)}
        except Exception:
            return
        _ball_last_t = now
        with _ball_lock:
            _ball_history.append(entry)
            _prune_list(_ball_history)
            snapshot = list(_ball_history)
        mb.set("ball_history", json.dumps(snapshot))


if __name__ == "__main__":
    try:
        val = mb.get("imu_pitch")
        if val is not None:
            _imu_pitch = float(val)
    except Exception:
        pass

    threading.Thread(target=_time_loop, daemon=True, name="time-publisher").start()

    mb.setcallback(
        ["lidar", "imu_pitch", "robot_position", "other_robots", "ball"],
        on_update,
    )
    print("[POSITIONING] Starting combined positioning node...")
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\n[POSITIONING] Stopped.")
        mb.close()
