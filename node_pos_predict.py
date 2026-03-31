from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import time
import numpy as np

# ── Field dimensions ──────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.82   # metres
FIELD_HEIGHT = 2.43
ROBOT_RADIUS = 0.09

# ── Robot tracking ────────────────────────────────────────────────────────────
MAX_ROBOTS      = 3
VEL_MIN_DT      = 0.05   # seconds — min elapsed time between history samples
VEL_HISTORY_N   = 10     # rolling history length per tracked robot
VEL_HISTORY_MIN = 3      # minimum samples before fitted velocity is trusted
MAX_ROBOT_SPEED = 2.0    # m/s — hard cap after fitting

_MAX_PRED_STEPS = 20
_MAX_PRED_DT    = 0.5

# ── Ball prediction ───────────────────────────────────────────────────────────
FOV_DEG              = 62.2
BALL_RADIUS_MM       = 21.0
BALL_VEL_HISTORY_N   = 10
BALL_VEL_HISTORY_MIN = 3
BALL_VEL_MIN_DT      = 0.05   # seconds
MAX_BALL_SPEED       = 3.0    # m/s

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_pos_predict", broker=mb)

# ── Context from broker ───────────────────────────────────────────────────────
_robot_pos = None   # (x, y) metres
_imu_pitch = None   # degrees
_sim_state = None   # {"robot": [x,y], "obstacles": [[x,y],...]}

# ── Robot tracking state ──────────────────────────────────────────────────────
_tracked  = {}   # id → {"x","y","vx","vy","t","lost","history"}
_next_id  = 1

# ── Ball prediction state ─────────────────────────────────────────────────────
_vel_history      = []       # [[t, x, y], ...]
_vel_last_t       = -999.0
_last_detection_t = -999.0
_last_ball_vx     = 0.0
_last_ball_vy     = 0.0
_hidden_state     = None     # [x, y, vx, vy] or None
_hidden_state_t   = None     # None = loaded from visible, not yet advanced
_ball_lost        = False    # latching: set in FOV with no detection, cleared on re-detection


# ── Robot prediction functions ────────────────────────────────────────────────

def _predict_with_bounce(x, y, vx, vy, dt):
    dt   = min(dt, _MAX_PRED_DT)
    n    = max(1, min(int(dt / 0.02) + 1, _MAX_PRED_STEPS))
    step = dt / n
    for _ in range(n):
        x += vx * step;  y += vy * step
        if   x < ROBOT_RADIUS:                x = ROBOT_RADIUS;                vx =  abs(vx)
        elif x > FIELD_WIDTH  - ROBOT_RADIUS:  x = FIELD_WIDTH  - ROBOT_RADIUS; vx = -abs(vx)
        if   y < ROBOT_RADIUS:                y = ROBOT_RADIUS;                vy =  abs(vy)
        elif y > FIELD_HEIGHT - ROBOT_RADIUS:  y = FIELD_HEIGHT - ROBOT_RADIUS; vy = -abs(vy)
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
        tid: _predict_with_bounce(tr["x"], tr["y"], tr["vx"], tr["vy"], now - tr["t"])
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
                                 "vx": new_vx, "vy": new_vy, "lost": 0, "history": history}
        else:
            if len(new_tracked) >= MAX_ROBOTS:
                continue
            tid = _next_id;  _next_id += 1
            new_tracked[tid] = {"x": det["x"], "y": det["y"], "t": now,
                                 "vx": 0.0, "vy": 0.0, "lost": 0,
                                 "history": [(now, det["x"], det["y"])]}
        det["id"] = tid
        det["vx"] = round(new_tracked[tid]["vx"], 3)
        det["vy"] = round(new_tracked[tid]["vy"], 3)

    result = list(detections)
    for tid, tr in _tracked.items():
        if tid in matched_track:
            continue
        dt     = now - tr["t"]
        px, py = _predict_with_bounce(tr["x"], tr["y"], tr["vx"], tr["vy"], dt)
        new_tracked[tid] = {**tr, "x": px, "y": py, "t": now, "lost": tr["lost"] + 1}
        result.append({"x": round(px, 3), "y": round(py, 3),
                        "pts": 0, "method": "predicted", "confidence": 0.0,
                        "id": tid, "vx": round(tr["vx"], 3), "vy": round(tr["vy"], 3)})

    _tracked = new_tracked
    return result


# ── Ball prediction functions ─────────────────────────────────────────────────

def _fit_ball_velocity(history):
    if len(history) < BALL_VEL_HISTORY_MIN:
        return 0.0, 0.0
    arr = np.array(history, dtype=float)
    ts  = arr[:, 0] - arr[0, 0]
    if ts[-1] < 1e-9:
        return 0.0, 0.0
    coeffs = np.polyfit(ts, arr[:, 1:3], 1)
    return float(coeffs[0, 0]), float(coeffs[0, 1])


def _all_robot_positions():
    if _sim_state is not None:
        robots = []
        r = _sim_state.get("robot")
        if r:
            robots.append((float(r[0]), float(r[1])))
        robots += [(float(p[0]), float(p[1])) for p in _sim_state.get("obstacles", [])]
        return robots
    if _robot_pos is not None:
        return [_robot_pos]
    return []


def _in_camera_fov(bx, by):
    if _robot_pos is None or _imu_pitch is None:
        return False
    heading_rad = math.radians(_imu_pitch)
    cam_x = _robot_pos[0] + ROBOT_RADIUS * math.cos(heading_rad)
    cam_y = _robot_pos[1] + ROBOT_RADIUS * math.sin(heading_rad)
    dx, dy  = bx - cam_x, by - cam_y
    local_z =  dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
    local_x = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)
    if local_z <= 0:
        return False
    return math.degrees(math.atan2(abs(local_x), local_z)) <= FOV_DEG / 2.0


def _extrapolate_ball(x, y, vx, vy, dt, robots=None):
    STEP = 0.02
    r    = BALL_RADIUS_MM / 1000.0
    sep  = ROBOT_RADIUS + r
    n    = max(1, round(dt / STEP))
    step = dt / n
    for _ in range(n):
        x += vx * step;  y += vy * step
        if   x < r:                x = r;                vx =  abs(vx)
        elif x > FIELD_WIDTH  - r: x = FIELD_WIDTH  - r; vx = -abs(vx)
        if   y < r:                y = r;                vy =  abs(vy)
        elif y > FIELD_HEIGHT - r: y = FIELD_HEIGHT - r; vy = -abs(vy)
        if robots:
            for rx, ry in robots:
                dx, dy = x - rx, y - ry
                dist   = math.hypot(dx, dy)
                if 0 < dist < sep:
                    nx, ny = dx / dist, dy / dist
                    x, y   = rx + nx * sep, ry + ny * sep
                    dot    = vx * nx + vy * ny
                    if dot < 0:
                        vx -= 2 * dot * nx
                        vy -= 2 * dot * ny
    return x, y, vx, vy


# ── Broker callback ───────────────────────────────────────────────────────────

def on_update(key, value):
    global _robot_pos, _imu_pitch, _sim_state
    global _vel_history, _vel_last_t, _last_detection_t
    global _last_ball_vx, _last_ball_vy
    global _hidden_state, _hidden_state_t, _ball_lost

    if value is None:
        return

    # ── Context keys ──────────────────────────────────────────────────────────
    if key == "robot_position":
        try:
            p = json.loads(value)
            _robot_pos = (float(p["x"]), float(p["y"]))
        except Exception:
            pass
        return

    if key == "imu_pitch":
        try:
            _imu_pitch = float(value)
        except Exception:
            pass
        return

    if key == "sim_state":
        try:
            _sim_state = json.loads(value)
        except Exception:
            pass
        return

    # ── Robot tracking ────────────────────────────────────────────────────────
    if key == "other_robots_raw":
        with _perf.measure("robots"):
            try:
                payload    = json.loads(value)
                origin     = payload.get("origin")
                robot_list = payload.get("robots", [])
            except Exception:
                return
            now    = time.monotonic()
            result = _match_and_track(robot_list, now)
            mb.set("other_robots", json.dumps({"origin": origin, "robots": result}))
        return

    # ── Ball prediction ───────────────────────────────────────────────────────
    if key == "ball_raw":
        with _perf.measure("ball"):
            try:
                raw = json.loads(value)
            except Exception:
                return

            now_t  = time.monotonic()
            gpos   = raw.get("global_pos")
            hidden_pos = None
            pub_vx, pub_vy = _last_ball_vx, _last_ball_vy

            if gpos is not None:
                _last_detection_t = now_t
                _ball_lost        = False
                _hidden_state     = [gpos["x"], gpos["y"], _last_ball_vx, _last_ball_vy]
                _hidden_state_t   = None   # sentinel: don't advance on first hidden frame

                if now_t - _vel_last_t >= BALL_VEL_MIN_DT:
                    ok = True
                    if _vel_history:
                        dt_s = now_t - _vel_history[-1][0]
                        if math.hypot(gpos["x"] - _vel_history[-1][1],
                                      gpos["y"] - _vel_history[-1][2]
                                      ) > MAX_BALL_SPEED * dt_s * 1.5:
                            ok = False
                    if ok:
                        _vel_history.append([now_t, gpos["x"], gpos["y"]])
                        if len(_vel_history) > BALL_VEL_HISTORY_N:
                            _vel_history.pop(0)
                    _vel_last_t = now_t
            else:
                if now_t - _last_detection_t > 1.0:
                    _vel_history.clear()
                    _vel_last_t = -999.0

            if len(_vel_history) >= BALL_VEL_HISTORY_MIN:
                vx_fit, vy_fit = _fit_ball_velocity(_vel_history)
                spd = math.hypot(vx_fit, vy_fit)
                if spd > MAX_BALL_SPEED:
                    vx_fit *= MAX_BALL_SPEED / spd
                    vy_fit *= MAX_BALL_SPEED / spd
                _last_ball_vx = vx_fit
                _last_ball_vy = vy_fit
                if _hidden_state is not None:
                    _hidden_state[2] = vx_fit
                    _hidden_state[3] = vy_fit

            if gpos is None and _hidden_state is not None:
                if _ball_lost or _in_camera_fov(_hidden_state[0], _hidden_state[1]):
                    _ball_lost = True
                    pub_vx = pub_vy = 0.0
                else:
                    if _hidden_state_t is None:
                        _hidden_state_t = now_t
                    else:
                        dt_frame = now_t - _hidden_state_t
                        if dt_frame > 0:
                            hx, hy, hvx, hvy = _extrapolate_ball(
                                _hidden_state[0], _hidden_state[1],
                                _hidden_state[2], _hidden_state[3],
                                dt_frame, robots=_all_robot_positions(),
                            )
                            _hidden_state   = [hx, hy, hvx, hvy]
                            _hidden_state_t = now_t
                    pub_vx = _hidden_state[2]
                    pub_vy = _hidden_state[3]
                hidden_pos = {"x": round(_hidden_state[0], 3),
                              "y": round(_hidden_state[1], 3)}

            result = dict(raw)
            result["vx"]         = round(pub_vx, 3)
            result["vy"]         = round(pub_vy, 3)
            result["hidden_pos"] = hidden_pos
            result["ball_lost"]  = _ball_lost

            mb.set("ball",      json.dumps(result))
            mb.set("ball_lost", json.dumps(_ball_lost))


if __name__ == "__main__":
    for key, target, parse in [
        ("robot_position", "_robot_pos", lambda v: (float(json.loads(v)["x"]),
                                                     float(json.loads(v)["y"]))),
        ("imu_pitch",      "_imu_pitch", float),
        ("sim_state",      "_sim_state", json.loads),
    ]:
        try:
            val = mb.get(key)
            if val is not None:
                globals()[target] = parse(val)
        except Exception:
            pass

    print("[PREDICT] Starting prediction node...")
    mb.setcallback(
        ["ball_raw", "other_robots_raw", "robot_position", "imu_pitch", "sim_state"],
        on_update,
    )
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\n[PREDICT] Stopped.")
        mb.close()
