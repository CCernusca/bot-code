import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import time
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor

# ── Field dimensions ──────────────────────────────────────────────────────────
FIELD_WIDTH  = 1.58   # metres — playing field only
FIELD_HEIGHT = 2.19

# ── Teams ─────────────────────────────────────────────────────────────────────
# Team 0 = our team  — bottom goal  (y = 0)
# Team 1 = enemy     — top    goal  (y = FIELD_HEIGHT)
TEAM_US    = 0
TEAM_ENEMY = 1

# ── Ball control ──────────────────────────────────────────────────────────────
ROBOT_RADIUS      = 0.09
BALL_RADIUS       = 0.021
BALL_CONTROL_DIST = ROBOT_RADIUS + BALL_RADIUS + 0.04   # ≈ 0.15 m

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_master", broker=mb, print_every=100)

# ── Broker state ──────────────────────────────────────────────────────────────
_robot_pos    = None   # {"x": float, "y": float}
_other_robots = None   # {"robots": [...]} from prediction node
_ball         = None   # {"global_pos": {x,y}, "ball_lost": bool, ...}
_ally_id      = None   # ID of allied robot (team 0); all others are team 1


# ── Position accessors ────────────────────────────────────────────────────────

def self_pos():
    """Own robot position, or None if unknown.

    Returns {"x": float, "y": float}.
    """
    if _robot_pos is None:
        return None
    return {"x": float(_robot_pos["x"]), "y": float(_robot_pos["y"])}


def all_robots():
    """All tracked other robots (detections and predictions).

    Returns a list of {"id": int, "x": float, "y": float,
                        "vx": float|None, "vy": float|None,
                        "predicted": bool, "team": int}.
    The ally robot (same team as us) has team=TEAM_US; all others TEAM_ENEMY.
    """
    if _other_robots is None:
        return []
    out = []
    for r in _other_robots.get("robots", []):
        x, y = r.get("x"), r.get("y")
        if x is None or y is None:
            continue
        rid = r.get("id")
        out.append({
            "id":        rid,
            "x":         float(x),
            "y":         float(y),
            "vx":        r.get("vx"),
            "vy":        r.get("vy"),
            "predicted": r.get("method") == "predicted",
            "team":      TEAM_US if rid == _ally_id else TEAM_ENEMY,
        })
    return out


def robot_by_id(robot_id):
    """Position of a specific tracked robot, or None."""
    for r in all_robots():
        if r["id"] == robot_id:
            return r
    return None


def ball_pos():
    """Ball position, or None if unavailable.

    Returns {"x": float, "y": float, "lost": bool}.
    """
    if _ball is None:
        return None
    gpos = _ball.get("global_pos")
    if gpos is None:
        return None
    return {
        "x":    float(gpos["x"]),
        "y":    float(gpos["y"]),
        "lost": bool(_ball.get("ball_lost", False)),
    }


# ── Ball control ──────────────────────────────────────────────────────────────

# The team that currently has the ball (TEAM_US, TEAM_ENEMY, or None).
# Updated by on_ball() on every call.
controlling_team = None


def _dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def on_ball(robot_id=None):
    """Return the robot closest to the ball that is within BALL_CONTROL_DIST.

    Also updates the module-level `controlling_team` variable.

    If `robot_id` is given, return that robot's entry only if it is in
    control (useful as a boolean: ``if on_ball(my_id): ...``).

    Returns a dict {"id": int|None, "x", "y", "predicted", "team", "dist"}
    where id=None represents the own robot.  Returns None when no robot
    is in ball-control range.
    """
    global controlling_team

    bp = ball_pos()
    if bp is None:
        controlling_team = None
        return None

    candidates = []

    sp = self_pos()
    if sp is not None:
        d = _dist(sp["x"], sp["y"], bp["x"], bp["y"])
        if d <= BALL_CONTROL_DIST:
            candidates.append({"id": None, "x": sp["x"], "y": sp["y"],
                                "predicted": False, "team": TEAM_US, "dist": d})

    for r in all_robots():
        d = _dist(r["x"], r["y"], bp["x"], bp["y"])
        if d <= BALL_CONTROL_DIST:
            candidates.append({**r, "dist": d})

    if not candidates:
        controlling_team = None
        return None

    closest = min(candidates, key=lambda c: c["dist"])
    controlling_team = closest["team"]

    if robot_id is not None:
        return closest if closest["id"] == robot_id else None

    return closest


def self_on_ball():
    """True if the own robot is the closest robot in ball-control range."""
    r = on_ball()
    return r is not None and r["id"] is None


def ball_controlled():
    """True if any robot (any team) is currently in ball-control range."""
    return on_ball() is not None


# ── Broker publish ────────────────────────────────────────────────────────────

def _publish(now):
    state = {"t": round(now, 3)}

    with _perf.measure("positions"):
        p = self_pos()
        if p is not None:
            state["self"] = p

        state["robots"] = [
            {"id": r["id"], "x": r["x"], "y": r["y"],
             "predicted": r["predicted"], "team": r["team"]}
            for r in all_robots()
        ]

        bp = ball_pos()
        if bp is not None:
            state["ball"] = bp

    with _perf.measure("ball_control"):
        ctrl = on_ball()
        state["ball_control"] = (
            {"id": ctrl["id"], "team": ctrl["team"], "dist": round(ctrl["dist"], 3)}
            if ctrl is not None else None
        )
        state["controlling_team"] = controlling_team

    mb.set("field_sectors", json.dumps(state))


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _robot_pos, _other_robots, _ball, _ally_id

    if value is None:
        return

    try:
        if key == "robot_position":
            _robot_pos = json.loads(value)
        elif key == "other_robots":
            _other_robots = json.loads(value)
        elif key == "ball":
            _ball = json.loads(value)
        elif key == "ally_id":
            _ally_id = int(value) if value else None
        else:
            return
    except (json.JSONDecodeError, TypeError, ValueError):
        return

    _publish(time.monotonic())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key in ("robot_position", "other_robots", "ball", "ally_id"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["robot_position", "other_robots", "ball", "ally_id"], on_update)
    print("[MASTER] Starting master node...")
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[MASTER] Stopped.")
        mb.close()
