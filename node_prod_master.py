import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import math
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
ROBOT_RADIUS       = 0.09
BALL_RADIUS        = 0.021
BALL_CONTROL_DIST  = ROBOT_RADIUS + BALL_RADIUS + 0.10  # ≈ 0.21 m
BALL_CONTROL_DWELL = 0.3   # seconds ball must stay in range to confirm control

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

# Per-robot dwell tracking: robot_key (id or None for self) → monotonic time
# when the ball first entered BALL_CONTROL_DIST for that robot continuously.
_control_first_seen = {}


def _dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def on_ball(robot_id=None):
    """Return the robot closest to the ball that is within BALL_CONTROL_DIST
    and has kept the ball in range for at least BALL_CONTROL_DWELL seconds.

    Also updates the module-level `controlling_team` variable.

    If `robot_id` is given, return that robot's entry only if it is in
    control (useful as a boolean: ``if on_ball(my_id): ...``).

    Returns a dict {"id": int|None, "x", "y", "predicted", "team", "dist"}
    where id=None represents the own robot.  Returns None when no robot
    has confirmed control.
    """
    global controlling_team, _control_first_seen

    bp = ball_pos()
    if bp is None:
        controlling_team = None
        _control_first_seen.clear()
        return None

    now = time.monotonic()

    # Build list of all robots currently within range
    in_range = []
    sp = self_pos()
    if sp is not None:
        d = _dist(sp["x"], sp["y"], bp["x"], bp["y"])
        if d <= BALL_CONTROL_DIST:
            in_range.append({"id": None, "x": sp["x"], "y": sp["y"],
                              "predicted": False, "team": TEAM_US, "dist": d})
    for r in all_robots():
        d = _dist(r["x"], r["y"], bp["x"], bp["y"])
        if d <= BALL_CONTROL_DIST:
            in_range.append({**r, "dist": d})

    # Evict robots that left range; record first-seen for new entrants
    in_range_keys = {c["id"] for c in in_range}
    _control_first_seen = {k: v for k, v in _control_first_seen.items()
                           if k in in_range_keys}
    for c in in_range:
        if c["id"] not in _control_first_seen:
            _control_first_seen[c["id"]] = now

    # Only robots that have dwelled long enough qualify
    eligible = [c for c in in_range
                if now - _control_first_seen[c["id"]] >= BALL_CONTROL_DWELL]

    if not eligible:
        controlling_team = None
        return None

    closest = min(eligible, key=lambda c: c["dist"])
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


# ── Strategy points ───────────────────────────────────────────────────────────

# Goal centres
_OUR_GOAL   = (FIELD_WIDTH / 2, 0.0)
_ENEMY_GOAL = (FIELD_WIDTH / 2, FIELD_HEIGHT)

_SHOOT_GRID_N = 15   # grid resolution for LOS search
_MAX_RANGE = 0.5     # Maximum shooting distance

def _closest_on_segment(ax, ay, bx, by, px, py):
    """Return the point on segment A→B that is closest to P."""
    dx, dy = bx - ax, by - ay
    lsq = dx * dx + dy * dy
    if lsq < 1e-12:
        return ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / lsq))
    return ax + t * dx, ay + t * dy


def _dist_to_segment(ax, ay, bx, by, px, py):
    cx, cy = _closest_on_segment(ax, ay, bx, by, px, py)
    return math.hypot(px - cx, py - cy)


def _find_shooting_position():
    """Return the field position closest to us that has a clear
    line of sight to the enemy goal (no robot within ROBOT_RADIUS of the path).

    Searches a _SHOOT_GRID_N × _SHOOT_GRID_N grid sorted by distance to us,
    returning the first unblocked cell which has clear sight to the goal and ally.  Returns None if every
    cell is blocked.
    """
    gx, gy  = _ENEMY_GOAL
    robots  = all_robots()
    n       = _SHOOT_GRID_N
    sp      = self_pos()
    # Build candidates sorted closest-to-self first
    candidates = sorted(
        (((xi + 0.5) / n * FIELD_WIDTH, (yi + 0.5) / n * FIELD_HEIGHT)
         for xi in range(n) for yi in range(n)),
        key=lambda p: math.hypot(p[0] - sp["x"], p[1] - sp["y"]),
    )
    for px, py in candidates:
        print(_dist(px, py, gx, gy))
        if _dist(px, py, gx, gy) <= _MAX_RANGE and not any(_dist_to_segment(px, py, gx, gy, r["x"], r["y"]) < ROBOT_RADIUS
                   for r in robots):
            return px, py
    return None


def _find_passing_position():
    """Return the field position closest to us that has a clear
    line of sight to the enemy goal (no robot within ROBOT_RADIUS of the path).

    Searches a _SHOOT_GRID_N × _SHOOT_GRID_N grid sorted by distance to us,
    returning the first unblocked cell which has clear sight to the goal and ally.  Returns None if every
    cell is blocked.
    """
    gx, gy  = _ENEMY_GOAL
    robots  = all_robots()
    n       = _SHOOT_GRID_N
    sp      = self_pos()
    ally    = next((r for r in all_robots() if r["team"] == TEAM_US), None)
    # Build candidates sorted closest-to-self first
    candidates = sorted(
        (((xi + 0.5) / n * FIELD_WIDTH, (yi + 0.5) / n * FIELD_HEIGHT)
         for xi in range(n) for yi in range(n)),
        key=lambda p: math.hypot(p[0] - sp["x"], p[1] - sp["y"]),
    )
    for px, py in candidates:
        if _dist(px, py, gx, gy) <= _MAX_RANGE and _dist(px, py, ally["x"], ally["y"]) <= _MAX_RANGE and not any(_dist_to_segment(px, py, gx, gy, r["x"], r["y"]) < ROBOT_RADIUS and _dist_to_segment(px, py, ally["x"], ally["y"], r["x"], r["y"]) < ROBOT_RADIUS
                   for r in robots):
            return px, py
    return None


def _compute_strategy_points(ctrl):
    """Return the robot_strategy_points list for the current game state.

    Our team has the ball
    ─────────────────────
    • We control it  → single point at the enemy goal (shoot).
    • Ally controls  → find the field position closest to the enemy goal with
                       a clear line of sight to it; single point there.

    Enemy has the ball
    ──────────────────
    The controlling enemy robot is always the last point (so the connecting
    line shows the intercept direction).  The first point is the suggested
    intercept position for our robot:
    • If we are closer to the line controller → our goal than the ally: block
      that shot lane.
    • Otherwise the ally covers the goal; we cover the pass to the closest
      other enemy.

    Returns a list of {"x", "y"} dicts (0, 1, or 2 entries).
    """
    if ctrl is None:
        return []

    if ctrl.get("team") == TEAM_US:
        if ctrl.get("id") is None:
            # We have the ball — point at enemy goal
            pos = _find_shooting_position()
            if pos is None:
                return []
            return [{"x": round(pos[0], 3), "y": round(pos[1], 3)}]
        else:
            # Ally has the ball — best open shooting position
            pos = _find_passing_position()
            if pos is None:
                return []
            return [{"x": round(pos[0], 3), "y": round(pos[1], 3)}]

    # ── Enemy control ─────────────────────────────────────────────────────────
    crx, cry = ctrl["x"], ctrl["y"]
    gx, gy   = _OUR_GOAL

    sp   = self_pos()
    ally = next((r for r in all_robots() if r["team"] == TEAM_US), None)

    d_self = (_dist_to_segment(crx, cry, gx, gy, sp["x"],   sp["y"])
              if sp   else float("inf"))
    d_ally = (_dist_to_segment(crx, cry, gx, gy, ally["x"], ally["y"])
              if ally else float("inf"))

    if d_self <= d_ally:
        if sp:
            ix, iy = _closest_on_segment(crx, cry, gx, gy, sp["x"], sp["y"])
        else:
            ix, iy = gx, gy
    else:
        others = [r for r in all_robots()
                  if r["team"] == TEAM_ENEMY and r["id"] != ctrl["id"]]
        if not others:
            if sp:
                ix, iy = _closest_on_segment(crx, cry, gx, gy, sp["x"], sp["y"])
            else:
                ix, iy = gx, gy
        else:
            target = min(others, key=lambda r: math.hypot(r["x"] - crx, r["y"] - cry))
            if sp:
                ix, iy = _closest_on_segment(
                    crx, cry, target["x"], target["y"], sp["x"], sp["y"])
            else:
                ix, iy = (crx + target["x"]) / 2, (cry + target["y"]) / 2

    return [
        {"x": round(ix,  3), "y": round(iy,  3)},
        {"x": round(crx, 3), "y": round(cry, 3)},
    ]


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
    mb.set("robot_strategy_points", json.dumps(_compute_strategy_points(ctrl)))


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
