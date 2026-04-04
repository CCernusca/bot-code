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

# ── Subdivision grid ──────────────────────────────────────────────────────────
# The field is split into COLS × ROWS = 4 × 4 = 16 equal subdivisions.
#
#   row 3 │ (0,3) (1,3) (2,3) (3,3) │
#   row 2 │ (0,2) (1,2) (2,2) (3,2) │
#   row 1 │ (0,1) (1,1) (2,1) (3,1) │
#   row 0 │ (0,0) (1,0) (2,0) (3,0) │
#           col 0  col 1  col 2  col 3
#
# col  (0–3) : vertical strip, increases left → right
# row  (0–3) : horizontal strip, increases bottom → top
# rank        : synonym for row  — "which horizontal band"
# vertical    : synonym for col  — "which vertical band"

COLS = 4
ROWS = 4

_COL_W = FIELD_WIDTH  / COLS   # 0.395 m per column
_ROW_H = FIELD_HEIGHT / ROWS   # 0.5475 m per row

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_prod_master", broker=mb, print_every=100)

# ── Broker state ──────────────────────────────────────────────────────────────
_robot_pos    = None   # {"x": float, "y": float}
_other_robots = None   # {"robots": [...]} from prediction node
_ball         = None   # {"global_pos": {x,y}, "ball_lost": bool, ...}


# ── Low-level subdivision math ────────────────────────────────────────────────

def col_of(x):
    """Vertical column index (0 = leftmost) for a global x coordinate."""
    return max(0, min(COLS - 1, int(x / _COL_W)))


def row_of(y):
    """Horizontal row index (0 = bottommost) for a global y coordinate."""
    return max(0, min(ROWS - 1, int(y / _ROW_H)))


def subdiv_of(x, y):
    """(col, row) subdivision indices for a global position."""
    return col_of(x), row_of(y)


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

    Returns a list of {"id": int, "x": float, "y": float, "predicted": bool}.
    """
    if _other_robots is None:
        return []
    out = []
    for r in _other_robots.get("robots", []):
        x, y = r.get("x"), r.get("y")
        if x is None or y is None:
            continue
        out.append({
            "id":        r.get("id"),
            "x":         float(x),
            "y":         float(y),
            "predicted": r.get("method") == "predicted",
        })
    return out


def robot_by_id(robot_id):
    """Position of a specific tracked robot, or None.

    Returns {"id": int, "x": float, "y": float, "predicted": bool}.
    """
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


# ── Generic location predicates ───────────────────────────────────────────────

def in_subdivision(x, y, col, row):
    """True if (x, y) falls in subdivision (col, row)."""
    return col_of(x) == col and row_of(y) == row


def in_rank(x, y, rank):
    """True if (x, y) is in horizontal row `rank` (0 = bottom, 3 = top)."""
    return row_of(y) == rank


def in_vertical(x, y, vertical):
    """True if (x, y) is in vertical column `vertical` (0 = left, 3 = right)."""
    return col_of(x) == vertical


# ── Own-robot location helpers ────────────────────────────────────────────────

def self_in_subdivision(col, row):
    """True if the own robot is in subdivision (col, row)."""
    p = self_pos()
    return p is not None and in_subdivision(p["x"], p["y"], col, row)


def self_in_rank(rank):
    """True if the own robot is in horizontal row `rank`."""
    p = self_pos()
    return p is not None and in_rank(p["x"], p["y"], rank)


def self_in_vertical(vertical):
    """True if the own robot is in vertical column `vertical`."""
    p = self_pos()
    return p is not None and in_vertical(p["x"], p["y"], vertical)


# ── Ball location helpers ─────────────────────────────────────────────────────

def ball_in_subdivision(col, row):
    """True if the ball is in subdivision (col, row)."""
    p = ball_pos()
    return p is not None and in_subdivision(p["x"], p["y"], col, row)


def ball_in_rank(rank):
    """True if the ball is in horizontal row `rank`."""
    p = ball_pos()
    return p is not None and in_rank(p["x"], p["y"], rank)


def ball_in_vertical(vertical):
    """True if the ball is in vertical column `vertical`."""
    p = ball_pos()
    return p is not None and in_vertical(p["x"], p["y"], vertical)


# ── Per-robot location helpers ────────────────────────────────────────────────

def robot_in_subdivision(robot_id, col, row):
    """True if robot `robot_id` is in subdivision (col, row)."""
    r = robot_by_id(robot_id)
    return r is not None and in_subdivision(r["x"], r["y"], col, row)


def robot_in_rank(robot_id, rank):
    """True if robot `robot_id` is in horizontal row `rank`."""
    r = robot_by_id(robot_id)
    return r is not None and in_rank(r["x"], r["y"], rank)


def robot_in_vertical(robot_id, vertical):
    """True if robot `robot_id` is in vertical column `vertical`."""
    r = robot_by_id(robot_id)
    return r is not None and in_vertical(r["x"], r["y"], vertical)


# ── Broker publish ────────────────────────────────────────────────────────────

def _publish(now):
    state = {"t": round(now, 3)}

    p = self_pos()
    if p is not None:
        c, r = subdiv_of(p["x"], p["y"])
        state["self"] = {"col": c, "row": r}

    robots_out = []
    for rb in all_robots():
        c, r = subdiv_of(rb["x"], rb["y"])
        robots_out.append({"id": rb["id"], "col": c, "row": r,
                            "predicted": rb["predicted"]})
    state["robots"] = robots_out

    bp = ball_pos()
    if bp is not None:
        c, r = subdiv_of(bp["x"], bp["y"])
        state["ball"] = {"col": c, "row": r, "lost": bp["lost"]}

    mb.set("field_sectors", json.dumps(state))


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _robot_pos, _other_robots, _ball

    if value is None:
        return

    try:
        if key == "robot_position":
            _robot_pos = json.loads(value)
        elif key == "other_robots":
            _other_robots = json.loads(value)
        elif key == "ball":
            _ball = json.loads(value)
        else:
            return
    except (json.JSONDecodeError, TypeError):
        return

    with _perf.measure("sectors"):
        _publish(time.monotonic())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key in ("robot_position", "other_robots", "ball"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["robot_position", "other_robots", "ball"], on_update)
    print("[MASTER] Starting master node...")
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[MASTER] Stopped.")
        mb.close()
