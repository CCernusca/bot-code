import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import math
import threading
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
from utils.cooperation_reader import SerialCooperationReader

# ── Reader factory ────────────────────────────────────────────────────────────
# To swap the transport layer, return a different BaseCooperationReader here.
def _make_reader():
    return SerialCooperationReader()
# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_cooperation", broker=mb, print_every=100)

# Broker state — updated by on_update()
_other_robots = None  # {"origin": ..., "robots": [{x, y, id, confidence, ...}, ...]}
_robot_pos    = None  # (x, y) metres — this system's own position
_ball_pos     = None  # {"x": ..., "y": ..., "confidence": ...} or None


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _xy(d):
    """Return (x, y) from a position dict, or None on failure."""
    if d is None:
        return None
    try:
        return float(d["x"]), float(d["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _conf(d, default=1.0):
    """Return the confidence value from a position dict, falling back to *default*."""
    if d is None:
        return default
    try:
        return float(d.get("confidence", default))
    except (TypeError, ValueError):
        return default


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _fuse(pos_a, conf_a, pos_b, conf_b):
    """Confidence-weighted average of two (x, y) positions."""
    total = conf_a + conf_b
    if total < 1e-9:
        return pos_a
    wa, wb = conf_a / total, conf_b / total
    return (
        round(wa * pos_a[0] + wb * pos_b[0], 3),
        round(wa * pos_a[1] + wb * pos_b[1], 3),
    )


# ── Frame handler ─────────────────────────────────────────────────────────────

def on_frame(data):
    """
    Process one cooperation frame from the ally robot.

    Steps:
      1. Publish raw ally fields to the broker under the ally_ prefix.
      2. Identify the ally in our other_robots list by proximity to
         main_robot_pos; update its position to the received value and tag it.
      3. Among other_pos_1/2/3, find and discard the entry closest to this
         system's own robot_position (the ally is observing us).
      4. Fuse the remaining two ally observations with our other_robots via
         nearest-neighbour matching and confidence-weighted averaging.
      5. Fuse ball positions (confidence-weighted, or direct if no local value).
      6. Publish updated other_robots and ball_pos.
    """
    with _perf.measure("frame"):
        # ── Step 1: Publish raw ally fields ───────────────────────────────────
        for key in ("main_robot_pos", "other_pos_1", "other_pos_2",
                    "other_pos_3", "ball_pos"):
            if key in data:
                mb.set(f"ally_{key}", json.dumps(data[key]))

        ally_main   = data.get("main_robot_pos")
        ally_others = [data.get(f"other_pos_{i}") for i in range(1, 4)]
        ally_ball   = data.get("ball_pos")

        if _other_robots is None:
            return  # not enough broker state to fuse yet

        robots = [dict(r) for r in _other_robots.get("robots", [])]
        origin = _other_robots.get("origin")

        # ── Step 2: Identify ally robot in our other_robots list ──────────────
        ally_idx      = None
        ally_main_pos = _xy(ally_main)
        if ally_main_pos is not None and robots:
            ally_idx = min(
                range(len(robots)),
                key=lambda i: _dist((robots[i]["x"], robots[i]["y"]), ally_main_pos),
            )
            # The ally knows its own position most accurately — use directly.
            robots[ally_idx]["x"]    = ally_main_pos[0]
            robots[ally_idx]["y"]    = ally_main_pos[1]
            robots[ally_idx]["ally"] = True
            mb.set("ally_id", str(robots[ally_idx].get("id", ally_idx)))

        # ── Step 3: Discard the ally's observation that matches our position ──
        remaining_ally = list(ally_others)
        if _robot_pos is not None:
            candidates = [
                (i, _xy(p))
                for i, p in enumerate(remaining_ally)
                if _xy(p) is not None
            ]
            if candidates:
                self_idx = min(candidates,
                               key=lambda t: _dist(t[1], _robot_pos))[0]
                remaining_ally[self_idx] = None

        # ── Step 4: Fuse remaining ally observations with our other_robots ────
        sys_indices = [i for i in range(len(robots)) if i != ally_idx]
        ally_valid  = [
            (_xy(p), _conf(p))
            for p in remaining_ally
            if p is not None and _xy(p) is not None
        ]

        matched = set()
        for ally_pos, ally_c in ally_valid:
            unmatched = [i for i in sys_indices if i not in matched]
            if not unmatched:
                break
            best    = min(unmatched,
                          key=lambda i: _dist((robots[i]["x"], robots[i]["y"]), ally_pos))
            matched.add(best)
            sys_pos = (robots[best]["x"], robots[best]["y"])
            sys_c   = float(robots[best].get("confidence", 1.0))
            fx, fy  = _fuse(sys_pos, sys_c, ally_pos, ally_c)
            robots[best]["x"] = fx
            robots[best]["y"] = fy

        mb.set("other_robots", json.dumps({"origin": origin, "robots": robots}))

        # ── Step 5: Fuse ball position ─────────────────────────────────────────
        if ally_ball is not None:
            ally_ball_pos  = _xy(ally_ball)
            ally_ball_conf = _conf(ally_ball)
            if ally_ball_pos is not None:
                if _ball_pos is not None:
                    sys_ball_pos  = _xy(_ball_pos)
                    sys_ball_conf = _conf(_ball_pos)
                    if sys_ball_pos is not None:
                        fx, fy = _fuse(sys_ball_pos, sys_ball_conf,
                                       ally_ball_pos, ally_ball_conf)
                        mb.set("ball_pos", json.dumps({
                            "x": fx, "y": fy,
                            "confidence": round(
                                (sys_ball_conf + ally_ball_conf) / 2, 3),
                        }))
                        return
                # No local ball reading — publish the ally's value directly.
                mb.set("ball_pos", json.dumps(ally_ball))


# ── Broker callbacks ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _other_robots, _robot_pos, _ball_pos

    if value is None:
        return

    if key == "other_robots":
        try:
            _other_robots = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass

    elif key == "robot_position":
        try:
            pos        = json.loads(value)
            _robot_pos = (float(pos["x"]), float(pos["y"]))
        except Exception:
            pass

    elif key == "ball_pos":
        try:
            _ball_pos = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for key in ("other_robots", "robot_position", "ball_pos"):
        try:
            val = mb.get(key)
            if val is not None:
                on_update(key, val)
        except Exception:
            pass

    mb.setcallback(["other_robots", "robot_position", "ball_pos"], on_update)
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    reader = _make_reader()
    reader.start(on_frame)

    _shutdown = threading.Event()
    try:
        _shutdown.wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[COOP] Stopped.")
        reader.stop()
        mb.close()
