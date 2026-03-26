import json
import math
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from robus_core.libs.lib_telemtrybroker import TelemetryBroker

# ── Field configuration ───────────────────────────────────────────────────────
FIELD_WIDTH  = 1.0   # metres
FIELD_HEIGHT = 2.0   # metres
ROBOT_RADIUS = 0.1   # metres
# ─────────────────────────────────────────────────────────────────────────────

mb = TelemetryBroker()

# ── Broker state ──────────────────────────────────────────────────────────────
_lidar        = {}    # {angle_deg (int): dist_mm (int)}  — sensor frame
_field_angle  = None  # float | None
_sim_heading  = None  # float | None
_robot_pos    = None  # (x, y) metres, field frame
_other_robots = []    # [[x, y], ...]  field frame
_corners      = []    # [[angle_deg, dist_mm], ...]  sensor frame
_wall_corners       = []   # [[x, y], ...]  robot-centred field-aligned metres
_walls              = []   # [{"gradient": 0|None, "offset": float}]  field frame
_positioning_corner = None # {"dx": float, "dy": float}  robot-centred field-aligned

_state_lock    = threading.Lock()
_needs_redraw  = threading.Event()
# ─────────────────────────────────────────────────────────────────────────────


def _effective_field_angle():
    if _field_angle is not None:
        return _field_angle
    if _sim_heading is not None:
        return _sim_heading
    return 0.0


# ── Matplotlib setup ──────────────────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(6, 9))
plt.tight_layout(pad=1.5)
plt.show(block=False)


def _lidar_to_field(angle_deg, dist_mm, fa_rad, origin=(0.0, 0.0)):
    """Convert a single sensor-frame reading to field-frame (x, y)."""
    d = dist_mm / 1000.0
    a = math.radians(angle_deg) + fa_rad
    return origin[0] + d * math.cos(a), origin[1] + d * math.sin(a)


def _redraw():
    ax.cla()

    fa          = _effective_field_angle()
    fa_rad      = math.radians(fa)
    known_pos   = _robot_pos is not None
    origin      = _robot_pos if known_pos else (0.0, 0.0)

    # ── Lidar points (rotated to field frame, shifted to robot position) ──────
    pts = None
    if _lidar:
        pts = np.array([
            _lidar_to_field(a, d, fa_rad, origin)
            for a, d in _lidar.items()
        ])
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=5, c='#222222', zorder=10, label='Lidar')

    # Centre point for the heading arrow: robot pos if known, else lidar centroid
    if known_pos:
        arrow_origin = origin
    elif pts is not None and len(pts):
        arrow_origin = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    else:
        arrow_origin = None

    # ── Detected walls ────────────────────────────────────────────────────────
    # Wall offsets are robot-centred; shift into field coordinates.
    for wall in _walls:
        off = wall["offset"]
        if wall["gradient"] == 0:          # horizontal: field_y = robot_y + offset
            field_off = origin[1] + off
            ax.axhline(field_off, color='steelblue', lw=1.5, ls='--', zorder=4)
        else:                              # vertical:   field_x = robot_x + offset
            field_off = origin[0] + off
            ax.axvline(field_off, color='steelblue', lw=1.5, ls='--', zorder=4)

    if known_pos:
        # ── Field boundary ────────────────────────────────────────────────────
        ax.add_patch(patches.Rectangle(
            (0, 0), FIELD_WIDTH, FIELD_HEIGHT,
            linewidth=2, edgecolor='#888888', facecolor='#f8f8f8', zorder=1,
        ))

        # ── Own robot ─────────────────────────────────────────────────────────
        rx, ry = origin
        ax.add_patch(patches.Circle(
            (rx, ry), ROBOT_RADIUS,
            linewidth=1.5, edgecolor='#2a7a2a', facecolor='#c8f0c8', zorder=7,
        ))

        # ── Other robots ──────────────────────────────────────────────────────
        for i, r in enumerate(_other_robots):
            ox, oy   = r[0], r[1]
            centroid = str(r[2]).startswith("centroid") if len(r) > 2 else False
            if centroid:
                ax.scatter(ox, oy, s=200, marker='X', c='yellow', edgecolors='#c07000',
                           linewidths=1.5, zorder=7)
            else:
                ax.add_patch(patches.Circle(
                    (ox, oy), ROBOT_RADIUS,
                    linewidth=1.5, edgecolor='#c07000', facecolor='#ffe0a0', zorder=7,
                ))
            ax.text(ox, oy + ROBOT_RADIUS + 0.03, str(i + 1),
                    ha='center', va='bottom', fontsize=8, color='#c07000', zorder=8)

    # ── Heading arrow (always drawn if an origin is available) ───────────────
    if arrow_origin is not None:
        ax_x, ax_y = arrow_origin
        arrow_len  = ROBOT_RADIUS * 1.8
        ax.annotate(
            "", xy=(ax_x + arrow_len * math.cos(fa_rad),
                    ax_y + arrow_len * math.sin(fa_rad)),
            xytext=(ax_x, ax_y),
            arrowprops=dict(arrowstyle='->', color='#2a7a2a', lw=2.0),
            zorder=8,
        )

    # ── Depth corners (sensor frame → field frame) ────────────────────────────
    if _corners:
        cpts = np.array([
            _lidar_to_field(a, d, fa_rad, origin)
            for a, d in _corners
        ])
        ax.scatter(cpts[:, 0], cpts[:, 1],
                   s=140, marker='X', c='red', edgecolors='black',
                   linewidths=0.8, zorder=6, label='Depth corners')

    # ── Wall corners (robot-centred field-aligned → field frame) ──────────────
    if _wall_corners:
        wcpts = np.array([[origin[0] + x, origin[1] + y] for x, y in _wall_corners])
        ax.scatter(wcpts[:, 0], wcpts[:, 1],
                   s=140, marker='X', c='steelblue', edgecolors='black',
                   linewidths=0.8, zorder=6, label='Wall corners')

    # ── Winning positioning corner (green, topmost) ────────────────────────────
    if _positioning_corner is not None:
        pcx = origin[0] + _positioning_corner["dx"]
        pcy = origin[1] + _positioning_corner["dy"]
        ax.scatter(pcx, pcy, s=200, marker='X', c='limegreen', edgecolors='black',
                   linewidths=1.0, zorder=11, label='Positioning corner')

    # ── Axes ──────────────────────────────────────────────────────────────────
    if known_pos:
        margin = 0.25
        ax.set_xlim(-margin, FIELD_WIDTH  + margin)
        ax.set_ylim(-margin, FIELD_HEIGHT + margin)
    elif arrow_origin is not None:
        cx, cy  = arrow_origin
        spread  = max(pts[:, 0].ptp(), pts[:, 1].ptp(), 0.5) / 2 + 0.3
        ax.set_xlim(cx - spread, cx + spread)
        ax.set_ylim(cy - spread, cy + spread)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    pos_str = f"({origin[0]:.2f}, {origin[1]:.2f})" if known_pos else "unknown"
    src     = ("field_angle" if _field_angle is not None
               else "sim_heading" if _sim_heading is not None
               else "default")
    ax.set_title(
        f"Twin Visualisation\n"
        f"pos={pos_str}  field_angle={fa:.1f}° ({src})"
        f"  walls={len(_walls)}  corners={len(_corners)}  bots={len(_other_robots)}"
    )
    ax.grid(True, alpha=0.25, zorder=0)

    fig.canvas.draw()
    fig.canvas.flush_events()


# ── Broker callbacks ──────────────────────────────────────────────────────────
def on_update(key, value):
    global _lidar, _field_angle, _sim_heading
    global _robot_pos, _other_robots, _corners, _wall_corners, _walls, _positioning_corner

    if value is None:
        return

    try:
        with _state_lock:
            if key == "lidar":
                raw = json.loads(value)
                _lidar = {int(k): int(v) for k, v in raw.items()}

            elif key == "field_angle":
                _field_angle = float(value)

            elif key == "sim_heading":
                _sim_heading = float(value)

            elif key == "robot_position":
                p = json.loads(value)
                _robot_pos = (float(p["x"]), float(p["y"]))

            elif key == "other_robots":
                _other_robots = [[float(r["x"]), float(r["y"]), r.get("method", "")]
                                  for r in json.loads(value)]

            elif key == "depth_corners":
                _corners = json.loads(value)

            elif key == "wall_corners":
                _wall_corners = json.loads(value)

            elif key == "positioning_corner":
                _positioning_corner = json.loads(value)

            elif key == "lidar_walls":
                _walls = json.loads(value)

    except Exception as e:
        print(f"[VIS] parse error on {key!r}: {e}")
        return

    _needs_redraw.set()


if __name__ == "__main__":
    # Seed existing broker values so the display is populated immediately
    _SEEDS = {
        "field_angle":    lambda v: float(v),
        "sim_heading":    lambda v: float(v),
        "lidar":          lambda v: {int(k): int(x) for k, x in json.loads(v).items()},
        "robot_position": lambda v: (float(json.loads(v)["x"]), float(json.loads(v)["y"])),
        "other_robots":   lambda v: [[float(r["x"]), float(r["y"]), r.get("method", "")] for r in json.loads(v)],
        "depth_corners":  lambda v: json.loads(v),
        "wall_corners":       lambda v: json.loads(v),
        "positioning_corner": lambda v: json.loads(v),
        "lidar_walls":    lambda v: json.loads(v),
    }
    _TARGETS = {
        "field_angle":    "_field_angle",
        "sim_heading":    "_sim_heading",
        "lidar":          "_lidar",
        "robot_position": "_robot_pos",
        "other_robots":   "_other_robots",
        "depth_corners":  "_corners",
        "wall_corners":       "_wall_corners",
        "positioning_corner": "_positioning_corner",
        "lidar_walls":    "_walls",
    }
    for key, parse in _SEEDS.items():
        try:
            val = mb.get(key)
            if val is not None:
                globals()[_TARGETS[key]] = parse(val)
        except Exception:
            pass

    _redraw()

    mb.setcallback(list(_SEEDS.keys()), on_update)
    threading.Thread(target=mb.receiver_loop, daemon=True, name="broker-receiver").start()

    try:
        while plt.fignum_exists(fig.number):
            if _needs_redraw.is_set():
                _needs_redraw.clear()
                with _state_lock:
                    _redraw()
            plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping twin visualisation.")
        plt.close(fig)
        mb.close()
