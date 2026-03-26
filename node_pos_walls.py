from robus_core.libs.lib_telemtrybroker import TelemetryBroker
import json
import math
import numpy as np

# ── Line-detection parameters ─────────────────────────────────────────────────
LINE_TOL       = 0.02   # metres — max gap between adjacent sorted values to stay
                        #          in the same cluster (comparable to sensor noise)
WALL_THICKNESS = 0.05   # metres — max total spread of a cluster in the normal
                        #          direction; rejects curved/scattered surfaces
MIN_POINTS     = 20     # minimum lidar points for a cluster to be accepted
MIN_SPAN       = 0.1   # metres — minimum extent along the wall axis

# ── Histogram-based detection parameters ─────────────────────────────────────
HIST_BINS         = 250   # histogram resolution
HIST_MIN_COUNTS   = 8     # minimum bin count to qualify as a wall peak
HIST_MIN_PEAK_SEP = 0.15  # metres — minimum separation between two peaks on the same axis
# ─────────────────────────────────────────────────────────────────────────────

mb = TelemetryBroker()

_lidar       = {}   # angle_deg (int) → dist_mm (int)
_field_angle = None # explicitly set from broker; None = not present
_sim_heading = None # fallback from lidar simulation


def _effective_field_angle():
    """field_angle → sim_heading → 0."""
    if _field_angle is not None:
        return _field_angle
    if _sim_heading is not None:
        return _sim_heading
    return 0.0


def _cluster_1d(pairs, tolerance):
    """
    `pairs` is a list of (primary_value, secondary_value).
    Sort by primary, group consecutive entries where adjacent primaries differ
    by at most `tolerance`.
    Returns a list of groups, each group being a list of (primary, secondary) tuples.
    """
    if not pairs:
        return []
    pairs_sorted = sorted(pairs, key=lambda p: p[0])
    clusters = [[pairs_sorted[0]]]
    for prev, curr in zip(pairs_sorted, pairs_sorted[1:]):
        if curr[0] - prev[0] <= tolerance:
            clusters[-1].append(curr)
        else:
            clusters.append([curr])
    return clusters


def _detect_walls(pts):
    """
    pts: list of (x, y) in field-aligned metres.
    Returns a list of wall dicts: {"gradient": 0, "offset": y} for horizontal
    or {"gradient": null, "offset": x} for vertical walls.
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    walls = []

    # ── Horizontal walls  (y ≈ constant, spread in x) ─────────────────────────
    for cluster in _cluster_1d(list(zip(ys, xs)), LINE_TOL):
        if len(cluster) < MIN_POINTS:
            continue
        c_ys, c_xs = zip(*cluster)
        normal_spread = max(c_ys) - min(c_ys)
        if normal_spread > WALL_THICKNESS:          # too thick → not a flat wall
            continue
        if max(c_xs) - min(c_xs) < MIN_SPAN:
            continue
        c_ys_sorted = sorted(c_ys)
        offset = c_ys_sorted[len(c_ys_sorted) // 2]   # median — robust to stragglers
        walls.append({"gradient": 0, "offset": round(offset, 3)})
        print(f"  [WALLS] Horizontal  y ≈ {offset:+.3f} m"
              f"  pts={len(cluster)}  x-span={max(c_xs)-min(c_xs):.2f} m"
              f"  thickness={normal_spread:.3f} m")

    # ── Vertical walls  (x ≈ constant, spread in y) ──────────────────────────
    for cluster in _cluster_1d(list(zip(xs, ys)), LINE_TOL):
        if len(cluster) < MIN_POINTS:
            continue
        c_xs, c_ys = zip(*cluster)
        normal_spread = max(c_xs) - min(c_xs)
        if normal_spread > WALL_THICKNESS:          # too thick → not a flat wall
            continue
        if max(c_ys) - min(c_ys) < MIN_SPAN:
            continue
        c_xs_sorted = sorted(c_xs)
        offset = c_xs_sorted[len(c_xs_sorted) // 2]   # median
        walls.append({"gradient": None, "offset": round(offset, 3)})
        print(f"  [WALLS] Vertical    x ≈ {offset:+.3f} m"
              f"  pts={len(cluster)}  y-span={max(c_ys)-min(c_ys):.2f} m"
              f"  thickness={normal_spread:.3f} m")

    return walls


def _detect_walls_histogram(pts_field_aligned):
    """
    Histogram-based wall detection.

    pts_field_aligned: list of (x, y) in field-aligned robot-centred metres
                       (same coordinate system as _detect_walls).

    Builds a 1-D histogram along each axis and extracts the two strongest
    peaks, which correspond to the two opposing walls in that direction.
    Returns wall dicts in the same format as _detect_walls.
    """
    if len(pts_field_aligned) < MIN_POINTS:
        return []

    pts = np.array(pts_field_aligned, dtype=float)
    walls = []

    for axis, gradient in [(0, None), (1, 0)]:
        data = pts[:, axis]
        counts, bin_edges = np.histogram(data, bins=HIST_BINS)
        centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Peak 1: highest bin
        i1 = int(np.argmax(counts))
        if counts[i1] < HIST_MIN_COUNTS:
            continue
        p1 = float(centres[i1])

        # Peak 2: highest bin at least HIST_MIN_PEAK_SEP away from peak 1
        masked = counts.copy()
        masked[np.abs(centres - p1) < HIST_MIN_PEAK_SEP] = 0
        i2 = int(np.argmax(masked))
        if masked[i2] < HIST_MIN_COUNTS:
            continue
        p2 = float(centres[i2])

        label = "Vertical" if gradient is None else "Horizontal"
        print(f"  [WALLS-H] {label}  peaks={p1:+.3f}, {p2:+.3f} m")
        walls.append({"gradient": gradient, "offset": round(p1, 3)})
        walls.append({"gradient": gradient, "offset": round(p2, 3)})

    return walls


def on_update(key, value):
    global _lidar, _field_angle, _sim_heading

    if value is None:
        return

    if key == "field_angle":
        try:
            _field_angle = float(value)
        except (ValueError, TypeError):
            pass
        return

    if key == "sim_heading":
        try:
            _sim_heading = float(value)
        except (ValueError, TypeError):
            pass
        return

    if key == "lidar":
        try:
            raw = json.loads(value)
            _lidar = {int(k): int(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            return

        fa     = _effective_field_angle()
        fa_rad = math.radians(fa)
        source = ("field_angle" if _field_angle is not None
                  else "sim_heading" if _sim_heading is not None
                  else "default (0°)")
        print(f"[WALLS] field_angle={fa:.1f}° (from {source})  points: {len(_lidar)}")

        # Rotate all lidar points into the field coordinate frame:
        # field_direction = lidar_angle + field_angle
        pts = [
            (
                (dist_mm / 1000.0) * math.cos(math.radians(angle_deg) + fa_rad),
                (dist_mm / 1000.0) * math.sin(math.radians(angle_deg) + fa_rad),
            )
            for angle_deg, dist_mm in _lidar.items()
        ]

        walls = _detect_walls(pts)
        print(f"  [WALLS] {len(walls)} wall(s) detected")
        mb.set("lidar_walls", json.dumps(walls))

        walls_hist = _detect_walls_histogram(pts)
        print(f"  [WALLS-H] {len(walls_hist)} wall(s) detected")
        mb.set("lidar_walls_hist", json.dumps(walls_hist))


if __name__ == "__main__":
    # Seed state from broker so existing values are used immediately
    for key, store in [("field_angle", "_field_angle"), ("sim_heading", "_sim_heading")]:
        try:
            val = mb.get(key)
            if val is not None:
                globals()[store] = float(val)
        except Exception:
            pass

    mb.setcallback(["lidar", "field_angle", "sim_heading"], on_update)
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping wall detection node.")
        mb.close()
