from robus_core.libs.lib_telemtrybroker import TelemetryBroker
import json
import math
import time

# ── Field & detection configuration ──────────────────────────────────────────
FIELD_WIDTH      = 1.82    # metres, X axis
FIELD_HEIGHT     = 2.43    # metres, Y axis
ROBOT_RADIUS     = 0.09    # metres — assumed radius of all robots

# A lidar hit is classified as a wall hit if it falls within this distance of
# any field boundary.  Slightly smaller than ROBOT_RADIUS so that robots near
# walls can still be detected.
WALL_MARGIN      = 0.08   # metres

# Two interior points belong to the same robot if they are within this distance.
# Upper bound: at 1° resolution and 5 m range, adjacent hits on the same robot
# are at most ~8.7 cm apart.  Keep well below the robot diameter (18 cm) so
# that two separate robots with a small angular gap are not merged.
CLUSTER_DIST     = 0.10   # metres

# Clusters smaller than this are discarded as noise.
MIN_CLUSTER_SIZE = 2

# Attempt circle fitting only for clusters at least this large.
MIN_CIRCLE_PTS   = 3

# Iterations of the centre-convergence loop.
CIRCLE_FIT_ITERS = 8

# If the RMS deviation from ROBOT_RADIUS exceeds this, fall back to centroid.
MAX_FIT_ERROR    = 0.06   # metres

# Maximum extent of a valid robot cluster along either field axis.
# A robot of radius ROBOT_RADIUS subtends at most 2*ROBOT_RADIUS in any
# direction.  Wall fragments that leak past WALL_MARGIN are typically much
# longer in one axis, so this gate rejects them without affecting real robots.
MAX_CLUSTER_EXTENT = ROBOT_RADIUS * 3   # metres  (≈ 27 cm)

# ── Confidence & tracking ─────────────────────────────────────────────────────
# After scoring all raw detections, keep only this many (highest confidence).
MAX_ROBOTS       = 3

# Two detections whose centres are closer than this are considered overlapping;
# the less-confident one is discarded.
OVERLAP_DIST     = ROBOT_RADIUS * 2   # metres

# Minimum elapsed time between two detections before adding a history sample.
VEL_MIN_DT       = 0.05   # seconds

# Rolling history length for least-squares velocity fitting (per tracked robot).
VEL_HISTORY_N    = 10

# Minimum history samples required before the fitted velocity is trusted.
VEL_HISTORY_MIN  = 3

# Hard cap applied after fitting — guards against residual noise spikes.
MAX_ROBOT_SPEED  = 2.0    # m/s

# ─────────────────────────────────────────────────────────────────────────────

mb = TelemetryBroker()

_lidar        = {}    # {angle_deg (int): dist_mm (int)}
_robot_pos    = None  # (x, y) metres, in field frame
_field_angle  = None  # degrees — explicitly set; None = not set
_sim_heading  = None  # degrees — fallback from simulation

# ── Tracking state ────────────────────────────────────────────────────────────
_tracked = {}   # id → {"x","y","vx","vy","t","lost","history"}
_next_id = 1    # monotonically increasing ID counter


def _effective_field_angle():
    if _field_angle is not None:
        return _field_angle
    if _sim_heading is not None:
        return _sim_heading
    return 0.0


# ── Coordinate conversion ─────────────────────────────────────────────────────

def _is_wall_hit(x, y):
    """True if the point is close enough to a field boundary to be a wall hit."""
    return (
        x <= WALL_MARGIN or x >= FIELD_WIDTH  - WALL_MARGIN or
        y <= WALL_MARGIN or y >= FIELD_HEIGHT - WALL_MARGIN
    )


def _inside_field(x, y):
    return 0 <= x <= FIELD_WIDTH and 0 <= y <= FIELD_HEIGHT


# ── Angular-continuity clustering ─────────────────────────────────────────────

def _cluster_contiguous(sorted_scan):
    """
    Cluster lidar points by angular continuity.

    sorted_scan: [(angle_deg, x, y, is_interior), ...] sorted by sensor angle.

    A cluster is broken whenever:
      - a non-interior point (wall hit or out-of-field) is encountered, OR
      - the Cartesian gap to the previous interior point exceeds CLUSTER_DIST.

    Returns a list of [(x, y)] clusters.
    """
    clusters = []
    current  = []

    for _, x, y, is_interior in sorted_scan:
        if not is_interior:
            # Wall hit / out-of-field — end the current run
            if current:
                clusters.append(current)
                current = []
        else:
            if current:
                lx, ly = current[-1]
                if math.hypot(x - lx, y - ly) > CLUSTER_DIST:
                    # Spatial gap too large — start a fresh cluster
                    clusters.append(current)
                    current = []
            current.append((x, y))

    if current:
        clusters.append(current)

    return clusters


# ── Circle fitting ────────────────────────────────────────────────────────────

def _fit_center(pts, rx=0.0, ry=0.0):
    """
    Estimate the centre of the robot given a cluster of lidar hit points.

    For clusters of MIN_CIRCLE_PTS or more:
      Iterative convergence — each point P estimates the centre as
          C_i = P - R * normalise(P - O_prev)
      The new estimate is the mean of all C_i.  Converges to the
      least-squares circle centre for a circle of known radius ROBOT_RADIUS.
      Falls back to centroid if the RMS fit error exceeds MAX_FIT_ERROR.

    For smaller clusters: centroid of the points.

    rx, ry: position of the observing robot (used to seed a better initial
    estimate — the centroid of a near-side arc sits on the robot surface,
    so pushing it outward by ROBOT_RADIUS gives a near-correct starting point
    and avoids the dist≈0 degenerate case).

    Returns (x, y, method_str).
    """
    n = len(pts)
    cx = sum(p[0] for p in pts) / n
    cy = sum(p[1] for p in pts) / n

    if n < MIN_CIRCLE_PTS:
        return cx, cy, "centroid"

    # Seed: move the arc centroid away from the observing robot by ROBOT_RADIUS.
    # For a narrow arc this lands near the true circle centre; for a wide arc
    # it is much closer to the true centre than the raw centroid.
    ddx, ddy = cx - rx, cy - ry
    dd = math.hypot(ddx, ddy)
    if dd > 1e-9:
        ox = cx + (ddx / dd) * ROBOT_RADIUS
        oy = cy + (ddy / dd) * ROBOT_RADIUS
    else:
        ox, oy = cx, cy

    # Iterative circle-centre convergence
    for _ in range(CIRCLE_FIT_ITERS):
        sx, sy = 0.0, 0.0
        for px, py in pts:
            dx, dy = px - ox, py - oy
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                sx += px
                sy += py
            else:
                scale = ROBOT_RADIUS / dist
                sx += px - dx * scale   # = P - R * normalise(P - O)
                sy += py - dy * scale
        ox, oy = sx / n, sy / n

    # Validate: RMS of |P - O| - R should be small
    rms = math.sqrt(
        sum((math.hypot(px - ox, py - oy) - ROBOT_RADIUS) ** 2 for px, py in pts) / n
    )
    if rms > MAX_FIT_ERROR:
        return cx, cy, f"centroid (fit rms={rms:.3f})"

    return ox, oy, f"circle (rms={rms:.3f})"


# ── Physics-aware position prediction ────────────────────────────────────────

_MAX_PRED_STEPS = 20   # hard cap — prevents O(dt) blow-up for long-lost robots
_MAX_PRED_DT    = 2.0  # seconds — prediction accuracy degrades beyond this anyway


def _predict_with_bounce(x, y, vx, vy, dt):
    """
    Extrapolate (x, y) forward by at most _MAX_PRED_DT seconds, using at most
    _MAX_PRED_STEPS sub-steps with wall reflection.  Both caps are O(1) so
    the cost is constant regardless of how long a robot has been undetected.
    """
    dt   = min(dt, _MAX_PRED_DT)
    n    = max(1, min(int(dt / 0.02) + 1, _MAX_PRED_STEPS))
    step = dt / n
    for _ in range(n):
        x += vx * step
        y += vy * step
        if x < ROBOT_RADIUS:
            x = ROBOT_RADIUS;  vx = abs(vx)
        elif x > FIELD_WIDTH - ROBOT_RADIUS:
            x = FIELD_WIDTH - ROBOT_RADIUS;  vx = -abs(vx)
        if y < ROBOT_RADIUS:
            y = ROBOT_RADIUS;  vy = abs(vy)
        elif y > FIELD_HEIGHT - ROBOT_RADIUS:
            y = FIELD_HEIGHT - ROBOT_RADIUS;  vy = -abs(vy)
    return x, y


# ── Overlap filtering ─────────────────────────────────────────────────────────

def _filter_overlapping(robots):
    """
    Given robots sorted by confidence (descending), always discard any robot
    whose centre falls within OVERLAP_DIST of a higher-confidence robot already kept.
    """
    kept = []
    for r in robots:
        if not any(math.hypot(r["x"] - k["x"], r["y"] - k["y"]) < OVERLAP_DIST
                   for k in kept):
            kept.append(r)
    return kept


# ── Confidence scoring ────────────────────────────────────────────────────────

def _confidence_score(pts_count, method):
    """
    Return a scalar confidence for one detection.

    Circle fit with low RMS → high score.
    Centroid fallback (bad fit) → moderate penalty.
    Centroid (too few points for fitting) → larger penalty.
    """
    if method.startswith("circle"):
        try:
            rms = float(method.split("rms=")[1].rstrip(")"))
        except Exception:
            rms = MAX_FIT_ERROR / 2
        quality = 1.0 - min(rms, MAX_FIT_ERROR) / MAX_FIT_ERROR
        return round(pts_count * quality, 3)
    elif "fit rms=" in method:           # centroid because circle fit was poor
        return round(pts_count * 0.2, 3)
    else:                                # centroid because cluster was too small
        return round(pts_count * 0.4, 3)


# ── Velocity fitting ─────────────────────────────────────────────────────────

def _fit_velocity(history):
    """
    Ordinary least-squares linear fit over a history of (t, rx, ry) tuples
    where rx/ry are positions relative to the main robot.

    Returns (vx_rel, vy_rel) in m/s.  Each axis is solved independently:
        pos = v * t + b   →   v = (n·Σtx − Σt·Σx) / (n·Σt² − (Σt)²)
    """
    n = len(history)
    if n < 2:
        return 0.0, 0.0

    t0     = history[0][0]
    ts     = [h[0] - t0 for h in history]
    xs     = [h[1]       for h in history]
    ys     = [h[2]       for h in history]

    sum_t  = sum(ts)
    sum_t2 = sum(t * t for t in ts)
    denom  = n * sum_t2 - sum_t ** 2
    if abs(denom) < 1e-9:
        return 0.0, 0.0

    sum_tx = sum(t * x for t, x in zip(ts, xs))
    sum_ty = sum(t * y for t, y in zip(ts, ys))
    vx = (n * sum_tx - sum_t * sum(xs)) / denom
    vy = (n * sum_ty - sum_t * sum(ys)) / denom
    return vx, vy


# ── ID tracking ───────────────────────────────────────────────────────────────

def _match_and_track(detections, now):
    """
    Assign persistent IDs to `detections` and fill in predicted positions for
    any tracked robots that were not detected this frame.

    Matching uses the globally best-fit pairs (greedy by distance, no cutoff).
    Unmatched tracked robots within MAX_LOST_FRAMES are appended to the result
    at their velocity-extrapolated position with method="predicted".

    detections: [{"x","y","pts","method","confidence"}, ...]  (already filtered)
    now:        current monotonic timestamp (seconds)

    Returns the combined list (matched detections + predicted ghosts).
    """
    global _tracked, _next_id

    # ── Predict current positions for every tracked robot ─────────────────────
    predictions = {}          # tid → (px, py)
    for tid, tr in _tracked.items():
        dt = now - tr["t"]
        predictions[tid] = _predict_with_bounce(
            tr["x"], tr["y"], tr["vx"], tr["vy"], dt,
        )

    # ── Greedy best-fit matching — no distance cutoff ──────────────────────────
    matched_det   = [None] * len(detections)
    matched_track = set()

    pairs = sorted(
        (math.hypot(det["x"] - px, det["y"] - py), di, tid)
        for di, det in enumerate(detections)
        for tid, (px, py) in predictions.items()
    )
    for _, di, tid in pairs:
        if matched_det[di] is None and tid not in matched_track:
            matched_det[di] = tid
            matched_track.add(tid)

    # ── Force-assign remaining detections to nearest unmatched tracked robot ──
    # New IDs are only minted when tracked robots are exhausted and we are still
    # below MAX_ROBOTS (i.e. not all robots have been seen for the first time).
    for di, det in enumerate(detections):
        if matched_det[di] is not None:
            continue
        remaining = [(tid, px, py) for tid, (px, py) in predictions.items()
                     if tid not in matched_track]
        if remaining:
            best_tid = min(remaining,
                           key=lambda t: math.hypot(det["x"] - t[1], det["y"] - t[2]))[0]
            matched_det[di] = best_tid
            matched_track.add(best_tid)
        elif len(_tracked) < MAX_ROBOTS:
            pass   # genuinely new robot — will receive a fresh ID below

    # ── Update / create tracked entries for each detection ────────────────────
    new_tracked = {}

    for di, det in enumerate(detections):
        tid = matched_det[di]

        if tid is not None:
            old     = _tracked[tid]
            dt      = now - old["t"]
            history = old.get("history", [])

            if dt >= VEL_MIN_DT:
                # History stores absolute field-frame positions; main-robot
                # drift is neutralised upstream by the EMA filter on _robot_pos.
                history = history + [(now, det["x"], det["y"])]
                history = history[-VEL_HISTORY_N:]

            if len(history) >= VEL_HISTORY_MIN:
                new_vx, new_vy = _fit_velocity(history)
            else:
                new_vx, new_vy = old["vx"], old["vy"]

            # Clamp to maximum plausible speed
            spd = math.hypot(new_vx, new_vy)
            if spd > MAX_ROBOT_SPEED:
                new_vx *= MAX_ROBOT_SPEED / spd
                new_vy *= MAX_ROBOT_SPEED / spd

            new_tracked[tid] = {
                "x": det["x"], "y": det["y"], "t": now,
                "vx": new_vx, "vy": new_vy,
                "lost": 0, "history": history,
            }
        else:
            # Only reached when tracked is empty or below MAX_ROBOTS
            tid = _next_id
            _next_id += 1
            new_tracked[tid] = {
                "x": det["x"], "y": det["y"], "t": now,
                "vx": 0.0, "vy": 0.0,
                "lost": 0, "history": [(now, det["x"], det["y"])],
            }

        det["id"] = tid
        det["vx"] = round(new_tracked[tid]["vx"], 3)
        det["vy"] = round(new_tracked[tid]["vy"], 3)

    result = list(detections)

    # ── Carry forward unmatched tracked robots ────────────────────────────────
    # Tracked entries are NEVER dropped so IDs survive long absences.
    # _predict_with_bounce is O(1) (capped steps) so this is safe even for
    # robots that have been undetected for a long time.
    for tid, tr in _tracked.items():
        if tid in matched_track:
            continue
        tr["lost"] += 1
        new_tracked[tid] = tr              # always keep the entry
        dt = now - tr["t"]
        px, py = _predict_with_bounce(tr["x"], tr["y"], tr["vx"], tr["vy"], dt)
        result.append({
            "x": round(px, 3), "y": round(py, 3),
            "pts": 0, "method": "predicted",
            "confidence": 0.0,
            "id": tid,
            "vx": round(tr["vx"], 3),
            "vy": round(tr["vy"], 3),
        })

    _tracked = new_tracked
    return result


# ── Main detection ────────────────────────────────────────────────────────────

def _detect_robots():
    """
    Returns (robots, origin) where origin is the coordinate-frame snapshot
    used for this detection cycle — embedded in the published payload so the
    vis can apply the identical transform to the raw lidar, eliminating the
    inter-process race that otherwise causes detected circles to appear shifted
    relative to the displayed scatter.
    """
    if not _lidar or _robot_pos is None:
        return [], None

    now    = time.monotonic()
    rx, ry = _robot_pos
    fa_rad = math.radians(_effective_field_angle())

    # Build angularly-sorted scan with field coordinates and interior flag
    sorted_scan = []
    for angle_deg in sorted(_lidar):
        dist_mm = _lidar[angle_deg]
        d = dist_mm / 1000.0
        a = math.radians(angle_deg) + fa_rad
        x = rx + d * math.cos(a)
        y = ry + d * math.sin(a)
        interior = _inside_field(x, y) and not _is_wall_hit(x, y)
        sorted_scan.append((angle_deg, x, y, interior))

    clusters = [
        c for c in _cluster_contiguous(sorted_scan)
        if len(c) >= MIN_CLUSTER_SIZE
    ]

    # ── Fit, score, sort, filter overlaps, keep top MAX_ROBOTS ───────────────
    robots = []
    for cluster in clusters:
        # Physical size gate: reject clusters whose bounding box along either
        # field axis is larger than a robot could produce.  Wall fragments that
        # leak past WALL_MARGIN are typically elongated along one axis and are
        # caught here before any fitting or scoring is attempted.
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        if max(xs) - min(xs) > MAX_CLUSTER_EXTENT or \
           max(ys) - min(ys) > MAX_CLUSTER_EXTENT:
            continue

        ox, oy, method = _fit_center(cluster, rx, ry)
        confidence = _confidence_score(len(cluster), method)
        robots.append({"x": round(ox, 3), "y": round(oy, 3),
                       "pts": len(cluster), "method": method,
                       "confidence": confidence})

    robots.sort(key=lambda r: r["confidence"], reverse=True)
    robots = _filter_overlapping(robots)[:MAX_ROBOTS]

    # ── Assign / update persistent IDs (returns detections + predicted ghosts) ─
    robots = _match_and_track(robots, now)

    for r in robots:
        print(f"  [ROBOTS] id={r['id']}  {r['pts']:2d} pts"
              f"  ({r['x']:.3f}, {r['y']:.3f})"
              f"  conf={r['confidence']:.2f}  [{r['method']}]"
              f"  v=({r['vx']:.2f}, {r['vy']:.2f})")

    # Only the position is included — heading (sim_heading) is published
    # per-scan and is always current in both processes; bundling a potentially
    # stale heading here would cause spurious rotation errors in the vis.
    origin = {"x": round(rx, 4), "y": round(ry, 4)}
    return robots, origin


# ── Broker interface ──────────────────────────────────────────────────────────

def on_update(key, value):
    global _lidar, _robot_pos, _field_angle, _sim_heading

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

    elif key == "field_angle":
        try:
            _field_angle = float(value)
        except (ValueError, TypeError):
            return

    elif key == "sim_heading":
        try:
            _sim_heading = float(value)
        except (ValueError, TypeError):
            return

    # Only re-detect on new lidar data; position/angle updates just refresh state.
    if key == "lidar":
        robots, origin = _detect_robots()
        # Bundle origin with robots so the vis can use the same coordinate
        # snapshot without a second broker write (avoids I/O contention).
        mb.set("other_robots", json.dumps({"origin": origin, "robots": robots}))


if __name__ == "__main__":
    for attr, broker_key in [("_field_angle", "field_angle"),
                              ("_sim_heading", "sim_heading")]:
        try:
            val = mb.get(broker_key)
            if val is not None:
                globals()[attr] = float(val)
        except Exception:
            pass
    try:
        raw = mb.get("robot_position")
        if raw:
            pos = json.loads(raw)
            _robot_pos = (float(pos["x"]), float(pos["y"]))
    except Exception:
        pass

    mb.setcallback(
        ["lidar", "robot_position", "field_angle", "sim_heading"],
        on_update,
    )
    try:
        mb.receiver_loop()
    except KeyboardInterrupt:
        print("\nStopping robot detection.")
        mb.close()
