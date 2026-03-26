import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def simple_corners(points, window=3, proximity=0.1):
	"""
	Returns (x, y) coordinates that are local depth maxima over `window`
	neighbours on each side, where every individual neighbour is within
	`proximity` metres of the candidate point.

	Fully vectorised — no Python loop over points.
	"""
	if len(points) < 2 * window + 1:
		return []

	pts    = np.array(points, dtype=float)                 # (N, 2)
	depths = np.hypot(pts[:, 0], pts[:, 1])                # (N,)
	w2     = 2 * window + 1

	# ── Depth check ───────────────────────────────────────────────────────────
	# sliding_window_view(depths, w2) → (M, w2),  M = N - 2*window
	d_wins   = sliding_window_view(depths, w2)              # (M, w2)
	center_d = d_wins[:, window]                            # (M,)
	other_d  = np.delete(d_wins, window, axis=1)            # (M, 2w)
	depth_ok = np.all(center_d[:, None] > other_d, axis=1) # (M,)

	# ── Per-neighbour proximity check ─────────────────────────────────────────
	# sliding_window_view(pts, (w2, 2)) → (M, 1, w2, 2); squeeze the size-1 axis
	p_wins    = sliding_window_view(pts, (w2, 2))[:, 0]    # (M, w2, 2)
	centers   = p_wins[:, window, :]                        # (M, 2)
	neighbors = np.delete(p_wins, window, axis=1)           # (M, 2w, 2)
	diffs     = centers[:, None, :] - neighbors             # (M, 2w, 2)
	nbr_dists = np.hypot(diffs[:, :, 0], diffs[:, :, 1])   # (M, 2w)
	prox_ok   = np.all(nbr_dists < proximity, axis=1)       # (M,)

	return [tuple(p) for p in centers[depth_ok & prox_ok]]


def intersection_corners(walls):
	"""
	Returns all intersection points between horizontal and vertical walls.

	walls: list of {"gradient": 0|None, "offset": float}
	  gradient == 0    → horizontal wall at y = offset  (robot-centred)
	  gradient == None → vertical   wall at x = offset  (robot-centred)

	Returns a list of (x, y) tuples, one per H×V pair.
	"""
	horizontals = [w["offset"] for w in walls if w.get("gradient") == 0]
	verticals   = [w["offset"] for w in walls if w.get("gradient") is None]
	return [(vx, hy) for hy in horizontals for vx in verticals]
