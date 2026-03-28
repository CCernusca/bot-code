"""
utils/perf_monitor.py — lightweight per-key timing for all nodes.

Set PERF_ENABLED = False to compile out all monitoring with zero overhead
(the .measure() context manager becomes a no-op _NullTimer).

Usage
-----
    from utils.perf_monitor import PerfMonitor
    _perf = PerfMonitor("node_name", broker=mb)

    def on_update(key, value):
        with _perf.measure(key):
            ...                     # code being timed

Stats are printed every PRINT_EVERY total measurements and published to the
broker key  perf_<node_name>  as JSON  {key: {avg_ms, peak_ms, n}, ...}.
"""

import time
import json
import collections

# ── Configuration ─────────────────────────────────────────────────────────────
PERF_ENABLED = True   # False → all .measure() calls are zero-cost no-ops
WINDOW       = 100    # rolling sample window per key
PRINT_EVERY  = 100    # print + publish every N total measurements
# ─────────────────────────────────────────────────────────────────────────────


class PerfMonitor:
    def __init__(self, node_name: str, broker=None,
                 window: int = WINDOW, print_every: int = PRINT_EVERY):
        self._name        = node_name
        self._broker      = broker        # TelemetryBroker instance, or None
        self._broker_key  = f"perf_{node_name}"
        self._window      = window
        self._print_every = print_every
        self._per_key: dict[str, collections.deque] = {}
        self._total       = 0

    def measure(self, key: str = ""):
        """Return a context manager that records elapsed time for *key*."""
        if not PERF_ENABLED:
            return _NullTimer()
        return _Timer(self, key)

    # ── internal ──────────────────────────────────────────────────────────────

    def _record(self, key: str, elapsed: float) -> None:
        if key not in self._per_key:
            self._per_key[key] = collections.deque(maxlen=self._window)
        self._per_key[key].append(elapsed)
        self._total += 1
        if self._total % self._print_every == 0:
            self._report()

    def _report(self) -> None:
        stats = {}
        parts = []
        for key in sorted(self._per_key):
            samples = self._per_key[key]
            if not samples:
                continue
            avg_ms  = sum(samples) / len(samples) * 1000
            peak_ms = max(samples) * 1000
            stats[key] = {
                "avg_ms":  round(avg_ms,  2),
                "peak_ms": round(peak_ms, 2),
                "n":       self._total,
            }
            parts.append(f"{key}: avg={avg_ms:.1f}ms peak={peak_ms:.1f}ms")

        body = "  |  ".join(parts) if parts else "(no data)"
        print(f"[PERF:{self._name}]  {body}  (n={self._total})")

        if self._broker is not None and stats:
            try:
                self._broker.set(self._broker_key, json.dumps(stats))
            except Exception:
                pass


class _NullTimer:
    """Zero-overhead stand-in used when PERF_ENABLED = False."""
    __slots__ = ()
    def __enter__(self): return self
    def __exit__(self, *_): pass


class _Timer:
    __slots__ = ("_monitor", "_key", "_t0")

    def __init__(self, monitor: PerfMonitor, key: str):
        self._monitor = monitor
        self._key     = key
        self._t0      = 0.0

    def __enter__(self):
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *_):
        self._monitor._record(self._key, time.monotonic() - self._t0)
