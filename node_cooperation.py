import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import json
import serial
import threading
from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor

# ── Configuration ──────────────────────────────────────────────────────────────
PORT = "/dev/ttyUSB1"
BAUD = 115200
# ──────────────────────────────────────────────────────────────────────────────

# Keys expected in each incoming serial message, mapped to their broker keys.
_FIELD_KEYS = [
    "main_robot_pos",
    "other_pos_1",
    "other_pos_2",
    "other_pos_3",
    "ball_pos",
]

mb    = TelemetryBroker()
_perf = PerfMonitor("node_cooperation", broker=mb, print_every=100)


def _publish(data: dict):
    """Publish all recognised fields from *data* with the 'ally_' prefix."""
    for key in _FIELD_KEYS:
        if key in data:
            mb.set(f"ally_{key}", json.dumps(data[key]))


def _read_serial():
    """Blocking loop: open serial port, read newline-delimited JSON, publish."""
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print(f"[COOP] Serial opened on {PORT} at {BAUD} baud.")
    except serial.SerialException as e:
        print(f"[COOP] Could not open {PORT}: {e}")
        return

    buf = b""
    try:
        while True:
            chunk = ser.read(ser.in_waiting or 1)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                with _perf.measure("parse"):
                    try:
                        data = json.loads(line.decode("utf-8", errors="replace"))
                        _publish(data)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"[COOP] Parse error: {e} — line: {line[:80]}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("[COOP] Serial closed.")


if __name__ == "__main__":
    # Run broker receiver in daemon thread so callbacks stay live.
    threading.Thread(target=mb.receiver_loop, daemon=True,
                     name="broker-receiver").start()

    try:
        _read_serial()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[COOP] Stopped.")
        mb.close()
