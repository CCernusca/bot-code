"""
Modular cooperation data readers.

To swap the transport layer, subclass BaseCooperationReader and implement
start() and stop(), then return an instance from _make_reader() in
node_cooperation.py.

Expected frame schema (one JSON object per newline for SerialCooperationReader):

    {
      "main_robot_pos": {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pos_1":    {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pos_2":    {"x": <float>, "y": <float>, "confidence": <float>},
      "other_pos_3":    {"x": <float>, "y": <float>, "confidence": <float>},
      "ball_pos":       {"x": <float>, "y": <float>, "confidence": <float>}
    }

All fields are optional; missing ones are simply ignored by the node.
"""

import json
import threading

try:
    import serial as _serial
    _serial_available = True
except ImportError:
    _serial_available = False


class BaseCooperationReader:
    """
    Abstract base for cooperation data readers.

    Subclasses must implement start(on_frame) and stop().
    on_frame(data: dict) is called from a background thread for each frame.
    """

    def start(self, on_frame):
        """Start reading.  Must return immediately; run I/O on a background thread."""
        raise NotImplementedError

    def stop(self):
        """Signal the reader to stop and release all resources."""
        raise NotImplementedError


class SerialCooperationReader(BaseCooperationReader):
    """Reads newline-delimited JSON frames from a serial port."""

    DEFAULT_PORT = "/dev/ttyUSB1"
    DEFAULT_BAUD = 115200

    def __init__(self, port=None, baud=None):
        if not _serial_available:
            raise ImportError("pyserial is required for SerialCooperationReader")
        self._port    = port or self.DEFAULT_PORT
        self._baud    = baud or self.DEFAULT_BAUD
        self._stop_ev = threading.Event()
        self._thread  = None

    def start(self, on_frame):
        self._stop_ev.clear()
        self._thread = threading.Thread(
            target=self._run, args=(on_frame,),
            daemon=True, name="coop-serial",
        )
        self._thread.start()

    def stop(self):
        self._stop_ev.set()

    def _run(self, on_frame):
        try:
            ser = _serial.Serial(self._port, self._baud, timeout=1)
            print(f"[COOP] Serial opened on {self._port} at {self._baud} baud.")
        except _serial.SerialException as e:
            print(f"[COOP] Could not open {self._port}: {e}")
            return

        buf = b""
        try:
            while not self._stop_ev.is_set():
                chunk = ser.read(ser.in_waiting or 1)
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8", errors="replace"))
                        on_frame(data)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"[COOP] Parse error: {e} — {line[:80]}")
        finally:
            ser.close()
            print("[COOP] Serial closed.")
