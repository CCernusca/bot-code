from robus_core.libs.lib_telemtrybroker import TelemetryBroker
from utils.perf_monitor import PerfMonitor
import json
import math
import time

# ── Camera specifications (Raspberry Pi Cam V2) ────────────────────────────────
FOV_DEG    = 62.2          # Horizontal field of view in degrees
RES_WIDTH  = 640           # Frame width in pixels
RES_HEIGHT = 480           # Frame height in pixels
CENTER_X   = RES_WIDTH / 2.0

# ── Colour filter (HSV) ────────────────────────────────────────────────────────
# Tune these values with a colour-picker / slider script for your lighting.
LOWER_ORANGE = (5,  120, 100)
UPPER_ORANGE = (25, 255, 255)

# ── Detection thresholds ───────────────────────────────────────────────────────
DEADZONE_PIXELS = 40   # Centre dead-zone width (pixels) before steering
MIN_RADIUS      = 5    # Minimum ball radius in pixels (filters noise)

# ── Ball physical size ─────────────────────────────────────────────────────────
BALL_RADIUS_MM = 21.0  # Real-world ball radius in mm (used for distance calc)

# ── Broker key ────────────────────────────────────────────────────────────────
BROKER_KEY = "ball"

# ─────────────────────────────────────────────────────────────────────────────

mb    = TelemetryBroker()
_perf = PerfMonitor("node_vision", broker=mb, print_every=100)

_hw_available = False
try:
    import cv2
    import numpy as np
    _hw_available = True
except ImportError as e:
    print(f"[VISION] OpenCV not available ({e}) — node will not run.")


def _process_frame(frame):
    """
    Detect the orange ball in a BGR frame.

    Returns a dict:
        command      – "FORWARD" | "LEFT" | "RIGHT" | "NO_BALL"
        distance_cm  – estimated distance in centimetres (0.0 if no ball)
        x_center     – detected pixel x-centre (-1 if no ball)
        radius       – detected pixel radius (-1 if no ball)
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array(LOWER_ORANGE),
                       np.array(UPPER_ORANGE))

    # Noise reduction: erode then dilate
    mask = cv2.erode(mask,  None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    largest = max(contours, key=cv2.contourArea)
    (x_center, _), radius = cv2.minEnclosingCircle(largest)

    if radius <= MIN_RADIUS:
        return {"command": "NO_BALL", "distance_cm": 0.0,
                "angle_deg": 0.0, "x_center": -1, "radius": -1}

    # ── Distance estimate via angular diameter ─────────────────────────────────
    # The ball subtends an angle of Ow_deg degrees in the image.
    # Using: distance = ball_radius / tan(half_angle)
    diameter_px = radius * 2.0
    Ow_deg = (FOV_DEG / RES_WIDTH) * diameter_px
    Ow_rad = math.radians(Ow_deg)
    distance_cm = (BALL_RADIUS_MM / math.tan(Ow_rad / 2.0)) / 10.0 if Ow_rad > 0 else 0.0

    # ── Angle to ball (horizontal) ─────────────────────────────────────────────
    # Map pixel offset from centre to degrees using the horizontal FOV.
    # Positive = ball is to the right, negative = ball is to the left.
    error_x   = x_center - CENTER_X
    angle_deg = (error_x / RES_WIDTH) * FOV_DEG

    # ── Steering command ───────────────────────────────────────────────────────
    if error_x < -DEADZONE_PIXELS:
        command = "LEFT"
    elif error_x > DEADZONE_PIXELS:
        command = "RIGHT"
    else:
        command = "FORWARD"

    return {
        "command":     command,
        "distance_cm": round(distance_cm, 1),
        "angle_deg":   round(angle_deg, 1),
        "x_center":    round(float(x_center), 1),
        "radius":      round(float(radius), 1),
    }


if __name__ == "__main__":
    if not _hw_available:
        raise SystemExit("[VISION] Cannot start: OpenCV is not installed.")

    print("[VISION] Starting headless vision system...")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RES_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_HEIGHT)

    if not cap.isOpened():
        raise SystemExit("[VISION] Could not open camera on device 0.")

    print(f"[VISION] Camera opened ({RES_WIDTH}x{RES_HEIGHT}). "
          "Running detection loop (Ctrl+C to stop)...")

    last_log_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[VISION] ERROR: Camera connection lost!")
                break

            with _perf.measure("frame"):
                result = _process_frame(frame)
                mb.set(BROKER_KEY, json.dumps(result))

            # Log to console at most once per second
            now = time.time()
            if now - last_log_time >= 1.0:
                if result["command"] == "NO_BALL":
                    print("[VISION] Status: NO BALL")
                else:
                    print(f"[VISION] Status: {result['command']} | "
                          f"Distance: {result['distance_cm']:.1f} cm | "
                          f"Angle: {result['angle_deg']:.1f} deg")
                last_log_time = now

    except KeyboardInterrupt:
        print("\n[VISION] Stopped by user.")
    finally:
        cap.release()
        mb.close()
        print("[VISION] Camera closed. System stopped.")
