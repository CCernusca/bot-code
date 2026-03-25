import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "robus-core"))

import time
import math
import json
from robus_core.libs.lib_telemtrybroker import TelemetryBroker

try:
    import board
    import busio
    import digitalio
    from adafruit_bno08x import (
        BNO_REPORT_ACCELEROMETER,
        BNO_REPORT_LINEAR_ACCELERATION,
        BNO_REPORT_GYROSCOPE,
        BNO_REPORT_ROTATION_VECTOR,
        PacketError,
    )
    from adafruit_bno08x.i2c import BNO08X_I2C
except ImportError as e:
    print(f"[IMU] Hardware libraries not available ({e}) — is this running on the robot? Exiting.")
    sys.exit(0)

# ── Configuration ──────────────────────────────────────────────────────────────
I2C_ADDRESS          = 0x4b
RESET_PIN            = board.D17
POLL_RATE            = 0.01   # seconds (100 Hz)

PUBLISH_ORIENTATION  = True   # roll, pitch, yaw (°) + quaternion  →  imu_orientation, imu_quaternion
PUBLISH_ACCELERATION = True   # linear acceleration (m/s², gravity removed)  →  imu_linear_accel
PUBLISH_ANGULAR_VEL  = True   # angular velocity (°/s)  →  imu_gyro
# ──────────────────────────────────────────────────────────────────────────────

# ── I2C baudrate check ─────────────────────────────────────────────────────────
def check_baudrate():
    try:
        with open("/sys/class/i2c-adapter/i2c-1/of_node/clock-frequency", "rb") as f:
            current_baud = int.from_bytes(f.read(), byteorder="big")
        print(f"[IMU] I2C baudrate: {current_baud} Hz")
        if current_baud > 50000:
            print("!" * 60)
            print("WARNING: I2C baudrate is too high for BNO08x.")
            print("Set 'dtparam=i2c_arm=on,i2c_arm_baudrate=40000' in /boot/config.txt")
            print("!" * 60)
    except Exception:
        print("[IMU] Could not read I2C baudrate, skipping check.")

# ── Hardware reset ─────────────────────────────────────────────────────────────
def reset_sensor(reset_pin):
    print("[IMU] Resetting sensor...")
    reset_pin.value = False
    time.sleep(0.2)
    reset_pin.value = True
    time.sleep(0.8)

# ── Sensor init ────────────────────────────────────────────────────────────────
def init_bno(i2c, reset_pin):
    reset_sensor(reset_pin)
    try:
        bno = BNO08X_I2C(i2c, address=I2C_ADDRESS)
        bno.enable_feature(BNO_REPORT_ACCELEROMETER)
        time.sleep(0.05)
        if PUBLISH_ORIENTATION:
            bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            time.sleep(0.05)
        if PUBLISH_ACCELERATION:
            bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
            time.sleep(0.05)
        if PUBLISH_ANGULAR_VEL:
            bno.enable_feature(BNO_REPORT_GYROSCOPE)
            time.sleep(0.05)
        print("[IMU] Sensor initialised.")
        return bno
    except Exception as e:
        print(f"[IMU] Init failed: {e}")
        return None

# ── Quaternion → Euler ─────────────────────────────────────────────────────────
def quaternion_to_euler(i, j, k, w):
    sinr_cosp = 2 * (w * i + j * k)
    cosr_cosp = 1 - 2 * (i * i + j * j)
    roll  = math.atan2(sinr_cosp, cosr_cosp)

    sinp  = 2 * (w * j - k * i)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

    siny_cosp = 2 * (w * k + i * j)
    cosy_cosp = 1 - 2 * (j * j + k * k)
    yaw_deg   = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    if yaw_deg < 0:
        yaw_deg += 360

    return math.degrees(roll), math.degrees(pitch), yaw_deg

# ── Main ───────────────────────────────────────────────────────────────────────
check_baudrate()

broker = TelemetryBroker()

reset_pin           = digitalio.DigitalInOut(RESET_PIN)
reset_pin.direction = digitalio.Direction.OUTPUT
reset_pin.value     = True

i2c    = busio.I2C(board.SCL, board.SDA)
sensor = init_bno(i2c, reset_pin)

print("[IMU] Starting orientation stream...")

while True:
    if sensor is None:
        sensor = init_bno(i2c, reset_pin)
        time.sleep(1)
        continue

    try:
        data = {}

        # Raw accelerometer (includes gravity) — always published
        accel = sensor.acceleration
        if accel is not None:
            data["imu_accel"] = json.dumps({
                "x": round(accel[0], 3),
                "y": round(accel[1], 3),
                "z": round(accel[2], 3),
            })

        # Orientation: roll / pitch / yaw + quaternion
        if PUBLISH_ORIENTATION:
            quat = sensor.quaternion
            if quat is not None:
                i, j, k, w = quat
                roll, pitch, yaw = quaternion_to_euler(i, j, k, w)
                data["imu_orientation"] = json.dumps({
                    "roll":  round(roll,  2),
                    "pitch": round(pitch, 2),
                    "yaw":   round(yaw,   2),
                })
                data["imu_quaternion"] = json.dumps({
                    "i": round(i, 4), "j": round(j, 4),
                    "k": round(k, 4), "w": round(w, 4),
                })

        # Linear acceleration (gravity removed)
        if PUBLISH_ACCELERATION:
            la = sensor.linear_acceleration
            if la is not None:
                data["imu_linear_accel"] = json.dumps({
                    "x": round(la[0], 3),
                    "y": round(la[1], 3),
                    "z": round(la[2], 3),
                })

        # Angular velocity
        if PUBLISH_ANGULAR_VEL:
            gyr = sensor.gyro
            if gyr is not None:
                data["imu_gyro"] = json.dumps({
                    "x": round(math.degrees(gyr[0]), 2),
                    "y": round(math.degrees(gyr[1]), 2),
                    "z": round(math.degrees(gyr[2]), 2),
                })

        if data:
            broker.setmulti(data)

        time.sleep(POLL_RATE)

    except KeyboardInterrupt:
        print("\n[IMU] Stopped.")
        broker.close()
        break
    except Exception as e:
        print(f"[IMU] {type(e).__name__}: {e} — reinitialising...")
        sensor = None
