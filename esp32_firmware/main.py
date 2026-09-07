"""
main.py - MicroPython ESP32-S3 Pan-Tilt Servo Controller
========================================================
Runs automatically on boot on the ESP32-S3.
Drives hardware PWM on:
  - Pan Servo:  GPIO 14 (50 Hz, 500-2500 us pulse width)
  - Tilt Servo: GPIO 13 (50 Hz, 500-2500 us pulse width)
Receives commands over USB CDC Serial (115200 baud).
"""

import sys
import select
import time
from machine import Pin, PWM

# Pin Configuration
PAN_PIN = 14
TILT_PIN = 13
LED_PIN = 2

# Microsecond limits
MIN_PULSE_US = 500
MAX_PULSE_US = 2500
CENTER_US = 1500

PAN_MIN_US = 600    # ~ -80 deg
PAN_MAX_US = 2400   # ~ +80 deg
TILT_MIN_US = 1000  # ~ -45 deg
TILT_MAX_US = 2000  # ~ +45 deg

# Setup hardware PWM at 50Hz (20ms period = 20,000,000 ns)
pan_pwm = PWM(Pin(PAN_PIN), freq=50)
tilt_pwm = PWM(Pin(TILT_PIN), freq=50)

def set_servo_us(pwm_dev, us):
    # Setting duty_ns(0) cuts PWM pulses completely, relaxing servo holding torque
    if us <= 0:
        pwm_dev.duty_ns(0)
    else:
        pwm_dev.duty_ns(int(us * 1000))

# Recenter servos on boot
set_servo_us(pan_pwm, CENTER_US)
set_servo_us(tilt_pwm, CENTER_US)

try:
    led = Pin(LED_PIN, Pin.OUT)
    led.value(0)
except Exception:
    led = None

print("OK: ESP32-S3 Pan-Tilt Controller Ready (Pan:14, Tilt:13)")

buf = ""
spoll = select.poll()
spoll.register(sys.stdin, select.POLLIN)
last_pkt_ms = time.ticks_ms()

while True:
    events = spoll.poll(10)
    if events:
        ch = sys.stdin.read(1)
        if not ch:
            continue
        if ch in ('\n', '\r'):
            line = buf.strip()
            buf = ""
            if not line:
                continue

            if line == "PING":
                print("PONG")
                continue
            elif line in ("CENTER", "RESET"):
                set_servo_us(pan_pwm, CENTER_US)
                set_servo_us(tilt_pwm, CENTER_US)
                print("ACK: CENTER")
                continue
            elif line in ("OFF", "DETACH", "STOP", "RELEASE"):
                set_servo_us(pan_pwm, 0)
                set_servo_us(tilt_pwm, 0)
                if led:
                    led.value(0)
                print("ACK: OFF")
                continue

            pan_val = None
            tilt_val = None
            if line.startswith("P:") and " T:" in line:
                try:
                    p_str, t_str = line.split(" T:")
                    pan_val = int(p_str[2:])
                    tilt_val = int(t_str)
                except Exception:
                    pass
            elif "," in line:
                try:
                    p_str, t_str = line.split(",")
                    pan_val = int(p_str)
                    tilt_val = int(t_str)
                except Exception:
                    pass

            if pan_val is not None and tilt_val is not None:
                if pan_val == 0 and tilt_val == 0:
                    set_servo_us(pan_pwm, 0)
                    set_servo_us(tilt_pwm, 0)
                    if led:
                        led.value(0)
                    print("ACK: OFF")
                    continue
                elif MIN_PULSE_US <= pan_val <= MAX_PULSE_US and MIN_PULSE_US <= tilt_val <= MAX_PULSE_US:
                    p_clamped = max(PAN_MIN_US, min(PAN_MAX_US, pan_val))
                    t_clamped = max(TILT_MIN_US, min(TILT_MAX_US, tilt_val))
                    set_servo_us(pan_pwm, p_clamped)
                    set_servo_us(tilt_pwm, t_clamped)
                    last_pkt_ms = time.ticks_ms()
                    if led:
                        led.value(1)
        else:
            if len(buf) < 64:
                buf += ch

    # Watchdog: relax servos (cut PWM) and turn off LED if no packets for 3 seconds
    if time.ticks_diff(time.ticks_ms(), last_pkt_ms) > 3000:
        set_servo_us(pan_pwm, 0)
        set_servo_us(tilt_pwm, 0)
        if led:
            led.value(0)
