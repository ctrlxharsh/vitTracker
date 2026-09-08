"""
main.py - MicroPython ESP32-S3 Pan-Tilt Servo Controller
========================================================
Runs automatically on boot on the ESP32-S3.
Drives hardware PWM on:
  - Pan Servo:  GPIO 14 (50 Hz, 600-2400 us pulse width)
  - Tilt Servo: GPIO 13 (50 Hz, 1000-2000 us pulse width)
Receives commands over USB CDC Serial (115200 baud).

Features:
  - Constant, stable 50 Hz PWM (never changes frequency)
  - Smooth hardware slew-rate limiting (6 us / 15 ms)
  - Continuous center-holding watchdog (prevents camera collapse)
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

# Ensure pins are driven LOW before attaching PWM
Pin(PAN_PIN, Pin.OUT, value=0)
Pin(TILT_PIN, Pin.OUT, value=0)

# Constant 50 Hz PWM frequency for standard hobby servos (20ms period)
pan_pwm = PWM(Pin(PAN_PIN), freq=50, duty_ns=0)
tilt_pwm = PWM(Pin(TILT_PIN), freq=50, duty_ns=0)

try:
    led = Pin(LED_PIN, Pin.OUT)
    led.value(0)
except Exception:
    led = None

def set_servo_us(pwm_dev, us):
    if us <= 0:
        pwm_dev.duty_ns(0)
    else:
        pwm_dev.duty_ns(int(us * 1000))

# Target and current pulse widths for smooth slew rate limiting
target_pan_us = CENTER_US
target_tilt_us = CENTER_US
curr_pan_us = CENTER_US
curr_tilt_us = CENTER_US
servos_active = True

# Engage at center on boot
set_servo_us(pan_pwm, CENTER_US)
set_servo_us(tilt_pwm, CENTER_US)

print("OK: ESP32-S3 Pan-Tilt Controller Ready (Pan:14, Tilt:13)")

buf = ""
spoll = select.poll()
spoll.register(sys.stdin, select.POLLIN)
last_pkt_ms = time.ticks_ms()
last_slew_ms = time.ticks_ms()
MAX_US_STEP = 6  # 6us per 15ms (~400 us/s) for silky-smooth motion

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
                target_pan_us = CENTER_US
                target_tilt_us = CENTER_US
                if not servos_active:
                    servos_active = True
                    set_servo_us(pan_pwm, curr_pan_us)
                    set_servo_us(tilt_pwm, curr_tilt_us)
                last_pkt_ms = time.ticks_ms()
                print("ACK: CENTER")
                continue
            elif line in ("OFF", "DETACH", "STOP", "RELEASE"):
                servos_active = False
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
                    servos_active = False
                    set_servo_us(pan_pwm, 0)
                    set_servo_us(tilt_pwm, 0)
                    if led:
                        led.value(0)
                    print("ACK: OFF")
                    continue
                elif MIN_PULSE_US <= pan_val <= MAX_PULSE_US and MIN_PULSE_US <= tilt_val <= MAX_PULSE_US:
                    if not servos_active:
                        servos_active = True
                        set_servo_us(pan_pwm, curr_pan_us)
                        set_servo_us(tilt_pwm, curr_tilt_us)
                    target_pan_us = max(PAN_MIN_US, min(PAN_MAX_US, pan_val))
                    target_tilt_us = max(TILT_MIN_US, min(TILT_MAX_US, tilt_val))
                    last_pkt_ms = time.ticks_ms()
                    if led:
                        led.value(1)
        else:
            if len(buf) < 64:
                buf += ch

    # Smooth slew rate interpolation (~66 Hz)
    now_ms = time.ticks_ms()
    if servos_active and time.ticks_diff(now_ms, last_slew_ms) >= 15:
        last_slew_ms = now_ms
        if curr_pan_us != target_pan_us:
            diff_p = target_pan_us - curr_pan_us
            curr_pan_us += max(-MAX_US_STEP, min(MAX_US_STEP, diff_p))
            set_servo_us(pan_pwm, curr_pan_us)
        if curr_tilt_us != target_tilt_us:
            diff_t = target_tilt_us - curr_tilt_us
            curr_tilt_us += max(-MAX_US_STEP, min(MAX_US_STEP, diff_t))
            set_servo_us(tilt_pwm, curr_tilt_us)

    # Watchdog: return smoothly to center and keep holding position (never drop servos limp)
    if servos_active and time.ticks_diff(time.ticks_ms(), last_pkt_ms) > 3000:
        target_pan_us = CENTER_US
        target_tilt_us = CENTER_US
        if led:
            led.value(0)
