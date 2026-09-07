"""
servo_controller.py
===================
Pan-Tilt Visual Servoing Controller.

Drives 2-DoF pan/tilt camera mount servos to center on target bounding-box coordinates.
Features:
- Incremental PD control with deadband, velocity damping, and EMA error smoothing.
- Hardware-timed PWM via pigpio when run on Raspberry Pi.
- Automatic mock/simulation fallback on macOS/Windows/Linux without pigpiod.
- Boresight angle limits, calibration signs, and watchdog target-loss reset.
"""

import time
from typing import Optional, Tuple

try:
    import pigpio
    _PIGPIO_AVAILABLE = True
except ImportError:
    pigpio = None
    _PIGPIO_AVAILABLE = False


class AxisController:
    """
    Incremental PD controller.

    The command is an angle that accumulates corrections, so the plant itself
    integrates and a plain proportional term removes steady-state error.
    kd damps the overshoot that integration would otherwise cause.
    """

    def __init__(
        self,
        kp: float = 0.45,
        kd: float = 0.06,
        deadband_px: float = 8.0,
        max_step_deg: float = 2.0,
        ema: float = 0.4,
    ):
        self.kp = kp
        self.kd = kd
        self.deadband_px = deadband_px
        self.max_step_deg = max_step_deg
        self.ema = ema
        self.filt_err = 0.0
        self.prev_err = 0.0

    def reset(self):
        self.filt_err = 0.0
        self.prev_err = 0.0

    def step(self, err_px: float, deg_per_px: float, dt: float) -> float:
        """Returns the angle increment in degrees for this tick."""
        if abs(err_px) < self.deadband_px:
            err_px = 0.0

        self.filt_err += self.ema * (err_px - self.filt_err)
        err_deg = self.filt_err * deg_per_px

        derr = (err_deg - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err_deg

        delta = self.kp * err_deg + self.kd * derr
        return max(-self.max_step_deg, min(self.max_step_deg, delta))


class Servo:
    """Maps degrees to pulse width. Uses pigpio when available, or runs in mock mode."""

    def __init__(
        self,
        pi,
        gpio: int,
        min_us: float = 500.0,
        max_us: float = 2500.0,
        sweep_deg: float = 180.0,
        lo_deg: float = -90.0,
        hi_deg: float = 90.0,
        start_deg: float = 0.0,
    ):
        self.pi = pi
        self.gpio = gpio
        self.min_us = min_us
        self.max_us = max_us
        self.sweep = sweep_deg
        self.lo = lo_deg
        self.hi = hi_deg
        self.angle = 0.0
        self.write(start_deg)

    def write(self, deg: float) -> float:
        deg = max(self.lo, min(self.hi, deg))
        self.angle = deg
        us = (self.min_us + self.max_us) / 2.0 + deg * (self.max_us - self.min_us) / self.sweep
        if self.pi is not None:
            try:
                self.pi.set_servo_pulsewidth(self.gpio, int(us))
            except Exception:
                pass
        return deg

    def release(self):
        if self.pi is not None:
            try:
                self.pi.set_servo_pulsewidth(self.gpio, 0)
            except Exception:
                pass


class PanTiltServoing:
    """
    Image-based visual servoing controller coordinating Pan (azimuth) and Tilt (elevation).
    """

    def __init__(
        self,
        pan_gpio: int = 17,
        tilt_gpio: int = 27,
        hfov_deg: float = 62.2,
        vfov_deg: float = 48.8,
        pan_sign: int = -1,
        tilt_sign: int = 1,
        pan_min: float = -80.0,
        pan_max: float = 80.0,
        tilt_min: float = -45.0,
        tilt_max: float = 45.0,
        force_mock: bool = False,
    ):
        self.pan_gpio = pan_gpio
        self.tilt_gpio = tilt_gpio
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.pan_sign = pan_sign
        self.tilt_sign = tilt_sign
        self.detection_timeout = 0.5

        # Initialize GPIO / pigpio connection
        self.pi = None
        self.is_hardware = False

        if not force_mock and _PIGPIO_AVAILABLE:
            try:
                pi_conn = pigpio.pi()
                if not pi_conn.connected:
                    pi_conn.stop()
                    # If pigpiod daemon is not running, attempt to launch it
                    import subprocess
                    try:
                        subprocess.run(["sudo", "pigpiod"], check=False, capture_output=True, timeout=2)
                        time.sleep(0.4)
                        pi_conn = pigpio.pi()
                    except Exception:
                        pass

                if pi_conn.connected:
                    self.pi = pi_conn
                    self.is_hardware = True
                else:
                    pi_conn.stop()
            except Exception:
                self.pi = None
                self.is_hardware = False

        self.pan_servo = Servo(self.pi, pan_gpio, lo_deg=pan_min, hi_deg=pan_max, start_deg=0.0)
        self.tilt_servo = Servo(self.pi, tilt_gpio, lo_deg=tilt_min, hi_deg=tilt_max, start_deg=0.0)

        self.pan_ctl = AxisController(kp=0.45, kd=0.06, deadband_px=8.0, max_step_deg=2.0)
        self.tilt_ctl = AxisController(kp=0.45, kd=0.06, deadband_px=8.0, max_step_deg=1.5)

        self.pan_cmd = 0.0
        self.tilt_cmd = 0.0
        self.last_seen_time = 0.0
        self.enabled = True

    @property
    def is_mock(self) -> bool:
        return not self.is_hardware

    @property
    def pan_angle(self) -> float:
        return self.pan_servo.angle

    @property
    def tilt_angle(self) -> float:
        return self.tilt_servo.angle

    def center_servos(self):
        """Reset servos to 0 deg center."""
        self.pan_cmd = self.pan_servo.write(0.0)
        self.tilt_cmd = self.tilt_servo.write(0.0)
        self.pan_ctl.reset()
        self.tilt_ctl.reset()

    def update(
        self,
        center: Optional[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Updates the servo commands toward target bbox center.
        Returns:
            (pan_angle, tilt_angle)
        """
        t_now = time.monotonic()

        if not self.enabled:
            return self.pan_angle, self.tilt_angle

        if center is not None:
            self.last_seen_time = t_now
            cx, cy = center

            center_x = frame_w / 2.0
            center_y = frame_h / 2.0
            deg_per_px_x = self.hfov_deg / max(1, frame_w)
            deg_per_px_y = self.vfov_deg / max(1, frame_h)

            err_x = cx - center_x
            err_y = cy - center_y

            self.pan_cmd += self.pan_sign * self.pan_ctl.step(err_x, deg_per_px_x, dt)
            self.tilt_cmd += self.tilt_sign * self.tilt_ctl.step(err_y, deg_per_px_y, dt)

            self.pan_cmd = self.pan_servo.write(self.pan_cmd)
            self.tilt_cmd = self.tilt_servo.write(self.tilt_cmd)

        elif (t_now - self.last_seen_time) > self.detection_timeout:
            # Clear controller state so re-acquisition doesn't cause derivative kick
            self.pan_ctl.reset()
            self.tilt_ctl.reset()

        return self.pan_angle, self.tilt_angle

    def release(self):
        """Releases pulse signals and disconnects pigpio client."""
        self.pan_servo.release()
        self.tilt_servo.release()
        if self.pi is not None:
            try:
                self.pi.stop()
            except Exception:
                pass
            self.pi = None
