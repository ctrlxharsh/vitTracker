"""
servo_controller.py
===================
ESP32 Serial Pan-Tilt Visual Servoing Controller.

Drives 2-DoF pan/tilt camera mount servos by streaming hardware PWM pulse widths
(microseconds) over USB Serial to a NodeMCU ESP32 microcontroller running `esp32_firmware.ino`.

Features:
- Incremental PD control with deadband, velocity damping, and EMA error smoothing.
- High-speed serial communication (115200 baud) with automatic port detection.
- Seamless mock/simulation fallback when ESP32 is not plugged in.
- Safe travel clamping (prevents servo stall/binding) and soft center watchdog.
"""

import time
from typing import List, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    _SERIAL_AVAILABLE = False


def deg_to_us(deg: float, sweep_deg: float = 180.0, min_us: float = 500.0, max_us: float = 2500.0) -> int:
    """Converts angle in degrees to pulse width in microseconds."""
    center_us = (min_us + max_us) / 2.0
    us = center_us + deg * ((max_us - min_us) / sweep_deg)
    return int(round(max(min_us, min(max_us, us))))


def us_to_deg(us: float, sweep_deg: float = 180.0, min_us: float = 500.0, max_us: float = 2500.0) -> float:
    """Converts pulse width in microseconds to angle in degrees."""
    center_us = (min_us + max_us) / 2.0
    return (us - center_us) * (sweep_deg / (max_us - min_us))


def find_esp32_port(preferred: Optional[str] = None) -> Optional[str]:
    """
    Scans system serial ports to find a connected NodeMCU ESP32 or USB-UART adapter.
    """
    if not _SERIAL_AVAILABLE or serial is None:
        return None

    try:
        available_ports = list(serial.tools.list_ports.comports())
    except Exception:
        return None

    if not available_ports:
        return None

    # If preferred port specified and present, use it
    if preferred:
        for p in available_ports:
            if p.device == preferred:
                return p.device

    # Look for known USB-serial adapters typically used by ESP32 boards
    # (CP210x, CH340, FTDI, WCH, Espressif native USB-JTAG/serial)
    keywords = ["usbserial", "wchusbserial", "slab_usbtouart", "usbmodem", "cp210", "ch340", "ftdi", "espressif", "uart"]

    for p in available_ports:
        dev_lower = (p.device or "").lower()
        desc_lower = (p.description or "").lower()
        hwid_lower = (p.hwid or "").lower()

        # Prioritize matching device name
        for kw in keywords:
            if kw in dev_lower:
                return p.device

        # Check description or hardware ID
        for kw in keywords:
            if kw in desc_lower or kw in hwid_lower:
                return p.device

    # Fallback: On Linux/Pi, pick first ttyUSB or ttyACM
    for p in available_ports:
        if "ttyusb" in p.device.lower() or "ttyacm" in p.device.lower():
            return p.device

    return None


class AxisController:
    """
    Incremental PD controller with exponential moving average (EMA) smoothing and deadband.
    """

    def __init__(
        self,
        kp: float = 0.70,
        kd: float = 0.08,
        deadband_px: float = 10.0,
        max_step_deg: float = 10.0,
        ema: float = 0.75,
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
        """Returns angle increment in degrees for this update tick."""
        if abs(err_px) < self.deadband_px:
            self.filt_err = 0.0
            self.prev_err = 0.0
            return 0.0

        self.filt_err += self.ema * (err_px - self.filt_err)
        err_deg = self.filt_err * deg_per_px

        derr = (err_deg - self.prev_err) / dt if dt > 0 else 0.0
        self.prev_err = err_deg

        delta = self.kp * err_deg + self.kd * derr
        return max(-self.max_step_deg, min(self.max_step_deg, delta))


class Servo:
    """Logical model of a servo axis tracking angle and microsecond pulse width."""

    def __init__(
        self,
        min_us: float = 500.0,
        max_us: float = 2500.0,
        sweep_deg: float = 180.0,
        lo_deg: float = -90.0,
        hi_deg: float = 90.0,
        start_deg: float = 0.0,
        **kwargs,
    ):
        self.min_us = min_us
        self.max_us = max_us
        self.sweep = sweep_deg
        self.lo = lo_deg
        self.hi = hi_deg
        self.angle = 0.0
        self.pulse_us = 1500
        self.write(start_deg)

    def write(self, deg: float) -> float:
        deg = max(self.lo, min(self.hi, deg))
        self.angle = deg
        self.pulse_us = deg_to_us(deg, self.sweep, self.min_us, self.max_us)
        return deg


class ESP32SerialBridge:
    """
    High-speed serial communication bridge between host and NodeMCU ESP32.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False
        self.last_send_time = 0.0
        self.min_interval = 0.015  # Limit transmit rate to ~66 Hz to match servo frequency

        if port:
            self.connect(port)

    def connect(self, port: str) -> bool:
        if not _SERIAL_AVAILABLE or serial is None:
            return False

        self.close()
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=self.baudrate,
                timeout=0.05,
                write_timeout=0.05,
            )
            try:
                self.ser.dtr = True
                self.ser.rts = True
            except Exception:
                pass
            time.sleep(0.15)  # Allow USB CDC connection to settle
            self.port = port
            self.is_connected = True
            print(f"[ESP32] Successfully connected on serial port: {port} ({self.baudrate} baud)")
            return True
        except Exception as e:
            self.ser = None
            self.is_connected = False
            print(f"[ESP32] Could not open serial port {port}: {e}")
            return False

    def send_pwm(self, pan_us: int, tilt_us: int) -> bool:
        """Sends 'P:<pan_us> T:<tilt_us>\\n' packet to ESP32."""
        if not self.is_connected or self.ser is None:
            return False

        t_now = time.monotonic()
        if (t_now - self.last_send_time) < self.min_interval:
            return True
        self.last_send_time = t_now

        packet = f"P:{pan_us} T:{tilt_us}\n".encode("ascii")
        try:
            self.ser.write(packet)
            return True
        except Exception as e:
            print(f"[ESP32] Serial write error on {self.port}: {e}")
            self.is_connected = False
            return False

    def send_command(self, cmd: str) -> bool:
        """Sends raw command string to ESP32."""
        if not self.is_connected or self.ser is None:
            return False
        try:
            self.ser.write(f"{cmd.strip()}\n".encode("ascii"))
            return True
        except Exception:
            return False

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self.is_connected = False


class PanTiltServoing:
    """
    Visual servoing controller that computes required Pan & Tilt angles,
    converts them to PWM pulse widths (microseconds), and streams them to NodeMCU ESP32.
    """

    def __init__(
        self,
        serial_port: Optional[str] = None,
        baud: int = 115200,
        hfov_deg: float = 62.2,
        vfov_deg: float = 48.8,
        pan_sign: int = 1,
        tilt_sign: int = -1,
        pan_min: float = -80.0,
        pan_max: float = 80.0,
        tilt_min: float = -45.0,
        tilt_max: float = 45.0,
        force_mock: bool = False,
        **kwargs,  # Gracefully absorbs legacy pan_gpio / tilt_gpio kwargs
    ):
        self.hfov_deg = hfov_deg
        self.vfov_deg = vfov_deg
        self.pan_sign = pan_sign
        self.tilt_sign = tilt_sign
        self.detection_timeout = 0.5
        self.baud = baud

        self.pan_servo = Servo(lo_deg=pan_min, hi_deg=pan_max, start_deg=0.0)
        self.tilt_servo = Servo(lo_deg=tilt_min, hi_deg=tilt_max, start_deg=0.0)

        self.pan_ctl = AxisController(kp=0.70, kd=0.08, deadband_px=10.0, max_step_deg=10.0, ema=0.75)
        self.tilt_ctl = AxisController(kp=0.70, kd=0.08, deadband_px=10.0, max_step_deg=8.0, ema=0.75)

        self.pan_cmd = 0.0
        self.tilt_cmd = 0.0
        self.last_seen_time = 0.0
        self.enabled = True

        # ESP32 Serial Subsystem
        self.serial_bridge: Optional[ESP32SerialBridge] = None
        self.active_port: Optional[str] = None

        if not force_mock:
            target_port = serial_port if (serial_port and serial_port != "auto") else find_esp32_port()
            if target_port:
                self.serial_bridge = ESP32SerialBridge(port=target_port, baudrate=baud)
                if self.serial_bridge.is_connected:
                    self.active_port = target_port
            else:
                print("[ESP32] No hardware serial device detected. Initializing in MOCK mode.")
        else:
            print("[ESP32] Mock mode forced by user.")

    @property
    def is_hardware(self) -> bool:
        return self.serial_bridge is not None and self.serial_bridge.is_connected

    @property
    def is_mock(self) -> bool:
        return not self.is_hardware

    @property
    def port_name(self) -> str:
        return self.active_port if self.is_hardware else "MOCK (Simulated)"

    @property
    def pan_angle(self) -> float:
        return self.pan_servo.angle

    @property
    def tilt_angle(self) -> float:
        return self.tilt_servo.angle

    @property
    def pan_us(self) -> int:
        return self.pan_servo.pulse_us

    @property
    def tilt_us(self) -> int:
        return self.tilt_servo.pulse_us

    def reconnect(self, port: Optional[str] = None) -> bool:
        """Attempts to discover and connect/reconnect to the ESP32."""
        target = port or find_esp32_port()
        if not target:
            return False

        if self.serial_bridge is None:
            self.serial_bridge = ESP32SerialBridge(port=target, baudrate=self.baud)
        else:
            self.serial_bridge.connect(target)

        if self.serial_bridge.is_connected:
            self.active_port = target
            self.center_servos()
            return True
        return False

    def center_servos(self):
        """Reset servos to 0 deg center (1500 us)."""
        self.pan_cmd = self.pan_servo.write(0.0)
        self.tilt_cmd = self.tilt_servo.write(0.0)
        self.pan_ctl.reset()
        self.tilt_ctl.reset()

        if self.is_hardware and self.serial_bridge:
            self.serial_bridge.send_command("CENTER")

    def update(
        self,
        center: Optional[Tuple[float, float]],
        frame_w: int,
        frame_h: int,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Updates pan-tilt commands toward target bbox center.
        Streams PWM values in microseconds to the NodeMCU ESP32.
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

            # Stream PWM microseconds to NodeMCU ESP32
            if self.is_hardware and self.serial_bridge:
                self.serial_bridge.send_pwm(self.pan_servo.pulse_us, self.tilt_servo.pulse_us)

        elif (t_now - self.last_seen_time) > self.detection_timeout:
            # Target lost watchdog
            self.pan_ctl.reset()
            self.tilt_ctl.reset()

        return self.pan_angle, self.tilt_angle

    def release(self):
        """Closes serial connection."""
        if self.serial_bridge is not None:
            self.serial_bridge.close()
            self.serial_bridge = None
        self.active_port = None
