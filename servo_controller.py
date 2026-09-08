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

import atexit
import threading
import time
from typing import List, Optional, Tuple

try:
    import serial
    import serial.tools.list_ports
    _SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    _SERIAL_AVAILABLE = False

_ACTIVE_SERVO_INSTANCES: List["PanTiltServoing"] = []


def _cleanup_all_servos():
    for inst in list(_ACTIVE_SERVO_INSTANCES):
        try:
            inst.release()
        except Exception:
            pass


atexit.register(_cleanup_all_servos)


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

    # 1. If preferred port specified and present, use it (case-insensitive)
    if preferred and preferred.lower() != "auto":
        for p in available_ports:
            if p.device.lower() == preferred.lower():
                return p.device

    # 2. Look for known USB-serial adapters and microcontrollers
    keywords = [
        "usbserial",
        "usb serial",
        "wchusbserial",
        "slab_usbtouart",
        "usbmodem",
        "cp210",
        "ch340",
        "ch341",
        "ch910",
        "ftdi",
        "espressif",
        "303a:",       # Espressif native USB-JTAG/CDC VID
        "10c4:",       # Silicon Labs CP210x VID
        "1a86:",       # WCH CH340 VID
        "0403:",       # FTDI VID
        "uart",
    ]

    for p in available_ports:
        dev_lower = (p.device or "").lower()
        desc_lower = (p.description or "").lower()
        hwid_lower = (p.hwid or "").lower()

        for kw in keywords:
            if kw in dev_lower or kw in desc_lower or kw in hwid_lower:
                return p.device

    # 3. Fallback: On Linux/Pi, pick first ttyUSB or ttyACM
    for p in available_ports:
        dev_lower = (p.device or "").lower()
        if "ttyusb" in dev_lower or "ttyacm" in dev_lower:
            return p.device

    # 4. Fallback: On Windows, pick active USB COM port (excluding COM1)
    windows_coms = [p.device for p in available_ports if p.device.upper().startswith("COM") and p.device.upper() != "COM1"]
    if len(windows_coms) == 1:
        return windows_coms[0]

    return None


class AxisController:
    """
    Smooth, time-normalized incremental PD controller with EMA filtering,
    derivative kick prevention, and velocity/slew-rate limiting.
    """

    def __init__(
        self,
        kp: float = 1.8,
        kd: float = 0.04,
        deadband_px: float = 10.0,
        max_step_deg: float = 0.8,
        max_speed_deg_s: float = 25.0,
        ema: float = 0.25,
        **kwargs,
    ):
        self.kp = kp
        self.kd = kd
        self.deadband_px = deadband_px
        self.max_step_deg = max_step_deg
        self.max_speed_deg_s = max_speed_deg_s
        self.ema = ema
        self.filt_err = 0.0
        self.prev_err = 0.0
        self.first_step = True

    def reset(self):
        self.filt_err = 0.0
        self.prev_err = 0.0
        self.first_step = True

    def step(self, err_px: float, deg_per_px: float, dt: float) -> float:
        """Returns smooth angle increment in degrees for this update tick."""
        if abs(err_px) < self.deadband_px:
            # Soft deadband: decay filtered error to eliminate abrupt kick when re-exiting deadband
            self.filt_err *= 0.5
            self.first_step = True
            return 0.0

        # EMA filter on error to suppress bounding box detection jitter
        if self.first_step:
            self.filt_err = err_px
        else:
            self.filt_err += self.ema * (err_px - self.filt_err)

        err_deg = self.filt_err * deg_per_px

        # Suppress derivative kick on initial lock or setpoint change
        if self.first_step or dt <= 0:
            derr = 0.0
            self.first_step = False
        else:
            derr = (err_deg - self.prev_err) / dt

        self.prev_err = err_deg

        # Angular velocity command (deg/s) based on proportional error and derivative damping
        vel_cmd = self.kp * err_deg + self.kd * derr

        # Convert velocity to incremental angle for this time step
        effective_dt = dt if dt > 0 else 0.033
        delta = vel_cmd * effective_dt

        # Strict slew-rate limiting: cap maximum step and velocity for slow, smooth motion
        max_allowed = min(self.max_step_deg, self.max_speed_deg_s * effective_dt)
        return max(-max_allowed, min(max_allowed, delta))


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
    High-speed, non-blocking serial communication bridge between host and ESP32.
    Decouples serial I/O into a dedicated background worker thread so GUI execution
    and object tracking never stutter or block.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.is_connected = False
        self.min_interval = 0.015  # Limit transmit rate to ~66 Hz to match servo frequency

        # Thread synchronization
        self._io_lock = threading.Lock()
        self._target_lock = threading.Lock()
        self._target_pan: Optional[int] = None
        self._target_tilt: Optional[int] = None
        self._has_new_target = threading.Event()
        self._worker_running = False
        self._worker_thread: Optional[threading.Thread] = None

        if port:
            self.connect(port)

    def connect(self, port: str) -> bool:
        if not _SERIAL_AVAILABLE or serial is None:
            return False

        self.close()
        try:
            with self._io_lock:
                self.ser = serial.Serial(
                    port=port,
                    baudrate=self.baudrate,
                    timeout=0.2,
                    write_timeout=0.5,
                )
                try:
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                except Exception:
                    pass
            time.sleep(0.2)  # Allow connection to settle
            self.port = port
            self.is_connected = True
            print(f"[ESP32] Successfully connected on serial port: {port} ({self.baudrate} baud)")
            self._start_worker()
            return True
        except Exception as e:
            self.ser = None
            self.is_connected = False
            print(f"[ESP32] Could not open serial port {port}: {e}")
            return False

    def _start_worker(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._io_loop,
            daemon=True,
            name="ESP32-Serial-IO",
        )
        self._worker_thread.start()

    def _io_loop(self):
        while self._worker_running:
            self._has_new_target.wait(timeout=0.02)
            if not self._worker_running:
                break

            target = None
            with self._target_lock:
                if self._target_pan is not None and self._target_tilt is not None:
                    target = (self._target_pan, self._target_tilt)
                    self._target_pan = None
                    self._target_tilt = None
                self._has_new_target.clear()

            t_start = time.monotonic()
            if target is not None and self.is_connected:
                pan_us, tilt_us = target
                packet = f"P:{pan_us} T:{tilt_us}\n".encode("ascii")
                with self._io_lock:
                    if self.ser is not None and self.is_connected:
                        try:
                            self.ser.write(packet)
                        except Exception as e:
                            print(f"[ESP32] Serial write error on {self.port}: {e}")
                            self.is_connected = False
                            try:
                                self.ser.close()
                            except Exception:
                                pass
                            self.ser = None

            # Drain incoming serial buffer (e.g. feedback, ACKs, logs from ESP32)
            if self.is_connected:
                with self._io_lock:
                    if self.ser is not None and self.is_connected:
                        try:
                            if self.ser.in_waiting > 0:
                                _ = self.ser.read(self.ser.in_waiting)
                        except Exception:
                            pass

            # Throttle output rate to max ~66 Hz
            elapsed = time.monotonic() - t_start
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

    def send_pwm(self, pan_us: int, tilt_us: int) -> bool:
        """Non-blocking: Enqueues PWM target for background worker thread."""
        with self._target_lock:
            self._target_pan = pan_us
            self._target_tilt = tilt_us
        self._has_new_target.set()
        return self.is_connected

    def send_command(self, cmd: str) -> bool:
        """Sends raw command string to ESP32."""
        if not self.is_connected:
            return False
        with self._io_lock:
            if not self.is_connected or self.ser is None:
                return False
            try:
                self.ser.write(f"{cmd.strip()}\n".encode("ascii"))
                return True
            except Exception:
                return False

    def close(self):
        self._worker_running = False
        self._has_new_target.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.3)
        self._worker_thread = None

        with self._io_lock:
            if self.ser is not None:
                try:
                    self.ser.write(b"CENTER\n")
                    self.ser.flush()
                except Exception:
                    pass
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

        # Smooth, slow, and stable visual tracking dynamics
        self.pan_ctl = AxisController(
            kp=1.8,
            kd=0.04,
            deadband_px=10.0,
            max_step_deg=0.8,
            max_speed_deg_s=25.0,
            ema=0.25,
        )
        self.tilt_ctl = AxisController(
            kp=1.8,
            kd=0.04,
            deadband_px=10.0,
            max_step_deg=0.6,
            max_speed_deg_s=18.0,
            ema=0.25,
        )

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
                    self.center_servos()
            else:
                print("[ESP32] No hardware serial device detected. Initializing in MOCK mode.")
        else:
            print("[ESP32] Mock mode forced by user.")

        _ACTIVE_SERVO_INSTANCES.append(self)

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

    def relax_servos(self):
        """Sends 0 PWM and OFF command to detach servos and shut off holding torque."""
        if self.is_hardware and self.serial_bridge:
            self.serial_bridge.send_pwm(0, 0)
            self.serial_bridge.send_command("OFF")

    def release(self):
        """Closes serial connection safely, parking servos at center."""
        if self in _ACTIVE_SERVO_INSTANCES:
            try:
                _ACTIVE_SERVO_INSTANCES.remove(self)
            except ValueError:
                pass

        if self.serial_bridge is not None:
            try:
                self.center_servos()
                time.sleep(0.06)
            except Exception:
                pass
            self.serial_bridge.close()
            self.serial_bridge = None
        self.active_port = None
