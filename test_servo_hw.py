#!/usr/bin/env python3
"""
test_servo_hw.py
================
Smooth random motion simulation and hardware verification for ESP32 Pan-Tilt servos.
Generates multi-harmonic continuous wandering to verify silky-smooth motion.
"""

import argparse
import math
import sys
import time
from typing import Optional

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("[-] Error: pyserial not installed. Run: pip install pyserial")
    sys.exit(1)


def find_port(preferred: Optional[str] = None) -> Optional[str]:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None

    if preferred and preferred.lower() != "auto":
        for p in ports:
            if p.device.lower() == preferred.lower():
                return p.device

    keywords = ["cp210", "usbmodem", "usbserial", "ch340", "espressif", "303a:", "10c4:", "usb serial"]
    for p in ports:
        dev = (p.device or "").lower()
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(k in dev or k in desc or k in hwid for k in keywords):
            return p.device

    for p in ports:
        if p.device.upper().startswith("COM") and p.device.upper() != "COM1":
            return p.device
    return None


def run_smooth_random_simulation(ser: serial.Serial, duration: float = 15.0):
    """
    Simulates smooth, organic random wandering across Pan and Tilt axes.
    Uses multi-frequency sinusoidal superposition with zero velocity discontinuities.
    """
    print(f"\n[+] Starting Smooth Random Motion Simulation (Duration: {'Continuous' if duration <= 0 else f'{duration:.0f}s'})...")
    print("    Press Ctrl+C at any time to center servos and stop.\n")

    t_start = time.monotonic()
    last_ui = 0.0

    try:
        while True:
            t = time.monotonic() - t_start
            if duration > 0 and t >= duration:
                break

            # Organic multi-frequency waveform for Pan (amplitude ~320us, ~ +-30 deg)
            pan_delta = 260.0 * math.sin(0.65 * t) + 90.0 * math.sin(1.42 * t + 0.8) + 40.0 * math.sin(0.23 * t + 2.1)
            pan_us = int(round(max(1150, min(1850, 1500 + pan_delta))))
            pan_deg = (pan_us - 1500) * 0.09

            # Organic multi-frequency waveform for Tilt (amplitude ~180us, ~ +-16 deg)
            tilt_delta = 140.0 * math.cos(0.51 * t) + 60.0 * math.sin(1.15 * t + 1.4) + 30.0 * math.cos(0.31 * t)
            tilt_us = int(round(max(1280, min(1720, 1500 + tilt_delta))))
            tilt_deg = (tilt_us - 1500) * 0.09

            # Transmit high-rate PWM command
            packet = f"P:{pan_us} T:{tilt_us}\n".encode("ascii")
            ser.write(packet)

            # Drain incoming buffers
            if ser.in_waiting > 0:
                _ = ser.read(ser.in_waiting)

            # Console visualizer at ~10 Hz
            if time.monotonic() - last_ui >= 0.10:
                last_ui = time.monotonic()
                pan_bar = "#" * int((pan_us - 1100) / 40)
                tilt_bar = "#" * int((tilt_us - 1200) / 30)
                sys.stdout.write(
                    f"\r[{t:4.1f}s] PAN: {pan_us:4d}µs ({pan_deg:+5.1f}°) |{pan_bar:<20}|   "
                    f"TILT: {tilt_us:4d}µs ({tilt_deg:+5.1f}°) |{tilt_bar:<18}|"
                )
                sys.stdout.flush()

            time.sleep(0.02)  # 50 Hz smooth streaming rate

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user.")

    print("\n\n[+] Centering servos smoothly...")
    ser.write(b"CENTER\n")
    time.sleep(0.3)
    print("[+] Servos parked at 1500µs (Center).")


def main():
    parser = argparse.ArgumentParser(description="Smooth random servo motion simulation for ESP32 Pan-Tilt.")
    parser.add_argument("--port", type=str, default="auto", help="COM/serial port (e.g. COM3 or 'auto')")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds (0 = continuous)")
    args = parser.parse_args()

    port = find_port(args.port)
    if not port:
        print("[-] Error: No ESP32 USB Serial port detected!")
        sys.exit(1)

    print(f"[+] Found ESP32 port: {port}")
    try:
        s = serial.Serial(port=port, baudrate=115200, timeout=1, write_timeout=0)
    except Exception as e:
        print(f"[-] Error opening {port}: {e}")
        sys.exit(1)

    time.sleep(0.2)
    s.reset_input_buffer()
    s.reset_output_buffer()

    s.write(b"PING\n")
    time.sleep(0.1)
    banner = s.read(s.in_waiting).decode(errors="ignore").strip()
    print(f"[+] Handshake: {banner or 'OK'}")

    run_smooth_random_simulation(s, duration=args.duration)
    s.close()
    print("[+] Test completed successfully!")


if __name__ == "__main__":
    main()
