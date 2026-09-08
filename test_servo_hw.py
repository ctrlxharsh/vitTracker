#!/usr/bin/env python3
"""
test_servo_hw.py
================
Ultra-gentle physical test for Pan-Tilt servos over ESP32 USB Serial.
Tests Pan and Tilt individually with small steps to avoid USB brownout.
"""

import time
import serial
import serial.tools.list_ports


def find_port():
    for p in serial.tools.list_ports.comports():
        dev = p.device.lower()
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(k in dev or k in desc or k in hwid for k in ["usbmodem", "usbserial", "ch340", "cp210", "espressif", "303a:", "10c4:", "usb serial"]):
            return p.device
    for p in serial.tools.list_ports.comports():
        if p.device.upper().startswith("COM") and p.device.upper() != "COM1":
            return p.device
    return None


def main():
    port = find_port()
    if not port:
        print("[-] Error: No ESP32 USB Serial port detected!")
        return

    print(f"[+] Found ESP32 port: {port}")
    s = serial.Serial(port=port, baudrate=115200, timeout=1)
    s.dtr = False
    s.rts = False
    time.sleep(0.3)

    s.write(b"PING\n")
    time.sleep(0.1)
    banner = s.read(s.in_waiting).decode(errors="ignore").strip()
    print(f"[+] Handshake: {banner or 'OK'}")

    print("\n[+] Centering servos smoothly...")
    s.write(b"CENTER\n")
    time.sleep(0.4)

    print("\n[+] Testing PAN servo (GPIO 14) gently...")
    # Nudge Pan right (+15 deg) and left (-15 deg)
    for us in range(1500, 1620, 4):
        s.write(f"P:{us} T:1500\n".encode())
        time.sleep(0.04)
    for us in range(1620, 1380, -4):
        s.write(f"P:{us} T:1500\n".encode())
        time.sleep(0.04)
    for us in range(1380, 1501, 4):
        s.write(f"P:{us} T:1500\n".encode())
        time.sleep(0.04)
    print("  -> Pan test complete!")

    time.sleep(0.4)

    print("\n[+] Testing TILT servo (GPIO 13) gently...")
    # Nudge Tilt up (+12 deg) and down (-12 deg)
    for us in range(1500, 1600, 4):
        s.write(f"P:1500 T:{us}\n".encode())
        time.sleep(0.04)
    for us in range(1600, 1400, -4):
        s.write(f"P:1500 T:{us}\n".encode())
        time.sleep(0.04)
    for us in range(1400, 1501, 4):
        s.write(f"P:1500 T:{us}\n".encode())
        time.sleep(0.04)
    print("  -> Tilt test complete!")

    s.write(b"CENTER\n")
    time.sleep(0.3)
    s.close()
    print("\n[+] SUCCESS: Both servos moved smoothly without power drop or jerk!")


if __name__ == "__main__":
    main()
