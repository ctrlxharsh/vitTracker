# NodeMCU ESP32 Pan-Tilt Servo Controller Firmware

This folder contains the firmware to flash onto your **NodeMCU ESP32** (or any standard ESP32 DevKit) to receive PWM pan-tilt servo commands over USB Serial from the AI Vision Tracker running on your computer.

---

## 1. Hardware Pinout & Wiring

| Component Wire | Color | Connects To |
|---|---|---|
| **Pan Servo Signal** | Yellow / Orange | ESP32-S3 **GPIO 14** |
| **Tilt Servo Signal** | Yellow / Orange | ESP32-S3 **GPIO 13** |
| **Servo Power** | Red (+5V) | External 5V Power Supply (+ terminal) |
| **Servo Ground** | Brown / Black (GND) | External 5V Power Supply (- terminal) |
| **Common Ground** | Black / Ground Wire | ESP32 **GND** pin |
| **USB Data / Power** | USB-C Cable | Mac / Host PC |

> [!CAUTION]
> **DO NOT USE GPIO 19 / 20 ON ESP32-S3**:
> On the ESP32-S3, **GPIO 19 is USB D-** and **GPIO 20 is USB D+**. If a servo wire is plugged into GPIO 19 or 20, it will short the USB data lines and crash the USB connection immediately. Always use **GPIO 14** (Pan) and **GPIO 13** (Tilt).

> [!IMPORTANT]
> **Separate Power**: Do not power both servos directly from the ESP32 3.3V or 5V (VIN) pin from your computer USB port. Two servos moving simultaneously draw 1.5A–2.0A peak current, which will trip USB overcurrent or brown out the ESP32, shutting it off. Use an external 5V power supply (or USB power bank / phone charger) for the servos, sharing common GND with the ESP32.

---

## 2. Flashing using Arduino IDE

1. Open **Arduino IDE**.
2. Go to **Tools** -> **Board** -> **esp32** -> Select **NodeMCU-32S** (or **ESP32 Dev Module**).
3. Install required library:
   - Go to **Sketch** -> **Include Library** -> **Manage Libraries...**
   - Search for `ESP32Servo` (by Kevin Harrington).
   - Click **Install**.
4. Open `esp32_firmware.ino` in Arduino IDE.
5. Plug the ESP32 into your computer via USB.
6. Select your Port under **Tools** -> **Port** (e.g., `/dev/cu.usbserial-xxx` or `/dev/cu.SLAB_USBtoUART` on Mac, `COMx` on Windows).
7. Click **Upload** (arrow icon).

When flashing finishes, the onboard blue LED will flash 3 times, and the servos will center to 1500 µs (neutral 90°).

---

## 3. Flashing using PlatformIO / CLI (Alternative)

If using PlatformIO:
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 115200
lib_deps =
    madhephaestus/ESP32Servo@^3.0.5
```
Upload with:
```bash
pio run -t upload
```

---

## 4. Testing over Serial Monitor

Open the Arduino Serial Monitor at **115200 baud**.
Send test commands:
- `P:1500 T:1500` -> Centers both pan and tilt
- `P:1200 T:1700` -> Pans left, tilts up
- `P:1800 T:1300` -> Pans right, tilts down
- `CENTER` -> Re-centers both axes
- `PING` -> Responds `PONG`
