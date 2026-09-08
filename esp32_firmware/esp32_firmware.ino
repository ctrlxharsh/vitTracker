/*
 * esp32_firmware.ino
 * ==================
 * ESP32 / ESP32-S3 Hardware-Timed PWM Pan-Tilt Servo Controller
 *
 * Receives servo PWM pulse widths in microseconds over USB Serial (115200 baud)
 * from the AI Vision Tracker running on the host computer.
 *
 * Wiring for ESP32 / ESP32-S3:
 * -------------------------------------------------------------
 * Pan Servo (Signal / Yellow or Orange)   -> GPIO 14
 * Tilt Servo (Signal / Yellow or Orange)  -> GPIO 13
 * Servo Power (Red wire)                  -> External 5V (2A+ recommended)
 * Servo Ground (Brown or Black wire)      -> External 5V GND
 * ESP32 GND                               -> Common Ground with Servo GND
 * ESP32 USB                               -> Host Computer (Mac/PC)
 * -------------------------------------------------------------
 *
 * IMPORTANT FOR ARDUINO IDE ON ESP32-S3:
 * 1. Tools -> Board -> ESP32S3 Dev Module
 * 2. Tools -> USB CDC On Boot -> "Enabled"  (REQUIRED for native USB Serial)
 * 3. Tools -> Port -> /dev/cu.usbmodem11201
 *
 * Required Arduino Library:
 * - "ESP32Servo" by Kevin Harrington (Install via Arduino Library Manager)
 */

#include <ESP32Servo.h>

// ---------------------------------------------------------------------------
// Hardware Pin Definitions
// ---------------------------------------------------------------------------
#define PAN_PIN   14    // Pan servo PWM output
#define TILT_PIN  13    // Tilt servo PWM output
#define LED_PIN   2     // Built-in status LED

// ---------------------------------------------------------------------------
// Servo Calibration & Travel Limits (Microseconds)
// ---------------------------------------------------------------------------
#define MIN_PULSE_US  500
#define MAX_PULSE_US  2500
#define CENTER_US     1500

// Safe mechanical travel limits (prevents stalling or mechanical binding)
#define PAN_MIN_US    600    // ~ -80 deg
#define PAN_MAX_US    2400   // ~ +80 deg
#define TILT_MIN_US   1000   // ~ -45 deg
#define TILT_MAX_US   2000   // ~ +45 deg

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------
Servo panServo;
Servo tiltServo;

int targetPanUs   = CENTER_US;
int targetTiltUs  = CENTER_US;
int currentPanUs  = CENTER_US;
int currentTiltUs = CENTER_US;
bool servosAttached = true;

char serialBuffer[64];
uint8_t bufferIndex = 0;
unsigned long lastPacketTime = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 2000;

// ---------------------------------------------------------------------------
// Helper: Send Serial Response to Both Active Ports (USB-CDC and UART0)
// ---------------------------------------------------------------------------
void sendFeedback(const char* msg) {
    Serial.println(msg);
    Serial0.println(msg);
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
    // Initialize both USB-CDC (native USB) and Hardware UART0 (CP2102)
    Serial.begin(115200);
    Serial0.begin(115200);

    // Brief wait for USB Serial host connection
    unsigned long startWait = millis();
    while (!Serial && !Serial0 && (millis() - startWait < 1500)) {
        delay(10);
    }

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Allow hardware timers for ESP32PWM
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    // Standard 50Hz PWM frequency for hobby servos
    panServo.setPeriodHertz(50);
    tiltServo.setPeriodHertz(50);

    // Set center position before attaching to prevent initial power-up jump
    panServo.writeMicroseconds(CENTER_US);
    tiltServo.writeMicroseconds(CENTER_US);
    panServo.attach(PAN_PIN, MIN_PULSE_US, MAX_PULSE_US);
    tiltServo.attach(TILT_PIN, MIN_PULSE_US, MAX_PULSE_US);

    // Startup LED blink sequence (3 quick flashes)
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(80);
        digitalWrite(LED_PIN, LOW);
        delay(80);
    }

    sendFeedback("OK: ESP32-S3 Pan-Tilt Controller Ready (Pan:14, Tilt:13)");
    lastPacketTime = millis();
}

// ---------------------------------------------------------------------------
// Packet Parsing
// Supported formats:
//   "P:1520 T:1480"
//   "1520,1480"
//   "CENTER"
//   "PING"
// ---------------------------------------------------------------------------
void processPacket(const char* packet) {
    int panVal = -1;
    int tiltVal = -1;

    if (strncmp(packet, "CENTER", 6) == 0 || strncmp(packet, "RESET", 5) == 0) {
        targetPanUs = CENTER_US;
        targetTiltUs = CENTER_US;
        servosAttached = true;
        sendFeedback("ACK: CENTER");
    } else if (strncmp(packet, "OFF", 3) == 0 || strncmp(packet, "DETACH", 6) == 0 || strncmp(packet, "STOP", 4) == 0) {
        servosAttached = false;
        panServo.writeMicroseconds(0);
        tiltServo.writeMicroseconds(0);
        digitalWrite(LED_PIN, LOW);
        sendFeedback("ACK: OFF");
        return;
    } else if (sscanf(packet, "P:%d T:%d", &panVal, &tiltVal) == 2) {
        // Formatted packet: P:<pan_us> T:<tilt_us>
    } else if (sscanf(packet, "%d,%d", &panVal, &tiltVal) == 2) {
        // CSV format: <pan_us>,<tilt_us>
    } else if (strncmp(packet, "PING", 4) == 0) {
        sendFeedback("PONG");
        return;
    }

    if (panVal == 0 && tiltVal == 0) {
        servosAttached = false;
        panServo.writeMicroseconds(0);
        tiltServo.writeMicroseconds(0);
        digitalWrite(LED_PIN, LOW);
        sendFeedback("ACK: OFF");
        return;
    }

    if (panVal >= MIN_PULSE_US && panVal <= MAX_PULSE_US &&
        tiltVal >= MIN_PULSE_US && tiltVal <= MAX_PULSE_US) {

        // Constrain to physical safe envelope and set smooth interpolation targets
        targetPanUs = constrain(panVal, PAN_MIN_US, PAN_MAX_US);
        targetTiltUs = constrain(tiltVal, TILT_MIN_US, TILT_MAX_US);
        servosAttached = true;

        lastPacketTime = millis();
        digitalWrite(LED_PIN, HIGH); // LED ON when active
    }
}

// ---------------------------------------------------------------------------
// Main Loop
// ---------------------------------------------------------------------------
void handleSerialChar(char c) {
    if (c == '\n' || c == '\r') {
        if (bufferIndex > 0) {
            serialBuffer[bufferIndex] = '\0';
            processPacket(serialBuffer);
            bufferIndex = 0;
        }
    } else if (bufferIndex < sizeof(serialBuffer) - 1) {
        serialBuffer[bufferIndex++] = c;
    }
}

void loop() {
    // Read from standard USB-CDC Serial
    while (Serial.available() > 0) {
        handleSerialChar((char)Serial.read());
    }
    // Read from Hardware UART0 (CP2102)
    while (Serial0.available() > 0) {
        handleSerialChar((char)Serial0.read());
    }

    // Smooth hardware slew rate limiter: gently ramp pulse width to target
    static unsigned long lastSlewTime = 0;
    if (servosAttached && (millis() - lastSlewTime >= 15)) {
        lastSlewTime = millis();
        const int MAX_US_STEP = 6; // Smooth 6us per 15ms prevents sudden jerk

        if (currentPanUs != targetPanUs) {
            int diff = targetPanUs - currentPanUs;
            currentPanUs += constrain(diff, -MAX_US_STEP, MAX_US_STEP);
            panServo.writeMicroseconds(currentPanUs);
        }
        if (currentTiltUs != targetTiltUs) {
            int diff = targetTiltUs - currentTiltUs;
            currentTiltUs += constrain(diff, -MAX_US_STEP, MAX_US_STEP);
            tiltServo.writeMicroseconds(currentTiltUs);
        }
    }

    // Safety watchdog: return gently to center and hold position if packets stop
    if (servosAttached && (millis() - lastPacketTime > WATCHDOG_TIMEOUT_MS)) {
        targetPanUs = CENTER_US;
        targetTiltUs = CENTER_US;
        digitalWrite(LED_PIN, LOW);
    }
}
