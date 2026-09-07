/*
 * esp32_firmware.ino
 * ==================
 * NodeMCU ESP32 Hardware-Timed PWM Pan-Tilt Servo Controller
 *
 * Receives servo PWM pulse widths in microseconds over Serial (115200 baud)
 * from the AI Vision Tracker running on the host computer.
 *
 * Wiring for NodeMCU ESP32:
 * -------------------------------------------------------------
 * Pan Servo (Signal / Yellow or Orange)   -> GPIO 18
 * Tilt Servo (Signal / Yellow or Orange)  -> GPIO 19
 * Servo Power (Red wire)                  -> External 5V (2A+ recommended)
 * Servo Ground (Brown or Black wire)      -> External 5V GND
 * ESP32 GND                               -> Common Ground with Servo GND
 * ESP32 USB                               -> Host Computer (Mac/PC)
 * -------------------------------------------------------------
 *
 * Required Arduino Library:
 * - "ESP32Servo" by Kevin Harrington (Install via Arduino Library Manager)
 */

#include <ESP32Servo.h>

// ---------------------------------------------------------------------------
// Hardware Pin Definitions
// ---------------------------------------------------------------------------
#define PAN_PIN   18    // Pan servo PWM output
#define TILT_PIN  19    // Tilt servo PWM output
#define LED_PIN   2     // Built-in blue status LED on standard NodeMCU ESP32

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

int currentPanUs  = CENTER_US;
int currentTiltUs = CENTER_US;

char serialBuffer[64];
uint8_t bufferIndex = 0;
unsigned long lastPacketTime = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 2000;

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 1000) {
        delay(10);
    }

    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Allow hardware timers for ESP32PWM
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    // Standard 50Hz PWM frequency for analog & digital hobby servos
    panServo.setPeriodHertz(50);
    tiltServo.setPeriodHertz(50);

    panServo.attach(PAN_PIN, MIN_PULSE_US, MAX_PULSE_US);
    tiltServo.attach(TILT_PIN, MIN_PULSE_US, MAX_PULSE_US);

    // Recenter servos on boot
    panServo.writeMicroseconds(CENTER_US);
    tiltServo.writeMicroseconds(CENTER_US);

    // Startup LED blink sequence (3 quick flashes)
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(80);
        digitalWrite(LED_PIN, LOW);
        delay(80);
    }

    Serial.println("OK: ESP32 Pan-Tilt Controller Ready (115200 baud)");
    lastPacketTime = millis();
}

// ---------------------------------------------------------------------------
// Packet Parsing
// Supported formats:
//   "P:1520 T:1480"
//   "1520,1480"
//   "CENTER"
// ---------------------------------------------------------------------------
void processPacket(const char* packet) {
    int panVal = -1;
    int tiltVal = -1;

    if (strncmp(packet, "CENTER", 6) == 0 || strncmp(packet, "RESET", 5) == 0) {
        panVal = CENTER_US;
        tiltVal = CENTER_US;
    } else if (sscanf(packet, "P:%d T:%d", &panVal, &tiltVal) == 2) {
        // Formatted packet: P:<pan_us> T:<tilt_us>
    } else if (sscanf(packet, "%d,%d", &panVal, &tiltVal) == 2) {
        // CSV format: <pan_us>,<tilt_us>
    } else if (strncmp(packet, "PING", 4) == 0) {
        Serial.println("PONG");
        return;
    }

    if (panVal >= MIN_PULSE_US && panVal <= MAX_PULSE_US &&
        tiltVal >= MIN_PULSE_US && tiltVal <= MAX_PULSE_US) {

        // Constrain to physical safe envelope
        currentPanUs = constrain(panVal, PAN_MIN_US, PAN_MAX_US);
        currentTiltUs = constrain(tiltVal, TILT_MIN_US, TILT_MAX_US);

        panServo.writeMicroseconds(currentPanUs);
        tiltServo.writeMicroseconds(currentTiltUs);

        lastPacketTime = millis();
        digitalWrite(LED_PIN, HIGH); // LED ON when active
    }
}

// ---------------------------------------------------------------------------
// Main Loop
// ---------------------------------------------------------------------------
void loop() {
    // Non-blocking serial receiver
    while (Serial.available() > 0) {
        char c = Serial.read();

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

    // Safety watchdog: turn off LED if no packets received recently
    if (millis() - lastPacketTime > WATCHDOG_TIMEOUT_MS) {
        digitalWrite(LED_PIN, LOW);
    }
}
