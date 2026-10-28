// SentryGun v2 - camera pan/tilt mount ONLY. No trigger, no weapon.
// The Python dashboard sends:
//   "PAN:90;TILT:90;ENEMY\n"  -> move servos to the enemy face centre,
//                               LED + buzzer ON
//   "FRIEND\n"                -> LED + buzzer OFF, servos HOLD position
// Friends never move the mount. Servos only ever aim a camera.
#include <Servo.h>

Servo panServo;
Servo tiltServo;

const int PAN_PIN    = 7;    // same pins as the original sketch
const int TILT_PIN   = 10;
const int LED_PIN    = 11;
const int BUZZER_PIN = 12;

int panAngle  = 90;
int tiltAngle = 90;

void setup() {
  Serial.begin(115200);
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(panAngle);
  tiltServo.write(tiltAngle);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    if (input.indexOf("ENEMY") >= 0) {
      int panStart = input.indexOf("PAN:");
      int panEnd = input.indexOf(";", panStart);
      if (panStart >= 0 && panEnd >= 0) {
        panAngle = constrain(input.substring(panStart + 4, panEnd).toInt(), 0, 180);
      }
      int tiltStart = input.indexOf("TILT:");
      int tiltEnd = input.indexOf(";", tiltStart + 5);
      if (tiltStart >= 0 && tiltEnd >= 0) {
        tiltAngle = constrain(input.substring(tiltStart + 5, tiltEnd).toInt(), 0, 180);
      }
      panServo.write(panAngle);
      tiltServo.write(tiltAngle);
      digitalWrite(LED_PIN, HIGH);
      digitalWrite(BUZZER_PIN, HIGH);
    } else if (input.indexOf("FRIEND") >= 0) {
      digitalWrite(LED_PIN, LOW);
      digitalWrite(BUZZER_PIN, LOW);
      // Servos deliberately NOT moved here.
    }
  }
}
