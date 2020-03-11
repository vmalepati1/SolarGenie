#include <Servo.h>

Servo servo;

const int lowerSoftLimit = 30;
const int upperSoftLimit = 120;

int currentAngle = lowerSoftLimit;
bool increasing = true;

void setup() {
  Serial.begin(9600);
  Serial.println("Solar Genie Servo Testing");
  
  servo.attach(2); // D4
  
  servo.write(currentAngle);
  
  delay(200);
}

void loop() {
  if (increasing) {
    currentAngle++;
  } else {
    currentAngle--;
  }

  if (currentAngle > upperSoftLimit) {
    currentAngle = upperSoftLimit;
    increasing = false;
  }

  if (currentAngle < lowerSoftLimit) {
    currentAngle = lowerSoftLimit;
    increasing = true;
  }

  Serial.println(currentAngle);
  servo.write(currentAngle);

  delay(200);
}
