/*
  Optional MCU sketch for Snug on Arduino UNO Q (STM32U585).

  Python (Debian / App Lab) calls:

      from arduino.app_utils import Bridge
      Bridge.call("set_alarm", level, timeout=2)  # 0=off, 1=low, 2=medium, 3=high

  Do not open Serial1 — arduino-router owns it.
  Docs: https://docs.arduino.cc/software/app-lab/bridge/bridge-api/

  Hook a buzzer or Modulino on the Qwiic / a GPIO if you have one.
  The 4 user RGB LEDs or a digital pin can also be used — adjust pins
  for the UNO Q core you have installed.
*/

#include <Arduino_RouterBridge.h>

#ifndef LED_BUILTIN
#define LED_BUILTIN 13
#endif

int alarmLevel = 0;

void set_alarm(int level) {
  alarmLevel = constrain(level, 0, 3);
  digitalWrite(LED_BUILTIN, alarmLevel > 0 ? HIGH : LOW);
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  // Name must match Bridge.call("set_alarm", ...) on the Python side.
  Bridge.provide("set_alarm", set_alarm);
}

void loop() {
  if (alarmLevel >= 3) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(80);
    digitalWrite(LED_BUILTIN, LOW);
    delay(80);
  } else if (alarmLevel >= 1) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(200);
    digitalWrite(LED_BUILTIN, LOW);
    delay(400);
  } else {
    delay(200);
  }
}
