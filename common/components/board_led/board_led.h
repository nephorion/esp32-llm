/*
 * board_led.h - switch the on-board LED on and off.
 *
 * Hides the difference between the two kinds of LED found on ESP32-S3 boards:
 *   - an addressable RGB ("NeoPixel", WS2812) LED, driven by a one-wire pulse train, or
 *   - a plain LED switched by a GPIO pin.
 * Which one, and which pin, is chosen in menuconfig under "Board LED" (see Kconfig).
 * The LED is shown as dim white so it is visible without being dazzling.
 */
#ifndef BOARD_LED_H
#define BOARD_LED_H

#include <stdbool.h>
#include "esp_err.h"

/* Set the LED up (off). Returns ESP_OK, or an error if the LED could not be configured. */
esp_err_t board_led_init(void);

/* Switch the LED on (dim white) or off. Does nothing if board_led_init() has not succeeded. */
void board_led_set(bool on);

#endif /* BOARD_LED_H */
