/* board_led.c - see board_led.h */
#include <stdbool.h>
#include "driver/gpio.h"
#include "sdkconfig.h"
#include "board_led.h"

#if CONFIG_LED_IS_RGB
#include "led_strip.h"
static led_strip_handle_t strip;     /* handle to our one-pixel "strip"; NULL until init succeeds */
#else
static bool ready;
#endif

esp_err_t board_led_init(void)
{
#if CONFIG_LED_IS_RGB
    /* Describe the LED: which pin, and how many pixels (one). */
    led_strip_config_t cfg = { .strip_gpio_num = CONFIG_LED_GPIO, .max_leds = 1 };
    /* RMT clock: 10 MHz gives 0.1 microsecond resolution, plenty for the LED's pulses.
     * RMT is hardware that generates precise pulse patterns, so the CPU is not tied up. */
    led_strip_rmt_config_t rmt = { .resolution_hz = 10 * 1000 * 1000 };
    esp_err_t err = led_strip_new_rmt_device(&cfg, &rmt, &strip);
    if (err != ESP_OK) { strip = NULL; return err; }
    led_strip_clear(strip);
    return ESP_OK;
#else
    gpio_set_direction(CONFIG_LED_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(CONFIG_LED_GPIO, 0);
    ready = true;
    return ESP_OK;
#endif
}

void board_led_set(bool on)
{
#if CONFIG_LED_IS_RGB
    if (!strip) return;
    if (on) {
        led_strip_set_pixel(strip, 0, 32, 32, 32);   /* pixel 0 = (R, G, B): dim white */
        led_strip_refresh(strip);                    /* actually send it to the LED */
    } else {
        led_strip_clear(strip);                      /* all off, and sent */
    }
#else
    if (ready) gpio_set_level(CONFIG_LED_GPIO, on);
#endif
}
