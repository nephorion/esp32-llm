// Serial LED control. Commands (one per line): on, off, exit
#include <stdio.h>
#include <string.h>
#include "driver/gpio.h"
#include "driver/uart.h"
#include "sdkconfig.h"
#if CONFIG_LED_IS_RGB
#include "led_strip.h"
static led_strip_handle_t strip;
#endif

static void led_init(void)
{
#if CONFIG_LED_IS_RGB
    led_strip_config_t cfg = { .strip_gpio_num = CONFIG_LED_GPIO, .max_leds = 1 };
    led_strip_rmt_config_t rmt = { .resolution_hz = 10 * 1000 * 1000 };
    ESP_ERROR_CHECK(led_strip_new_rmt_device(&cfg, &rmt, &strip));
    led_strip_clear(strip);
#else
    gpio_set_direction(CONFIG_LED_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(CONFIG_LED_GPIO, 0);
#endif
}

static void led_set(bool on)
{
#if CONFIG_LED_IS_RGB
    if (on) { led_strip_set_pixel(strip, 0, 32, 32, 32); led_strip_refresh(strip); }
    else    { led_strip_clear(strip); }
#else
    gpio_set_level(CONFIG_LED_GPIO, on);
#endif
}

static void reply(const char *s) { uart_write_bytes(UART_NUM_0, s, strlen(s)); }

void app_main(void)
{
    uart_driver_install(UART_NUM_0, 256, 0, 0, NULL, 0);
    led_init();

    char line[32];
    int n = 0;
    for (;;) {
        uint8_t c;
        if (uart_read_bytes(UART_NUM_0, &c, 1, portMAX_DELAY) != 1) continue;
        if (c != '\n' && c != '\r') {
            if (n < (int)sizeof(line) - 1) line[n++] = c;
            continue;
        }
        if (n == 0) continue;
        line[n] = 0; n = 0;
        if (!strcmp(line, "on"))        { led_set(true);  reply("OK on\r\n"); }
        else if (!strcmp(line, "off"))  { led_set(false); reply("OK off\r\n"); }
        else if (!strcmp(line, "exit")) { led_set(false); reply("OK bye\r\n"); }
        else                            { reply("ERR unknown command\r\n"); }
    }
}
