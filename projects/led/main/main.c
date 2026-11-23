// Serial LED control. Commands (one per line): on, off, exit
#include <stdio.h>
#include <string.h>
#include "driver/uart.h"
#include "board_led.h"      /* shared on-board LED (common/components/board_led) */

static void reply(const char *s) { uart_write_bytes(UART_NUM_0, s, strlen(s)); }

void app_main(void)
{
    uart_driver_install(UART_NUM_0, 256, 0, 0, NULL, 0);
    board_led_init();

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
        if (!strcmp(line, "on"))        { board_led_set(true);  reply("OK on\r\n"); }
        else if (!strcmp(line, "off"))  { board_led_set(false); reply("OK off\r\n"); }
        else if (!strcmp(line, "exit")) { board_led_set(false); reply("OK bye\r\n"); }
        else                            { reply("ERR unknown command\r\n"); }
    }
}
