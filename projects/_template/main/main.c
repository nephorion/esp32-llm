/*
 * __NAME__ - a starting point: a serial command shell that the host tools can talk to.
 *
 * The protocol is the same one every project in this repository uses, and the one
 * esp32tools.Board (common/python) understands:
 *
 *     board, once after every reset :  READY
 *     PC -> board                   :  one command per line
 *     board -> PC                   :  some reply lines, then OK   (or one line "ERR ...")
 *
 * Two commands are built in so you can see it work: `ping` and `echo <text>`.
 * Add your own in handle().
 */
#include <stdio.h>
#include <string.h>
#include "driver/uart.h"
#include "board_led.h"      /* the shared on-board LED (common/components/board_led) */

static void handle(char *line)
{
    if (!strcmp(line, "ping")) {
        printf("pong\n");
    } else if (!strncmp(line, "echo ", 5)) {
        printf("%s\n", line + 5);
    } else {
        printf("ERR unknown command\n");
        return;
    }
    printf("OK\n");
}

void app_main(void)
{
    uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0);   /* needed to READ the serial port */
    board_led_init();

    printf("READY\n");
    fflush(stdout);

    char line[256];
    int n = 0;
    for (;;) {
        uint8_t c;
        if (uart_read_bytes(UART_NUM_0, &c, 1, portMAX_DELAY) != 1) continue;
        if (c != '\n' && c != '\r') {
            if (n < (int)sizeof(line) - 1) line[n++] = (char)c;
            continue;
        }
        if (n == 0) continue;
        line[n] = 0;
        n = 0;
        board_led_set(true);        /* LED on while a command runs */
        handle(line);
        fflush(stdout);
        board_led_set(false);
    }
}
