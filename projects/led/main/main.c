// Serial LED control. Commands (one per line): on, off, ident, exit
//
// Speaks the same protocol as every project here (see common/python/esp32tools/board.py):
//   board, after every reset : READY
//   PC -> board              : one command per line
//   board -> PC              : some reply lines, then OK   (or one line "ERR ...")
#include <stdio.h>
#include <string.h>
#include "driver/uart.h"
#include "board_ident.h"    /* the `ident` command (common/components/board_ident) */
#include "board_led.h"      /* shared on-board LED (common/components/board_led) */

/* Version of this project's serial protocol. Bump it when a change would break the host
 * tools. It must equal PROTO_VERSION in tools/led_repl.py. */
#define PROTO_VERSION 1

static void handle(const char *line)
{
    if (!strcmp(line, "on"))         { board_led_set(true); }
    else if (!strcmp(line, "off"))   { board_led_set(false); }
    else if (!strcmp(line, "exit"))  { board_led_set(false); }
    else if (!strcmp(line, "ident")) { board_ident_print(PROTO_VERSION, NULL); }
    else { printf("ERR unknown command\n"); fflush(stdout); return; }
    printf("OK\n");
    fflush(stdout);
}

void app_main(void)
{
    uart_driver_install(UART_NUM_0, 256, 0, 0, NULL, 0);
    board_led_init();
    printf("READY\n");
    fflush(stdout);

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
        handle(line);
    }
}
