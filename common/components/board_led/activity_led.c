/*
 * ============================================================================
 *  activity_led.c  -  see activity_led.h for what this is for
 * ============================================================================
 *
 * THE DESIGN IN ONE PICTURE
 *
 *      main task (core 0)                         LED task (core 1)
 *      ------------------                         -----------------
 *                                                 sleeps, using no CPU,
 *                                                 waiting for a notification
 *      command arrives
 *      activity_led_start()  --- "wake up!" --->  wakes: ON 100 ms, OFF 100 ms,
 *      ...seconds of maths...                            repeat while busy
 *      activity_led_stop()   --- busy = false -->  finishes its current blink,
 *                                                  then goes back to sleep
 *
 * Two small pieces of FreeRTOS are used:
 *
 *   TASK NOTIFICATION  a lightweight "doorbell" to one specific task. The task
 *                      sleeps in xTaskNotifyWait() and is woken when someone calls
 *                      xTaskNotify(). The notification carries a few bits, so one
 *                      doorbell can say which thing happened (bit 0: "a command
 *                      started", bit 1: "show the ready pattern").
 *   volatile flag      `busy` is written by one task and read by another. `volatile`
 *                      tells the compiler not to cache it in a register, so the LED
 *                      task always sees the latest value. (For a single yes/no flag
 *                      on this CPU that is enough; we are not sharing anything bigger.)
 */
#include <stdbool.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "sdkconfig.h"
#include "activity_led.h"

#include "board_led.h"

static const char *TAG = "led";

/* Bits in the task notification value. */
#define EVT_ACTIVITY  (1u << 0)      /* a command started       */
#define EVT_READY     (1u << 1)      /* show the "ready" flashes */

#define BLINK_MS      100            /* on for 100 ms, off for 100 ms = 5 flashes a second */

static TaskHandle_t led_task;        /* NULL until init succeeds; the other calls check it */
static volatile bool busy;           /* true while a command is running */

/* ---------------------------------------------------------------------------
 * The LED task. It lives forever in this loop.
 * ------------------------------------------------------------------------- */
static void led_task_fn(void *arg)
{
    for (;;) {
        /* Sleep until woken. portMAX_DELAY = wait as long as it takes. The first
         * argument 0 and second UINT32_MAX mean: don't touch the notification bits
         * on the way in, clear them all on the way out. */
        uint32_t events = 0;
        xTaskNotifyWait(0, UINT32_MAX, &events, portMAX_DELAY);

        if (events & EVT_READY) {
            for (int i = 0; i < 3; i++) {            /* three quick flashes */
                board_led_set(true);  vTaskDelay(pdMS_TO_TICKS(60));
                board_led_set(false); vTaskDelay(pdMS_TO_TICKS(100));
            }
        }

        if (events & EVT_ACTIVITY) {
            /* do..while, not while: the check comes AFTER the first blink, so even
             * a command that is already finished by now still gets one full flash. */
            do {
                board_led_set(true);  vTaskDelay(pdMS_TO_TICKS(BLINK_MS));
                board_led_set(false); vTaskDelay(pdMS_TO_TICKS(BLINK_MS));
            } while (busy);
        }
    }
}

void activity_led_init(void)
{
    if (board_led_init() != ESP_OK) {
        ESP_LOGW(TAG, "LED on GPIO%d could not be set up; running without it", CONFIG_LED_GPIO);
        return;                                      /* led_task stays NULL: all calls become no-ops */
    }

    /* Create the task on core 1 (the neural network runs on core 0). Priority 1 is
     * low: if it ever competed with real work, the real work would win. 3 KB of
     * stack is ample for this small loop. */
    if (xTaskCreatePinnedToCore(led_task_fn, "activity_led", 3072, NULL, 1, &led_task, 1) != pdPASS) {
        led_task = NULL;
        ESP_LOGW(TAG, "could not start the LED task");
    }
}

void activity_led_ready(void)
{
    if (led_task) xTaskNotify(led_task, EVT_READY, eSetBits);   /* eSetBits: add this bit to any already set */
}

void activity_led_start(void)
{
    busy = true;                                     /* set BEFORE waking the task, so it sees it */
    if (led_task) xTaskNotify(led_task, EVT_ACTIVITY, eSetBits);
}

void activity_led_stop(void)
{
    busy = false;                                    /* the task finishes its current blink and sleeps */
}
