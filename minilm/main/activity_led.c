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

#if CONFIG_LED_IS_RGB
#include "led_strip.h"
static led_strip_handle_t strip;     /* handle to our one-pixel "strip" */
#else
#include "driver/gpio.h"
#endif

static const char *TAG = "led";

/* Bits in the task notification value. */
#define EVT_ACTIVITY  (1u << 0)      /* a command started       */
#define EVT_READY     (1u << 1)      /* show the "ready" flashes */

#define BLINK_MS      100            /* on for 100 ms, off for 100 ms = 5 flashes a second */

static TaskHandle_t led_task;        /* NULL until init succeeds; the other calls check it */
static volatile bool busy;           /* true while a command is running */

/* ---------------------------------------------------------------------------
 * Low-level: switch the LED on or off.
 *
 * An addressable RGB LED has a tiny chip inside that must be sent colour data in
 * a special timing-critical pulse train. led_strip does that using the ESP32's RMT
 * peripheral (hardware that generates precise pulse patterns), so the CPU is not
 * tied up. We send one pixel: dim white = red, green, blue all equal and low.
 * 32 out of 255 is bright enough to see in a room without being dazzling.
 * ------------------------------------------------------------------------- */
static void led_set(bool on)
{
#if CONFIG_LED_IS_RGB
    if (on) {
        led_strip_set_pixel(strip, 0, 32, 32, 32);   /* pixel 0 = (R, G, B) */
        led_strip_refresh(strip);                    /* actually send it to the LED */
    } else {
        led_strip_clear(strip);                      /* all off, and sent */
    }
#else
    gpio_set_level(CONFIG_LED_GPIO, on);             /* plain LED: just the pin level */
#endif
}

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
                led_set(true);  vTaskDelay(pdMS_TO_TICKS(60));
                led_set(false); vTaskDelay(pdMS_TO_TICKS(100));
            }
        }

        if (events & EVT_ACTIVITY) {
            /* do..while, not while: the check comes AFTER the first blink, so even
             * a command that is already finished by now still gets one full flash. */
            do {
                led_set(true);  vTaskDelay(pdMS_TO_TICKS(BLINK_MS));
                led_set(false); vTaskDelay(pdMS_TO_TICKS(BLINK_MS));
            } while (busy);
        }
    }
}

void activity_led_init(void)
{
#if CONFIG_LED_IS_RGB
    /* Describe the LED: which pin, and how many pixels (one). */
    led_strip_config_t cfg = { .strip_gpio_num = CONFIG_LED_GPIO, .max_leds = 1 };
    /* RMT clock: 10 MHz gives 0.1 microsecond resolution, plenty for the LED's pulses. */
    led_strip_rmt_config_t rmt = { .resolution_hz = 10 * 1000 * 1000 };
    if (led_strip_new_rmt_device(&cfg, &rmt, &strip) != ESP_OK) {
        ESP_LOGW(TAG, "LED on GPIO%d could not be set up; running without it", CONFIG_LED_GPIO);
        return;                                      /* led_task stays NULL: all calls become no-ops */
    }
    led_strip_clear(strip);
#else
    gpio_set_direction(CONFIG_LED_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(CONFIG_LED_GPIO, 0);
#endif

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
