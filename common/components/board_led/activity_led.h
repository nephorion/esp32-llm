/*
 * ============================================================================
 *  activity_led.h  -  a blinking LED that shows "the board is working"
 * ============================================================================
 *
 * A single embedding takes seconds, during which the board looks dead. This tiny
 * module blinks the on-board LED while a command is running, so you can SEE that
 * it is busy and not hung:
 *
 *      command arrives  ->  LED starts blinking (5 times a second)
 *      reply sent       ->  LED goes dark
 *
 * Even a command that finishes in a millisecond produces one visible flash, so
 * every command is acknowledged. After start-up, three quick flashes mean "ready".
 *
 * HOW IT STAYS OUT OF THE WAY
 * ---------------------------
 * The blinking is done by its own FreeRTOS task, pinned to CPU core 1. The neural
 * network runs on core 0, so the LED never steals time from it. When nothing is
 * happening, the task sleeps and uses no CPU at all (it waits to be woken by a
 * "task notification", see activity_led.c).
 *
 * The LED pin and type are chosen in menuconfig ("Board LED"); see board_led.h.
 */
#ifndef ACTIVITY_LED_H
#define ACTIVITY_LED_H

/* Set up the LED and start its task. Call once at start-up. If the LED cannot be
 * set up, the other functions quietly do nothing: a missing LED must never stop
 * the model from working. */
void activity_led_init(void);

/* Three quick flashes: "I have started and I am ready for commands". Returns
 * immediately; the flashing happens in the background. */
void activity_led_ready(void);

/* A command has started: begin blinking. Returns immediately. */
void activity_led_start(void);

/* The command has finished: stop blinking after the current flash. */
void activity_led_stop(void);

#endif /* ACTIVITY_LED_H */
