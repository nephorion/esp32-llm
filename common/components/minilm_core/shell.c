/*
 * ============================================================================
 *  shell.c  -  a tiny command shell on the serial port
 * ============================================================================
 *
 * This is the program's "user interface" (each project's main.c just calls
 * minilm_shell_run). It is only about talking to the PC; all the AI lives in bert.c.
 *
 * THE PROTOCOL
 * ------------
 * The chip and the PC exchange plain text lines over the USB serial port:
 *
 *     PC    ->  embed hello world         one command per line
 *     board ->  TOKENS 4                  zero or more reply lines...
 *     board ->  VEC 0.0123 -0.0345 ...
 *     board ->  TIME_MS 1420
 *     board ->  OK                        ...and ALWAYS a last line: "OK" or "ERR ..."
 *
 * Ending every reply with a marker line is the key trick. Without it the PC would
 * not know whether the board has finished or is still thinking (an embedding
 * takes seconds). tools/device.py reads lines until it sees OK or ERR.
 *
 * COMMANDS
 *   help                      list commands
 *   ident                     which firmware and model are on the board (for the host tools)
 *   info                      memory and model facts
 *   tokens <text>             show how the text is split into word pieces
 *   embed <text>              print the 384-number embedding
 *   dist <text a> | <text b>  similarity and distance between two texts
 *
 * Because it is just text, you can also type to it with any serial terminal.
 */
#include <stdio.h>
#include <string.h>
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "activity_led.h"
#include "board_ident.h"
#include "bert.h"
#include "model.h"
#include "tokenizer.h"
#include "minilm_shell.h"

static model_t model;                    /* the table of pointers into flash (small) */
static const minilm_hooks_t *hooks;      /* optional additions by the project, or NULL */

/* The version of the serial protocol this firmware speaks. Bump it when a change would
 * break the host tools (commands renamed or removed, reply format changed). It must equal
 * PROTO_VERSION in projects/minilm/tools/device.py: the host refuses a board with a different number and
 * offers to flash the right firmware. */
#define PROTO_VERSION 1

/* Is a usable model in flash? If not we still run, so the host can ask `ident`, see
 * model=missing and offer to flash the model, instead of finding a dead board. */
static bool model_ok;
static const char *model_state = "missing";   /* "ok", "missing" or "incompatible" */

/* Two result vectors, kept as globals so they don't use the (small) task stack.
 * `dist` needs both; `embed` uses only vec_a. */
static float vec_a[HIDDEN], vec_b[HIDDEN];

/* Milliseconds since boot, as a double. esp_timer_get_time() counts microseconds
 * with a 64-bit counter, so it never wraps in practice. */
static double now_ms(void) { return esp_timer_get_time() / 1000.0; }

/* ---- timing report -------------------------------------------------------
 * Prints one "TIMING <name> <microseconds>" line per phase, using the stopwatches
 * inside bert.c (see bert_timing_t in bert.h). The PC script turns these into a
 * table with percentages. Microseconds keep the numbers whole and exact. */
static void print_timing(void)
{
    const bert_timing_t *t = bert_timing_get();
    printf("TIMING total %lld\n",          (long long)t->total_us);
    printf("TIMING tokenize %lld\n",       (long long)t->tokenize_us);
    printf("TIMING lookup %lld\n",         (long long)t->lookup_us);
    printf("TIMING weight_fetch %lld\n",   (long long)t->weight_fetch_us);
    printf("TIMING dot %lld\n",            (long long)t->dot_us);
    printf("TIMING attention %lld\n",      (long long)t->attention_us);
    printf("TIMING layer_norm %lld\n",     (long long)t->layer_norm_us);
    printf("TIMING gelu %lld\n",           (long long)t->gelu_us);
    printf("TIMING residual %lld\n",       (long long)t->residual_us);
    printf("TIMING pool %lld\n",           (long long)t->pool_us);
}

/* ---- command: ident ------------------------------------------------------ *
 * Tells the host which firmware this is and what state the model data is in. The model
 * fields are what let tools/ensure reload the model when it is missing or is a different
 * file from the one on the PC (model_id is the content id in the file's header). */
static void cmd_ident(void)
{
    char extra[48];
    snprintf(extra, sizeof extra, "model=%s model_id=%08lx", model_state, (unsigned long)model.id);
    board_ident_print(PROTO_VERSION, extra);
}

/* ---- command: info ------------------------------------------------------- */
static void cmd_info(void)
{
    printf("model: %lu tokens in vocabulary, %d layers, %d-number embeddings\n",
           (unsigned long)model.vocab_size, N_LAYERS, HIDDEN);
    printf("limit: %d tokens per text (including [CLS] and [SEP])\n", MAX_TOKENS);
    /* A handy way to see how close we are to running out of fast RAM. */
    printf("internal RAM free: %u bytes\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
}

/* ---- command: tokens ----------------------------------------------------- *
 * Shows what the tokenizer did: very useful for understanding why two similar
 * sentences score differently (e.g. a rare word split into odd pieces). */
static void cmd_tokens(const char *text)
{
    int ids[MAX_TOKENS];
    int n = tokenize(&model, text, ids, MAX_TOKENS);
    printf("TOKENS %d:", n);
    for (int i = 0; i < n; i++) printf(" %s", token_string(&model, ids[i]));
    printf("\n");
}

/* ---- command: embed ------------------------------------------------------ *
 * Runs the network and prints all 384 numbers. The reply line "VEC ..." is
 * machine-readable (tools/device.py parses it); the REPL shows only the first 8. */
static void cmd_embed(const char *text)
{
    int n;
    bert_timing_reset();                          /* start the per-phase stopwatches from zero */
    double t0 = now_ms();
    bert_embed(&model, text, vec_a, &n);          /* the slow part: seconds */
    double ms = now_ms() - t0;

    printf("TOKENS %d\n", n);
    printf("VEC");
    for (int i = 0; i < HIDDEN; i++) printf(" %.6f", vec_a[i]);
    printf("\nTIME_MS %.0f\n", ms);               /* inference time only, not printing time */
    print_timing();
}

/* ---- command: dist ------------------------------------------------------- *
 * Input looks like  "the cat sat | a kitten rested".  We cut the line at the
 * '|' to get two texts, embed both, and compare them. */
static void cmd_dist(char *args)
{
    char *bar = strchr(args, '|');
    if (!bar) { printf("ERR usage: dist <text a> | <text b>\n"); return; }
    *bar = 0;                       /* overwrite '|' with a string terminator: `args` now ends
                                     * right before it, so the line is split into two C strings */
    const char *a = args, *b = bar + 1;

    bert_timing_reset();                        /* the report below covers BOTH embeddings */
    double t0 = now_ms();
    bert_embed(&model, a, vec_a, NULL);
    bert_embed(&model, b, vec_b, NULL);
    double ms = now_ms() - t0;

    /* Both vectors have length 1, so their dot product IS the cosine of the angle
     * between them: 1 = same direction (same meaning), 0 = unrelated. */
    float sim = 0;
    for (int i = 0; i < HIDDEN; i++) sim += vec_a[i] * vec_b[i];
    printf("SIM %.4f\nDIST %.4f\nTIME_MS %.0f\n", sim, 1.0f - sim, ms);   /* DIST = 1 - SIM */
    print_timing();
}

/* ---- dispatcher ---------------------------------------------------------- *
 * Look at the start of the line to decide which command it is. Each branch runs
 * the command, which prints its reply lines; after that we print the closing "OK".
 * (The command word ends in a space for commands that take text, e.g. "embed ",
 * so that "embedding" is not mistaken for "embed".) */
static void handle_command(char *line)
{
    while (*line == ' ') line++;                /* ignore leading spaces */

    if (!strncmp(line, "help", 4)) {
        printf("commands: help | ident | info | tokens <text> | embed <text> | dist <a> | <b>%s\n",
               (hooks && hooks->help) ? hooks->help : "");
    } else if (!strcmp(line, "ident")) {
        cmd_ident();
    } else if (!model_ok) {
        /* Everything below needs the model. */
        printf("ERR no usable model in flash (run: make flash-model)\n"); fflush(stdout); return;
    } else if (!strncmp(line, "info", 4)) {
        cmd_info();
    } else if (!strncmp(line, "tokens ", 7)) {
        cmd_tokens(line + 7);                   /* line + 7 skips the "tokens " prefix */
    } else if (!strncmp(line, "embed ", 6)) {
        cmd_embed(line + 6);
    } else if (!strncmp(line, "dist ", 5)) {
        /* Check the '|' here so a bad command ends with ERR instead of a stray OK. */
        if (!strchr(line + 5, '|')) { printf("ERR usage: dist <text a> | <text b>\n"); fflush(stdout); return; }
        cmd_dist(line + 5);
    } else {
        int handled = (hooks && hooks->command) ? hooks->command(&model, line) : 0;
        if (handled == 0) printf("ERR unknown command (try: help)\n");
        if (handled <= 0) { fflush(stdout); return; }
    }
    printf("OK\n");                             /* the end-of-reply marker the PC waits for */
    fflush(stdout);                             /* push it out NOW; stdout is buffered */
}

/* Run one command, with the activity LED blinking for as long as it takes.
 *
 * This wrapper exists so the LED start/stop happens in exactly ONE place. handle_command()
 * has several early `return`s (for errors); putting the LED calls inside it would mean
 * remembering to switch the LED off on every one of them. Wrapping it makes that
 * impossible to forget. */
static void handle(char *line)
{
    activity_led_start();
    handle_command(line);
    activity_led_stop();
}

/* ============================================================================
 *  minilm_shell_run - set up, then answer commands forever
 * ============================================================================ */
void minilm_shell_run(const minilm_hooks_t *project_hooks)
{
    hooks = project_hooks;

    /* Install the UART driver so we can READ bytes from the serial port. (Output
     * via printf works without it; input does not.) 512 = the size in bytes of
     * the receive buffer; the zeros mean "no transmit buffer, no event queue". */
    uart_driver_install(UART_NUM_0, 1024, 0, 0, NULL, 0);   /* receive buffer: room for a full-length command line */

    /* Locate the model in flash and allocate the working buffers.
     *
     * If the MODEL is missing or wrong we do NOT stop: we carry on and answer `ident` (with
     * model=missing), so the PC can tell what is wrong and offer to flash it. If the buffers
     * cannot be allocated that is a real fault, and there is nothing useful to run. */
    esp_err_t merr = model_load(&model);
    if (merr == ESP_OK) {
        if (bert_init() != ESP_OK) {
            printf("FATAL could not allocate the working buffers (see log above)\n");
            for (;;) vTaskDelay(1000);          /* sleep forever, politely */
        }
        model_ok = true;
        model_state = "ok";
        if (hooks && hooks->model_ready) hooks->model_ready(&model);
    } else if (merr == ESP_ERR_INVALID_VERSION) {
        model_state = "incompatible";           /* a model file, but not one this firmware understands */
    }                                           /* anything else: no model in flash ("missing") */

    /* Start the activity LED AFTER the model buffers are allocated: the neural network's
     * ~245 KB of working memory must be guaranteed first; the LED needs only a little. */
    activity_led_init();
    activity_led_ready();                       /* three quick flashes: "I am up" */

    /* The PC resets the board when it opens the port, then waits for this exact
     * word before sending anything, so no command arrives while we are booting. */
    printf("READY\n");
    fflush(stdout);

    /* The command loop: gather bytes into a line buffer; when a newline arrives,
     * run the line; repeat forever. */
    /* The longest command line we accept. The text is cut to 16 tokens anyway, but a
     * `dist` line carries TWO texts plus the command word, so it needs real room:
     * two 30-word sentences are about 350 characters. (An earlier 160-character buffer
     * silently chopped such lines in half and gave a wrong answer - see `overflow`.) */
    char line[512];
    int n = 0;                                  /* bytes collected so far */
    bool overflow = false;                      /* true once the current line has outgrown the buffer */
    for (;;) {
        uint8_t c;
        /* Read ONE byte, waiting as long as it takes (portMAX_DELAY). While we
         * wait, FreeRTOS runs other tasks and the CPU can idle. */
        if (uart_read_bytes(UART_NUM_0, &c, 1, portMAX_DELAY) != 1) continue;

        if (c != '\n' && c != '\r') {           /* an ordinary character: store it */
            if (n < (int)sizeof(line) - 1) line[n++] = (char)c;
            else overflow = true;               /* too long: remember it, and REFUSE the line below.
                                                 * Never run a half-received command - it could give
                                                 * a plausible but wrong answer. */
            continue;
        }
        if (overflow) {                         /* the line ended, but it did not fit */
            printf("ERR line too long (max %d characters)\n", (int)sizeof(line) - 1);
            fflush(stdout);
            n = 0;
            overflow = false;
            continue;
        }
        if (n == 0) continue;                   /* blank line, or the '\n' of a "\r\n" pair */
        line[n] = 0;                            /* terminate the string */
        n = 0;
        handle(line);                           /* takes seconds for embed/dist; input meanwhile
                                                 * waits safely in the UART's receive buffer */
    }
}
