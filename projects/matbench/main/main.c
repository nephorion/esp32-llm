/*
 * matbench - time the matmath kernels (common/components/matmath) on the chip.
 *
 * The usual protocol (READY, one command per line, reply lines then OK). Commands:
 *
 *   ident      which firmware this is
 *   info       CPU clock and free memory
 *   selftest   every implementation against plain C, for many lengths and offsets
 *   run        all the benchmarks; one line per case:
 *                RESULT,<op>,<type>,<impl>,<mem>,<shape>,<macs>,<cycles>,<ok|BAD>
 *              or SKIP,<op>,<type>,<mem>,<shape>,<reason>
 *
 * "cycles" is the CPU cycle count of ONE operation: the best of TRIALS trials, after one
 * warm-up call. "mem" is where the weight matrix lives; the vectors and results are
 * always in internal RAM, as activations would be.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_cpu.h"
#include "esp_clk_tree.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_random.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "matmath.h"
#include "driver/uart.h"
#include "board_ident.h"    /* the `ident` command (common/components/board_ident) */
#include "board_led.h"      /* the shared on-board LED (common/components/board_led) */

/* Version of this project's serial protocol. Bump it when a change would break the host
 * tools. The host side (see Makefile) must use the same number. */
#define PROTO_VERSION 1

#define TRIALS      5
#define MIN_CYCLES  2000000      /* repeat short operations until a trial lasts this long */

enum { MEM_INTERNAL, MEM_PSRAM, MEM_FLASH };
static const char *MEM_NAME[] = { "internal", "psram", "flash" };

static const struct { const char *name; mm_dot_s8_fn fn; } S8[] = {
    { "c", mm_dot_s8_c }, { "c_unroll4", mm_dot_s8_c4 },
    { "asm", mm_dot_s8_asm }, { "asm_simd", mm_dot_s8_simd },
};
static const struct { const char *name; mm_dot_f32_fn fn; } F32[] = {
    { "c", mm_dot_f32_c }, { "c_unroll4", mm_dot_f32_c4 }, { "asm", mm_dot_f32_asm },
};
#define COUNT(a) (sizeof(a) / sizeof((a)[0]))

/* One benchmark case. For dot: n = length. matvec: m = rows, k = cols. matmul: m x k x n. */
typedef enum { OP_DOT, OP_MATVEC, OP_MATMUL } op_t;
typedef struct { op_t op; size_t m, k, n; } shape_t;

static const shape_t SHAPES[] = {
    { OP_DOT, 1, 64, 1 }, { OP_DOT, 1, 384, 1 }, { OP_DOT, 1, 1536, 1 }, { OP_DOT, 1, 4096, 1 },
    { OP_MATVEC, 64, 64, 1 }, { OP_MATVEC, 128, 384, 1 }, { OP_MATVEC, 384, 384, 1 },
    { OP_MATMUL, 16, 64, 64 }, { OP_MATMUL, 16, 384, 128 },
};
static const char *OP_NAME[] = { "dot", "matvec", "matmul" };

/* a = the left operand (vector x, or matrix A) in internal RAM; w = the weights. */
typedef struct { const void *a, *w; void *out; shape_t s; } job_t;

static void run_s8(const job_t *j, mm_dot_s8_fn dot)
{
    switch (j->s.op) {
    case OP_DOT:    *(int32_t *)j->out = dot(j->w, j->a, j->s.k); break;
    case OP_MATVEC: mm_matvec_s8(dot, j->w, j->a, j->out, j->s.m, j->s.k); break;
    case OP_MATMUL: mm_matmul_s8(dot, j->a, j->w, j->out, j->s.m, j->s.k, j->s.n); break;
    }
}
static void run_f32(const job_t *j, mm_dot_f32_fn dot)
{
    switch (j->s.op) {
    case OP_DOT:    *(float *)j->out = dot(j->w, j->a, j->s.k); break;
    case OP_MATVEC: mm_matvec_f32(dot, j->w, j->a, j->out, j->s.m, j->s.k); break;
    case OP_MATMUL: mm_matmul_f32(dot, j->a, j->w, j->out, j->s.m, j->s.k, j->s.n); break;
    }
}

/* Cycles for one operation: warm up, choose a repeat count, keep the fastest trial. */
static uint32_t measure(const job_t *j, bool is_f32, int impl)
{
#define ONCE() do { if (is_f32) run_f32(j, F32[impl].fn); else run_s8(j, S8[impl].fn); } while (0)
    ONCE();
    uint32_t t0 = esp_cpu_get_cycle_count();
    ONCE();
    uint32_t one = esp_cpu_get_cycle_count() - t0;
    uint32_t reps = one >= MIN_CYCLES ? 1 : MIN_CYCLES / (one ? one : 1) + 1;
    uint32_t best = UINT32_MAX;
    for (int t = 0; t < TRIALS; t++) {
        t0 = esp_cpu_get_cycle_count();
        for (uint32_t r = 0; r < reps; r++) ONCE();
        uint32_t per = (esp_cpu_get_cycle_count() - t0) / reps;
        if (per < best) best = per;
    }
#undef ONCE
    return best;
}

static void *alloc16(size_t bytes, uint32_t caps)
{
    return heap_caps_aligned_alloc(16, bytes, caps | MALLOC_CAP_8BIT);
}

static void fill_s8(int8_t *p, size_t n) { esp_fill_random(p, n); }
static void fill_f32(float *p, size_t n)
{
    for (size_t i = 0; i < n; i++) p[i] = (int)(esp_random() % 2001) / 1000.0f - 1.0f;
}

/* Does `got` match the plain C answer `want`? int8 must be exact. Float versions add in a
 * different order, so allow for rounding (the inputs are within -1..1). */
static bool same(const void *got, const void *want, size_t count, bool is_f32, size_t terms)
{
    if (!is_f32) return memcmp(got, want, count * 4) == 0;
    const float *g = got, *w = want;
    for (size_t i = 0; i < count; i++)
        if (!(fabsf(g[i] - w[i]) <= 1e-6f * terms)) return false;
    return true;
}

static void bench_case(shape_t s, bool is_f32, int mem, const void *flash, size_t flash_len)
{
    size_t esize = is_f32 ? 4 : 1;
    size_t w_n = s.op == OP_MATMUL ? s.n * s.k : s.m * s.k;
    size_t a_n = s.op == OP_MATMUL ? s.m * s.k : s.k;
    size_t out_n = s.op == OP_MATMUL ? s.m * s.n : s.op == OP_MATVEC ? s.m : 1;
    size_t macs = s.op == OP_MATMUL ? s.m * s.k * s.n : s.m * s.k;
    const char *type = is_f32 ? "f32" : "s8";
    char shape[24];
    if (s.op == OP_DOT) snprintf(shape, sizeof shape, "%u", (unsigned)s.k);
    else if (s.op == OP_MATVEC) snprintf(shape, sizeof shape, "%ux%u", (unsigned)s.m, (unsigned)s.k);
    else snprintf(shape, sizeof shape, "%ux%ux%u", (unsigned)s.m, (unsigned)s.k, (unsigned)s.n);

    void *a = alloc16(a_n * esize, MALLOC_CAP_INTERNAL);
    void *out = alloc16(out_n * 4, MALLOC_CAP_INTERNAL);
    void *want = alloc16(out_n * 4, MALLOC_CAP_INTERNAL);
    void *ram = NULL;
    const void *w = NULL;
    if (mem == MEM_FLASH) {
        if (w_n * esize <= flash_len) w = flash;
    } else {
        ram = alloc16(w_n * esize, mem == MEM_PSRAM ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL);
        w = ram;
    }

    if (!a || !out || !want || !w) {
        printf("SKIP,%s,%s,%s,%s,does not fit\n", OP_NAME[s.op], type, MEM_NAME[mem], shape);
    } else {
        if (is_f32) fill_f32(a, a_n); else fill_s8(a, a_n);
        if (ram) { if (is_f32) fill_f32(ram, w_n); else fill_s8(ram, w_n); }

        job_t j = { .a = a, .w = w, .out = want, .s = s };
        if (is_f32) run_f32(&j, mm_dot_f32_c); else run_s8(&j, mm_dot_s8_c);
        j.out = out;

        int impls = is_f32 ? COUNT(F32) : COUNT(S8);
        for (int i = 0; i < impls; i++) {
            memset(out, 0xff, out_n * 4);
            uint32_t cycles = measure(&j, is_f32, i);
            bool ok = same(out, want, out_n, is_f32, s.k);
            printf("RESULT,%s,%s,%s,%s,%s,%u,%u,%s\n", OP_NAME[s.op], type,
                   is_f32 ? F32[i].name : S8[i].name, MEM_NAME[mem], shape,
                   (unsigned)macs, (unsigned)cycles, ok ? "ok" : "BAD");
            fflush(stdout);
            vTaskDelay(1);
        }
    }
    free(a); free(out); free(want); free(ram);
}

static void run_all(void)
{
    /* For the flash cases, map the app partition and use its bytes as int8 weights, so
     * they are read through the cache exactly as a memory-mapped model is. float32 is not
     * run from flash: arbitrary bytes read as floats include NaNs and denormals. */
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_APP,
                                                           ESP_PARTITION_SUBTYPE_ANY, NULL);
    const void *flash = NULL;
    size_t flash_len = 0;
    esp_partition_mmap_handle_t h;
    if (part && esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &flash, &h) == ESP_OK)
        flash_len = part->size;

    for (int is_f32 = 0; is_f32 <= 1; is_f32++)
        for (size_t i = 0; i < COUNT(SHAPES); i++)
            for (int mem = MEM_INTERNAL; mem <= (is_f32 ? MEM_PSRAM : MEM_FLASH); mem++)
                bench_case(SHAPES[i], is_f32, mem, flash, flash_len);

    if (flash_len) esp_partition_munmap(h);
}

/* Compare every implementation with plain C for lengths 0..100 at several start offsets,
 * which exercises the leftover-element and misaligned paths. Also the int8 extremes. */
static void selftest(void)
{
    enum { N = 4096, PAD = 16 };
    int8_t *a = alloc16(N + PAD, MALLOC_CAP_INTERNAL), *b = alloc16(N + PAD, MALLOC_CAP_INTERNAL);
    float *x = alloc16((N + PAD) * 4, MALLOC_CAP_INTERNAL), *y = alloc16((N + PAD) * 4, MALLOC_CAP_INTERNAL);
    unsigned checks = 0, bad = 0;
    if (a && b && x && y) {
        fill_s8(a, N + PAD); fill_s8(b, N + PAD);
        fill_f32(x, N + PAD); fill_f32(y, N + PAD);
        static const size_t OFF[] = { 0, 1, 4, 15 };
        for (size_t o = 0; o < COUNT(OFF); o++)
            for (size_t n = 0; n <= 100; n++) {
                size_t off = OFF[o];
                int32_t want = mm_dot_s8_c(a + off, b + off, n);
                for (size_t i = 1; i < COUNT(S8); i++, checks++)
                    bad += S8[i].fn(a + off, b + off, n) != want;
                size_t foff = off % 4;
                float fwant = mm_dot_f32_c(x + foff, y + foff, n);
                for (size_t i = 1; i < COUNT(F32); i++, checks++)
                    bad += !(fabsf(F32[i].fn(x + foff, y + foff, n) - fwant) <= 1e-4f);
            }
        /* Largest possible magnitudes, both signs: 4096 * (-128 * -128) and * (-128 * 127). */
        static const int8_t PAIR[][2] = { { -128, -128 }, { -128, 127 }, { 127, 127 } };
        for (size_t p = 0; p < COUNT(PAIR); p++) {
            memset(a, PAIR[p][0], N); memset(b, PAIR[p][1], N);
            for (size_t i = 0; i < COUNT(S8); i++, checks++)
                bad += S8[i].fn(a, b, N) != (int32_t)N * PAIR[p][0] * PAIR[p][1];
        }
        printf("SELFTEST checks=%u failed=%u\n", checks, bad);
    } else {
        printf("SELFTEST alloc_failed\n");
    }
    free(a); free(b); free(x); free(y);
}

static void info(void)
{
    uint32_t hz = 0;
    esp_clk_tree_src_get_freq_hz(SOC_MOD_CLK_CPU, ESP_CLK_TREE_SRC_FREQ_PRECISION_CACHED, &hz);
    printf("INFO cpu_mhz=%u internal_free=%u psram_free=%u\n", (unsigned)(hz / 1000000),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

static void handle(char *line)
{
    if (!strcmp(line, "info")) {
        info();
    } else if (!strcmp(line, "selftest")) {
        selftest();
    } else if (!strcmp(line, "run")) {
        run_all();
    } else if (!strcmp(line, "ident")) {
        board_ident_print(PROTO_VERSION, NULL);
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
