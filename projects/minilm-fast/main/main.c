/*
 * minilm-fast - MiniLM with switchable ways of doing the matrix multiply, to compare them.
 *
 * The program is the shared one (common/components/minilm_core). This file only replaces
 * linear(), through bert_set_linear(). Everything else (attention, GELU, norms) is unchanged.
 *
 * Serial command:
 *   mode                      show the switches
 *   mode key=value ...        change them
 *
 *     fetch=ref|fast      float kernel only: widen the int8 weight row one byte at a time
 *                         (as the original does) or from 32-bit reads
 *     kernel=float|simd   float: float32 multiply-adds (matmath mm_dot_f32_c4)
 *                         simd:  int8 weights x int8 activations (matmath mm_dot_s8_simd).
 *                         Each token's input vector is rounded to int8 with its own scale,
 *                         so the answers change slightly.
 *     cores=1|2           2: core 1 computes the second half of each matrix's rows
 *     v=int8|float        simd only: float keeps the attention "value" matrix on the float
 *                         path, which loses less accuracy
 *
 * At start-up: fetch=ref kernel=float cores=1, which runs the original linear() untouched.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "bert.h"
#include "matmath.h"
#include "minilm_shell.h"

static bool fetch_fast, simd, v_float;
static int cores = 1;
static const model_t *model;

/* One matrix multiply, shared by both cores. */
typedef struct {
    const qmat_t *W;
    const float *bias, *in;
    float *out;
    int T, in_dim, out_dim;
    bool simd, fetch_fast;
} job_t;

static job_t job;
static int64_t worker_fetch_us;
static SemaphoreHandle_t go, done;
static bool worker_ok;

/* Working memory in internal RAM. One weight row per core; the quantised activations are
 * written by core 0 before the cores start, then only read. 16-byte alignment is what the
 * SIMD load needs. */
static float row_f[2][INTERMEDIATE];
static int8_t row_q[2][INTERMEDIATE] __attribute__((aligned(16)));
static int8_t act_q[MAX_TOKENS * INTERMEDIATE] __attribute__((aligned(16)));
static float act_scale[MAX_TOKENS];

/* int8 -> float from 32-bit reads: a quarter as many loads from flash. Rows start on a
 * 4-byte boundary (the model file is laid out that way) and n is a multiple of 4. */
static void widen_fast(float *dst, const int8_t *src, int n)
{
    const uint32_t *p = (const uint32_t *)src;
    for (int i = 0; i < n; i += 4) {
        uint32_t w = *p++;
        dst[i]     = (float)(int8_t)w;                 /* little-endian: lowest byte first */
        dst[i + 1] = (float)(int8_t)(w >> 8);
        dst[i + 2] = (float)(int8_t)(w >> 16);
        dst[i + 3] = (float)((int32_t)w >> 24);
    }
}

/* Rows first..last-1 of the current job, using core `slot`'s buffers. */
static int64_t run_rows(int first, int last, int slot)
{
    const job_t *j = &job;
    int64_t fetch_us = 0;
    for (int r = first; r < last; r++) {
        const int8_t *w = j->W->w + (size_t)r * j->in_dim;
        const float scale = j->W->scale[r], b = j->bias[r];

        int64_t t0 = esp_timer_get_time();
        if (j->simd) {
            memcpy(row_q[slot], w, j->in_dim);
        } else if (j->fetch_fast) {
            widen_fast(row_f[slot], w, j->in_dim);
        } else {
            for (int i = 0; i < j->in_dim; i++) row_f[slot][i] = (float)w[i];
        }
        fetch_us += esp_timer_get_time() - t0;

        for (int t = 0; t < j->T; t++) {
            float sum = j->simd
                ? (float)mm_dot_s8_simd(row_q[slot], act_q + (size_t)t * j->in_dim, j->in_dim) * act_scale[t]
                : mm_dot_f32_c4(row_f[slot], j->in + (size_t)t * j->in_dim, j->in_dim);
            j->out[(size_t)t * j->out_dim + r] = sum * scale + b;
        }
    }
    return fetch_us;
}

static void worker(void *arg)
{
    for (;;) {
        xSemaphoreTake(go, portMAX_DELAY);
        worker_fetch_us = run_rows(job.out_dim / 2, job.out_dim, 1);
        xSemaphoreGive(done);
    }
}

/* Round each token's vector to int8: scale = largest magnitude / 127. */
static void quantise(const float *in, int T, int dim)
{
    for (int t = 0; t < T; t++) {
        const float *x = in + (size_t)t * dim;
        int8_t *q = act_q + (size_t)t * dim;
        float top = 0;
        for (int i = 0; i < dim; i++) { float a = fabsf(x[i]); if (a > top) top = a; }
        act_scale[t] = top > 0 ? top / 127.0f : 1.0f;
        const float inv = 1.0f / act_scale[t];
        for (int i = 0; i < dim; i++) {
            float v = x[i] * inv;
            int n = (int)(v >= 0 ? v + 0.5f : v - 0.5f);
            q[i] = n > 127 ? 127 : n < -127 ? -127 : n;
        }
    }
}

static bool is_value_matrix(const qmat_t *W)
{
    for (int l = 0; l < N_LAYERS; l++)
        if (W == &model->layers[l].v) return true;
    return false;
}

static int64_t fast_linear(const qmat_t *W, const float *bias, const float *in, float *out,
                           int T, int in_dim, int out_dim)
{
    bool use_simd = simd && !(v_float && is_value_matrix(W));
    if (use_simd) quantise(in, T, in_dim);
    /* In the simd modes the matrix kept in float uses the fast fetch too. */
    job = (job_t){ W, bias, in, out, T, in_dim, out_dim, use_simd, fetch_fast || simd };

    if (cores == 2) {
        xSemaphoreGive(go);
        int64_t mine = run_rows(0, out_dim / 2, 0);
        xSemaphoreTake(done, portMAX_DELAY);
        /* The two cores fetch at the same time: report the longer of the two. */
        return mine > worker_fetch_us ? mine : worker_fetch_us;
    }
    return run_rows(0, out_dim, 0);
}

static void show(void)
{
    printf("MODE fetch=%s kernel=%s cores=%d v=%s\n", fetch_fast ? "fast" : "ref",
           simd ? "simd" : "float", cores, v_float ? "float" : "int8");
}

static int command(model_t *m, const char *line)
{
    if (strncmp(line, "mode", 4) || (line[4] && line[4] != ' ')) return 0;

    /* Read all the settings before applying any, so a typo changes nothing. */
    bool f = fetch_fast, s = simd, vf = v_float;
    int c = cores;
    char args[96];
    if (strlen(line + 4) >= sizeof args) {       /* never apply a silently shortened line */
        printf("ERR mode line too long\n");
        return -1;
    }
    snprintf(args, sizeof args, "%s", line + 4);
    for (char *a = strtok(args, " "); a; a = strtok(NULL, " ")) {
        if      (!strcmp(a, "fetch=ref"))    f = false;
        else if (!strcmp(a, "fetch=fast"))   f = true;
        else if (!strcmp(a, "kernel=float")) s = false;
        else if (!strcmp(a, "kernel=simd"))  s = true;
        else if (!strcmp(a, "cores=1"))      c = 1;
        else if (!strcmp(a, "cores=2")) {
            if (!worker_ok) { printf("ERR the second-core worker did not start\n"); return -1; }
            c = 2;
        }
        else if (!strcmp(a, "v=int8"))       vf = false;
        else if (!strcmp(a, "v=float"))      vf = true;
        else {
            printf("ERR usage: mode [fetch=ref|fast] [kernel=float|simd] [cores=1|2] [v=int8|float]\n");
            return -1;
        }
    }
    fetch_fast = f; simd = s; cores = c; v_float = vf;
    /* With nothing switched on, use the original linear() itself. */
    bert_set_linear(fetch_fast || simd || cores == 2 ? fast_linear : NULL);
    show();
    return 1;
}

static void model_ready(model_t *m)
{
    model = m;
    go = xSemaphoreCreateBinary();
    done = xSemaphoreCreateBinary();
    /* Priority above the LED blinker (1), which also lives on core 1. */
    worker_ok = go && done && xTaskCreatePinnedToCore(worker, "matmul1", 4096, NULL, 5, NULL, 1) == pdPASS;
    if (!worker_ok) printf("warning: no second-core worker; cores=2 is unavailable\n");
}

void app_main(void)
{
    static const minilm_hooks_t hooks = { .model_ready = model_ready, .command = command,
                                          .help = " | mode [key=value ...]" };
    minilm_shell_run(&hooks);
}
