/*
 * minilm-psram - MiniLM-L6 with as many layers' weights as fit copied into PSRAM.
 *
 * The program is the shared one (common/components/minilm_core). The neural network reads
 * every weight matrix through a pointer, so nothing in it has to change: at start-up we
 * copy whole layers from flash into PSRAM, then point the model at the copies.
 *
 * Only the int8 matrices are moved (q, k, v, o, f1, f2: about 1.77 MB per layer). They
 * are read once per embedding, all equally, so which layers we pick does not matter.
 * Scales, biases, norms, the word table and the vocabulary stay in flash.
 *
 * Extra commands:
 *   psram        PSRAM layers=<in use> max=<copied> copy_ms=<time to copy> free=<bytes>
 *   psram <n>    read the first n layers from PSRAM and the rest from flash (0..max)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "minilm_shell.h"

#define KEEP_FREE (256 * 1024)       /* PSRAM we leave unused */
#define MATRICES  6

static const int8_t *in_flash[N_LAYERS][MATRICES];
static int8_t *in_psram[N_LAYERS][MATRICES];
static int copied, in_use;           /* layers copied, and layers currently read from PSRAM */
static int64_t copy_us;

static const size_t SIZE[MATRICES] = {
    HIDDEN * HIDDEN, HIDDEN * HIDDEN, HIDDEN * HIDDEN, HIDDEN * HIDDEN,   /* q k v o */
    INTERMEDIATE * HIDDEN, HIDDEN * INTERMEDIATE,                         /* f1 f2   */
};

static qmat_t *matrix(model_t *m, int layer, int i)
{
    layer_t *L = &m->layers[layer];
    qmat_t *all[MATRICES] = { &L->q, &L->k, &L->v, &L->o, &L->f1, &L->f2 };
    return all[i];
}

static void use_layers(model_t *m, int n)
{
    for (int l = 0; l < N_LAYERS; l++)
        for (int i = 0; i < MATRICES; i++)
            matrix(m, l, i)->w = l < n ? in_psram[l][i] : in_flash[l][i];
    in_use = n;
}

static void copy_to_psram(model_t *m)
{
    size_t layer_bytes = 0;
    for (int i = 0; i < MATRICES; i++) layer_bytes += SIZE[i];

    int64_t t0 = esp_timer_get_time();
    for (int l = 0; l < N_LAYERS; l++)
        for (int i = 0; i < MATRICES; i++) in_flash[l][i] = matrix(m, l, i)->w;

    /* Whole layers only. */
    while (copied < N_LAYERS &&
           heap_caps_get_free_size(MALLOC_CAP_SPIRAM) >= layer_bytes + KEEP_FREE) {
        bool ok = true;
        for (int i = 0; i < MATRICES && ok; i++) {
            in_psram[copied][i] = heap_caps_malloc(SIZE[i], MALLOC_CAP_SPIRAM);
            if (in_psram[copied][i]) memcpy(in_psram[copied][i], in_flash[copied][i], SIZE[i]);
            else ok = false;
        }
        if (!ok) {
            for (int i = 0; i < MATRICES; i++) { free(in_psram[copied][i]); in_psram[copied][i] = NULL; }
            break;
        }
        copied++;
    }
    copy_us = esp_timer_get_time() - t0;
    use_layers(m, copied);
}

static int command(model_t *m, const char *line)
{
    if (strncmp(line, "psram", 5) || (line[5] && line[5] != ' ')) return 0;
    if (line[5]) {
        char *end;
        long n = strtol(line + 6, &end, 10);
        if (end == line + 6 || *end || n < 0 || n > copied) {
            printf("ERR usage: psram [0..%d]\n", copied);
            return -1;
        }
        use_layers(m, (int)n);
    }
    printf("PSRAM layers=%d max=%d copy_ms=%lld free=%u\n", in_use, copied,
           (long long)(copy_us / 1000), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    return 1;
}

void app_main(void)
{
    static const minilm_hooks_t hooks = { .model_ready = copy_to_psram, .command = command,
                                          .help = " | psram [n]" };
    minilm_shell_run(&hooks);
}
