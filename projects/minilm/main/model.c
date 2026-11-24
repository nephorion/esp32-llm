/*
 * ============================================================================
 *  model.c  -  find the model in flash and work out where everything is
 * ============================================================================
 *
 * This file does NOT load weights. It computes ADDRESSES. When it finishes,
 * `model_t` holds ~100 pointers, each aimed at a tensor sitting in the
 * memory-mapped flash. The weights themselves are only fetched later, one cache
 * line at a time, as bert.c reads through those pointers.
 *
 * THE BOOKMARK TECHNIQUE
 * ----------------------
 * model.bin is one long run of bytes with no names or separators. The only way
 * to find tensor N is to know the size of tensors 0..N-1. So we keep a
 * "bookmark" (`cur`) that starts just after the header and moves forward past
 * each tensor as we claim it. The ORDER OF THE take_...() CALLS BELOW MUST
 * EXACTLY MATCH the order in tools/export_model.py. If they disagree, every
 * pointer after the first mismatch is garbage and the output is nonsense (with
 * no error message!). The numpy reference, minilm_ref.py, walks the file in the
 * same order, which is how we caught format mistakes during development.
 *
 * ALIGNMENT - a classic embedded-systems trap
 * -------------------------------------------
 * The Xtensa CPU can only load a 32-bit value from an address divisible by 4.
 * Reading a float from an odd address raises a hardware exception and the chip
 * reboots. That is why export_model.py pads every section to a multiple of 4
 * bytes, and why take_qmat() rounds its byte count up to a multiple of 4.
 * (int8 data itself may sit at any address: bytes have no alignment need.)
 */
#include <string.h>
#include "esp_log.h"
#include "esp_partition.h"
#include "model.h"

static const char *TAG = "model";   /* prefix for log messages */

/* The bookmark: points at the next unclaimed byte of the mapped file. */
static const uint8_t *cur;

/* Claim `n` consecutive float32 values and advance the bookmark past them.
 * Returns a pointer to the first one (it points into flash; nothing is copied). */
static const float *take_f32(size_t n)
{
    const float *p = (const float *)cur;
    cur += n * sizeof(float);
    return p;
}

/* Claim a quantised matrix of rows x cols. On disk it is:
 *      [ rows*cols int8 values ][ padding to 4 bytes ][ rows float32 scales ]
 * The `+ 3) & ~3` idiom rounds a size UP to the next multiple of 4:
 *      size 10 -> (10+3) & ~3 = 13 & ...11111100 = 12.   size 12 -> 12. */
static qmat_t take_qmat(size_t rows, size_t cols)
{
    qmat_t q;
    q.w = (const int8_t *)cur;
    cur += (rows * cols + 3) & ~(size_t)3;   /* skip the data plus its padding */
    q.scale = take_f32(rows);                /* then the per-row scales        */
    return q;
}

esp_err_t model_load(model_t *m)
{
    /* STEP 1 - find the partition by its label in partitions.csv. The type is
     * "data"; we leave the sub-type as ANY because ours is a custom one (0x40). */
    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part) {
        ESP_LOGE(TAG, "no 'model' partition - was the partition table flashed?");
        return ESP_ERR_NOT_FOUND;
    }

    /* STEP 2 - memory-map it. The chip's MMU (memory management unit) assigns
     * the flash region a range of virtual addresses and returns the start in
     * `base`. From now on `base[i]` reads flash byte i. `handle` would be used
     * to unmap later; we keep the mapping for the life of the program, so we
     * never need it again.
     *
     * Why not esp_partition_read() into a buffer? Because there is no RAM big
     * enough for a 15 MB buffer. Mapping costs no RAM at all. */
    const void *base;
    esp_partition_mmap_handle_t handle;
    esp_err_t err = esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &base, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "mmap failed: %s", esp_err_to_name(err));
        return err;
    }

    /* STEP 3 - read and sanity-check the 48-byte header (12 x uint32).
     *
     *   h[0] magic       0x364C4D4D = the bytes "MML6", a fingerprint that says
     *                    "this is one of our model files". An EMPTY flash reads
     *                    as 0xFFFFFFFF, so this catches "forgot flash-model".
     *   h[1] version     format version (1)
     *   h[2] vocab_size  rows in the word table
     *   h[3] hidden      must equal HIDDEN
     *   h[4] n_layers    must equal N_LAYERS
     *   h[5] n_heads     must equal N_HEADS
     *   h[6] inter       must equal INTERMEDIATE
     *   h[7] max_pos     position embeddings stored
     *   h[8..10]         ids of [UNK], [CLS], [SEP]
     *   h[11]            content id: the CRC-32 of every byte after this header, written
     *                    by export_model.py. It tells the host tools WHICH model.bin is in
     *                    flash (reported by the `ident` command), so they can reload it if it
     *                    is missing or out of date.
     */
    const uint32_t *h = (const uint32_t *)base;
    if (h[0] != 0x364C4D4D) {
        ESP_LOGE(TAG, "no model in flash (bad magic 0x%08lx). Run: make flash-model", (unsigned long)h[0]);
        return ESP_ERR_INVALID_STATE;
    }
    m->id = h[11];                               /* remember it even if the checks below fail */
    if (h[1] != 1 || h[3] != HIDDEN || h[4] != N_LAYERS || h[5] != N_HEADS || h[6] != INTERMEDIATE) {
        ESP_LOGE(TAG, "model file does not match this firmware's dimensions");
        return ESP_ERR_INVALID_VERSION;
    }
    m->vocab_size = h[2];
    m->max_pos    = h[7];
    m->unk_id = h[8];
    m->cls_id = h[9];
    m->sep_id = h[10];

    /* STEP 4 - walk the file. Put the bookmark just past the header. */
    cur = (const uint8_t *)base + 48;            /* 12 values * 4 bytes */

    /* 4a. Embedding stage. The word table has one row per vocabulary entry. */
    m->word  = take_qmat(m->vocab_size, HIDDEN);
    m->pos   = take_f32((size_t)m->max_pos * HIDDEN);
    m->type  = take_f32(HIDDEN);
    m->emb_g = take_f32(HIDDEN);
    m->emb_b = take_f32(HIDDEN);

    /* 4b. The six transformer layers, each in this fixed internal order. */
    for (int i = 0; i < N_LAYERS; i++) {
        layer_t *L = &m->layers[i];
        /* attention: query, key, value, output - matrix then its bias */
        L->q = take_qmat(HIDDEN, HIDDEN);  L->q_b = take_f32(HIDDEN);
        L->k = take_qmat(HIDDEN, HIDDEN);  L->k_b = take_f32(HIDDEN);
        L->v = take_qmat(HIDDEN, HIDDEN);  L->v_b = take_f32(HIDDEN);
        L->o = take_qmat(HIDDEN, HIDDEN);  L->o_b = take_f32(HIDDEN);
        L->attn_g = take_f32(HIDDEN);      L->attn_b = take_f32(HIDDEN);
        /* feed-forward: widen to 1536, then narrow back to 384 */
        L->f1 = take_qmat(INTERMEDIATE, HIDDEN);  L->f1_b = take_f32(INTERMEDIATE);
        L->f2 = take_qmat(HIDDEN, INTERMEDIATE);  L->f2_b = take_f32(HIDDEN);
        L->out_g = take_f32(HIDDEN);       L->out_b = take_f32(HIDDEN);
    }

    /* 4c. The vocabulary for the tokenizer. Layout:
     *        uint32 n
     *        uint32 offsets[n]            where each string starts in the blob
     *        uint16 ids[n]  + padding     (n*2 bytes rounded up to a multiple of 4)
     *        char   blob[]                the strings, each ending in '\0'    */
    m->vocab_n    = *(const uint32_t *)cur;  cur += 4;
    m->vocab_off  = (const uint32_t *)cur;   cur += 4 * (size_t)m->vocab_n;
    m->vocab_id   = (const uint16_t *)cur;   cur += (2 * (size_t)m->vocab_n + 3) & ~(size_t)3;
    m->vocab_blob = (const char *)cur;

    /* Sanity check for the format: the bookmark now sits at the START of the
     * string blob, so "MB used" below should be just under the size of model.bin
     * (short by the ~0.1 MB of vocabulary strings). A big gap is a quick hint
     * that this file and export_model.py disagree about the layout. */
    ESP_LOGI(TAG, "model mapped: %lu vocab entries, %.2f MB used of %.2f MB partition",
             (unsigned long)m->vocab_n, (double)(cur - (const uint8_t *)base) / 1e6,
             (double)part->size / 1e6);
    return ESP_OK;
}
