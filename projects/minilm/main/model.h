/*
 * ============================================================================
 *  model.h  -  how the MiniLM-L6 model is laid out and found in flash
 * ============================================================================
 *
 * READ THIS FIRST. It introduces the data structures every other file uses.
 *
 * WHAT IS "THE MODEL"?
 * --------------------
 * A neural network is, at heart, a big pile of numbers ("weights") plus a fixed
 * recipe for combining them with your input. MiniLM-L6 has about 22 million of
 * these numbers. This file defines C structures that say WHERE each group of
 * numbers lives, so the maths code in bert.c can simply ask for "layer 3's
 * query matrix" without worrying about file offsets.
 *
 * WHERE DO THE NUMBERS LIVE?
 * --------------------------
 * In a dedicated FLASH partition called "model" (see partitions.csv). They are
 * 15.4 MB; the chip has only ~390 KB of free RAM, so they can NEVER be copied
 * into RAM. Instead we use "memory mapping" (esp_partition_mmap): the chip's
 * flash controller makes the flash chip appear at a normal memory address.
 * After that, reading `weights[i]` in C transparently fetches the byte from
 * the flash chip, through a small cache.
 *
 *     RAM-resident data:   fast (292 MB/s measured),   tiny   (~390 KB)
 *     Flash-mapped data:   slow (14.5 MB/s measured),  large  (15 MB)
 *
 * Every pointer in the structures below points INTO FLASH, not into RAM. The
 * structures themselves are only a few hundred bytes of pointers.
 *
 * WHY "const"?
 * ------------
 * Flash-mapped memory is read-only. Writing through these pointers would fault.
 * Marking them `const` makes the compiler stop us from trying.
 *
 * THE FILE FORMAT
 * ---------------
 * The byte-by-byte layout of model.bin is documented at the top of
 * tools/export_model.py. model.c reads it in exactly that order.
 */
#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include "esp_err.h"

/* ---------------------------------------------------------------------------
 * Fixed shape of MiniLM-L6-v2.
 *
 * These are compile-time constants (not read from the file) so the compiler can
 * optimise loops with known sizes. model_load() checks the file agrees, and
 * refuses to run if someone flashes a different model by mistake.
 * ------------------------------------------------------------------------- */

/* Every token is represented by a vector of this many numbers. A bigger number
 * means a richer representation, but more memory and more maths. */
#define HIDDEN        384

/* The network is a stack of identical-shaped "layers". Each one refines the
 * token vectors a little more. More layers = deeper understanding, slower. */
#define N_LAYERS      6

/* Attention is split into several independent "heads", each looking at the text
 * in its own way (one might track grammar, another topic...). Each head gets an
 * equal slice of the vector: 384 / 12 = 32 numbers per head. */
#define N_HEADS       12

/* Inside each layer, the "feed-forward" block widens the vector to this size,
 * applies a non-linearity, then narrows back to HIDDEN. Always 4 x HIDDEN. */
#define INTERMEDIATE  1536

/* ---------------------------------------------------------------------------
 * qmat_t - a QUANTISED matrix (a grid of numbers stored in compressed form)
 * ---------------------------------------------------------------------------
 *
 * The original weights are 32-bit floats (4 bytes each). To fit in flash we
 * store each as an 8-bit integer (1 byte, range -127..127) and keep ONE float
 * "scale" per matrix row. To recover the approximate original:
 *
 *        real_value  =  w[row * cols + col]  *  scale[row]
 *
 * Worked example, one row of three weights:
 *
 *        original floats:  [ 0.50  -1.27   0.03 ]
 *        biggest |value|:  1.27   ->   scale = 1.27 / 127 = 0.01
 *        stored int8:      [  50   -127      3  ]
 *        recovered:        [ 0.50  -1.27   0.03 ]   (here exactly; usually close)
 *
 * The matrix is stored ROW BY ROW ("row-major"): all of row 0, then all of row
 * 1, and so on. So row r starts at  w + r * cols.
 *
 * In a neural network a matrix usually means "turn an input vector of length
 * `cols` into an output vector of length `rows`": output number r is the
 * input vector multiplied element by element with row r, then summed. That is
 * a "dot product", and it is the operation this whole program spends its time
 * on (see linear() in bert.c).
 */
typedef struct {
    const int8_t *w;       /* rows x cols int8 values, row-major, in flash */
    const float  *scale;   /* one float per ROW, in flash                  */
} qmat_t;

/* ---------------------------------------------------------------------------
 * layer_t - every parameter of ONE transformer layer
 * ---------------------------------------------------------------------------
 *
 * Names follow the original BERT paper / Hugging Face:
 *
 *   q, k, v   "query", "key", "value" matrices for attention (384 x 384 each).
 *             Attention works like a library search: each token makes a QUERY
 *             ("what am I looking for?"), every token offers a KEY ("what do I
 *             contain?") and a VALUE ("what do I contribute if chosen?").
 *   o         "output" matrix: mixes the 12 heads' results back together.
 *   f1, f2    the feed-forward block's two matrices: 384->1536 and 1536->384.
 *   *_b       "biases": a fixed number added to each output (plain floats, a
 *             vector the same length as the matrix has rows).
 *   *_g, *_b  "gamma" and "beta" of a LayerNorm: a per-number scale and shift
 *             applied after normalising (see layer_norm() in bert.c). Note that
 *             attn_b / out_b here are LayerNorm betas, not matrix biases.
 */
typedef struct {
    qmat_t q, k, v, o;                  /* attention matrices, 384 x 384   */
    const float *q_b, *k_b, *v_b, *o_b; /* their biases, 384 floats each   */
    const float *attn_g, *attn_b;       /* LayerNorm after the attention   */
    qmat_t f1;                          /* feed-forward UP:   1536 x 384   */
    const float *f1_b;                  /* 1536 floats                     */
    qmat_t f2;                          /* feed-forward DOWN: 384 x 1536   */
    const float *f2_b;                  /* 384 floats                      */
    const float *out_g, *out_b;         /* LayerNorm after the feed-forward*/
} layer_t;

/* ---------------------------------------------------------------------------
 * model_t - the whole model: a table of pointers into flash
 * ---------------------------------------------------------------------------
 * Filled in once by model_load() at start-up, then only read.
 */
typedef struct {
    uint32_t vocab_size;   /* number of rows in the (pruned) word table      */
    uint32_t max_pos;      /* how many positions have an embedding (64)      */
    uint32_t unk_id;       /* id of the special token [UNK]  (unknown word)  */
    uint32_t cls_id;       /* id of [CLS]  (marks the start of every text)   */
    uint32_t sep_id;       /* id of [SEP]  (marks the end of every text)     */

    /* --- the embedding stage: turns a token id into its first vector ---- */
    qmat_t       word;     /* one row of 384 numbers per vocabulary entry    */
    const float *pos;      /* one vector per POSITION: tells the model that
                            * a token is 1st, 2nd, 3rd... in the text, since
                            * attention alone has no sense of order          */
    const float *type;     /* the "segment" vector (always segment 0 here;
                            * BERT can take two sentences and mark which is
                            * which - we only ever give it one)              */
    const float *emb_g, *emb_b;   /* LayerNorm after adding the three above  */

    layer_t layers[N_LAYERS];     /* the six transformer layers              */

    /* --- vocabulary: used ONLY by the tokenizer ------------------------- *
     * Three parallel arrays describe the n vocabulary strings, sorted
     * alphabetically so that tokenizer.c can binary-search them:
     *   entry i  =  the string  vocab_blob + vocab_off[i]
     *               whose word-table row (token id) is  vocab_id[i]
     * Note the entries are sorted by STRING, so vocab_id[i] is NOT i. */
    uint32_t        vocab_n;      /* number of entries                       */
    const uint32_t *vocab_off;    /* byte offset of each string in the blob  */
    const uint16_t *vocab_id;     /* the token id of each string             */
    const char     *vocab_blob;   /* all strings, each ending with '\0'      */
} model_t;

/* Find and map the "model" flash partition, check the header, and fill in *m
 * with pointers to every tensor. Returns ESP_OK, or an error code and logs why.
 * Takes microseconds: it only computes addresses, it reads no weights. */
esp_err_t model_load(model_t *m);

#endif /* MODEL_H */
