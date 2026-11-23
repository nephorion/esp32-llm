/*
 * ============================================================================
 *  bert.h  -  the public face of the neural network
 * ============================================================================
 *
 * One function does the real work: bert_embed(). Give it text, get back 384
 * numbers (a "sentence embedding") that capture the text's meaning. Texts with
 * similar meaning get vectors that point in similar directions.
 *
 * The vector has length exactly 1 ("unit vector"). That convenient fact means the
 * similarity of two texts is simply the dot product of their vectors:
 *
 *      similarity = a[0]*b[0] + a[1]*b[1] + ... + a[383]*b[383]
 *
 * (1.0 = same direction = same meaning; 0.0 = unrelated.) main.c does this.
 */
#ifndef BERT_H
#define BERT_H

#include "model.h"

/* The longest text we accept, counted in TOKENS and INCLUDING [CLS] and [SEP]
 * (so 16 means at most 14 word pieces). Longer text is silently cut off.
 *
 * Why a small limit? Every token needs a 384-float vector (1.5 KB) in each of
 * several working buffers, and attention compares every token with every other
 * (cost grows with the square of the length). At 16 tokens the buffers use about
 * 245 KB: they fit in the chip's FAST internal RAM. Raising this needs more RAM
 * (see the PSRAM exercise in the README) and makes inference slower. */
#define MAX_TOKENS 16

/* Allocate the working buffers. Call ONCE at start-up, before bert_embed().
 * Returns ESP_ERR_NO_MEM if the fast RAM is not available. */
esp_err_t bert_init(void);

/* Compute the sentence embedding of `text`.
 *
 *   m         the model from model_load()
 *   text      NUL-terminated text (ASCII; see tokenizer.h)
 *   out       caller-provided array of HIDDEN (384) floats; receives the result
 *   n_tokens  optional (may be NULL): receives how many tokens were used
 *
 * Takes roughly 1 to 3.5 seconds depending on the text length. NOT thread-safe:
 * it uses shared static buffers, so call it from one task at a time. */
esp_err_t bert_embed(const model_t *m, const char *text, float *out, int *n_tokens);

/* ---------------------------------------------------------------------------
 * Timing breakdown: WHERE did the time go?
 *
 * bert_embed() measures how long each phase takes, in MICROSECONDS, and ADDS it
 * to a running total. Call bert_timing_reset() before the work you want to
 * measure, then bert_timing_get() afterwards. (So a `dist` command, which embeds
 * two texts, naturally reports the combined time of both.)
 *
 * The phases do not overlap, so they add up to (almost) total_us. The small
 * remainder is loop overhead and the cost of the timer calls themselves.
 * ------------------------------------------------------------------------- */
typedef struct {
    int64_t total_us;        /* whole of bert_embed()                                     */
    int64_t tokenize_us;     /* text -> token ids                                         */
    int64_t lookup_us;       /* token id -> starting vector (word + position + segment)   */
    int64_t weight_fetch_us; /* inside linear(): read int8 weights from FLASH, make floats */
    int64_t dot_us;          /* inside linear(): the multiply-adds (pure arithmetic)      */
    int64_t attention_us;    /* tokens scoring/mixing each other                          */
    int64_t layer_norm_us;   /* rescaling the vectors                                     */
    int64_t gelu_us;         /* the activation function                                   */
    int64_t residual_us;     /* the "x += block output" additions                         */
    int64_t pool_us;         /* averaging the token vectors and scaling to length 1       */
} bert_timing_t;

void bert_timing_reset(void);
const bert_timing_t *bert_timing_get(void);

#endif /* BERT_H */
