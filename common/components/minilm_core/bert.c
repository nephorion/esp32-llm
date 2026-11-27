/*
 * ============================================================================
 *  bert.c  -  the neural network itself
 * ============================================================================
 *
 * This is the C twin of tools/minilm_ref.py. If something here is confusing,
 * read the same function in that file: it is the same maths in a few lines of
 * numpy, with no memory management in the way.
 *
 * ----------------------------------------------------------------------------
 * THE BIG PICTURE
 * ----------------------------------------------------------------------------
 * A "sentence" arrives as T token ids. We give each token a vector of 384
 * numbers, and keep ALL T vectors in one array called `x` (T rows of 384).
 * Six identical-shaped LAYERS then rewrite `x` in place, each time making the
 * vectors more aware of their context. Inside every layer there are two steps:
 *
 *   1. ATTENTION   tokens look at EACH OTHER. After this step the vector for
 *                  "bank" in "river bank" differs from "bank" in "bank loan",
 *                  because it has absorbed information from its neighbours.
 *   2. FEED-FORWARD each token is then transformed ON ITS OWN by a small
 *                  two-layer network (more thinking per token, no mixing).
 *
 * At the end we AVERAGE the T vectors into one ("mean pooling") and scale it to
 * length 1. That single 384-number vector is the sentence embedding.
 *
 * ----------------------------------------------------------------------------
 * HOW MUCH WORK IS THIS?
 * ----------------------------------------------------------------------------
 * Per token, per layer, the matrix multiplies cost (multiply-adds, "MACs"):
 *
 *      q, k, v, o :  4 matrices of 384 x 384   =    589,824
 *      feed-forward: 384 x 1536  +  1536 x 384 =  1,179,648
 *                                       total  =  1,769,472  per token per layer
 *      x 6 layers                              = 10.6 million MACs per token
 *
 * So a 10-token sentence is ~106 million multiply-adds. The attention maths
 * (scores, softmax) is tiny by comparison. Nearly ALL the time is in linear().
 *
 * ----------------------------------------------------------------------------
 * HOW THE DATA IS STORED
 * ----------------------------------------------------------------------------
 * "T vectors of 384 numbers" is a flat float array, row after row:
 *
 *      x[ t*384 + i ]   =   number i of token t
 *
 *      index:    0 ......... 383 | 384 ......... 767 | 768 ...
 *                [  token 0    ] | [  token 1     ] | [ token 2 ...
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "bert.h"
#include "tokenizer.h"

/* Numbers per attention head: 384 / 12 = 32. */
#define HEAD_DIM (HIDDEN / N_HEADS)

/* ----------------------------------------------------------------------------
 * Working buffers. They live in `static` pointers so every function can use
 * them, and are allocated once in bert_init(). All hold floats.
 * ------------------------------------------------------------------------- */
static float *x;          /* T x 384   the token vectors; every layer rewrites these    */
static float *q, *k, *v;  /* T x 384   attention: queries, keys, values for each token  */
static float *ctx;        /* T x 384   attention's result: a mix of the value vectors   */
static float *tmp;        /* T x 384   scratch space for the output of a matrix multiply*/
static float *ffn;        /* T x 1536  the widened hidden layer of the feed-forward     */

/* One weight row converted from int8 to float (see linear()). Sized for the
 * longest row we ever handle (1536 = INTERMEDIATE). It is a plain global array
 * rather than a local variable because a local 6 KB array on the stack would
 * risk overflowing the task's stack. */
static float rowbuf[INTERMEDIATE];

/* ----------------------------------------------------------------------------
 * TIMING (observation only - nothing below changes any result)
 *
 * We want to know where the seconds go. The trick is a stopwatch around each
 * phase: read the microsecond clock before, read it after, add the difference
 * to that phase's total.
 *
 *      int64_t t = esp_timer_get_time();     <- start
 *      ...the work...
 *      tm.some_phase_us += esp_timer_get_time() - t;      <- stop and accumulate
 *
 * The TIMED() macro below is just that pattern packaged up, so the code that is
 * being measured stays easy to read.
 *
 * Two rules keep the measurement honest:
 *  - Never put a stopwatch INSIDE the innermost loops. Reading the clock costs a
 *    little; doing it millions of times would distort the very thing we measure.
 *    We time whole function calls, or once per weight ROW at the finest.
 *  - The stopwatches must not overlap, or the same time would be counted twice.
 * ------------------------------------------------------------------------- */
static bert_timing_t tm;                 /* running totals, in microseconds */
static int64_t lin_total_us;             /* all time spent inside linear()  */
static int64_t lin_fetch_us;             /* ...of which: fetching weights   */

/* TIMED(field, statements...): run the statements and add their duration to
 * tm.field. The "..." is so statements containing commas (like function calls
 * with several arguments) pass through the macro intact. */
#define TIMED(field, ...)                                           \
    do {                                                            \
        int64_t _t0 = esp_timer_get_time();                         \
        __VA_ARGS__;                                                \
        tm.field += esp_timer_get_time() - _t0;                     \
    } while (0)

void bert_timing_reset(void)
{
    memset(&tm, 0, sizeof tm);
    lin_total_us = lin_fetch_us = 0;
}

const bert_timing_t *bert_timing_get(void)
{
    /* linear() was timed as one block plus its "fetch" part; split it now:
     *   arithmetic = (total time in linear) - (time spent fetching weights) */
    tm.weight_fetch_us = lin_fetch_us;
    tm.dot_us = lin_total_us - lin_fetch_us;
    return &tm;
}

/* Allocate `n` floats from FAST internal RAM.
 *
 * ESP32-S3 has two kinds of RAM: ~390 KB of fast internal SRAM (17 ns latency)
 * and 8 MB of PSRAM (940 ns, about 50x slower - see the benchmark). Plain
 * malloc() might hand us either; asking for MALLOC_CAP_INTERNAL forces the fast
 * one. MALLOC_CAP_8BIT says "byte-addressable", which ordinary data needs. */
static float *alloc_floats(size_t n)
{
    return heap_caps_malloc(n * sizeof(float), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

esp_err_t bert_init(void)
{
    /* 6 buffers of 16 x 384 floats (24.6 KB each) + 1 of 16 x 1536 (98 KB) = ~245 KB */
    x   = alloc_floats(MAX_TOKENS * HIDDEN);
    q   = alloc_floats(MAX_TOKENS * HIDDEN);
    k   = alloc_floats(MAX_TOKENS * HIDDEN);
    v   = alloc_floats(MAX_TOKENS * HIDDEN);
    ctx = alloc_floats(MAX_TOKENS * HIDDEN);
    tmp = alloc_floats(MAX_TOKENS * HIDDEN);
    ffn = alloc_floats(MAX_TOKENS * INTERMEDIATE);
    /* heap_caps_malloc returns NULL on failure; check all of them at once. */
    return (x && q && k && v && ctx && tmp && ffn) ? ESP_OK : ESP_ERR_NO_MEM;
}

/* ============================================================================
 *  linear  -  out = in x W^T + bias        THE WORKHORSE: ~99% OF RUN TIME
 * ============================================================================
 *
 * What it computes. We have T input vectors of length `in_dim` and a weight
 * matrix W with `out_dim` rows of `in_dim` numbers. For every token t and every
 * output number j:
 *
 *      out[t][j] = ( W[j][0]*in[t][0] + W[j][1]*in[t][1] + ... ) * scale[j] + bias[j]
 *                  \_______________ dot product _______________/
 *
 * i.e. "how strongly does token t's vector line up with weight row j?". Each
 * row of W is a learned "feature detector"; the output vector says how much of
 * each of the `out_dim` features is present.
 *
 *      in  [T x in_dim]       T token vectors
 *      W   [out_dim x in_dim] int8 numbers + one float scale per row, IN FLASH
 *      out [T x out_dim]
 *
 * THE LOOP ORDER IS THE KEY DESIGN DECISION
 * -----------------------------------------
 * There are two ways to loop over (token, row):
 *
 *      for each token t:  for each row j: ...     <- re-reads ALL of W for every token
 *      for each row j:    for each token t: ...   <- reads each row of W ONCE   (ours)
 *
 * W is ~1.8 MB per layer and lives in slow flash (about 25 MB/s). Re-reading it for
 * every token would multiply the flash traffic by T. Ours fetches row j from
 * flash a single time, then reuses it for all T tokens while it is in fast RAM.
 * This is exactly why BERT suits a microcontroller: weights are reused across
 * the whole sentence. (An LLM generating one token at a time cannot do this.)
 *
 * WHY CONVERT THE ROW TO FLOAT FIRST?
 * -----------------------------------
 * The int8 -> float conversion costs a few CPU cycles. If we converted inside the
 * dot product, we would pay it once per token per weight (T times). Converting
 * the row once into `rowbuf` and reusing it pays it once per weight.
 *
 * WHY APPLY THE SCALE AT THE END?
 * -------------------------------
 * scale[j] is the same for the whole row, so   sum(w*x)*scale == sum((w*scale)*x).
 * Multiplying once after the sum is far cheaper than multiplying every weight.
 *
 * WHY FOUR RUNNING TOTALS (a0..a3)?
 * ---------------------------------
 * A single `acc += w*x` makes each step wait for the previous add to finish.
 * The floating-point unit has a pipeline several cycles deep, so it sits idle
 * most of the time. Four independent totals let it work on four multiply-adds
 * at once; we add them together at the end. (This is called "loop unrolling"
 * with multiple accumulators.) It requires in_dim to be a multiple of 4: both
 * 384 and 1536 are.
 *
 * Accuracy note: float addition is not perfectly associative, so splitting the
 * sum in four gives a result differing from the numpy version in about the 7th
 * decimal place. Harmless; validate_device.py confirms cosine = 1.000000.
 * ========================================================================= */
static void linear(const qmat_t *W, const float *bias, const float *in, float *out,
                   int T, int in_dim, int out_dim)
{
    for (int j = 0; j < out_dim; j++) {                  /* each weight row (a feature) */

        /* Step 1: fetch row j from flash and widen int8 -> float, ONCE. */
        const int8_t *wrow = W->w + (size_t)j * in_dim;  /* start of row j (row-major) */
        int64_t t_fetch = esp_timer_get_time();          /* stopwatch: weight fetch (once per ROW,
                                                          * never per element - see TIMING notes) */
        for (int i = 0; i < in_dim; i++) rowbuf[i] = (float)wrow[i];
        lin_fetch_us += esp_timer_get_time() - t_fetch;

        const float scale = W->scale[j];                 /* undoes the int8 quantisation */
        const float b = bias[j];

        /* Step 2: dot product of this row with EVERY token vector. */
        for (int t = 0; t < T; t++) {
            const float *xt = in + (size_t)t * in_dim;   /* token t's vector */
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;        /* four independent partial sums */
            for (int i = 0; i < in_dim; i += 4) {
                a0 += rowbuf[i]     * xt[i];
                a1 += rowbuf[i + 1] * xt[i + 1];
                a2 += rowbuf[i + 2] * xt[i + 2];
                a3 += rowbuf[i + 3] * xt[i + 3];
            }
            /* combine, undo quantisation, add bias, store as output feature j of token t */
            out[(size_t)t * out_dim + j] = (a0 + a1 + a2 + a3) * scale + b;
        }
    }
}

/* ============================================================================
 *  layer_norm  -  keep each token's numbers in a healthy range
 * ============================================================================
 *
 * After many additions and multiplications, the numbers in a vector can drift
 * to huge or tiny values, which makes later layers unstable. LayerNorm rescales
 * EACH TOKEN'S vector (independently) so its 384 numbers have mean 0 and
 * variance 1, then applies a learned per-number stretch (g, "gamma") and shift
 * (b, "beta") so the model can still choose a different range if it wants.
 *
 *      normalised[i] = (x[i] - mean) / sqrt(variance + eps)
 *      result[i]     = normalised[i] * g[i] + b[i]
 *
 * Tiny example, 4 numbers [2, 4, 6, 8]:
 *      mean = 5,  variance = ((-3)^2+(-1)^2+1^2+3^2)/4 = 5,  sqrt = 2.236
 *      normalised = [-1.34, -0.45, 0.45, 1.34]    (mean 0, spread 1)
 *
 * `eps` (1e-12 here, the model's own value) just prevents dividing by zero if
 * every number in a vector happens to be identical.
 *
 * The operation is done IN PLACE: `vec` is overwritten with the result.
 * ========================================================================= */
static void layer_norm(float *vec, int T, const float *g, const float *b)
{
    for (int t = 0; t < T; t++) {
        float *p = vec + (size_t)t * HIDDEN;             /* this token's vector */

        float mean = 0;
        for (int i = 0; i < HIDDEN; i++) mean += p[i];
        mean /= HIDDEN;

        float var = 0;                                   /* average squared distance from mean */
        for (int i = 0; i < HIDDEN; i++) { float d = p[i] - mean; var += d * d; }
        var /= HIDDEN;

        float inv = 1.0f / sqrtf(var + 1e-12f);          /* one division, reused 384 times */
        for (int i = 0; i < HIDDEN; i++) p[i] = (p[i] - mean) * inv * g[i] + b[i];
    }
}

/* ============================================================================
 *  gelu  -  the "activation function" of the feed-forward block
 * ============================================================================
 *
 * Why is it needed? If a network only ever did matrix multiplies, stacking
 * layers would be pointless: two multiplies in a row equal one bigger multiply.
 * An activation function adds a bend between them, which is what lets deep
 * networks represent complicated things.
 *
 * GELU is a smooth version of ReLU (which is just max(0, x)):
 *      large positive x  ->  ~x         (passes through)
 *      large negative x  ->  ~0         (squashed away)
 *      near zero         ->  a smooth blend
 *
 *      gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))      0.7071 = 1/sqrt(2)
 *
 * erff() is the single-precision "error function" from <math.h>. We use the
 * exact formula (not a cheaper approximation) so results match the original.
 * Done IN PLACE over n values.
 * ========================================================================= */
static void gelu(float *vec, size_t n)
{
    for (size_t i = 0; i < n; i++)
        vec[i] = 0.5f * vec[i] * (1.0f + erff(vec[i] * 0.70710678f));
}

/* ============================================================================
 *  attention  -  tokens deciding which other tokens matter to them
 * ============================================================================
 *
 * Inputs (already computed by linear()): q, k, v, each T x 384.
 * Output: ctx, T x 384.
 *
 * The library analogy. For every token t:
 *   - its QUERY q[t] is a question ("I'm a verb, who is my subject?").
 *   - every token s has a KEY k[s] (a label: "I'm a noun, singular") and a
 *     VALUE v[s] (the information it hands over if chosen).
 *   - how well the question matches each label gives a SCORE.
 *   - softmax turns the scores into weights that are positive and sum to 1.
 *   - the answer is the weighted average of the values.
 *
 *      score[s]  = ( q[t] . k[s] ) / sqrt(32)         "how relevant is s to t?"
 *      weight[s] = exp(score[s]) / sum of exp(score)  "softmax"
 *      ctx[t]    = sum over s of weight[s] * v[s]
 *
 * Why divide by sqrt(32)? Dot products of 32 numbers grow with the length; left
 * alone, the scores get large, softmax becomes extremely "sharp" (one weight
 * ~1, the rest ~0) and learning/behaviour degrades. Dividing by sqrt(HEAD_DIM)
 * keeps the scores in a sensible range.
 *
 * Softmax worked example, scores [2.0, 1.0, 0.0]:
 *      exp -> [7.39, 2.72, 1.00]   sum = 11.11
 *      weights = [0.665, 0.245, 0.090]   (sums to 1; the biggest score wins most)
 *
 * MULTI-HEAD: the 384 numbers are cut into 12 slices of 32 ("heads"). Each head
 * does the whole process above on its own slice only. Different heads can learn
 * to pay attention to different things (nearby words, the previous noun...).
 * Their results sit side by side in ctx, and the `o` matrix mixes them later.
 *
 * Note there is NO masking: we process exactly T real tokens with no padding,
 * so every token may look at every other token (BERT is "bidirectional").
 * ========================================================================= */
static void attention(int T)
{
    const float inv_sqrt_d = 1.0f / sqrtf((float)HEAD_DIM);
    float score[MAX_TOKENS];                     /* one query's scores against all keys */

    for (int h = 0; h < N_HEADS; h++) {
        const int off = h * HEAD_DIM;            /* this head owns numbers off .. off+31 */

        for (int t = 0; t < T; t++) {
            const float *qt = q + (size_t)t * HIDDEN + off;   /* this query, this head's slice */

            /* 1. Score this query against every key (dot product of 32 numbers).
             *    Remember the largest score for the next step. */
            float maxv = -1e30f;
            for (int s = 0; s < T; s++) {
                const float *ks = k + (size_t)s * HIDDEN + off;
                float d = 0;
                for (int i = 0; i < HEAD_DIM; i++) d += qt[i] * ks[i];
                score[s] = d * inv_sqrt_d;
                if (score[s] > maxv) maxv = score[s];
            }

            /* 2. Softmax. We subtract the maximum before exp(): this changes
             *    nothing mathematically (it cancels in the division) but stops
             *    exp() overflowing to infinity when a score is large. */
            float sum = 0;
            for (int s = 0; s < T; s++) { score[s] = expf(score[s] - maxv); sum += score[s]; }
            for (int s = 0; s < T; s++) score[s] /= sum;       /* now they sum to 1 */

            /* 3. Weighted average of the value vectors (this head's slice). */
            float *c = ctx + (size_t)t * HIDDEN + off;
            for (int i = 0; i < HEAD_DIM; i++) c[i] = 0;
            for (int s = 0; s < T; s++) {
                const float *vs = v + (size_t)s * HIDDEN + off;
                for (int i = 0; i < HEAD_DIM; i++) c[i] += score[s] * vs[i];
            }
        }
    }
}

/* ============================================================================
 *  encoder_layer  -  one full transformer layer: attention, then feed-forward
 * ============================================================================
 *
 * Reads and rewrites `x` (the T token vectors). Called six times, each time with
 * a different layer's weights.
 *
 * RESIDUAL CONNECTIONS ("x += tmp"): after each block we ADD the block's output
 * to the vector it started from, rather than replacing it. Each block therefore
 * only has to learn a small CORRECTION, and information from earlier layers is
 * never lost. This is what makes deep networks trainable at all.
 *
 *      x  ->  [ block ]  ->  tmp          x_new = LayerNorm( x + tmp )
 * ========================================================================= */
static void encoder_layer(const layer_t *L, int T)
{
    /* Stopwatch helper for linear(): adds to the "all of linear()" total. The
     * fetch-vs-arithmetic split is worked out inside linear() and in
     * bert_timing_get(). */
#define LINEAR(...) do { int64_t _l = esp_timer_get_time(); linear(__VA_ARGS__); \
                         lin_total_us += esp_timer_get_time() - _l; } while (0)

    /* ---- Block 1: self-attention ---------------------------------------- */

    /* Project every token three ways: as a query, a key and a value. */
    LINEAR(&L->q, L->q_b, x, q, T, HIDDEN, HIDDEN);
    LINEAR(&L->k, L->k_b, x, k, T, HIDDEN, HIDDEN);
    LINEAR(&L->v, L->v_b, x, v, T, HIDDEN, HIDDEN);

    TIMED(attention_us, attention(T));                   /* fills ctx */

    /* Mix the 12 heads' results together, add the original (residual), normalise. */
    LINEAR(&L->o, L->o_b, ctx, tmp, T, HIDDEN, HIDDEN);
    TIMED(residual_us, for (int i = 0; i < T * HIDDEN; i++) x[i] += tmp[i]);
    TIMED(layer_norm_us, layer_norm(x, T, L->attn_g, L->attn_b));

    /* ---- Block 2: feed-forward (each token independently) --------------- *
     * Widen 384 -> 1536 (room to compute many features), bend with GELU, then
     * narrow 1536 -> 384 back to the working size. */
    LINEAR(&L->f1, L->f1_b, x, ffn, T, HIDDEN, INTERMEDIATE);
    TIMED(gelu_us, gelu(ffn, (size_t)T * INTERMEDIATE));
    LINEAR(&L->f2, L->f2_b, ffn, tmp, T, INTERMEDIATE, HIDDEN);
    TIMED(residual_us, for (int i = 0; i < T * HIDDEN; i++) x[i] += tmp[i]);   /* residual again */
    TIMED(layer_norm_us, layer_norm(x, T, L->out_g, L->out_b));
#undef LINEAR
}

/* ============================================================================
 *  bert_embed  -  the whole pipeline, start to finish
 * ============================================================================ */
esp_err_t bert_embed(const model_t *m, const char *text, float *out, int *n_tokens)
{
    /* ---- Step 1: text -> token ids -------------------------------------- *
     * e.g. "cats!" -> four ids standing for  [CLS] cats ! [SEP]
     * MAX_TOKENS caps the length so our fixed-size buffers cannot overflow. */
    int64_t t_start = esp_timer_get_time();              /* stopwatch for the whole call */
    int ids[MAX_TOKENS];
    int T;
    TIMED(tokenize_us, T = tokenize(m, text, ids, MAX_TOKENS));
    if (n_tokens) *n_tokens = T;

    /* ---- Step 2: token ids -> starting vectors --------------------------- *
     * Each token's first vector is the SUM of three learned vectors:
     *   - its WORD embedding   (what the word is)        row `ids[t]` of the word table
     *   - its POSITION vector  (where it is: 1st, 2nd..) row `t` of the position table
     *   - the SEGMENT vector   (which sentence; always 0 for us)
     * The word table is stored as int8 in flash, so we undo the quantisation for
     * the one row we need:  real = int8 * scale. We only touch T rows out of
     * ~12,600, which is why a huge vocabulary costs storage but not time. */
    int64_t t_lookup = esp_timer_get_time();
    for (int t = 0; t < T; t++) {
        const int8_t *row = m->word.w + (size_t)ids[t] * HIDDEN;   /* row `ids[t]` of the word table */
        const float s = m->word.scale[ids[t]];
        float *xt = x + (size_t)t * HIDDEN;
        for (int i = 0; i < HIDDEN; i++)
            xt[i] = row[i] * s + m->pos[(size_t)t * HIDDEN + i] + m->type[i];
    }
    tm.lookup_us += esp_timer_get_time() - t_lookup;
    TIMED(layer_norm_us, layer_norm(x, T, m->emb_g, m->emb_b));

    /* ---- Step 3: six transformer layers ---------------------------------- *
     * Each call rewrites x in place; the vectors gradually absorb context. */
    for (int l = 0; l < N_LAYERS; l++) encoder_layer(&m->layers[l], T);

    /* ---- Step 4: mean pooling -------------------------------------------- *
     * We have T vectors but want ONE for the whole sentence. The simplest good
     * choice: average them. (The vectors already contain context, so [CLS] and
     * [SEP] participate like any other token. This matches how the model was
     * trained.) First sum, then divide by T. */
    int64_t t_pool = esp_timer_get_time();
    for (int i = 0; i < HIDDEN; i++) out[i] = 0;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < HIDDEN; i++) out[i] += x[(size_t)t * HIDDEN + i];

    float norm = 0;                                      /* sum of squares, for the length */
    for (int i = 0; i < HIDDEN; i++) { out[i] /= T; norm += out[i] * out[i]; }

    /* ---- Step 5: normalise to length 1 ----------------------------------- *
     * Divide by the vector's length (sqrt of the sum of squares). The result is a
     * "unit vector": it keeps its DIRECTION (which encodes the meaning) and drops
     * its size (which does not). Comparing two unit vectors by dot product then
     * gives their cosine similarity directly, with no extra division. */
    norm = 1.0f / sqrtf(norm);
    for (int i = 0; i < HIDDEN; i++) out[i] *= norm;
    tm.pool_us += esp_timer_get_time() - t_pool;         /* pooling + normalising */
    tm.total_us += esp_timer_get_time() - t_start;       /* the whole call */
    return ESP_OK;
}
