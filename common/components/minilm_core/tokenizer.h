/*
 * ============================================================================
 *  tokenizer.h  -  turning text into numbers
 * ============================================================================
 *
 * WHY DO WE NEED A TOKENIZER?
 * ---------------------------
 * A neural network can only do arithmetic on numbers, so text must be converted
 * first. The obvious scheme (one number per letter) is wasteful and loses word
 * meaning; another obvious one (one number per whole word) needs an enormous
 * dictionary and cannot handle new words ("ChatGPT-ish", typos, names).
 *
 * BERT uses a middle path called WordPiece: a fixed vocabulary of common whole
 * words PLUS common word fragments. Any word can be built from fragments, so
 * nothing is ever truly "unknown":
 *
 *      "cats"          ->  cats                    (whole word is in the vocabulary)
 *      "unbelievable"  ->  unbelievable            (also a whole word)
 *      "descaling"     ->  desc  ##aling           (not a word we know: use pieces)
 *
 * The "##" prefix marks a piece that CONTINUES a word rather than starting one.
 * It matters because "##ing" and "ing" are different vocabulary entries.
 *
 * Each piece has an integer ID, which is simply its row number in the word
 * table (model.word). The network never sees letters, only these ids.
 *
 * SPECIAL TOKENS
 * --------------
 * Every text is wrapped:  [CLS]  piece piece piece ...  [SEP]
 *   [CLS]  "classification": marks the start. (BERT was trained with it.)
 *   [SEP]  "separator": marks the end.
 *   [UNK]  "unknown": used only when a word cannot be built from any pieces.
 *
 * THE RULES (these MUST match tools/minilm_ref.py exactly, or the board and the
 * PC will disagree about the same sentence)
 * ------------------------------------------------------------------------
 *   1. Lower-case everything. (The model is "uncased": "Cat" == "cat".)
 *   2. Letters and digits form words; any other character ends the current word.
 *   3. Spaces vanish. Every punctuation character becomes its own one-letter word.
 *   4. Split each word into the LONGEST known pieces, greedily from the left.
 *   5. A word that cannot be fully split becomes a single [UNK].
 *
 * LIMITATION: ASCII only. Accented and non-English characters are dropped.
 */
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "model.h"

/* Convert `text` into token ids.
 *   m        the model (we need its vocabulary and the ids of [CLS]/[SEP]/[UNK])
 *   text     NUL-terminated input string
 *   ids      output array, written from index 0
 *   max_ids  capacity of `ids`. Long text is TRUNCATED to fit; [SEP] is always
 *            kept as the last id.
 * Returns how many ids were written (always >= 2: [CLS] and [SEP]). */
int tokenize(const model_t *m, const char *text, int *ids, int max_ids);

/* Look up the printable string of a token id (e.g. 2023 -> "this"), for display
 * and debugging. Returns "?" for an id not in the vocabulary. SLOW (a linear
 * scan), so only use it for printing, never inside the inference path. */
const char *token_string(const model_t *m, int id);

#endif /* TOKENIZER_H */
