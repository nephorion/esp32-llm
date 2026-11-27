/*
 * ============================================================================
 *  tokenizer.c  -  WordPiece tokenizer (see tokenizer.h for the big picture)
 * ============================================================================
 *
 * The program flow, for the text "Unbelievable cats!":
 *
 *   tokenize()  walks the text one character at a time, collecting letters and
 *               digits into a word buffer:   "unbelievable"  (lower-cased)
 *   at the space it has a complete word, so it calls
 *   wordpiece() to split that word into vocabulary pieces -> [unbelievable]
 *   then "cats" -> [cats]; then "!" is punctuation, so it is a word by itself
 *   -> [!]. Finally [SEP] is appended. Result:
 *
 *        [CLS] unbelievable cats ! [SEP]
 */
#include <ctype.h>
#include <string.h>
#include "tokenizer.h"

/* BERT's own limit: a "word" longer than 100 characters is not split at all but
 * replaced by [UNK]. This also bounds the size of our buffers below. */
#define MAX_WORD 100

/* ---------------------------------------------------------------------------
 * vocab_find - is the string `s` in the vocabulary? If so, what is its id?
 *
 * The vocabulary has ~12,600 strings. A linear search would compare against
 * thousands of strings per lookup, and wordpiece() does many lookups per word.
 * Instead export_model.py SORTED the strings alphabetically, so we can use
 * BINARY SEARCH: look at the middle string; if ours sorts before it, throw away
 * the upper half, otherwise the lower half; repeat. That needs at most
 * log2(12600) = 14 comparisons instead of up to 12,600.
 *
 *      [ ... a ... cat ... dog ... mat ... zoo ... ]
 *      looking for "dog":  middle = "mat" -> too high, search the left half...
 *
 * strcmp() returns <0, 0 or >0 like a "compare" function. It treats characters
 * as UNSIGNED bytes, and Python's sort of bytes strings does too, so the two
 * sides agree on the order. (If they disagreed, binary search would miss
 * entries and quietly produce [UNK]s.)
 *
 * Returns the token id, or -1 if `s` is not in the vocabulary.
 * ------------------------------------------------------------------------- */
static int vocab_find(const model_t *m, const char *s)
{
    int lo = 0, hi = (int)m->vocab_n - 1;        /* the part still being searched */
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        /* The strings live in a blob; vocab_off[mid] is where string `mid` starts. */
        int c = strcmp(s, m->vocab_blob + m->vocab_off[mid]);
        if (c == 0) return m->vocab_id[mid];     /* found it: return its token id */
        if (c < 0) hi = mid - 1;                 /* ours sorts earlier: go left   */
        else       lo = mid + 1;                 /* ours sorts later:   go right  */
    }
    return -1;
}

const char *token_string(const model_t *m, int id)
{
    /* The table is sorted by string, not by id, so finding an id means scanning. */
    for (uint32_t i = 0; i < m->vocab_n; i++)
        if (m->vocab_id[i] == id) return m->vocab_blob + m->vocab_off[i];
    return "?";
}

/* ---------------------------------------------------------------------------
 * wordpiece - split ONE word into vocabulary pieces
 *
 * The algorithm is "greedy longest-match-first":
 *   - Stand at the start of the word. Try the WHOLE rest of the word as a piece.
 *   - Not in the vocabulary? Chop one letter off the end and try again.
 *   - Once a piece matches, emit it, move past it, and repeat on what is left.
 *   - Pieces after the first are looked up with "##" in front.
 *
 * Worked example: "kitten", with a vocabulary that has "kit" and "##ten" but not
 * "kitten":
 *
 *      start=0:  try "kitten" (no)  "kitte" (no)  "kitt" (no)  "kit" (YES) -> emit kit
 *      start=3:  try "##ten"  (YES)                                       -> emit ##ten
 *      result:   kit ##ten
 *
 * Greedy is not guaranteed to find the best split, but it is what BERT was
 * trained with, so we must do exactly the same thing.
 *
 * If at some position NOTHING matches (not even a single letter), the whole word
 * is replaced by one [UNK] - never a half-split word.
 *
 *   word   the characters (NOT NUL-terminated; we are given the length n)
 *   out    where to write the ids (needs room for n entries: worst case is one
 *          piece per letter)
 * Returns how many ids were written.
 * ------------------------------------------------------------------------- */
static int wordpiece(const model_t *m, const char *word, int n, int *out)
{
    if (n > MAX_WORD) { out[0] = m->unk_id; return 1; }

    char cand[MAX_WORD + 3];       /* candidate string: "##" + up to 100 letters + NUL */
    int count = 0, start = 0;
    while (start < n) {
        int end = n, found = -1;
        while (end > start) {                    /* try the longest piece first, then shorter */
            int len = 0;
            if (start > 0) { cand[len++] = '#'; cand[len++] = '#'; }   /* continuation marker */
            memcpy(cand + len, word + start, end - start);
            len += end - start;
            cand[len] = 0;                       /* terminate so strcmp can use it */
            found = vocab_find(m, cand);
            if (found >= 0) break;               /* matched: stop shortening */
            end--;
        }
        if (found < 0) { out[0] = m->unk_id; return 1; }   /* even 1 letter failed: whole word is [UNK] */
        out[count++] = found;
        start = end;                             /* continue after the piece we just used */
    }
    return count;
}

int tokenize(const model_t *m, const char *text, int *ids, int max_ids)
{
    int n = 0;
    ids[n++] = m->cls_id;                        /* every text starts with [CLS] */

    char word[MAX_WORD + 2];                     /* the word being collected */
    int wlen = 0;
    int pieces[MAX_WORD + 1];                    /* scratch: ids of one split word */

    /* Visit every character, INCLUDING the final '\0' terminator: reaching it is
     * what flushes the last word, so there is no special "end of text" code. */
    for (const char *p = text;; p++) {
        /* Cast to unsigned char before using <ctype.h> functions: they take
         * values 0..255, and a plain (signed) char holding a byte above 127 would
         * be negative, which is undefined behaviour. */
        unsigned char ch = (unsigned char)*p;
        int is_word_char = (ch < 128 && isalnum(ch));    /* ASCII letter or digit */

        if (is_word_char) {
            /* Keep collecting. We store up to MAX_WORD+1 characters: one more
             * than allowed, purely so wordpiece() can SEE the word is too long. */
            if (wlen < MAX_WORD + 1) word[wlen++] = (char)tolower(ch);
            continue;
        }

        /* Anything else ends the current word (if we have one). */
        if (wlen > 0) {
            int np = wordpiece(m, word, wlen, pieces);
            /* Append, but stop early if the output is full. `max_ids - 1` keeps a
             * slot free for the closing [SEP]. This is how long text is truncated. */
            for (int i = 0; i < np && n < max_ids - 1; i++) ids[n++] = pieces[i];
            wlen = 0;
        }

        /* Punctuation (printable ASCII that is not a letter/digit/space) is a
         * word on its own. Spaces, control characters and non-ASCII bytes fall
         * through and are simply dropped: they only served as separators. */
        if (ch < 128 && ch > 32 && ch != 127) {
            char one[2] = { (char)ch, 0 };
            int np = wordpiece(m, one, 1, pieces);
            for (int i = 0; i < np && n < max_ids - 1; i++) ids[n++] = pieces[i];
        }
        if (ch == 0) break;                      /* end of the string */
    }

    ids[n++] = m->sep_id;                        /* every text ends with [SEP] */
    return n;
}
