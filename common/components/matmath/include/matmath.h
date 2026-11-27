/*
 * matmath.h - small matrix kernels for the ESP32-S3, each written in more than one way.
 *
 * Every operation is built on a dot product, so there is one dot product per
 * (type, implementation) pair, and matrix-vector / matrix-matrix take it as an argument.
 *
 * int8 (quantised weights; the result is an exact int32)
 *   mm_dot_s8_c        plain C loop
 *   mm_dot_s8_c4       C, four accumulators
 *   mm_dot_s8_asm      hand-written scalar assembly, one multiply-add at a time
 *   mm_dot_s8_simd     S3 vector instructions, 16 multiply-adds per instruction
 *
 * float32 (the S3 FPU is scalar: there are no float vector instructions)
 *   mm_dot_f32_c       plain C loop
 *   mm_dot_f32_c4      C, four accumulators
 *   mm_dot_f32_asm     hand-written assembly, four accumulators, zero-overhead loop
 *
 * Limits
 *   - int8 results are int32: safe up to n = 131,071 (n * 128 * 128 < 2^31).
 *   - mm_dot_s8_simd is fast only when both pointers are 16-byte aligned (allocate with
 *     heap_caps_aligned_alloc(16, ...)). Otherwise it quietly uses C. Lengths that are
 *     not a multiple of 16 are fine; the tail is done in C. In a matrix, rows stay
 *     aligned only when `cols` is a multiple of 16.
 *   - mm_dot_s8_simd uses the vector registers, so call it from a task, never from an ISR.
 *   - The *_f32_asm / _c4 versions add in a different order from the plain loop, so their
 *     results can differ from it in the last bits.
 */
#ifndef MATMATH_H
#define MATMATH_H

#include <stddef.h>
#include <stdint.h>

typedef int32_t (*mm_dot_s8_fn)(const int8_t *a, const int8_t *b, size_t n);
typedef float (*mm_dot_f32_fn)(const float *a, const float *b, size_t n);

int32_t mm_dot_s8_c(const int8_t *a, const int8_t *b, size_t n);
int32_t mm_dot_s8_c4(const int8_t *a, const int8_t *b, size_t n);
int32_t mm_dot_s8_asm(const int8_t *a, const int8_t *b, size_t n);
int32_t mm_dot_s8_simd(const int8_t *a, const int8_t *b, size_t n);

float mm_dot_f32_c(const float *a, const float *b, size_t n);
float mm_dot_f32_c4(const float *a, const float *b, size_t n);
float mm_dot_f32_asm(const float *a, const float *b, size_t n);

/* y[r] = dot(w row r, x). w is rows x cols, row after row. */
void mm_matvec_s8(mm_dot_s8_fn dot, const int8_t *w, const int8_t *x, int32_t *y,
                  size_t rows, size_t cols);
void mm_matvec_f32(mm_dot_f32_fn dot, const float *w, const float *x, float *y,
                   size_t rows, size_t cols);

/* out = A x B^T:  out[i][j] = dot(A row i, B row j). A is m x k, B is n x k (that is, the
 * right-hand matrix is stored TRANSPOSED, so every dot product reads contiguous memory),
 * out is m x n. */
void mm_matmul_s8(mm_dot_s8_fn dot, const int8_t *a, const int8_t *bt, int32_t *out,
                  size_t m, size_t k, size_t n);
void mm_matmul_f32(mm_dot_f32_fn dot, const float *a, const float *bt, float *out,
                   size_t m, size_t k, size_t n);

#endif /* MATMATH_H */
