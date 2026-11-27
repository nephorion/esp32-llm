/* matmath.c - the C versions and the wrappers around the assembly. See matmath.h. */
#include "matmath.h"

/* In matmath_s3.S. */
int32_t mm_asm_dot_s8(const int8_t *a, const int8_t *b, size_t n);
int32_t mm_asm_dot_s8_x16(const int8_t *a, const int8_t *b, size_t blocks);
float mm_asm_dot_f32_x4(const float *a, const float *b, size_t blocks);

int32_t mm_dot_s8_c(const int8_t *a, const int8_t *b, size_t n)
{
    int32_t s = 0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

int32_t mm_dot_s8_c4(const int8_t *a, const int8_t *b, size_t n)
{
    int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return s0 + s1 + s2 + s3;
}

int32_t mm_dot_s8_asm(const int8_t *a, const int8_t *b, size_t n)
{
    return mm_asm_dot_s8(a, b, n);
}

int32_t mm_dot_s8_simd(const int8_t *a, const int8_t *b, size_t n)
{
    /* The vector load ignores the low 4 address bits, so a misaligned pointer would
     * silently read the wrong bytes. */
    if (((uintptr_t)a | (uintptr_t)b) & 15) return mm_dot_s8_c(a, b, n);

    size_t body = n & ~(size_t)15;
    int32_t s = mm_asm_dot_s8_x16(a, b, body / 16);
    for (size_t i = body; i < n; i++) s += a[i] * b[i];
    return s;
}

float mm_dot_f32_c(const float *a, const float *b, size_t n)
{
    float s = 0;
    for (size_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

float mm_dot_f32_c4(const float *a, const float *b, size_t n)
{
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

float mm_dot_f32_asm(const float *a, const float *b, size_t n)
{
    size_t body = n & ~(size_t)3;
    float s = mm_asm_dot_f32_x4(a, b, body / 4);
    for (size_t i = body; i < n; i++) s += a[i] * b[i];
    return s;
}

void mm_matvec_s8(mm_dot_s8_fn dot, const int8_t *w, const int8_t *x, int32_t *y,
                  size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) y[r] = dot(w + r * cols, x, cols);
}

void mm_matvec_f32(mm_dot_f32_fn dot, const float *w, const float *x, float *y,
                   size_t rows, size_t cols)
{
    for (size_t r = 0; r < rows; r++) y[r] = dot(w + r * cols, x, cols);
}

void mm_matmul_s8(mm_dot_s8_fn dot, const int8_t *a, const int8_t *bt, int32_t *out,
                  size_t m, size_t k, size_t n)
{
    /* Outer loop over B: each row of the (usually larger, slower) weight matrix is read
     * once and reused for every row of A while it is still in cache. */
    for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < m; i++) out[i * n + j] = dot(a + i * k, bt + j * k, k);
}

void mm_matmul_f32(mm_dot_f32_fn dot, const float *a, const float *bt, float *out,
                   size_t m, size_t k, size_t n)
{
    for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < m; i++) out[i * n + j] = dot(a + i * k, bt + j * k, k);
}
