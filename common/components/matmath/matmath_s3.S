/*
 * matmath_s3.S - the assembly kernels, for the ESP32-S3 (Xtensa LX7).
 *
 * Calling convention (windowed ABI): after `entry`, the arguments are in a2, a3, a4 and
 * the return value goes back in a2. Floats are also returned in a2, as raw bits.
 * `loopnez aN, label` is a zero-overhead loop: it repeats the block aN times with no
 * counter or branch instructions, and skips it when aN is 0.
 */
    .text

/* int32_t mm_asm_dot_s8(const int8_t *a, const int8_t *b, size_t n) - scalar */
    .align  4
    .global mm_asm_dot_s8
    .type   mm_asm_dot_s8, @function
mm_asm_dot_s8:
    entry   a1, 32
    movi    a5, 0                   /* sum */
    loopnez a4, .Ls8_end
    l8ui    a6, a2, 0
    l8ui    a7, a3, 0
    sext    a6, a6, 7               /* there is no signed byte load */
    sext    a7, a7, 7
    addi    a2, a2, 1
    addi    a3, a3, 1
    mull    a6, a6, a7
    add     a5, a5, a6
.Ls8_end:
    mov     a2, a5
    retw.n
    .size   mm_asm_dot_s8, . - mm_asm_dot_s8

/* int32_t mm_asm_dot_s8_x16(const int8_t *a, const int8_t *b, size_t blocks)
 * Vector version. a and b must be 16-byte aligned; processes blocks * 16 elements. */
    .align  4
    .global mm_asm_dot_s8_x16
    .type   mm_asm_dot_s8_x16, @function
mm_asm_dot_s8_x16:
    entry   a1, 32
    ee.zero.accx                    /* the 40-bit accumulator */
    loopnez a4, .Lv_end
    ee.vld.128.ip   q0, a2, 16      /* load 16 bytes, advance pointer by 16 */
    ee.vld.128.ip   q1, a3, 16
    ee.vmulas.s8.accx q0, q1        /* ACCX += sum of the 16 products */
.Lv_end:
    rur.accx_0  a2                  /* low 32 bits of ACCX */
    retw.n
    .size   mm_asm_dot_s8_x16, . - mm_asm_dot_s8_x16

/* float mm_asm_dot_f32_x4(const float *a, const float *b, size_t blocks)
 * Processes blocks * 4 elements with four independent accumulators (f0..f3), so a
 * multiply-add does not have to wait for the previous one to leave the FPU pipeline. */
    .align  4
    .global mm_asm_dot_f32_x4
    .type   mm_asm_dot_f32_x4, @function
mm_asm_dot_f32_x4:
    entry   a1, 32
    movi    a5, 0
    wfr     f0, a5
    wfr     f1, a5
    wfr     f2, a5
    wfr     f3, a5
    loopnez a4, .Lf_end
    lsi     f4, a2, 0
    lsi     f8, a3, 0
    lsi     f5, a2, 4
    lsi     f9, a3, 4
    lsi     f6, a2, 8
    lsi     f10, a3, 8
    lsi     f7, a2, 12
    lsi     f11, a3, 12
    madd.s  f0, f4, f8
    madd.s  f1, f5, f9
    madd.s  f2, f6, f10
    madd.s  f3, f7, f11
    addi    a2, a2, 16
    addi    a3, a3, 16
.Lf_end:
    add.s   f0, f0, f1
    add.s   f2, f2, f3
    add.s   f0, f0, f2
    rfr     a2, f0
    retw.n
    .size   mm_asm_dot_f32_x4, . - mm_asm_dot_f32_x4
