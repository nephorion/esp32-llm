# matbench

Times the matrix kernels in [`common/components/matmath`](../../common/components/matmath/include/matmath.h)
on the chip: plain C, unrolled C, hand-written assembly and the S3 vector (SIMD) instructions,
for int8 and float32, as dot product, matrix-vector and matrix-matrix.

```bash
. ~/esp/esp-idf/export.sh       # only needed if the tool has to build
make run                        # from this folder
```

`make run` asks the board which firmware it runs and offers to write matbench if it is
something else. It then runs a self-test (every implementation against plain C, including odd
lengths, misaligned pointers and the int8 extremes), runs all the cases, checks each result
against plain C and saves a CSV in `results/`. After **changing the code**, run
`make build flash` first: a board with the same protocol version is accepted as it is (the
tool warns when its build differs from yours).

## Findings (240 MHz, million multiply-adds per second)

From `results/matbench_20261127_200042.csv`
(flash in QIO mode at 80 MHz, PSRAM at 40 MHz, 32 KB data cache).

| Matrix-vector, 384 columns | plain C | best scalar | SIMD |
|---|---|---|---|
| int8, internal RAM (128 rows) | 19.9 | 29.4 (unrolled C) | 643 |
| int8, PSRAM (128 rows) | 14.6 | 19.0 | 33.6 |
| int8, flash (128 rows) | 15.0 | 19.9 | 25.5 |
| float32, internal RAM (128 rows) | 47.2 | 66.0 (unrolled C = asm) | none |
| float32, PSRAM (128 rows) | 8.3 | 8.3 | none |

- **SIMD int8 is the only large win from assembly**: about 32x over plain C in internal RAM, and
  up to 45x on a long dot product.
- **Scalar assembly does not beat good C.** For float32, GCC turns the unrolled C into the same
  instruction mix as the hand-written loop, in a different order (cycle counts within 1.5%). For int8, the scalar assembly
  beats the plain loop (GCC uses a branch there, not a zero-overhead loop) but loses to unrolled C.
- **Once the weights do not fit the 32 KB data cache, memory speed decides.** From flash, even SIMD
  gets only about 25 MMAC/s.
- **Reusing each weight row helps.** `matmul 16x384x128` reads each row once for 16 inputs and
  reaches 255 MMAC/s from flash with SIMD, against 26 for matrix-vector.
- Small cases show the same time for every memory because the weights sit in the cache.

**QIO flash** (now set in `common/sdkconfig.common`) is what moved the flash row: in DIO mode
SIMD matrix-vector from flash managed 15.2 MMAC/s (`results/matbench_20261127_191240.csv`).
Not tried here: PSRAM at 80 MHz, or a larger data cache.
