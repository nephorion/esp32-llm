# minilm-fast

[MiniLM](../minilm/README.md) with switchable ways of doing the matrix multiply and the GELU
activation, to compare them on the chip. The program is the same shared code (`common/components/minilm_core`); this
project only replaces `linear()` and `gelu()`. The switches change while it runs, and with all of them off
it runs the original code.

The model must already be in flash (`make model flash-model` in `projects/minilm`).

```bash
. ~/esp/esp-idf/export.sh       # only needed if the tool has to build
make run                        # from this folder: compare the matrix switches, save CSVs
make gelu                       # tune the GELU switches (about 12 minutes)
```

After changing the code, run `make build flash` first.

| Serial command | Meaning |
|---|---|
| `mode` | show the switches |
| `mode fetch=ref\|fast` | float kernel: widen weights to float from single bytes, or from 32-bit reads |
| `mode kernel=float\|simd` | simd: int8 weights times int8 activations, using the [`matmath`](../../common/components/matmath/include/matmath.h) SIMD kernel |
| `mode cores=1\|2` | 2: core 1 computes half of each matrix's rows |
| `mode v=int8\|float` | simd only: float keeps the attention value matrix in float |
| `mode gelu=exact\|table\|relu` | GELU by `erff()`, by table lookup, or replaced by `max(0, x)` for comparison |
| `mode table=8..4096` | number of table intervals (default 256) |
| `mode gcores=1\|2` | 2: core 1 computes half of the GELU values |

Several can be given at once: `mode kernel=simd cores=2`. At start-up everything is off.
With `kernel=simd` the `fetch` switch has no effect.

## Speed

Seconds per embedding, from `results/speed_20261127_205744.csv` (240 MHz, flash QIO 80 MHz).

| Switches | 3 tokens | 8 tokens | 16 tokens | Speedup at 16 |
|---|---|---|---|---|
| original | 1.040 | 1.897 | 3.294 | 1.00x |
| fast fetch | 0.994 | 1.866 | 3.287 | 1.00x |
| 2 cores | 0.733 | 1.204 | 1.982 | 1.66x |
| fast fetch, 2 cores | 0.653 | 1.119 | 1.898 | 1.74x |
| simd | 0.544 | 0.707 | 0.994 | 3.31x |
| **simd, 2 cores** | **0.460** | **0.547** | **0.725** | **4.54x** |
| simd, v float | 0.581 | 0.803 | 1.184 | 2.78x |
| simd, v float, 2 cores | 0.476 | 0.593 | 0.824 | 4.00x |

## Accuracy

Against the original path on 40 sentences, from `results/accuracy_20261127_205744.csv`. The
second core does not change any answer.

| Switches | Cosine mean (worst) | Similarity score change, mean (worst) | Same closest sentence |
|---|---|---|---|
| fast fetch | 1.0000 (1.0000) | 0.0000 (0.0000) | 40 of 40 |
| simd | 0.9970 (0.9917) | 0.0055 (0.030) | 40 of 40 |
| simd, v float | 0.9988 (0.9966) | 0.0033 (0.016) | 38 of 40 |

The sentences are a small hand-written set, not a standard similarity benchmark.

## GELU

Times are for 16 tokens with `kernel=simd cores=2`. Accuracy is against the original float
code on the same 40 sentences, from `results/gelu_20261127_221914.csv`.

| GELU setting | Total | GELU part | Cosine, GELU change alone: mean (worst) | Cosine with simd: mean (worst) |
|---|---|---|---|---|
| exact | 0.722 s | 0.160 s | 1.0000 (1.0000) | 0.9970 (0.9917) |
| exact, `gcores=2` | 0.662 s | 0.100 s | 1.0000 (1.0000) | 0.9970 (0.9917) |
| table 8 | 0.582 s | 0.020 s | 0.9901 (0.9784) | 0.9870 (0.9747) |
| table 16 | 0.580 s | 0.020 s | 0.9993 (0.9987) | 0.9965 (0.9913) |
| table 64 | 0.582 s | 0.020 s | 0.999997 (0.999994) | 0.9971 (0.9935) |
| **table 256** | **0.580 s** | **0.020 s** | **1.000000 (1.000000)** | **0.9971 (0.9928)** |
| table 256, `gcores=2` | 0.570 s | 0.010 s | same as above | same as above |
| relu | 0.563 s | 0.003 s | 0.61 (0.49) | 0.61 (0.49) |

- **A table of 64 or more intervals makes GELU 8x faster at no meaningful cost in accuracy.**
  The table size does not change the speed, so 256 is a comfortable choice.
- **`relu` is unusable** for this model.
- **Do not compare two simd runs with each other.** int8 rounding is so sensitive that a
  change of 0.000001 anywhere reshuffles it (cosine about 0.998 between the two in a numpy
  test), without moving either further from the float answer. Always compare with float.

## What it shows

- **SIMD is the largest win.** It needs int8 activations, so answers move slightly.
- **The second core helps without changing answers:** 1.7x on long texts with the float kernel.
- **A faster int8 to float loop gains little** (0.503 to about 0.45 s of fetch). The fetch is now
  close to what the flash can deliver (`bench` reads it at 24.4 MB/s).
- With 2 cores, "weight fetch" and "arithmetic" are approximate: both cores work at once, and
  the longer of their two fetch times is reported.
- Attention and the norms still run in float on one core.
- The extra buffers and task leave about 55 KB of internal RAM free (117 KB in `minilm`).
