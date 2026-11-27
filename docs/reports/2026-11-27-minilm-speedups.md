# MiniLM speed-ups on the ESP32-S3 (2026-11-27)

## Summary

A 16-token embedding went from **3.54 s to 0.73 s (4.9x)** on the same chip and model.

| Seconds per embedding | 3 tokens | 8 tokens | 16 tokens |
|---|---|---|---|
| Starting point (flash DIO, float, 1 core) | 1.283 | 2.140 | 3.538 |
| + QIO flash (now the default everywhere) | 1.040 | 1.897 | 3.294 |
| + int8 SIMD and 2 cores (`minilm-fast`, switchable) | 0.460 | 0.547 | 0.725 |
| **Total speedup** | **2.8x** | **3.9x** | **4.9x** |

QIO changes no answers. SIMD does, slightly: cosine 0.9970 (worst 0.9917) against the float
path. `projects/minilm` still runs float on one core; the SIMD and two-core paths are only in
`projects/minilm-fast`.

## Changes

Times are local (+11:00), all on 2026-11-27.

| Committed | Commits | PR | Change |
|---|---|---|---|
| 19:09 | `9fc0ba7`, `db20be5` | [#4](https://github.com/nephorion/esp32-llm/pull/4) | `matmath` kernels and the `matbench` benchmark |
| 19:27 | `4d9ffab`, `e66c371` | [#5](https://github.com/nephorion/esp32-llm/pull/5) | Shared MiniLM code; `minilm-psram` |
| 20:06 | `5a78699`, `375607a` | [#6](https://github.com/nephorion/esp32-llm/pull/6) | QIO flash in every project; host tools apply changed settings |
| 21:35 | `173ad99` | [#7](https://github.com/nephorion/esp32-llm/pull/7) | `minilm-fast` (SIMD, 2 cores, faster fetch); `bench` flash test fixed |

## Where in the code

| What | File | Look for |
|---|---|---|
| SIMD int8 dot product | [`matmath_s3.S`](../../common/components/matmath/matmath_s3.S) | `mm_asm_dot_s8_x16` |
| Kernel API and its limits | [`matmath.h`](../../common/components/matmath/include/matmath.h) | `mm_dot_s8_simd`, `mm_dot_f32_c4` |
| Second-core worker | [`minilm-fast/main/main.c`](../../projects/minilm-fast/main/main.c) | `worker`, `fast_linear`, `run_rows` |
| int8 activations | same file | `quantise` |
| Faster int8 to float | same file | `widen_fast` |
| Hook that replaces the matrix multiply | [`bert.c`](../../common/components/minilm_core/bert.c) | `bert_set_linear` |
| QIO flash | [`sdkconfig.common`](../../common/sdkconfig.common) | `CONFIG_ESPTOOLPY_FLASHMODE_QIO` |
| Applying changed settings on rebuild | [`firmware.py`](../../common/python/esp32tools/firmware.py) | `_refresh_sdkconfig` |
| Layers copied to PSRAM | [`minilm-psram/main/main.c`](../../projects/minilm-psram/main/main.c) | `copy_to_psram` |

## Findings

### What worked

| Technique | Gain at 16 tokens | Cost |
|---|---|---|
| int8 SIMD (1 core) | 3.31x over QIO float | Answers move slightly; more internal RAM |
| Second core, float | 1.66x | No meaningful change to answers |
| SIMD and 2 cores | 4.54x | Both of the above |
| QIO flash | weight fetch 0.743 s to 0.503 s | The flash chip must support it |

In the kernel benchmark, int8 SIMD is about 32x plain C on matrix-vector in internal RAM (643
million multiply-adds per second) and up to 45x on a long dot product. From flash, matrix-vector
is held to about 25 million multiply-adds per second (only 1.7x plain C), because memory cannot
feed it.

Keeping the attention value matrix in float (`v=float`) improves cosine to 0.9988 (worst
0.9966), for 0.824 s instead of 0.725 s on 2 cores. The closest sentence then matched for 38 of 40,
against 40 of 40 with all int8.

### What did not help much

| Tried | Result |
|---|---|
| Hand-written scalar assembly | No better than unrolled C |
| Faster int8 to float loop | Fetch 0.503 s to about 0.45 s only: flash speed is now the limit |
| 4 of 6 layers in PSRAM (40 MHz) | Helped with DIO (fetch 0.743 to 0.597 s); slightly worse with QIO (0.517 vs 0.503 s) |
| PSRAM at 80 MHz with QIO ¹ | Fetch 0.454 s with 4 layers in PSRAM, 0.487 s all in flash; not enabled |
| Lossless compression of weights ¹ | Shrinks them only to about 90%, too little to pay for decoding |
| 4-bit weights (numpy simulation) ¹ | Too inaccurate: cosine 0.92 to 0.97 for all layers |

¹ From one-off runs; no results file or script is in the repository.

### Rules of thumb for later (LLM) work

- Weights are read once per input, so a single input (token-by-token generation) is limited by
  memory speed. Sharing each weight row between several inputs helps a lot.
- All matrix weights are read equally often: there are no hot spots to cache.
- Measure before optimising: two of our guesses (the size of the PSRAM gain, and the fetch loop
  as the limit) were wrong.

## Limits of these results

- One board, short runs at room temperature; long-term stability is not tested.
- Accuracy is measured on 40 hand-written sentences, against the float path, not on a standard
  similarity benchmark.
- With 2 cores, the split between "weight fetch" and "arithmetic" is approximate.

## Open items

- Attention, GELU and the norms are still float on one core, and are now a larger share.
- The SIMD loop reaches 3.8 multiply-adds per cycle on a long dot product; its instruction
  order could be improved.
- Finer activation scales (per 32 values) looked much more accurate in simulation (cosine
  0.9997 ¹) but their speed cost is unmeasured.
- Not tried: a larger data cache, or PSRAM at 80 MHz in `matbench` and `minilm-fast`.

## How to reproduce

The model must be in flash (`make model flash-model` in `projects/minilm`).

```bash
. ~/esp/esp-idf/export.sh
make p=minilm-fast build flash && make p=minilm-fast run     # switches: speed and accuracy
make p=matbench build flash && make p=matbench run           # kernels
make p=bench build flash && make p=bench run                 # raw memory speeds
make p=minilm-psram build flash && make p=minilm-psram run   # layers in PSRAM
```

For the DIO starting point, remove `CONFIG_ESPTOOLPY_FLASHMODE_QIO=y` from
`common/sdkconfig.common`, then `make build flash` again.

Data behind the numbers:

- Starting point: [`psram_20261127_193333.csv`](../../projects/minilm-psram/results/psram_20261127_193333.csv) (rows with 0 layers)
- QIO: [`psram_20261127_200240.csv`](../../projects/minilm-psram/results/psram_20261127_200240.csv)
- Switches: [`speed_…205744.csv`](../../projects/minilm-fast/results/speed_20261127_205744.csv), [`accuracy_…205744.csv`](../../projects/minilm-fast/results/accuracy_20261127_205744.csv)
- Kernels: [`matbench_…200042.csv`](../../projects/matbench/results/matbench_20261127_200042.csv) (QIO), [`matbench_…191240.csv`](../../projects/matbench/results/matbench_20261127_191240.csv) (DIO)
- Memory: [`bench_20261127_205819.csv`](../../projects/bench/results/bench_20261127_205819.csv)
