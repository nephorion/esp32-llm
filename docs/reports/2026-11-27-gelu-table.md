# GELU table lookup, and the total speed-up so far (2026-11-27)

Follows [MiniLM speed-ups](2026-11-27-minilm-speedups.md).

## Summary

Looking GELU up in a table instead of calling `erff()` makes that step **8x faster with no
meaningful loss of accuracy**. A 16-token embedding is now **6.2x faster than where we
started: 3.54 s to 0.57 s**.

| Seconds per embedding | 3 tokens | 8 tokens | 16 tokens |
|---|---|---|---|
| Starting point (flash DIO, float, 1 core) | 1.283 | 2.140 | 3.538 |
| Now (QIO, int8 SIMD, 2 cores, GELU table) | 0.431 | 0.471 | 0.570 |
| **Speedup** | **3.0x** | **4.5x** | **6.2x** |

Where the 16-token gain came from, adding one step at a time (the order changes each step's
share):

| Step | Time after | Gain of this step |
|---|---|---|
| Start | 3.538 s | |
| QIO flash | 3.294 s | 1.07x |
| int8 SIMD | 0.994 s | 3.31x |
| Second core for the matrix multiplies | 0.725 s | 1.37x |
| GELU table on both cores | 0.570 s | 1.27x |

The cost: cosine 0.9971 on average against the original float model (worst 0.9928), almost
all from int8 SIMD, and less free internal RAM (about 117 KB in `minilm`, 55 KB in
`minilm-fast`). Only QIO is on everywhere. The rest are switches in `minilm-fast`, off at
start-up.

## Changes

| Committed (2026-11-27, +11:00) | Commit | Change |
|---|---|---|
| 22:22 | `1c4d371` | GELU switches, `bert_set_gelu` hook, `tools/gelu.py` sweep |

## Where in the code

| What | File | Look for |
|---|---|---|
| The table and the three methods | [`minilm-fast/main/main.c`](../../projects/minilm-fast/main/main.c) | `build_table`, `gelu_range` |
| GELU split across cores | same file | `fast_gelu`, `worker` |
| Hook that replaces GELU | [`bert.c`](../../common/components/minilm_core/bert.c) | `bert_set_gelu` |
| Sweep tool | [`tools/gelu.py`](../../projects/minilm-fast/tools/gelu.py) | |

## How the table works

`gelu(x) = max(x, 0) - c(|x|)`, where `c(a) = a * (1 - erf(a / sqrt 2)) / 2`. The correction
`c` is the same for `x` and `-x` and fades to nothing, so the table only covers 0 to 5, with
linear interpolation. Beyond 5 the answer is simply `max(x, 0)`.

Tweaking parameters (serial): `mode gelu=exact|table|relu table=8..4096 gcores=1|2`.

## Findings

16 tokens with `kernel=simd cores=2`. Cosines are against the original float code on 40
sentences.

| GELU setting | Total | GELU part | Cosine, GELU alone: mean (worst) | Cosine with simd: mean (worst) |
|---|---|---|---|---|
| exact | 0.722 s | 0.160 s | 1.0000 (1.0000) | 0.9970 (0.9917) |
| exact, 2 cores | 0.662 s | 0.100 s | 1.0000 (1.0000) | 0.9970 (0.9917) |
| table 8 | 0.582 s | 0.020 s | 0.9901 (0.9784) | 0.9870 (0.9747) |
| table 16 | 0.580 s | 0.020 s | 0.9993 (0.9987) | 0.9965 (0.9913) |
| table 64 | 0.582 s | 0.020 s | 0.999997 (0.999994) | 0.9971 (0.9935) |
| table 256 | 0.580 s | 0.020 s | 1.000000 (1.000000) | 0.9971 (0.9928) |
| table 256, 2 cores | 0.570 s | 0.010 s | as above | as above |
| relu | 0.563 s | 0.003 s | 0.61 (0.49) | 0.61 (0.49) |

- 64 intervals or more is accurate. The size does not change the speed.
- With the table, a second core for GELU adds little (0.010 s).
- `relu` is unusable for this model.

### A measurement trap

The first sweep compared simd runs with *each other* and made even a 1024-entry table look
inaccurate. int8 rounding is so sensitive that a change of 0.000001 anywhere reshuffles it:
in a numpy test, two int8 runs then agree only to a cosine of about 0.998 ¹, while neither
moves further from the float answer. **Compare with the float answer, never int8 with int8.**

¹ From a one-off script that is not in the repository.

## Limits of these results

One board, short runs, and 40 hand-written sentences rather than a standard benchmark.

## Open items

- At 3 tokens most of the time is the weight fetch, so faster maths does little for short
  texts. PSRAM at 80 MHz is the next thing to try there.
- Attention and the norms still run in float on one core.

## How to reproduce

```bash
. ~/esp/esp-idf/export.sh
make p=minilm-fast build flash && make p=minilm-fast gelu     # about 12 minutes
```

Data: [`gelu_20261127_221914.csv`](../../projects/minilm-fast/results/gelu_20261127_221914.csv);
starting point [`psram_…193333.csv`](../../projects/minilm-psram/results/psram_20261127_193333.csv)
(rows with 0 layers); QIO [`psram_…200240.csv`](../../projects/minilm-psram/results/psram_20261127_200240.csv);
SIMD and cores [`speed_…205744.csv`](../../projects/minilm-fast/results/speed_20261127_205744.csv).
