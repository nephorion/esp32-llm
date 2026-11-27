# minilm-psram

[MiniLM](../minilm/README.md) with as many layers' weight matrices as fit (4 of 6) copied from
flash into PSRAM at start-up. The program is the same shared code
(`common/components/minilm_core`); this project only copies the matrices and repoints the
model's pointers at them, so the answers are bit-for-bit identical.

The model must already be in flash (`make model flash-model` in `projects/minilm`).

```bash
. ~/esp/esp-idf/export.sh       # only needed if the tool has to build
make run                        # from this folder
```

`make run` checks the board runs this firmware (and offers to write it), then times three
texts with 0 to 4 layers read from PSRAM, checks every embedding is identical to the
all-flash one, and saves a CSV in `results/`. After changing the code, run
`make build flash` first. The serial command `psram [n]` shows or sets the number of layers.

## Result

From `results/psram_20261127_191839.csv` (240 MHz, flash DIO 80 MHz, PSRAM 40 MHz).

| Tokens | All in flash | 4 layers in PSRAM | Speedup |
|---|---|---|---|
| 3 | 1.283 s | 1.137 s | 1.13x |
| 8 | 2.140 s | 1.995 s | 1.07x |
| 16 | 3.538 s | 3.392 s | 1.04x |

- Each layer moved saves about **36 ms** of weight fetch, the same for every layer and length.
  With 4 layers the fetch drops from 0.743 s to 0.597 s.
- A layer therefore takes about 124 ms to fetch from flash and 87 ms from PSRAM: only 1.4x
  faster, although PSRAM streamed about twice as fast as flash in `matbench`. A likely reason
  is that the byte-by-byte int8 to float loop now limits it. This has not been measured.
- Arithmetic is unchanged and dominates longer texts.
- Copying the 4 layers adds 0.84 s to start-up.
