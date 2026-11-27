# minilm-psram

[MiniLM](../minilm/README.md) with as many layers' weight matrices as fit (4 of 6) copied from
flash into PSRAM at start-up. The program is the same shared code
(`common/components/minilm_core`); this project only copies the matrices and repoints the
model's pointers at them. The weights are the same bytes, and the embeddings come out identical
(compared as printed, to 6 decimals).

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

**With QIO flash (the current setting), copying to PSRAM at 40 MHz no longer helps.** From
`results/psram_20261127_200240.csv` (240 MHz, flash QIO 80 MHz, PSRAM 40 MHz):

| Tokens | All in flash | 4 layers in PSRAM |
|---|---|---|
| 3 | 1.040 s | 1.054 s |
| 8 | 1.897 s | 1.911 s |
| 16 | 3.294 s | 3.308 s |

The weight fetch goes from 0.503 s to 0.517 s, and copying adds 0.65 s to start-up.

Before QIO, with flash in DIO mode (`results/psram_20261127_193333.csv`), the same 4 layers cut
the fetch from 0.743 s to 0.597 s (1.04x to 1.13x faster embeddings).

In a one-off trial (console output only, no results file), PSRAM at 80 MHz **with** QIO brought the fetch to 0.454 s with 4 layers
(0.487 s all in flash). That setting is not enabled here. A likely limit in all these cases is
the byte-by-byte int8 to float loop, which has not been measured on its own.
