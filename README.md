# esp32-llm

**A neural network that understands sentences, running on a US$10 microcontroller, written
in plain C and explained line by line.**

This repo runs [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
a real 22-million-parameter sentence-embedding model, on an **ESP32-S3**. There's no Linux,
no GPU, and no cloud. You type two sentences over USB serial, and the chip tells you how close
their meanings are.

```
minilm> dist After a long day she walked home through the quiet park | After a hard day he went home across the calm gardens
SIM 0.5424  (somewhat related)
DIST 0.4576
tokens: 13 + 13
timing: round trip 6.06 s = inference 6.02 s (99%) + serial/other 0.04 s (1%)
```

It is also a **teaching project**. The C code is small (about 430 lines of code) and heavily
commented, and the tutorial README is written to be turned into a talk. Everything is measured
on the real chip, with timing charts drawn in your terminal.

## What you get

- **A working sentence-embedding model on a microcontroller.** It runs int8 weights, mapped
  straight from flash, with a vocabulary pruned to the 12,603 most useful word pieces. It
  matches a numpy reference exactly (cosine 1.000000) and the original Hugging Face model to
  about 0.999.
- **Commands over serial:** `embed` (a 384-number vector), `dist` (similarity and distance of
  two texts), `tokens` (how text is split into word pieces) and `info`.
- **A timing breakdown for every call:** how long the round trip took, how much of it was
  inference, and which phase used the time. The weight fetch from flash costs a fixed 0.74 s
  per embedding, and the arithmetic adds 0.17 s per token.
- **A history of every call, with ASCII charts** drawn with plotext, plus a straight-line fit of
  time against text length (R² = 1.000). No board is needed to look at them.
- **Python tools:** an interactive prompt with arrow-key history, and a one-shot CLI with
  `--json`.
- **A [Claude Code](https://claude.com/claude-code) skill** (`.claude/skills/minilm/`) so an AI
  assistant can query the board directly.
- **An activity LED** that blinks while the board is working.
- **A benchmark project** for memory and IO speed, and a small LED on/off example.

## Why it is interesting

The original model is about 90 MB in float32, but the chip has 512 KB of fast RAM. The tutorial
shows how it fits:

1. **Quantise** to int8, one scale per row, for about a 4x saving.
2. **Prune the vocabulary** by word frequency, so it fits a 16 MB flash chip.
3. **Memory-map the weights** from a flash partition, so nothing is copied into RAM.
4. **Stream** each weight row once and reuse it for every token.

Embedding time turns out to be almost exactly `0.77 s + 0.17 s x tokens`. A fixed cost
comes from reading the weights, and a per-token cost from the arithmetic. This is why short
texts are limited by flash and long texts by the CPU.

## Quick start

You need an ESP32-S3 board with 8 MB PSRAM and 16 MB flash, a USB data cable, and Linux.
The full steps (serial permissions, ESP-IDF install) are in **[docs/SETUP.md](docs/SETUP.md)**.

```bash
uv sync                                  # Python tools
. ~/esp/esp-idf/export.sh                # ESP-IDF v5.4, once per shell
cd minilm
make download model build flash-model flash    # fetch the model, convert it, build, flash
make repl                                # talk to the board
```

Then read the tutorial: **[minilm/README.md](minilm/README.md)**. It covers how embeddings
work, a guided tour of the C code, the measurements, and a talk outline.

## Repo layout

| Folder | What it is |
|---|---|
| [`minilm/`](minilm/README.md) | **The main project**: MiniLM-L6 on the chip, with the Python tools and the tutorial |
| `bench/` | Memory and GPIO speed benchmark (results in `results/`) |
| `main/` | LED on/off over serial, the first example |
| `tools/` | Host scripts for `main/` and `bench/` |
| `.claude/skills/minilm/` | Claude Code skill for querying the board |
| [`docs/SETUP.md`](docs/SETUP.md) | Board capabilities, prerequisites, troubleshooting |

Each project folder is its own ESP-IDF project. **Flashing `main/` or `bench/` overwrites part
of the model partition**, so run `make flash-model` in `minilm/` again afterwards.

## Limits

- Text is **English and ASCII only**, and is cut off at **16 tokens** (about 12 words).
- The pruned vocabulary splits rare words into pieces and lowers accuracy for them.
- Each embedding takes 1.3 to 3.5 s, so this is a learning tool, not a fast one.

## Licence

[MIT](LICENSE). The model weights are not in this repo. `make download` fetches
all-MiniLM-L6-v2 from Hugging Face, where it is published under the Apache 2.0 licence.
