# Sentence embeddings on an ESP32-S3: running MiniLM-L6 in C

Give the chip a sentence, get back 384 numbers that capture its *meaning*. Give it two
sentences, get back how similar they are. All of it runs on the ESP32-S3, in C, with the
model weights read straight out of flash.

```
minilm> dist how do I reset my password? | I forgot my login
SIM 0.6113  (closely related)
DIST 0.3887
TIME_MS 4213

minilm> dist the dog barked | interest rates rose
SIM 0.0020
DIST 0.9980
```

This tutorial explains how and why it works, step by step, and every source file is written
to be read. It is also the outline of a tech talk (see the end).

**Contents**
1. [The idea in five minutes](#1-the-idea-in-five-minutes)
2. [Quick start](#2-quick-start)
3. [Step 1: make the model fit (quantise and prune)](#3-step-1-make-the-model-fit)
4. [Step 2: a readable reference in numpy](#4-step-2-a-readable-reference-in-numpy)
5. [Step 3: prove the shrunken model is still good](#5-step-3-prove-the-shrunken-model-is-still-good)
6. [Step 4: the flash layout](#6-step-4-the-flash-layout)
7. [Step 5: the firmware tour](#7-step-5-the-firmware-tour)
8. [Step 6: talk to it from the PC](#8-step-6-talk-to-it-from-the-pc)
9. [Results](#9-results)
10. [Limitations](#10-limitations)
11. [Exercises and next steps](#11-exercises-and-next-steps)
12. [Tech talk outline](#12-tech-talk-outline)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. The idea in five minutes

### What is an embedding?

An **embedding** turns text into a list of numbers (here, 384 of them) so that **texts with
similar meaning get similar numbers**. "How do I reset my password?" and "I forgot my login"
share no words, but their number lists point in nearly the same direction. A computer can
now measure meaning with arithmetic.

### What is "distance"?

The model does *not* output a distance. It outputs one vector per text. You compare two
vectors yourself. The standard way is **cosine similarity**: the cosine of the angle between
them.

* `1.0`: same direction, same meaning
* `0.0`: unrelated (at right angles)
* `-1.0`: opposite

We scale every vector to length 1 at the end, which makes cosine similarity just a **dot
product**: multiply the 384 numbers pairwise and add them up. We call `1 - similarity` the
*distance*.

### What is MiniLM-L6?

`all-MiniLM-L6-v2` is a small **BERT**-style transformer trained to produce good sentence
embeddings. It has 6 layers, 384 numbers per token, and 22.7 million parameters. (BERT-base,
the famous original, has 110 million and would not fit on this chip.)

### The pipeline

```
"a cat sat"
    |  1. TOKENISE      split into known word pieces, look up their ids
    v
[CLS] a cat sat [SEP]                         5 token ids
    |  2. EMBED         each id -> a 384-number vector (+ its position in the text)
    v
5 vectors of 384 numbers
    |  3. 6 x LAYER     tokens look at each other (attention), then each is refined
    v                   (feed-forward). The vectors now know about their context.
5 vectors of 384 numbers
    |  4. MEAN POOL     average the 5 vectors
    v
1 vector of 384 numbers
    |  5. NORMALISE     scale to length 1
    v
the sentence embedding
```

Comparing two sentences = run this twice and take a dot product.

### Why this suits a microcontroller better than a chatbot

A chatbot (LLM) re-reads **every weight for every word it writes**, so memory speed caps it
at a token or so per second. BERT processes the **whole sentence in one pass**, so each
weight is read once and reused for every token. It is far less memory-starved, which is why
a 22M-parameter model is practical here.

---

## 2. Quick start

### One-time setup

The project-level [`docs/SETUP.md`](../../docs/SETUP.md) covers the serial port permissions
(`dialout` group), `uv`, and installing ESP-IDF. In short:

```bash
sudo usermod -aG dialout $USER      # then log out and back in
uv sync                             # Python tools (from the repo root)
. ~/esp/esp-idf/export.sh           # in every new shell, before building
```

### Build and run

```bash
cd projects/minilm
make model          # 1. download MiniLM and shrink it -> model_build/model.bin  (15.98 MB)
make build          # 2. compile the firmware
make flash          # 3. write firmware + partition table to the chip
make flash-model    # 4. write the model into its partition (~3 min, one time only)
make repl           # 5. play!
make validate       # 6. optional: check the board against the numpy reference
```

`make flash` only rewrites the ~215 KB firmware. The 15 MB model partition is left alone, so
after the first time you never wait for it again unless the model changes.

You do not have to remember any of this: **the tools check the board for you**. See
[Is the right firmware on the board?](#is-the-right-firmware-on-the-board) below.

### One-shot command line (for scripts and AI assistants)

`tools/minilm_cli.py` does the same as the prompt but runs a single command and exits:

```bash
uv run python projects/minilm/tools/minilm_cli.py dist "the cat sat" "a kitten rested"
uv run python projects/minilm/tools/minilm_cli.py --json dist "a" "b"          # machine-readable
uv run python projects/minilm/tools/minilm_cli.py embed --full "hello world"   # all 384 numbers
uv run python projects/minilm/tools/minilm_cli.py tokens "unbelievable cats!"
uv run python projects/minilm/tools/minilm_cli.py info
printf 'cat | kitten\ndog | bank\n' | uv run python projects/minilm/tools/minilm_cli.py batch
```

Opening the serial port resets the board, so every call spends about 2.5 s booting it. Use
`batch` to compare many pairs over one connection. Options such as `--json` and `--no-timing`
can go before or after the command (`--port` goes before it). Exit codes: 0 ok, 1 the board rejected the
command, 2 bad arguments, 3 cannot reach the board (for example the port is in use), 4 the
board runs the wrong firmware or model and was not changed.

The tools take an exclusive lock on the serial port, so the REPL, the CLI and `make validate`
refuse to run at the same time (with a clear message) instead of garbling each other's replies.

### Is the right firmware on the board?

The CLI, the prompt and `make validate` only work against this project's firmware and model, and
other projects in the repo share the same chip. So before doing anything, every tool asks the
board one question, `ident`, and the board answers with one line:

```
IDENT project=minilm proto=1 build=3f2a91bc model=ok model_id=efe6e2da
```

| Field | Meaning |
|---|---|
| `project` | which firmware this is (the name in `CMakeLists.txt`) |
| `proto` | the **protocol version**: bumped whenever a change would break the PC tools |
| `build` | which exact build (a fingerprint of the compiled program; changes on every rebuild) |
| `model` | `ok`, `missing` (nothing in the model partition) or `incompatible` |
| `model_id` | which model file: a checksum stored in `model.bin`'s header by `export_model.py` |

The tool accepts the board when the **project and protocol match** and the model id equals the
one in `model_build/model.bin`. A firmware you rebuilt without changing the protocol is fine as it
is. If something is wrong, it says what, and asks before writing anything:

```
the board is running 'led', not 'minilm'.
Write the minilm firmware now? This replaces what is on the board and takes under a minute. [y/N]
```

- **Firmware:** written from `build/` (a few seconds, plus a rebuild first when the sources are
  newer than `build/`, which needs ESP-IDF exported). The model is not touched.
- **Model:** written from `model_build/model.bin` (about 3 minutes), only when it is missing or is
  a different file. If `model.bin` does not exist yet, the tool tells you to run `make model`.
- **`--yes`** writes without asking; **`--no-flash`** never writes. Without a terminal, or with
  `--json`, the tools never write unless you also pass `--yes` (this is what keeps scripts and the
  Claude skill safe). They exit with code 4 and say how to fix it.

Where the check lives: `common/python/esp32tools/` (`ident.py`, `firmware.py`, `ensure.py`) is
generic, so any project can use it. This project's side is the `REQUIREMENT` in `tools/device.py`
and `PROTO_VERSION` in `common/components/minilm_core/shell.c`. **The two `PROTO_VERSION` numbers must match.** After
writing the firmware the tool checks that the board reports the build it just wrote, which would
catch them drifting apart.

Why a checksum in the model header instead of just a version? Two different exports (a different
vocabulary size, say) would both be "version 1" yet give different answers. The checksum says
which file is really in flash. The header word it uses was unused before, so the file format did
not change.

### Metrics, history and ASCII charts

After every `embed` or `dist`, the CLI and the prompt draw a [plotext](https://github.com/piccolomo/plotext)
bar chart of where that call's time went, with the seconds and percentage on each bar:

```
                         where the 3.76 s went (seconds)
                       ┌───────────────────────────────────────────────────────┐
  arithmetic 2.107s 56%┤███████████████████████████████████████████████████    │
weight fetch 1.484s 39%┤████████████████████████████████████                   │
         GELU 0.133s 4%┤████                                                   │
    attention 0.022s 1%┤██                                                     │
```

Use `--no-chart` (CLI) or `chart off` (prompt) for a plain table instead.

**Every call is recorded** to `projects/minilm/history/calls.jsonl`, one JSON line per call (the
folder is git-ignored because it holds your own text). Each record holds the time, whether it
came from the CLI or the prompt, the texts, token counts, similarity, and the full timing
breakdown. `--no-history` skips recording. Three commands read it back, **with no board
needed**:

```bash
uv run python projects/minilm/tools/minilm_cli.py history -n 10     # the last calls
uv run python projects/minilm/tools/minilm_cli.py stats             # summary + straight-line fit
uv run python projects/minilm/tools/minilm_cli.py plot latency      # ASCII chart: see below
```

(The same commands work in the prompt as `history`, `stats` and `plot latency`, and as
`make history`, `make stats` and `make plot KIND=latency`.)

| `plot` kind | Shows |
|---|---|
| `phases` | average time per phase over the whole history |
| `latency` | seconds against text length, with the fitted straight line |
| `bottleneck` | weight fetch against arithmetic as text grows |
| `timeline` | the time of every call, oldest to newest |
| `similarity` | histogram of your `dist` scores |
| `all` | every chart above |

**The fit is the interesting part.** Embedding time is almost exactly a straight line in text
length, so `stats` fits one to your own calls:

```
straight-line fit of your history (17 points, R^2 = 1.000):
  seconds per embedding = 0.761 s fixed  +  0.173 s x tokens
```

The fixed part is streaming the weights out of flash (the firmware's own stopwatch says
0.742 s); the per-token part is the arithmetic (0.165 s) plus small extras like GELU. Two
independent measurements, the board's stopwatches and a line fitted from the outside, agree.
`dist` runs the network twice, so the history works per embedding (time divided by the number
of embeddings) to compare `embed` and `dist` calls fairly. The charts need a handful of calls
of different lengths before the fit is meaningful, and say so until then.

The code is split so each file does one thing: `history.py` (storing and the statistics),
`charts.py` (all the plotext code in one place), `calls.py` (run, record and present a call;
shared by the CLI and the prompt) and `device.py` (the serial link).

### A Claude Code skill

`.claude/skills/minilm/SKILL.md` teaches Claude Code how to use the CLI: which command to
run, how to read the similarity bands and the timing phases, the limits to mention
(pruned vocabulary, 16 tokens), the exit codes, and the guardrail that it must never flash
the board. In a Claude Code session started in this repo, ask for example: "how similar are
'reset my password' and 'I forgot my login' on the ESP32?" and Claude will run the board and
report real device results.

### Reading a `dist` result

`SIM` runs from -1 to 1, but for sentences you will mostly see 0 to 1. The REPL adds a
plain-English label under it. As a rule of thumb for this model:

| SIM | Meaning | Example |
|---|---|---|
| 0.80 and up | almost the same meaning | "how do I reset my password" / "how can I change my password" |
| 0.60 to 0.80 | closely related | "reset my password" / "I forgot my login" (0.61) |
| 0.40 to 0.60 | somewhat related | same topic, different point |
| 0.20 to 0.40 | weakly related | a loose connection |
| under 0.20 | unrelated | "the dog barked" / "interest rates rose" (0.00) |

These bands are guidelines, not laws: scores depend on the text and on how many words were
pruned (see [limitations](#10-limitations)). What matters most is **comparing scores with
each other**: if A is closer to B than to C, A's SIM with B is higher.
`DIST` is just `1 - SIM`, so 0 means identical and about 1 means unrelated.

Inside the REPL:

| Command | What it does |
|---|---|
| `embed <text>` | Show the first 8 of the 384 numbers |
| `dist <text a> \| <text b>` | Similarity and distance between two texts |
| `tokens <text>` | Show how the text is split into word pieces |
| `info` | Model and memory facts from the board |
| `exit` | Quit |

---

## 3. Step 1: make the model fit

The original model is **90 MB** (float32). The chip has **16 MB** of flash. Two tricks get it
to **15.98 MB**. The code is `tools/export_model.py`.

### Trick 1: quantise to int8 (4x smaller)

A float32 weight takes 4 bytes. We store each weight as an **int8** (1 byte, -127 to 127)
plus **one float scale per matrix row**:

```
row (floats):  [ 0.50, -1.27,  0.03 ]
scale = max(|row|) / 127 = 1.27 / 127 = 0.01
row (int8):    [   50,  -127,     3 ]        <- what we store
reconstruct:   int8 * scale = [0.50, -1.27, 0.03]
```

We lose a tiny bit of precision for a 4x saving. Biases and LayerNorm parameters are few and
stay float32. The accuracy cost is measured in [step 3](#5-step-3-prove-the-shrunken-model-is-still-good): essentially zero.

### Trick 2: prune the vocabulary

Half the model (11.7 M parameters) is the **word table**: one 384-number row for each of
BERT's 30,522 word pieces. Reading it is cheap (we only fetch the few rows we need) but
*storing* it is not. We keep **12,603** pieces:

* the **9,000 most common English words**, ranked with the `wordfreq` package (BERT's
  vocabulary file is *not* in frequency order, so "first N ids" is a poor rule: we tried it),
* **2,500 sub-word pieces** such as `##ing` and `##ly`,
* **every single ASCII character**, so any word can still be spelled out letter by letter.

A word that was pruned is split into smaller pieces. This is the one real cost of fitting on
the chip:

```
kitten  ->  kit ##ten        (the original model has a single token for "kitten")
```

### The storage budget

| Section | Size |
|---|---|
| 6 transformer layers (int8 + scales + biases) | 10.8 MB |
| Word table, 12,603 rows x (384 int8 + 1 float scale) | 4.9 MB |
| Position embeddings (64 x 384 float32) | 0.1 MB |
| Vocabulary strings and index | 0.2 MB |
| **Total `model.bin`** | **15.98 MB** |
| Flash partition available | 16.19 MB |

The export script refuses to write a file that would not fit the partition. To trade
vocabulary for space, change `WORDS_KEEP`.

### The file format

One flat file, read front to back with no parsing. The firmware just moves a pointer.
Everything is 4-byte aligned because the Xtensa CPU faults on misaligned 32-bit loads.

```
header (12 x uint32) | word table | positions | LayerNorm | layer 0 ... layer 5 | vocabulary
```

The exact order is in the docstring at the top of `tools/export_model.py`. Both the numpy
reference and the C firmware read it in that same order.

---

## 4. Step 2: a readable reference in numpy

Before writing C, `tools/minilm_ref.py` implements the whole model in about 150 lines of
numpy. It reads the same `model.bin`, so it also tests the file format. **Read this file
first.** The C code is a line-for-line translation.

```bash
# from projects/minilm/
uv run --project ../.. python tools/minilm_ref.py dist "a cat sat on the mat" "stock markets fell"
```

The key functions, each with a twin in `common/components/minilm_core/bert.c`:

| Function | What it does |
|---|---|
| `wordpiece()` | Greedy longest-match splitting into pieces |
| `layer_norm()` | Rescale each token's vector to mean 0, variance 1 |
| `encoder_layer()` | Attention, then feed-forward, each with a residual connection |
| `embed()` | The whole pipeline, ending in mean pooling and normalising |

---

## 5. Step 3: prove the shrunken model is still good

Never trust a re-implementation without checking it. `tools/verify_hf.py` runs the same
sentences through the **original float32 model** (PyTorch, from Hugging Face) and through
ours, and prints the cosine similarity between the two answers. (1.0 = identical.)

```bash
uv run --project ../.. --with torch --with transformers \
    --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
    python tools/verify_hf.py
```

(torch is large, so it is not a project dependency; `--with` uses a throwaway environment.)

| Sentence | cos(ours, original) | Note |
|---|---|---|
| hello world | 0.9990 | |
| a cat sat on the mat | 0.9995 | |
| stock markets fell sharply today | 0.9997 | |
| how do i reset my password? | 0.9996 | |
| the quick brown fox jumps over the lazy dog | 0.9996 | |
| espresso machines need regular descaling | 0.9997 | |
| I forgot my login credentials | 0.9156 | `login` and `credentials` were split into pieces |
| a kitten rested on the rug | 0.2567 | `kitten`, `rested`, `rug` were split into pieces |

**How to read this.** When every word is in our vocabulary the match is 0.999+, so the
**maths is right and int8 quantisation costs almost nothing**. When a sentence contains a
pruned word the result moves, sometimes a lot. That is the vocabulary trade-off, and it is
the honest limit of this design.

On pairs, which is how you actually use embeddings:

| Pair | Original | Ours |
|---|---|---|
| cat on the mat / stock markets fell | 0.058 | 0.058 |
| hello world / stock markets fell | 0.012 | 0.013 |
| the quick brown fox / cat on the mat | 0.228 | 0.224 |
| how do i reset my password / I forgot my login credentials | 0.678 | 0.571 |
| cat on the mat / kitten rested on the rug | 0.618 | 0.203 |

Unrelated and common-word pairs agree closely; pairs involving pruned words drift.

---

## 6. Step 4: the flash layout

`partitions.csv` gives almost all of the 16 MB flash to the model:

```
Name       Type  SubType  Offset    Size
nvs        data  nvs      0x9000    0x6000
phy_init   data  phy      0xf000    0x1000
factory    app   factory  0x10000   0x80000     firmware: 512 KB (it uses 215 KB)
model      data  0x40     0x90000   0xF70000    15.4 MiB for model.bin
```

The firmware does **not** copy the weights into RAM (there is not enough). It calls
`esp_partition_mmap()`, which makes flash readable like memory. A `const int8_t *` then
points *into the flash chip*, and the CPU reads it through a cache. `make flash-model`
writes the file at offset `0x90000`, which must match the `model` row.

---

## 7. Step 5: the firmware tour

Read the files in this order. About 1214 lines of C in total, of which roughly 431 are code and the rest are teaching comments.

The C code is in [`common/components/minilm_core/`](../../common/components/minilm_core/) so that [`minilm-psram`](../minilm-psram/README.md) can share it; this project's own `main/main.c` only starts it.

| File | Role |
|---|---|
| `model.h`, `model.c` | Map the flash partition and compute pointers to every tensor. Nothing is copied. |
| `tokenizer.h`, `tokenizer.c` | WordPiece: text to token ids (binary search over the sorted vocabulary). |
| `bert.h`, `bert.c` | The neural network: embeddings, 6 layers, pooling. |
| `shell.c` | The serial command shell (`embed`, `dist`, `tokens`, `info`). |
| `../../common/components/board_led/` | The blinking activity LED (see below). Shared with the other projects, so it lives in `common/`. |

### The tokenizer (`tokenizer.c`)

Lower-case, split on spaces, make each punctuation mark its own word, then split every word
into the **longest** pieces found in the vocabulary. Pieces after the first are looked up
with a `##` prefix. A word that cannot be split becomes `[UNK]`. It is ASCII-only: other
bytes are treated as separators.

### The workhorse: `linear()` (`bert.c`)

About 99% of the run time is one function that computes `out = in x W^T + bias`:

```c
for (int j = 0; j < out_dim; j++) {           // each row of the weight matrix
    // 1. fetch row j from flash ONCE and convert int8 -> float
    for (i...) rowbuf[i] = (float)wrow[i];
    // 2. dot it with EVERY token vector
    for (t...) out[t][j] = dot(rowbuf, in[t]) * scale[j] + bias[j];
}
```

The loop order is the key idea: rows on the outside, tokens on the inside. Each weight row
crosses the slow flash bus **once** and is reused for every token. Four running sums in the
dot product let the CPU overlap its floating-point operations.

### Attention (`attention()`)

For each of the 12 heads, each token scores every other token (query . key), turns the
scores into weights with `softmax`, and takes a weighted average of the other tokens'
values. This is how "bank" learns that it sits next to "river" and not "money".

### The activity LED

An embedding takes seconds, so the board would otherwise look dead. While a command runs, the
on-board LED blinks (dim white, 5 flashes a second) and goes dark when the reply is sent. Even
a command that takes a millisecond gives one visible flash, and three quick flashes after
start-up mean "ready".

How it stays out of the way: the blinking is done by its own FreeRTOS task, **pinned to core 1**
while the neural network runs on core 0, and the task sleeps (using no CPU) until a *task
notification* wakes it. Measured effect: the timings are unchanged (for example `hello world`
1.45 s before and after) and it costs about 10 KB of internal RAM.

The code is in the shared component `common/components/board_led/` (`activity_led.c`, built on `board_led.c`). The pin and LED type are set in its `Kconfig`: GPIO48 and an addressable RGB LED by
default. Most ESP32-S3 dev boards use GPIO48 (some older ones GPIO38). If yours does not light up,
run `make menuconfig`, open "Board LED", change the GPIO number, and `make build flash`. If the
LED cannot be set up at all, the firmware carries on without it.

### Memory

| Where | What | Size |
|---|---|---|
| Flash (memory-mapped) | all weights | 15.4 MB |
| Internal RAM | 7 working buffers (token vectors, q, k, v, ...) | about 245 KB |
| Internal RAM | everything else (FreeRTOS, stacks, ...) | the rest |

After start-up about 128 KB of internal RAM is still free. PSRAM is not used. The limit of
16 tokens (`MAX_TOKENS` in `bert.h`) exists so the working buffers fit in fast internal RAM.

---

## 8. Step 6: talk to it from the PC

The protocol is plain text over the USB serial port (115200 baud). The PC sends one line, the
board replies with some lines and then `OK`:

```
PC    -> embed hello world
board -> TOKENS 4
board -> VEC 0.012 -0.034 ... (384 numbers)
board -> TIME_MS 1420
board -> TIMING total 1417000        (one line per phase, in microseconds)
board -> TIMING dot 659000
board -> ...
board -> OK
```

`tools/device.py` wraps this, `tools/embed_repl.py` is the interactive prompt, `tools/minilm_cli.py` is the one-shot command line, and
`tools/validate_device.py` sends test sentences and compares the board's vectors with the
numpy reference.

Opening the serial port **resets the board** (the USB chip toggles the reset line), so the
scripts wait for the board's `READY` message first, then ask `ident`.

---

## 9. Results

### Accuracy on the board

`make validate` compares the board against numpy on the same `model.bin`:

| Sentence | Tokens | cos(board, numpy) | Time |
|---|---|---|---|
| hello world | 4 | 1.000000 | 1.42 s |
| a cat sat on the mat | 8 | 1.000000 | 2.10 s |
| stock markets fell sharply today | 7 | 1.000001 | 1.93 s |
| how do i reset my password? | 9 | 1.000000 | 2.28 s |
| the quick brown fox jumps over the lazy dog | 11 | 1.000000 | 2.63 s |
| (a long sentence, cut to the limit) | 16 | 1.000000 | 3.50 s |

The C implementation reproduces the numpy one to six decimal places. Any remaining
difference from the original model comes from quantisation and pruning, not from bugs.

### Where the time goes

After every `embed` and `dist`, the REPL prints a breakdown measured by stopwatches inside
the firmware (`TIMED()` and `bert_timing_t` in `bert.c` / `bert.h`):

```
timing: round trip 1.79 s = inference 1.45 s (81%) + serial/other 0.34 s (19%)
where the 1.45 s went:
  arithmetic: multiply-adds      0.659 s  45.4%  █████████████
  weight fetch: flash -> float   0.742 s  51.1%  ██████████████
  attention (tokens mixing)      0.004 s   0.3%
  layer norm                     0.005 s   0.3%
  GELU activation                0.041 s   2.8%  █
  ...
```

"Round trip" is the PC's stopwatch from sending the line to getting `OK`; "inference" is the
board's own stopwatch around `bert_embed()`. For `embed`, the gap is mostly printing 384
numbers over a 115200 baud link.

**Measured with 4 tokens vs 16 tokens:**

| Phase | 4 tokens ("hello world") | 16 tokens |
|---|---|---|
| Arithmetic (multiply-adds) | 0.659 s (45%) | 2.552 s (72%) |
| Weight fetch (flash to float) | 0.742 s (51%) | 0.742 s (21%) |
| GELU | 0.041 s (3%) | 0.159 s (5%) |
| Attention | 0.004 s (0.3%) | 0.063 s (1.8%) |
| Everything else | under 0.01 s | under 0.02 s |
| **Total** | **1.45 s** | **3.53 s** |

**What this teaches**

* **Weight fetch is a fixed cost: 0.742 s, whatever the length.** Each weight row is read
  from flash once and reused for every token (the loop order in `linear()`). It is also
  exactly what the benchmark predicted: about 10.8 MB of weights at 14.5 MB/s is 0.74 s.
* **Arithmetic grows with the number of tokens**: 0.659 s at 4 tokens, 2.552 s at 16, almost
  exactly 4 times as much. That is 0.165 s per token, about 64 million multiply-adds per
  second.
* **So short texts are limited by flash, long texts by the CPU.** At 4 tokens fetching is
  51% of the time; at 16 tokens it is 21%. Different bottlenecks need different fixes:
  the second core and SIMD instructions help arithmetic, while faster flash settings or
  keeping weights in RAM help the fetch.
* **Attention is almost free** (under 2%). It is the part of a transformer people talk about
  most, yet on this chip nearly all the time goes into the plain matrix multiplies.
  Attention cost grows with the square of the length, so it would matter more for long text.
* **GELU is surprisingly costly (3 to 5%)** for such a small job, because each `erff()` call
  takes roughly 250 CPU cycles. A cheaper approximation would save most of it.
* **`dist` doubles everything.** It embeds two texts; the weight fetch (1.48 s) is twice a
  single embedding, because the weights are fetched again for the second text.

**How trustworthy are the numbers?** The stopwatches read the clock once per function call,
and once per weight row inside `linear()`, never inside the innermost loops, so they barely
disturb the result. The "untimed" row (loop overhead and the cost of the timer reads) comes
out at about 0.000 s. "Weight fetch" includes the int8-to-float conversion as well as the
flash read, and these cannot be separated without a finer stopwatch.

---

## 10. Limitations

* **Pruned vocabulary.** Rare words are spelled out in pieces and give worse embeddings.
  This is the main quality loss (see step 3).
* **16 tokens maximum**, including `[CLS]` and `[SEP]`, so about 14 word pieces. Longer text
  is silently cut. (Raise `MAX_TOKENS` and put the buffers in PSRAM to go longer.)
* **ASCII, English only.** Accents and other scripts are dropped.
* **Slow**: 1.5 to 3.5 s per text, on one core of two, without SIMD.
* **Serial speed**: printing 384 numbers over 115200 baud adds a noticeable fraction of a
  second to `embed`.
* The tokenizer does not strip accents or handle special tokens typed in the text.

---

## 11. Exercises and next steps

1. **Dig one level deeper.** The breakdown lumps flash reading and int8-to-float conversion
   together as "weight fetch". Split them: copy one layer into RAM first and time the
   conversion alone, to see how much is truly the flash bus.
2. **Use the second core.** Split the rows of `linear()` between core 0 and core 1. Arithmetic
   should speed up; flash streaming will not.
3. **Use the S3's vector (SIMD) instructions** for the int8 dot product. This is the biggest
   potential win for arithmetic, and a natural place for hand-written Xtensa assembly.
4. **Bigger vocabulary.** Store the word table as int4, or stream the model from an SD card,
   to keep more words.
5. **Longer inputs.** Allocate the buffers in PSRAM and raise `MAX_TOKENS`.
6. **Real use.** Build a tiny search: embed 20 stored sentences, then rank them by similarity to
   a query.
7. **Break it on purpose.** Try `WORDS_KEEP = 2000` and watch which sentences degrade.

---

## 12. Tech talk outline

A 30-minute talk. Each bullet is a slide; the **demo** points use the REPL.

1. **Title and hook.** "Meaning, measured, on a $10 chip." Demo: `dist` on a similar pair and
   an unrelated pair.
2. **What is an embedding?** Text to numbers where similar meaning means nearby numbers.
   Draw the angle between two arrows; cosine similarity.
3. **Why MiniLM-L6?** BERT family, 22.7 M parameters, built for sentences. BERT-base is
   110 M and would not fit.
4. **The chip.** ESP32-S3: 2 cores at 240 MHz, 512 KB RAM, 8 MB PSRAM, 16 MB flash. Show the
   benchmark: internal RAM 292 MB/s, PSRAM 32 MB/s, flash 14.5 MB/s. *Memory speed is the
   story.*
5. **The pipeline.** Tokenise, embed, 6 layers, mean pool, normalise (the diagram from
   section 1).
6. **Problem: 90 MB into 16 MB.** Two tricks.
7. **Trick 1: int8 quantisation.** The `[0.50, -1.27, 0.03]` worked example.
8. **Trick 2: vocabulary pruning.** Half the model is the word table. Anecdote: BERT's
   vocabulary is *not* frequency-ordered; the first attempt kept "cat" but dropped "mat".
9. **Prove it.** The accuracy table: 0.999+ when words are present; honest about the
   failures (`kitten`).
10. **Reading weights from flash.** `mmap`: a pointer straight into the chip. The partition
    table.
11. **Why BERT suits a microcontroller.** Every weight is read once and reused for every
    token. Show the `linear()` loop order.
12. **Attention in one slide.** Tokens deciding who to listen to. Heads.
13. **Results.** Exact match with numpy; 1.5 to 3.5 s per text; the measured time
    breakdown (0.74 s fetching weights, a fixed cost, plus 0.165 s per token of arithmetic).
    Live demo: run `embed` on a short and a long text and watch the bottleneck move from
    flash to the CPU.
14. **What I'd do next.** SIMD, second core, int4, SD card.
15. **Demo and Q&A.** Live `dist` calls; let the audience suggest sentence pairs, and show a
    pruned-word failure on purpose.

**Likely questions:** *Why not a bigger model?* (storage and speed; see the benchmark.)
*Why not word2vec?* (no context: "bank" always the same.) *Could it run on Wi-Fi?* (yes,
add a TCP server around `bert_embed`.) *How accurate is int8?* (see the table.)

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `the board is running 'led', not 'minilm'` (exit 4) | Another project is on the board. Answer `y` to the prompt, or `make flash`. |
| `the model on the board ... is not the one in model.bin` (exit 4) | Answer `y` (3 minutes), or `make flash-model`. |
| `board did not say READY` | The firmware crashed or the port is busy. Run `make flash`, then `make repl`. |
| `ERR no usable model in flash` / `model=missing` | The `model` partition is empty. The tools offer to write it, or `make flash-model`. |
| `no 'model' partition` | The partition table is old. `make flash` rewrites it. |
| `model is too big for the flash partition` | Lower `WORDS_KEEP` in `tools/export_model.py`. |
| `Permission denied: /dev/ttyACM0` | Not in the `dialout` group, or you have not logged out and in since adding it. |
| `Device busy` | Another program (a monitor, a second REPL) holds the serial port. |
| `idf.py: command not found` | Run `. ~/esp/esp-idf/export.sh` in this shell. |
| `No module named minilm_ref` | Run the scripts without `python -I`; they import their neighbours. |
| A text longer than 14 words gives odd results | It was cut to `MAX_TOKENS = 16`. |
