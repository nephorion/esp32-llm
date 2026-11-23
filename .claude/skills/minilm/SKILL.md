---
name: minilm
description: Query the MiniLM-L6 sentence-embedding model running on the ESP32-S3 board over serial. Use it to measure how similar in meaning two texts are (similarity/distance), get a text's embedding, see how text is split into word pieces, read model/memory info, or get a per-phase timing breakdown (arithmetic vs flash weight fetch etc.). Trigger on requests like "how similar are these sentences on the board", "run this through the ESP32", "embed this", "test the model on the chip", "what is the inference latency".
allowed-tools: Bash(uv run python minilm/tools/minilm_cli.py:*), Bash(fuser -v /dev/ttyACM0:*)
---

# MiniLM on the ESP32-S3

The board runs `all-MiniLM-L6-v2` (int8, pruned vocabulary) in C. `minilm/tools/minilm_cli.py`
sends one command over USB serial and prints the result. Everything runs on the chip, so
results are real device output, not a PC simulation.

## How to run it

Always run from the repo root, and prefer `--json` so you can read exact numbers:

```bash
cd <repo root>   # the folder containing minilm/
uv run python minilm/tools/minilm_cli.py --json dist "text a" "text b"
```

| Need | Command |
|---|---|
| Similarity / distance of two texts | `dist "a" "b"` |
| Many comparisons | `batch` with lines `text a \| text b` on stdin (one connection, much faster) |
| The embedding | `embed "text"` (add `--full` for all 384 numbers; default shows the first 8) |
| How text is split into pieces | `tokens "text"` |
| Model and RAM facts | `info` |

Global options (`--json`, `--no-chart`, `--no-timing`, `--no-history`, `--port /dev/ttyACM0`) can go before or after the command.

Batch example (one connection, any number of pairs):

```bash
printf 'the cat sat | a kitten rested\nreset my password | I forgot my login\n' \
  | uv run python minilm/tools/minilm_cli.py --json batch
```

## History, statistics and charts (no board needed)

Every call through the CLI or REPL is appended to `minilm/history/calls.jsonl` (one JSON line
per call: time, source, command, texts, token counts, similarity, full timing). These commands
read it back and do NOT open the serial port, so they work even while the user has a REPL open:

| Need | Command |
|---|---|
| The last calls | `--json history -n 10` |
| Summary and time-vs-length fit | `--json stats` |
| ASCII chart | `plot phases \| latency \| bottleneck \| timeline \| similarity \| all` |

`stats` returns, among other things, `fit`: `seconds per embedding = fixed_s + per_token_s x tokens`.
`fixed_s` is the flash weight-streaming cost, `per_token_s` the arithmetic. Charts are plain
text when piped, so you can paste them into an answer in a code block. If a chart says
"not enough data", more calls of different lengths are needed.

Calls you make are recorded too (source `cli`). For throwaway or diagnostic calls the user did
not ask to keep, add `--no-history` so their history is not padded with your experiments.

## Reading the results

`similarity` is the cosine similarity (1 = same meaning, 0 = unrelated); `distance` is
`1 - similarity`. The `meaning` field is a rule-of-thumb label:

| similarity | meaning |
|---|---|
| >= 0.80 | almost the same meaning |
| 0.60 - 0.80 | closely related |
| 0.40 - 0.60 | somewhat related |
| 0.20 - 0.40 | weakly related |
| < 0.20 | unrelated |

These are guidelines; comparing scores with each other is more reliable than the bands.
Report the number AND the label, and say which pair is closer when comparing several.

Timing (`timing` in JSON, all seconds): `round_trip_s` is the PC's stopwatch,
`inference_s` is the board's own, and `phases_s` splits it:

- `dot` = arithmetic (multiply-adds); `weight_fetch` = reading int8 weights from flash and
  converting to float (a fixed ~0.74 s per embedding, whatever the text length);
- `attention`, `layer_norm`, `gelu`, `residual`, `lookup`, `tokenize`, `pool` = the rest;
- `total` = whole call. `dist` embeds two texts, so its phases cover both.

Typical latency: about 1.3 s for a 3-token text up to about 3.5 s for 16 tokens, per text
(the fit from the history: roughly 0.76 s + 0.173 s per token).
Each CLI call also spends about 2.5 s because opening the serial port resets the board.

## Limits you must mention when they matter

- **Pruned vocabulary (the main quality limit).** Only 12,603 word pieces are kept. A rarer
  word is split into smaller pieces (for example kitten becomes `kit ##ten`) and scores
  can differ a lot from the original model. If a result looks surprising, or a key word may
  be rare, run `tokens` on the text and check whether important words were split.
- **16 tokens maximum**, including [CLS] and [SEP] (about 14 word pieces). Longer text is
  silently cut off; check the `tokens` output if a text is long.
- **ASCII and English only.** Other characters are dropped.
- **Do not put `|` inside a text** (it is the separator for `dist`; the tool replaces it).
- Scores are for this shrunken model, not the original float32 one. Against the original
  they agree to about 0.999 when every word is in the vocabulary.

## Errors and exit codes

| Exit | Meaning | What to do |
|---|---|---|
| 0 | success | |
| 1 | the board rejected the command | read the `error:` line |
| 2 | bad command-line arguments | fix the command |
| 3 | cannot talk to the board | see below |

Exit 3 usually means the port is in use ("close the REPL or monitor"), the board is
unplugged, or the model is not flashed. Check what holds the port with
`fuser -v /dev/ttyACM0`. If it is the user's own REPL or monitor, ask them to close it. Do
NOT kill their process.

## Guardrails

- Only the read-only commands above are in scope. **Never** run `make flash`, `make
  flash-model`, `idf.py flash` or `esptool` from this skill: flashing the `main/` or `bench/`
  firmware overwrites part of the model partition, and flashing the model writes ~15 MB of
  flash. Only do those if the user explicitly asks.
- Do not run several CLI calls at once; they share one serial port (the second one fails with
  exit 3). Use `batch` instead.
- The numpy reference (`uv run python minilm/tools/minilm_ref.py dist "a" "b"`) gives the same
  answer on the PC without the board, and is a good cross-check if the board is unavailable.
