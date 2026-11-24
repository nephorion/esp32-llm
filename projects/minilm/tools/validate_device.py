#!/usr/bin/env python3
"""
validate_device.py - check the ESP32 gives the same answers as the numpy reference.

Both use the exact same model.bin, so the vectors should agree to ~6 decimal places
(cosine ~ 1.0000); tiny differences come from float32 rounding order.
It also reports how long each embedding took on the board.

Usage:  uv run python projects/minilm/tools/validate_device.py [serial-port]
"""
import sys

import numpy as np

import minilm_ref as ref
from esp32tools import FlashError, Mismatch
from device import connect

SENTENCES = [
    "hello world",
    "a cat sat on the mat",
    "stock markets fell sharply today",
    "how do i reset my password?",
    "the quick brown fox jumps over the lazy dog",
    "this sentence is quite a bit longer than the others and will be cut off at the token limit",
]
PAIRS = [(1, 2), (3, 4), (0, 1)]

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
model = ref.load()
try:
    dev = connect(port, "ask")
except (Mismatch, FlashError) as e:
    sys.exit(f"error: {e}")

print(f"{'sentence':<50} tokens   cos(board, numpy)   board time")
board = []
for s in SENTENCES:
    v, ms = dev.embed(s)
    board.append(np.array(v))
    r = ref.embed(model, s)
    ntok = len(ref.tokenize(model, s))
    print(f"{s[:48]:<50} {ntok:>5}   {float(np.dot(board[-1], r)):.6f}          {ms / 1000:5.2f} s")

print("\nSimilarity between pairs (board vs numpy):")
for i, j in PAIRS:
    on_board = float(np.dot(board[i], board[j]))
    on_pc = ref.cosine(ref.embed(model, SENTENCES[i]), ref.embed(model, SENTENCES[j]))
    print(f"  {SENTENCES[i][:25]!r} vs {SENTENCES[j][:25]!r}:  {on_board:.4f}  {on_pc:.4f}")
dev.close()
