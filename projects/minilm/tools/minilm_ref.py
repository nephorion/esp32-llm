#!/usr/bin/env python3
"""
minilm_ref.py - a plain-numpy implementation of MiniLM-L6 sentence embeddings.

This is the READABLE version of what the ESP32 firmware does. It reads the very
same model.bin the board reads (so it also checks the file format), and every step
below has a matching function in projects/minilm/main/bert.c. Read this first, then the C.

The whole pipeline, for one piece of text:

    text --tokenize--> [CLS] piece piece ... [SEP]            (list of token ids)
         --embed-----> one 384-number vector per token         (lookup + position)
         --6 x transformer layer--> still one vector per token (tokens "look at" each other)
         --mean pool--> ONE 384-number vector for the whole text
         --normalise--> scaled to length 1

Two texts are compared with cosine similarity: because both vectors have length 1,
that is just their dot product (1.0 = same meaning, ~0 = unrelated).

USAGE
-----
  uv run python projects/minilm/tools/minilm_ref.py embed "hello world"
  uv run python projects/minilm/tools/minilm_ref.py dist  "a cat sat" "a kitten rested"
"""
import struct
import sys
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parent.parent / "model_build" / "model.bin"
MAX_TOKENS = 16        # must match MAX_TOKENS in main/bert.h
EPS = 1e-12            # LayerNorm epsilon used by this model


# --------------------------------------------------------------------------
# Loading model.bin  (mirror of main/model.c)
# --------------------------------------------------------------------------
class Reader:
    """Walks through the file in the exact order export_model.py wrote it."""

    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def f32(self, n):
        a = np.frombuffer(self.data, dtype="<f4", count=n, offset=self.pos)
        self.pos += 4 * n
        return a

    def qmat(self, rows, cols):
        """Read int8 data + per-row scales and return the dequantised float32 matrix."""
        q = np.frombuffer(self.data, dtype=np.int8, count=rows * cols, offset=self.pos)
        self.pos += (rows * cols + 3) // 4 * 4
        scale = self.f32(rows)
        return q.reshape(rows, cols).astype(np.float32) * scale[:, None]


def load(path=MODEL):
    data = Path(path).read_bytes()
    (magic, version, vocab_n, hid, n_layers, n_heads, inter, max_pos,
     unk, cls, sep, _) = struct.unpack_from("<12I", data, 0)
    assert magic == 0x364C4D4D, "not a model.bin"
    r = Reader(data)
    r.pos = 48
    m = dict(hid=hid, n_heads=n_heads, unk=unk, cls=cls, sep=sep)
    m["word"] = r.qmat(vocab_n, hid)
    m["pos"] = r.f32(max_pos * hid).reshape(max_pos, hid)
    m["type"] = r.f32(hid)
    m["emb_g"], m["emb_b"] = r.f32(hid), r.f32(hid)
    m["layers"] = []
    for _ in range(n_layers):
        L = {}
        for name in ("q", "k", "v", "o"):
            L[name + "_w"] = r.qmat(hid, hid)
            L[name + "_b"] = r.f32(hid)
        L["attn_g"], L["attn_b"] = r.f32(hid), r.f32(hid)
        L["f1_w"], L["f1_b"] = r.qmat(inter, hid), r.f32(inter)
        L["f2_w"], L["f2_b"] = r.qmat(hid, inter), r.f32(hid)
        L["out_g"], L["out_b"] = r.f32(hid), r.f32(hid)
        m["layers"].append(L)
    # vocabulary: sorted strings -> id
    n = struct.unpack_from("<I", data, r.pos)[0]
    offs = np.frombuffer(data, "<u4", n, r.pos + 4)
    ids = np.frombuffer(data, "<u2", n, r.pos + 4 + 4 * n)
    blob_at = (r.pos + 4 + 4 * n + 2 * n + 3) // 4 * 4
    m["vocab"] = {}
    for off, i in zip(offs, ids):
        end = data.index(b"\0", blob_at + off)
        m["vocab"][data[blob_at + off:end].decode()] = int(i)
    return m


# --------------------------------------------------------------------------
# Step 1: tokenizer  (mirror of main/tokenizer.c)
# --------------------------------------------------------------------------
def basic_split(text):
    """Lower-case, split on whitespace, and make every punctuation mark its own word.
    Non-ASCII characters are dropped (the firmware tokenizer is ASCII-only)."""
    words, cur = [], ""
    for ch in text.lower():
        if ch.isascii() and (ch.isalnum()):
            cur += ch
        else:
            if cur:
                words.append(cur)
                cur = ""
            if ch.isascii() and ch.isprintable() and not ch.isspace():
                words.append(ch)               # punctuation: a word by itself
    if cur:
        words.append(cur)
    return words


def wordpiece(word, vocab, unk):
    """Greedy longest-match-first: 'embeddings' -> 'em', '##bed', '##ding', '##s' ...
    Pieces after the first carry a '##' prefix. If no piece matches, the whole
    word becomes [UNK]."""
    if len(word) > 100:
        return [unk]
    pieces, start = [], 0
    while start < len(word):
        end, found = len(word), None
        while end > start:
            sub = word[start:end]
            if start > 0:
                sub = "##" + sub
            if sub in vocab:
                found = vocab[sub]
                break
            end -= 1
        if found is None:
            return [unk]
        pieces.append(found)
        start = end
    return pieces


def tokenize(m, text):
    ids = []
    for w in basic_split(text):
        ids += wordpiece(w, m["vocab"], m["unk"])
    ids = ids[: MAX_TOKENS - 2]                 # leave room for [CLS] and [SEP]
    return [m["cls"]] + ids + [m["sep"]]


# --------------------------------------------------------------------------
# Steps 2-4: the neural network (mirror of main/bert.c)
# --------------------------------------------------------------------------
def layer_norm(x, g, b):
    """Normalise each token's vector to mean 0 / variance 1, then scale and shift."""
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + EPS) * g + b


def gelu(x):
    """The smooth 'ReLU-like' activation BERT uses (exact erf form)."""
    from math import erf, sqrt
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def encoder_layer(x, L, n_heads):
    T, H = x.shape
    d = H // n_heads                            # 32 numbers per head
    # --- self-attention: every token mixes in information from every other token
    q = x @ L["q_w"].T + L["q_b"]
    k = x @ L["k_w"].T + L["k_b"]
    v = x @ L["v_w"].T + L["v_b"]
    ctx = np.zeros_like(x)
    for h in range(n_heads):                    # 12 independent "heads", 32 numbers each
        sl = slice(h * d, (h + 1) * d)
        scores = q[:, sl] @ k[:, sl].T / np.sqrt(d)    # how much token i cares about token j
        ctx[:, sl] = softmax(scores) @ v[:, sl]        # weighted average of the values
    x = layer_norm(x + ctx @ L["o_w"].T + L["o_b"], L["attn_g"], L["attn_b"])   # residual + norm
    # --- feed-forward: each token is processed on its own (384 -> 1536 -> 384)
    f = gelu(x @ L["f1_w"].T + L["f1_b"])
    x = layer_norm(x + f @ L["f2_w"].T + L["f2_b"], L["out_g"], L["out_b"])     # residual + norm
    return x


def embed(m, text):
    ids = tokenize(m, text)
    # token embedding + position embedding + segment embedding, then LayerNorm
    x = m["word"][ids] + m["pos"][: len(ids)] + m["type"]
    x = layer_norm(x, m["emb_g"], m["emb_b"])
    for L in m["layers"]:
        x = encoder_layer(x, L, m["n_heads"])
    v = x.mean(axis=0)                          # mean pooling: average all token vectors
    return v / np.linalg.norm(v)                # unit length -> dot product == cosine


def cosine(a, b):
    return float(np.dot(a, b))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("embed", "dist"):
        sys.exit(__doc__)
    model = load()
    if sys.argv[1] == "embed":
        v = embed(model, sys.argv[2])
        print("tokens:", tokenize(model, sys.argv[2]))
        print("first 8:", np.round(v[:8], 4))
    else:
        a, b = embed(model, sys.argv[2]), embed(model, sys.argv[3])
        print(f"similarity {cosine(a, b):.4f}   distance {1 - cosine(a, b):.4f}")
