#!/usr/bin/env python3
"""
export_model.py - turn the Hugging Face MiniLM-L6 checkpoint into one small file
that the ESP32 can read straight out of flash.

WHY THIS STEP EXISTS
--------------------
The original model is 90 MB (float32). The ESP32-S3 has 16 MB of flash. To make it
fit we do two things:

  1. QUANTISE: store every big weight matrix as int8 (1 byte) instead of float32
     (4 bytes). Each matrix ROW gets its own float32 "scale", so that
         real_weight  ~=  int8_value * scale[row]
     This is called per-row symmetric int8 quantisation. It loses a little
     precision (we measure how much in verify_hf.py) but is 4x smaller.

  2. PRUNE THE VOCABULARY: the word-embedding table has 30,522 rows (11.7M
     parameters - half the whole model!). We keep
        - the ~8,000 whole words that are most common in English (ranked with the
          `wordfreq` package - the BERT vocabulary itself is NOT in frequency order),
        - ~2,500 sub-word pieces ("##ing", "##ly", ...),
        - every single ASCII character and "##" character, so ANY word can still be
          spelled out letter by letter.
     Rarer words get split into smaller pieces ("descaling" -> "desc", "##aling").
     This is the main place where our embeddings differ from the original model.

THE FILE FORMAT (all little-endian, every section 4-byte aligned)
-----------------------------------------------------------------
  header      12 x uint32:
              magic, version, vocab_size, hidden, n_layers, n_heads,
              intermediate, max_pos, unk_id, cls_id, sep_id, reserved
  word_emb    qmat [vocab_size x hidden]
  pos_emb     float32 [max_pos x hidden]
  type_emb    float32 [hidden]              (only segment 0 is ever used)
  emb_ln      gamma[hidden], beta[hidden]   (LayerNorm after the embeddings)
  for each of the n_layers layers, in this exact order:
      q   qmat [hidden x hidden]   + bias float32[hidden]
      k   qmat [hidden x hidden]   + bias
      v   qmat [hidden x hidden]   + bias
      o   qmat [hidden x hidden]   + bias      (attention output projection)
      attn_ln  gamma, beta
      ffn1 qmat [intermediate x hidden] + bias float32[intermediate]
      ffn2 qmat [hidden x intermediate] + bias float32[hidden]
      out_ln   gamma, beta
  vocab       uint32 n; uint32 str_off[n]; uint16 id[n]; pad to 4; string blob
              (strings are sorted so the C code can binary-search them)

  A "qmat" is:  int8 data[rows*cols]  then  float32 scale[rows].

The C code (main/model.c) and the numpy reference (minilm_ref.py) both read this
format in exactly this order. If you change anything here, change them too.

USAGE
-----
  uv run python projects/minilm/tools/export_model.py            # writes model_build/model.bin
"""
import struct
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

HERE = Path(__file__).resolve().parent
BUILD = HERE.parent / "model_build"
HF = BUILD / "hf"           # downloaded by `make download` (see README)
OUT = BUILD / "model.bin"

MAGIC = 0x364C4D4D          # bytes "MML6" little-endian
VERSION = 1
WORDS_KEEP = 9000           # how many whole words to keep (most frequent English words)
PIECES_KEEP = 2500          # how many "##xyz" sub-word pieces to keep (lowest original ids)
MAX_POS = 64                # only the first 64 position embeddings are stored
                            # (the firmware only uses MAX_TOKENS of them)
PARTITION_BYTES = 0xF70000  # size of the "model" flash partition (see partitions.csv)


def quantize_rows(w: np.ndarray):
    """Symmetric per-row int8 quantisation.

    For each row: scale = max(|row|) / 127, int8 = round(row / scale).
    Reconstruct with  int8 * scale.  Returns (int8 matrix, float32 scales).
    """
    scale = np.abs(w).max(axis=1) / 127.0
    scale[scale == 0] = 1.0                      # avoid divide-by-zero on all-zero rows
    q = np.clip(np.round(w / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


class Writer:
    """Tiny helper that appends sections to the output file, keeping 4-byte alignment."""

    def __init__(self):
        self.buf = bytearray()

    def pad4(self):
        while len(self.buf) % 4:
            self.buf.append(0)

    def f32(self, a):
        self.buf += np.ascontiguousarray(a, dtype="<f4").tobytes()
        self.pad4()

    def qmat(self, w):
        q, scale = quantize_rows(np.asarray(w, dtype=np.float32))
        self.buf += q.tobytes()
        self.pad4()
        self.f32(scale)

    def u32(self, *vals):
        self.buf += struct.pack("<%dI" % len(vals), *vals)


def choose_vocab(vocab):
    """Decide which original token ids survive pruning. Returns a sorted list of ids."""
    from wordfreq import zipf_frequency          # how common a word is in English text

    keep = set()
    words = []                                    # (frequency, id) for whole-word tokens
    pieces = []                                   # ids of "##xyz" tokens
    for i, tok in enumerate(vocab):
        if tok.startswith("[unused") or not tok.isascii():
            continue                              # placeholders and non-ASCII: never needed
        core = tok[2:] if tok.startswith("##") else tok
        if tok.startswith("[") or len(core) == 1:
            keep.add(i)                           # [CLS] [SEP] [UNK] ... and single characters
        elif tok.startswith("##"):
            pieces.append(i)
        elif core.isalpha():
            words.append((zipf_frequency(tok, "en"), i))
        else:
            keep.add(i)                           # numbers like "1990" - keep, they are few
    words.sort(reverse=True)
    keep.update(i for _, i in words[:WORDS_KEEP])
    keep.update(pieces[:PIECES_KEEP])
    return sorted(keep)


def main():
    cfg_hidden, n_layers, n_heads, inter = 384, 6, 12, 1536
    t = load_file(str(HF / "model.safetensors"))
    vocab = [ln.rstrip("\n") for ln in open(HF / "vocab.txt", encoding="utf-8")]

    keep = choose_vocab(vocab)
    new_id = {old: new for new, old in enumerate(keep)}        # old id -> new id
    print(f"vocabulary: kept {len(keep)} of {len(vocab)} tokens")

    w = Writer()
    w.u32(MAGIC, VERSION, len(keep), cfg_hidden, n_layers, n_heads, inter, MAX_POS,
          new_id[vocab.index("[UNK]")], new_id[vocab.index("[CLS]")],
          new_id[vocab.index("[SEP]")], 0)

    # --- embeddings -------------------------------------------------------
    w.qmat(t["embeddings.word_embeddings.weight"][keep])       # only the rows we kept
    w.f32(t["embeddings.position_embeddings.weight"][:MAX_POS])
    w.f32(t["embeddings.token_type_embeddings.weight"][0])
    w.f32(t["embeddings.LayerNorm.weight"])
    w.f32(t["embeddings.LayerNorm.bias"])

    # --- transformer layers ----------------------------------------------
    for i in range(n_layers):
        p = f"encoder.layer.{i}."
        for name in ("query", "key", "value"):
            w.qmat(t[p + f"attention.self.{name}.weight"])
            w.f32(t[p + f"attention.self.{name}.bias"])
        w.qmat(t[p + "attention.output.dense.weight"])
        w.f32(t[p + "attention.output.dense.bias"])
        w.f32(t[p + "attention.output.LayerNorm.weight"])
        w.f32(t[p + "attention.output.LayerNorm.bias"])
        w.qmat(t[p + "intermediate.dense.weight"])
        w.f32(t[p + "intermediate.dense.bias"])
        w.qmat(t[p + "output.dense.weight"])
        w.f32(t[p + "output.dense.bias"])
        w.f32(t[p + "output.LayerNorm.weight"])
        w.f32(t[p + "output.LayerNorm.bias"])

    # --- vocabulary (sorted strings -> new id) -----------------------------
    entries = sorted((vocab[old].encode("ascii", "ignore"), new_id[old])
                     for old in keep if vocab[old].isascii())
    blob, offs = bytearray(), []
    for s, _ in entries:
        offs.append(len(blob))
        blob += s + b"\0"
    w.u32(len(entries))
    w.u32(*offs)
    w.buf += struct.pack("<%dH" % len(entries), *[i for _, i in entries])
    w.pad4()
    w.buf += blob
    w.pad4()

    OUT.write_bytes(bytes(w.buf))
    mb = len(w.buf) / 1e6
    print(f"wrote {OUT}  ({mb:.2f} MB)")
    if len(w.buf) > PARTITION_BYTES:
        raise SystemExit("model is too big for the flash partition - lower WORDS_KEEP")


if __name__ == "__main__":
    main()
