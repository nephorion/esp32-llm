#!/usr/bin/env python3
"""
verify_hf.py - prove that our shrunken model still behaves like the real one.

We embed the same sentences with
    (a) the ORIGINAL float32 model from Hugging Face (needs torch + transformers), and
    (b) our int8 + pruned-vocabulary model (minilm_ref.py reading model.bin),
and print the cosine similarity between the two answers for each sentence.
1.0 would be identical. Anything above ~0.98 means the shrinking cost very little.

It also compares the *distances* between sentence pairs, which is what you actually use.

torch is big, so it is not a project dependency. Run this with a throwaway env:

  uv run --with torch --with transformers \
      --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
      python minilm/tools/verify_hf.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import minilm_ref as ref                              # noqa: E402

import torch                                          # noqa: E402
from transformers import AutoModel, AutoTokenizer    # noqa: E402

HF = Path(__file__).resolve().parent.parent / "model_build" / "hf"
SENTENCES = [
    "hello world",
    "a cat sat on the mat",
    "a kitten rested on the rug",
    "stock markets fell sharply today",
    "how do i reset my password?",
    "I forgot my login credentials",
    "the quick brown fox jumps over the lazy dog",
    "espresso machines need regular descaling",
]
PAIRS = [(1, 2), (1, 3), (4, 5), (0, 3), (6, 1)]


def hf_embed(tok, model, text):
    """Reference answer: tokenizer -> BertModel -> mean pool -> normalise."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=ref.MAX_TOKENS)
    with torch.no_grad():
        out = model(**enc).last_hidden_state[0]
    v = out.mean(dim=0).numpy()
    return v / np.linalg.norm(v)


def main():
    tok = AutoTokenizer.from_pretrained(str(HF))
    model = AutoModel.from_pretrained(str(HF)).eval()
    ours = ref.load()

    print(f"{'sentence':<48} cos(ours, original)")
    a_vecs, b_vecs = [], []
    for s in SENTENCES:
        a, b = hf_embed(tok, model, s), ref.embed(ours, s)
        a_vecs.append(a)
        b_vecs.append(b)
        print(f"{s:<48} {float(np.dot(a, b)):.4f}")

    print(f"\n{'pair':<62} original   ours")
    for i, j in PAIRS:
        label = f"{SENTENCES[i][:28]!r} vs {SENTENCES[j][:28]!r}"
        print(f"{label:<62} {np.dot(a_vecs[i], a_vecs[j]):.4f}    {np.dot(b_vecs[i], b_vecs[j]):.4f}")


if __name__ == "__main__":
    main()
