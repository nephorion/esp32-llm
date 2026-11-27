#!/usr/bin/env python3
"""Compare the minilm-fast switches on the board: time per embedding, and how far the
answers move from the unmodified path.

Usage:  uv run python projects/minilm-fast/tools/compare.py [serial-port] [--yes | --no-flash]

Uses the MiniLM host code and model from projects/minilm. Saves two CSVs in results/.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "minilm" / "tools"))

import device as minilm                                                  # noqa: E402
from esp32tools import BoardError, FlashError, Mismatch, Requirement, connect  # noqa: E402

REQUIREMENT = Requirement(project="minilm-fast", proto=minilm.PROTO_VERSION, project_dir=HERE,
                          data=minilm.REQUIREMENT.data)
BASE = "fetch=ref kernel=float cores=1 v=int8"
CONFIGS = [   # name, switches
    ("original",            BASE),
    ("fast fetch",          "fetch=fast kernel=float cores=1 v=int8"),
    ("2 cores",             "fetch=ref kernel=float cores=2 v=int8"),
    ("fast fetch, 2 cores", "fetch=fast kernel=float cores=2 v=int8"),
    ("simd",                "fetch=fast kernel=simd cores=1 v=int8"),
    ("simd, 2 cores",       "fetch=fast kernel=simd cores=2 v=int8"),
    ("simd, v float",       "fetch=fast kernel=simd cores=1 v=float"),
    ("simd, v float, 2 cores", "fetch=fast kernel=simd cores=2 v=float"),
]
ACCURACY = ["fast fetch", "simd", "simd, v float"]    # cores do not change the answers (checked)
TIMED_TEXTS = ["hello",
               "a cat sat on the mat",
               "after a long day she walked home through the quiet park and then went to bed early"]
REPEATS = 3
SENTENCES = """a cat sat on the mat
a kitten rested on the rug
the dog chased the ball in the park
a puppy played fetch outside
stock markets fell sharply today
shares dropped as investors sold
the central bank raised interest rates
how do i reset my password?
I forgot my login credentials
my account is locked
espresso machines need regular descaling
clean your coffee maker often
the quick brown fox jumps over the lazy dog
hello world
good morning everyone
the weather is sunny and warm
heavy rain is expected tomorrow
she walked home through the quiet park
he went home across the calm gardens
the flight was delayed by two hours
our train arrived late
this phone has a great camera
the battery drains too quickly
i love this movie
that film was terrible
the meeting is moved to friday
please send me the invoice
where is the nearest hospital
the doctor prescribed antibiotics
water boils at one hundred degrees
the earth orbits the sun
he plays guitar in a band
she is learning to play piano
the recipe needs two eggs and flour
bake the cake for thirty minutes
the car would not start this morning
my engine is making a strange noise
python is a programming language
the code failed to compile
we are hiring a software engineer""".split("\n")


def embed(dev, text):
    reply = dev.command("embed " + text)
    return next(l for l in reply if l.startswith("VEC"))


def vector(line):
    return np.array([float(v) for v in line.split()[1:]])


def save(name, rows):
    out = HERE / "results" / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} rows to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="/dev/ttyACM0")
    ap.add_argument("--yes", action="store_true", help="write the firmware/model without asking if needed")
    ap.add_argument("--no-flash", action="store_true", help="never write; report a mismatch and stop")
    args = ap.parse_args()

    try:
        dev = connect(minilm.Device, args.port, REQUIREMENT,
                      "yes" if args.yes else "never" if args.no_flash else "ask")
    except (Mismatch, FlashError) as e:
        sys.exit(f"error: {e}")

    timing, accuracy, answers, mismatched = [], [], {}, []
    try:
        for name, switches in CONFIGS:
            dev.command("mode " + switches)
            for text in TIMED_TEXTS:
                tokens = len(dev.tokens(text))
                runs = []
                for _ in range(REPEATS):
                    vec = embed(dev, text)
                    runs.append(dict(dev.last_phases))
                    if answers.setdefault((name, text), vec) != vec:
                        mismatched.append(f"{name}: repeat differs")
                best = min(runs, key=lambda r: r["total"])
                timing.append(dict(config=name, tokens=tokens, total_s=round(best["total"], 4),
                                   weight_fetch_s=round(best["weight_fetch"], 4),
                                   arithmetic_s=round(best["dot"], 4)))
            print(f"timed: {name}", flush=True)

        # A second core must not change any answer.
        for name, _ in CONFIGS:
            if name.endswith("2 cores"):
                one = name.removesuffix(", 2 cores") if name != "2 cores" else "original"
                for text in TIMED_TEXTS:
                    a, b = vector(answers[name, text]), vector(answers[one, text])
                    # "2 cores" alone uses a different float kernel from the original, so
                    # only its last printed digit may differ.
                    if name == "2 cores":
                        if np.abs(a - b).max() > 5e-6:
                            mismatched.append(f"{name} differs from {one}")
                    elif answers[name, text] != answers[one, text]:
                        mismatched.append(f"{name} differs from {one}")

        vectors = {}
        for name, switches in [CONFIGS[0]] + [c for c in CONFIGS if c[0] in ACCURACY]:
            dev.command("mode " + switches)
            vectors[name] = np.stack([vector(embed(dev, s)) for s in SENTENCES])
            print(f"embedded {len(SENTENCES)} sentences: {name}", flush=True)
        dev.command("mode " + BASE)
    except (BoardError, TimeoutError) as e:
        sys.exit(f"error: {e}")
    finally:
        dev.close()

    base = {r["tokens"]: r["total_s"] for r in timing if r["config"] == "original"}
    print(f"\n{'switches':<26}{'tokens':>7}{'total s':>10}{'weight fetch s':>16}{'arithmetic s':>14}{'speedup':>9}")
    for r in timing:
        r["speedup"] = round(base[r["tokens"]] / r["total_s"], 2)
        print(f"{r['config']:<26}{r['tokens']:>7}{r['total_s']:>10.3f}{r['weight_fetch_s']:>16.3f}"
              f"{r['arithmetic_s']:>14.3f}{r['speedup']:>8.2f}x")

    ref = vectors["original"] / np.linalg.norm(vectors["original"], axis=1, keepdims=True)
    pairs = np.triu_indices(len(SENTENCES), 1)

    def nearest(m):
        m = m.copy()
        np.fill_diagonal(m, -9)
        return m.argmax(1)

    print(f"\nagainst the original path, {len(SENTENCES)} sentences:")
    for name in ACCURACY:
        e = vectors[name] / np.linalg.norm(vectors[name], axis=1, keepdims=True)
        cos = (e * ref).sum(1)
        shift = np.abs(e @ e.T - ref @ ref.T)[pairs]
        same = int((nearest(e @ e.T) == nearest(ref @ ref.T)).sum())
        accuracy.append(dict(config=name, cos_mean=round(float(cos.mean()), 5), cos_min=round(float(cos.min()), 5),
                             sim_shift_mean=round(float(shift.mean()), 5), sim_shift_max=round(float(shift.max()), 5),
                             same_nearest=same, sentences=len(SENTENCES)))
        print(f"  {name:<18} cosine mean {cos.mean():.4f} min {cos.min():.4f} | similarity shift mean "
              f"{shift.mean():.4f} max {shift.max():.4f} | same nearest {same}/{len(SENTENCES)}")

    print()
    save("speed", timing)
    save("accuracy", accuracy)
    if mismatched:
        sys.exit("error: " + "; ".join(sorted(set(mismatched))))


if __name__ == "__main__":
    main()
