#!/usr/bin/env python3
"""Tune the GELU switches: time per embedding and how far the answers move.

Usage:  uv run python projects/minilm-fast/tools/gelu.py [serial-port] [--tables 8 64 256] [--yes | --no-flash]

Times use kernel=simd cores=2, the fastest matrix path. Accuracy is always measured against
the ORIGINAL float code with exact GELU, two ways:

  gelu alone   on the float path, so the GELU change is the only difference
  with simd    on the simd path, which is what you would actually run

Do not compare two simd runs with each other: int8 rounding is so sensitive that a change of
0.000001 anywhere reshuffles it (cosine about 0.998 between the two), without making either
one further from the float answer.

Takes about 12 minutes. Saves a CSV in results/.
"""
import argparse
import sys

import numpy as np

import compare as c      # the sibling script: sentences, helpers, requirement

FAST = "fetch=fast kernel=simd cores=2 v=int8"
FLOAT = "fetch=fast kernel=float cores=2 v=int8"


def unit(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="/dev/ttyACM0")
    ap.add_argument("--tables", type=int, nargs="+", default=[8, 16, 64, 256, 1024], help="table sizes to try")
    ap.add_argument("--yes", action="store_true", help="write the firmware/model without asking if needed")
    ap.add_argument("--no-flash", action="store_true", help="never write; report a mismatch and stop")
    args = ap.parse_args()

    try:
        dev = c.connect(c.minilm.Device, args.port, c.REQUIREMENT,
                        "yes" if args.yes else "never" if args.no_flash else "ask")
    except (c.Mismatch, c.FlashError) as e:
        sys.exit(f"error: {e}")

    # name, gelu method, table size, gelu cores
    configs = [("exact", "exact", 256, 1), ("exact, 2 cores", "exact", 256, 2)]
    configs += [(f"table {n}", "table", n, 1) for n in args.tables]
    configs += [("table 256, 2 cores", "table", 256, 2), ("relu", "relu", 256, 1)]

    def embed_all():
        return unit(np.stack([c.vector(c.embed(dev, s)) for s in c.SENTENCES]))

    def against(ref, vectors):
        cos = (vectors * ref).sum(1)
        pairs = np.triu_indices(len(c.SENTENCES), 1)
        shift = np.abs(vectors @ vectors.T - ref @ ref.T)[pairs]
        return round(float(cos.mean()), 6), round(float(cos.min()), 6), round(float(shift.max()), 6)

    rows, alone = [], {}
    try:
        dev.command("mode " + c.BASE)
        original = embed_all()
        print("done: original float reference", flush=True)
        for name, method, size, cores in configs:
            switches = f"gelu={method} table={size} gcores={cores}"
            dev.command(f"mode {FAST} {switches}")
            timed = {}
            for text in c.TIMED_TEXTS:
                tokens = len(dev.tokens(text))
                runs = []
                for _ in range(c.REPEATS):
                    c.embed(dev, text)
                    runs.append(dict(dev.last_phases))
                timed[tokens] = min(runs, key=lambda r: r["total"])
            row = dict(config=name, gelu=method, table=size if method == "table" else "", gcores=cores)
            for tokens, r in timed.items():
                row[f"total_s_{tokens}"] = round(r["total"], 4)
                row[f"gelu_s_{tokens}"] = round(r["gelu"], 4)
            simd = against(original, embed_all())
            # A second GELU core computes the same values, so reuse the one-core accuracy.
            key = (method, size)
            if key not in alone:
                dev.command(f"mode {FLOAT} gelu={method} table={size} gcores=1")
                alone[key] = against(original, embed_all())
            for prefix, (mean, worst, shift) in (("alone", alone[key]), ("simd", simd)):
                row.update({f"{prefix}_cos_mean": mean, f"{prefix}_cos_min": worst, f"{prefix}_shift_max": shift})
            rows.append(row)
            print(f"done: {name}", flush=True)
        dev.command("mode " + c.BASE)
    except (c.BoardError, TimeoutError) as e:
        sys.exit(f"error: {e}")
    finally:
        dev.close()

    long = max(int(k.rsplit("_", 1)[1]) for k in rows[0] if k.startswith("total_s_"))
    short = min(int(k.rsplit("_", 1)[1]) for k in rows[0] if k.startswith("total_s_"))
    print(f"\ntimes with kernel=simd cores=2; accuracy against the original float code, {len(c.SENTENCES)} sentences")
    print(f"{'':<20}{f'{short} tok':>8}{f'{long} tok':>8}{'gelu':>7}  | {'GELU alone (float path)':^33} | {'with simd':^22}")
    print(f"{'gelu':<20}{'total s':>8}{'total s':>8}{'s':>7}  | {'cos mean':>10}{'cos min':>10}{'shift max':>11}  | {'cos mean':>10}{'cos min':>10}")
    for r in rows:
        print(f"{r['config']:<20}{r[f'total_s_{short}']:>8.3f}{r[f'total_s_{long}']:>8.3f}{r[f'gelu_s_{long}']:>7.3f}  | "
              f"{r['alone_cos_mean']:>10.6f}{r['alone_cos_min']:>10.6f}{r['alone_shift_max']:>11.6f}  | "
              f"{r['simd_cos_mean']:>10.6f}{r['simd_cos_min']:>10.6f}")
    print()
    c.save("gelu", rows)


if __name__ == "__main__":
    main()
