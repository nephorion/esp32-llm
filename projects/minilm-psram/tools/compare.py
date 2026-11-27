#!/usr/bin/env python3
"""Time embeddings with 0, 1, 2 ... layers read from PSRAM, and check the answers are identical.

Usage:  uv run python projects/minilm-psram/tools/compare.py [serial-port] [--yes | --no-flash]

Uses the MiniLM host code and model from projects/minilm. Saves a CSV in results/.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "minilm" / "tools"))

import device as minilm                                                  # noqa: E402
from esp32tools import BoardError, FlashError, Mismatch, Requirement, connect  # noqa: E402

REQUIREMENT = Requirement(project="minilm-psram", proto=minilm.PROTO_VERSION, project_dir=HERE,
                          data=minilm.REQUIREMENT.data)
TEXTS = ["hello",
         "a cat sat on the mat",
         "after a long day she walked home through the quiet park and then went to bed early"]
REPEATS = 3


def psram(dev, n=None):
    line = dev.command("psram" if n is None else f"psram {n}")[0]
    return {k: int(v) for k, v in (kv.split("=") for kv in line.split()[1:])}


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

    rows, reference, different = [], {}, 0
    try:
        status = psram(dev)
        print(f"{status['max']} layers copied to PSRAM in {status['copy_ms']} ms, "
              f"{status['free'] / 1e6:.2f} MB of PSRAM left")
        for n in range(status["max"] + 1):
            psram(dev, n)
            for text in TEXTS:
                tokens = len(dev.tokens(text))
                runs = []
                for _ in range(REPEATS):
                    reply = dev.command("embed " + text)
                    vec = next(l for l in reply if l.startswith("VEC"))
                    different += vec != reference.setdefault(text, vec)
                    runs.append(dict(dev.last_phases))
                best = min(runs, key=lambda r: r["total"])
                rows.append(dict(psram_layers=n, tokens=tokens, total_s=round(best["total"], 4),
                                 weight_fetch_s=round(best["weight_fetch"], 4),
                                 arithmetic_s=round(best["dot"], 4)))
        psram(dev, status["max"])
    except (BoardError, TimeoutError) as e:
        sys.exit(f"error: {e}")
    finally:
        dev.close()

    base = {r["tokens"]: r["total_s"] for r in rows if r["psram_layers"] == 0}
    print(f"\n{'PSRAM layers':>12}{'tokens':>8}{'total s':>10}{'weight fetch s':>16}{'arithmetic s':>14}{'speedup':>9}")
    for r in rows:
        print(f"{r['psram_layers']:>12}{r['tokens']:>8}{r['total_s']:>10.3f}{r['weight_fetch_s']:>16.3f}"
              f"{r['arithmetic_s']:>14.3f}{base[r['tokens']] / r['total_s']:>8.2f}x")

    out = HERE / "results" / f"psram_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}")
    if different:
        sys.exit(f"error: {different} embeddings differed from the all-flash answer")
    print("Every embedding was identical to the all-flash one.")


if __name__ == "__main__":
    main()
