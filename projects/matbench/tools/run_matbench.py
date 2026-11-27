#!/usr/bin/env python3
"""Run the matrix benchmarks on the board, print tables and save a CSV in results/.

Usage:  uv run python projects/matbench/tools/run_matbench.py [serial-port] [--yes | --no-flash]

On start-up the board is asked which firmware it runs. If it is not matbench, the tool says
so and asks before writing it (--yes writes without asking, --no-flash never writes).
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from esp32tools import Board, BoardError, FlashError, Mismatch, Requirement, connect
from esp32tools.firmware import app_build_id
from esp32tools.ident import identify

# Version of the serial protocol: must equal PROTO_VERSION in main/main.c.
PROTO_VERSION = 1
HERE = Path(__file__).resolve().parent.parent
REQUIREMENT = Requirement(project="matbench", proto=PROTO_VERSION, project_dir=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", default="/dev/ttyACM0")
    ap.add_argument("--yes", action="store_true", help="write the matbench firmware without asking if needed")
    ap.add_argument("--no-flash", action="store_true", help="never write; report a mismatch and stop")
    args = ap.parse_args()

    try:
        board = connect(Board, args.port, REQUIREMENT,
                        "yes" if args.yes else "never" if args.no_flash else "ask")
    except (Mismatch, FlashError) as e:
        sys.exit(f"error: {e}")

    try:
        ident, why = identify(board)
        if ident is None:
            sys.exit(f"error: {why}")
        local = app_build_id(REQUIREMENT.image)
        print(f"board: {ident.describe()}")
        if local and local != ident.build:
            print(f"WARNING: the local build is {local}. These numbers are from the OLDER firmware "
                  f"on the board; run `make build flash` to measure the current code.")
        info = board.command("info")[0].split()[1:]
        print(" ".join(info))
        selftest = board.command("selftest", timeout=60)[0]
        print(selftest)
        lines = board.command("run", timeout=600)
    except (BoardError, TimeoutError) as e:
        sys.exit(f"error: {e}")
    finally:
        board.close()

    mhz = int(dict(kv.split("=") for kv in info)["cpu_mhz"])
    rows, skips = [], []
    for line in lines:
        p = line.split(",")
        if p[0] == "RESULT" and len(p) == 9:
            op, typ, impl, mem, shape, macs, cycles, ok = p[1:]
            rows.append(dict(op=op, type=typ, impl=impl, mem=mem, shape=shape,
                             macs=int(macs), cycles=int(cycles), ok=ok))
        elif p[0] == "SKIP":
            skips.append(p[1:])
    if not rows:
        sys.exit("No benchmark output received.")

    base = {(r["op"], r["type"], r["mem"], r["shape"]): r["cycles"] for r in rows if r["impl"] == "c"}
    for r in rows:
        r["mac_per_cycle"] = round(r["macs"] / r["cycles"], 3)
        r["mmac_per_s"] = round(r["macs"] / r["cycles"] * mhz, 1)
        r["speedup_vs_c"] = round(base[r["op"], r["type"], r["mem"], r["shape"]] / r["cycles"], 2)

    last = None
    for r in rows:
        key = (r["type"], r["op"], r["shape"], r["mem"])
        if key != last:
            print(f"\n{r['type']} {r['op']} {r['shape']}, weights in {r['mem']}")
            print(f"  {'impl':<10}{'cycles':>11}{'time us':>11}{'MAC/cycle':>11}{'MMAC/s':>9}{'vs C':>7}")
            last = key
        print(f"  {r['impl']:<10}{r['cycles']:>11}{r['cycles'] / mhz:>11.1f}{r['mac_per_cycle']:>11.3f}"
              f"{r['mmac_per_s']:>9.1f}{r['speedup_vs_c']:>6.2f}x" + ("" if r["ok"] == "ok" else "  WRONG RESULT"))
    for s in skips:
        print(f"\nskipped {s[1]} {s[0]} {s[3]} in {s[2]}: {s[4]}")

    out = HERE / "results" / f"matbench_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out}")

    bad = sum(r["ok"] != "ok" for r in rows)
    if bad or "failed=0" not in selftest.split():
        sys.exit(f"error: {bad} wrong results; {selftest}")


if __name__ == "__main__":
    main()
