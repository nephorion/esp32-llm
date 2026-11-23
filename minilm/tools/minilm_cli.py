#!/usr/bin/env python3
"""
minilm_cli.py - one-shot command line for the MiniLM model running on the ESP32.

It does the same things as the interactive prompt (embed_repl.py) but runs a single
command and exits, so it can be used from scripts, from `make`, or by an AI assistant.

TALK TO THE BOARD
    minilm_cli.py dist   "the cat sat" "a kitten rested"     similarity + distance
    minilm_cli.py embed  "hello world"                       the embedding
    minilm_cli.py tokens "unbelievable cats!"                how the text is split up
    minilm_cli.py info                                       model and memory facts
    minilm_cli.py batch  < pairs.txt                         many `a | b` pairs, one connection

EVERY CALL IS RECORDED to minilm/history/calls.jsonl (see history.py). These commands
read that record and need NO board:
    minilm_cli.py history [-n 15]       the last calls
    minilm_cli.py stats                 summary, plus a straight-line fit of time vs length
    minilm_cli.py plot KIND             ASCII chart: phases | latency | bottleneck |
                                        timeline | similarity | all

After a call that runs the network, a plotext bar chart shows where its time went.

OPTIONS (before or after the command)
    --port /dev/ttyACM0   serial port
    --json                machine-readable output (one JSON document on stdout)
    --no-chart            print the numbers as a table instead of a chart
    --no-timing           leave timing out of the output
    --no-history          do not record this call
    --full                (embed) show all 384 numbers, not just the first 8

EXIT CODES
    0  success
    1  the board rejected the command
    2  bad command-line arguments (argparse's standard code)
    3  could not talk to the board (port in use, not plugged in, model not flashed)

NOTES
    * Opening the serial port RESETS the board, so each run spends about 2.5 s waiting for
      it to boot. Use `batch` to compare many pairs with one connection.
    * Only one program can use the serial port at a time (close the REPL first).
    * Text is ASCII only and is cut off at 16 tokens (see the README, "Limitations").
"""
import argparse
import json
import sys

import serial

import calls
import charts
import history
from device import BoardError, Device


def present(res, args):
    """Print one call's result as JSON or as human-readable text."""
    if args.json:
        out = calls.to_json(res, full=getattr(args, "full", False))
        if args.no_timing:
            out.pop("timing", None)
        print(json.dumps(out, indent=2))
    else:
        shown = dict(res, timing=None) if args.no_timing else res
        print(calls.format_result(shown, full=getattr(args, "full", False), chart=not args.no_chart))


def run(dev, args, command, texts):
    return calls.run_call(dev, "cli", command, texts, record=not args.no_history)


# ---------------------------------------------------------------------------
# Commands that talk to the board
# ---------------------------------------------------------------------------
def cmd_dist(dev, args):
    present(run(dev, args, "dist", [args.a, args.b]), args)


def cmd_embed(dev, args):
    present(run(dev, args, "embed", [args.text]), args)


def cmd_tokens(dev, args):
    present(run(dev, args, "tokens", [args.text]), args)


def cmd_info(dev, args):
    present(run(dev, args, "info", []), args)


def cmd_batch(dev, args):
    """Read `text a | text b` lines from stdin; compare each pair on one connection."""
    results = []
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("|") != 1:
            sys.exit(f"bad line (need exactly one '|'): {line!r}")
        a, b = (s.strip() for s in line.split("|"))
        res = run(dev, args, "dist", [a, b])
        results.append(res)
        if not args.json:
            print(f"SIM {res['similarity']:+.4f}  ({res['meaning']:<24}) {a!r} | {b!r}   "
                  f"[{res['timing']['inference_s']:.1f} s]", flush=True)
    if args.json:
        print(json.dumps([calls.to_json(r) for r in results], indent=2))


# ---------------------------------------------------------------------------
# Commands that only read the history (no board needed)
# ---------------------------------------------------------------------------
def cmd_history(args):
    records = history.load()
    if args.json:
        print(json.dumps(records[-args.n:], indent=2))
    else:
        print(history.format_list(records, args.n))


def cmd_stats(args):
    s = history.stats(history.load())
    print(json.dumps(s, indent=2) if args.json else history.format_stats(s))


def cmd_plot(args):
    records = history.load()
    kinds = list(charts.KINDS) if args.kind == "all" else [args.kind]
    for k in kinds:
        print(charts.render(k, records))


# ---------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="One-shot command line for MiniLM on the ESP32.",
                                 epilog="See the top of this file for exit codes and notes.")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--no-chart", action="store_true", help="timing as a table, not a chart")
    ap.add_argument("--no-timing", action="store_true", help="omit the timing")
    ap.add_argument("--no-history", action="store_true", help="do not record this call")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_global_flags(p):
        """Accept the global flags AFTER the command too (`stats --json` as well as
        `--json stats`). SUPPRESS means "leave the value alone unless it was given"."""
        for flag, help_ in (("--json", "machine-readable output"), ("--no-chart", "table, not chart"),
                            ("--no-timing", "omit the timing"), ("--no-history", "do not record")):
            p.add_argument(flag, action="store_true", default=argparse.SUPPRESS, help=help_)

    def board(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn, needs_board=True)
        add_global_flags(p)
        return p

    def offline(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn, needs_board=False)
        add_global_flags(p)
        return p

    p = board("dist", cmd_dist, "similarity and distance between two texts")
    p.add_argument("a")
    p.add_argument("b")
    p = board("embed", cmd_embed, "the 384-number embedding of a text")
    p.add_argument("text")
    p.add_argument("--full", action="store_true", help="show all 384 numbers")
    p = board("tokens", cmd_tokens, "how a text is split into word pieces")
    p.add_argument("text")
    board("info", cmd_info, "model and memory facts from the board")
    board("batch", cmd_batch, "compare many `a | b` pairs from stdin")

    p = offline("history", cmd_history, "the last calls (read from the history file)")
    p.add_argument("-n", type=int, default=15, help="how many (default 15)")
    offline("stats", cmd_stats, "summary of the history, with a time-vs-length fit")
    p = offline("plot", cmd_plot, "ASCII chart drawn from the history")
    p.add_argument("kind", choices=list(charts.KINDS) + ["all"],
                   help="; ".join(f"{k}: {v[1]}" for k, v in charts.KINDS.items()))
    return ap


def main():
    args = build_parser().parse_args()

    if not args.needs_board:                       # history / stats / plot: no serial port
        args.fn(args)
        return 0

    try:
        dev = Device(args.port)
    except (ConnectionError, serial.SerialException, OSError) as e:
        why = "the port is in use by another program (close the REPL or monitor)" \
            if "lock" in str(e).lower() else str(e)
        print(f"error: cannot talk to the board on {args.port}: {why}", file=sys.stderr)
        return 3
    try:
        args.fn(dev, args)
    except BoardError as e:
        print(f"error: board replied: {e}", file=sys.stderr)
        return 1
    except (TimeoutError, serial.SerialException, OSError) as e:
        print(f"error: lost contact with the board: {e}", file=sys.stderr)
        return 3
    finally:
        dev.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
