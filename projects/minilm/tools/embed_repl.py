#!/usr/bin/env python3
"""
embed_repl.py - an interactive prompt for the ESP32 embedding model.

TALK TO THE BOARD
    embed <text>              show the embedding (first 8 of 384 numbers)
    dist <text a> | <text b>  similarity (1 = same meaning) and distance (1 - similarity)
    tokens <text>             show how the text is split into word pieces
    info                      memory / model facts from the board

AFTER EACH CALL a chart shows where its time went, and the call is added to the history
file (projects/minilm/history/calls.jsonl, shared with minilm_cli.py).

LOOK AT THE HISTORY (no board needed)
    history [n]               the last n calls (default 15)
    stats                     summary, with a straight-line fit of time against length
    plot <kind>               phases | latency | bottleneck | timeline | similarity | all
    chart on|off              draw the per-call chart, or print a plain table instead

OTHER
    help                      this list
    exit                      quit (Up/Down arrows recall earlier commands)

Usage:  uv run python projects/minilm/tools/embed_repl.py [serial-port] [--yes | --no-flash]

On start-up the board is asked which firmware it runs. If it is not this project's firmware, or
the model is missing or out of date, the tool says so and asks before writing anything
(--yes writes without asking, --no-flash never writes).
"""
import argparse
import atexit
import readline          # importing this gives input() arrow-key history and line editing
import sys
from pathlib import Path

import calls
import charts
import history
from esp32tools import FlashError, Mismatch
from device import connect

# Remember typed commands between sessions (~/.minilm_history). This is only the
# keyboard recall; the record of calls and their timings is history/calls.jsonl.
HISTORY = Path.home() / ".minilm_history"
try:
    readline.read_history_file(HISTORY)
except FileNotFoundError:
    pass
readline.set_history_length(500)
atexit.register(readline.write_history_file, HISTORY)

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("port", nargs="?", default="/dev/ttyACM0")
ap.add_argument("--yes", action="store_true")
ap.add_argument("--no-flash", action="store_true")
args = ap.parse_args()
port = args.port
print(f"Connecting to {port} (the board resets when the port opens)...")
try:
    dev = connect(port, "yes" if args.yes else "never" if args.no_flash else "ask")
except (Mismatch, FlashError) as e:
    sys.exit(f"error: {e}")
print("Ready. Type 'help' for the commands.")

show_chart = True

while True:
    try:
        line = input("minilm> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not line:
        continue
    word, _, rest = line.partition(" ")
    rest = rest.strip()

    try:
        if word == "exit":
            break
        elif word == "help":
            print(__doc__.split("Usage:")[0].strip("\n"))
        elif word == "chart":
            if rest not in ("on", "off"):
                print("usage: chart on|off")
            else:
                show_chart = rest == "on"
                print("per-call chart is", rest)
        elif word == "history":
            print(history.format_list(history.load(), int(rest) if rest.isdigit() else 15))
        elif word == "stats":
            print(history.format_stats(history.stats(history.load())))
        elif word == "plot":
            kinds = list(charts.KINDS) if rest == "all" else [rest]
            if rest != "all" and rest not in charts.KINDS:
                print("usage: plot " + " | ".join(charts.KINDS) + " | all")
            else:
                for k in kinds:
                    print(charts.render(k, history.load()))
        elif word in ("embed", "tokens") and rest:
            print(calls.format_result(calls.run_call(dev, "repl", word, [rest]), chart=show_chart))
        elif word == "dist" and rest.count("|") == 1:
            a, b = (s.strip() for s in rest.split("|"))
            print(calls.format_result(calls.run_call(dev, "repl", "dist", [a, b]), chart=show_chart))
        elif word == "dist":
            print("usage: dist <text a> | <text b>")
        elif word == "info":
            print(calls.format_result(calls.run_call(dev, "repl", "info"), chart=show_chart))
        else:
            print("unknown command (type 'help')")
    except Exception as e:                       # keep the prompt alive on errors
        print("error:", e)
dev.close()
