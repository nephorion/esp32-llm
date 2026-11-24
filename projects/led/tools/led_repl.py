#!/usr/bin/env python3
"""REPL to control the ESP32 LED over serial. Commands: on, off, exit.

Usage:  uv run python projects/led/tools/led_repl.py [serial-port] [--yes | --no-flash]

On start-up the board is asked which firmware it runs. If it is not the led firmware, the
tool says so and asks before writing it (--yes writes without asking, --no-flash never writes).
"""
import argparse
import sys
from pathlib import Path

from esp32tools import BoardError, Board, FlashError, Mismatch, Requirement, connect

# Version of the serial protocol: must equal PROTO_VERSION in main/main.c.
PROTO_VERSION = 1
REQUIREMENT = Requirement(project="led", proto=PROTO_VERSION,
                          project_dir=Path(__file__).resolve().parent.parent)

ap = argparse.ArgumentParser()
ap.add_argument("port", nargs="?", default="/dev/ttyACM0")
ap.add_argument("--yes", action="store_true", help="write the led firmware without asking if needed")
ap.add_argument("--no-flash", action="store_true", help="never write; report a mismatch and stop")
args = ap.parse_args()

try:
    board = connect(Board, args.port, REQUIREMENT, "yes" if args.yes else "never" if args.no_flash else "ask")
except (Mismatch, FlashError) as e:
    sys.exit(f"error: {e}")

print(f"Connected to {args.port}. Commands: on, off, exit")
while True:
    try:
        cmd = input("led> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        cmd = "exit"
    if cmd not in ("on", "off", "exit"):
        if cmd:
            print("unknown command (use: on, off, exit)")
        continue
    try:
        board.command(cmd)
        print("OK", cmd)
    except BoardError as e:
        print(e)
    if cmd == "exit":
        break
board.close()
