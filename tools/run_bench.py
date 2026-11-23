#!/usr/bin/env python3
"""Read BENCH lines from the board, print a table and save a CSV in results/."""
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
TIMEOUT = 180

s = serial.Serial()
s.port, s.baudrate, s.timeout = PORT, 115200, 1
s.dtr = False
s.rts = False
s.open()
# Reset the board so the benchmark starts from the beginning.
s.rts = True
time.sleep(0.1)
s.rts = False

rows = []
end = time.time() + TIMEOUT
while time.time() < end:
    line = s.readline().decode(errors="replace").strip()
    if not line.startswith("BENCH,"):
        continue
    parts = line.split(",")
    if len(parts) != 6:
        continue
    if parts[1] == "done":
        break
    rows.append(parts[1:])
    print(f"{parts[1]:<13}{parts[2]:<16}{int(parts[3]):>9}  {parts[4]:>10} {parts[5]}", flush=True)
s.close()

if not rows:
    sys.exit("No benchmark output received.")
out = Path(__file__).resolve().parent.parent / "results" / f"bench_{datetime.now():%Y%m%d_%H%M%S}.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["memory", "test", "size_bytes", "value", "unit"])
    w.writerows(rows)
print(f"\nSaved {len(rows)} rows to {out}")
