#!/usr/bin/env python3
"""REPL to control the ESP32 LED over serial. Commands: on, off, exit."""
import sys
import time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"

# Configure before opening to minimise DTR/RTS toggling.
s = serial.Serial()
s.port, s.baudrate, s.timeout = PORT, 115200, 1
s.dtr = False
s.rts = False
s.open()
time.sleep(2.5)  # opening the port may reset the board; let it boot
s.reset_input_buffer()

print(f"Connected to {PORT}. Commands: on, off, exit")
while True:
    try:
        cmd = input("led> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        cmd = "exit"
    if cmd not in ("on", "off", "exit"):
        if cmd:
            print("unknown command (use: on, off, exit)")
        continue
    s.write((cmd + "\n").encode())
    print(s.readline().decode(errors="replace").strip() or "(no reply)")
    if cmd == "exit":
        break
s.close()
