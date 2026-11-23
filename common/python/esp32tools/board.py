"""
board.py - a line-based serial connection to an ESP32 project.

Every project in this repository speaks the same simple text protocol:

    PC  -> board :  one command per line             e.g.  "embed hello world"
    board -> PC  :  some reply lines, then "OK"      (or one line starting "ERR")
    board, once after every reset:  "READY"

Board handles the parts that are the same for every project: opening the port safely,
waiting for the board to boot, and the send-then-collect-replies loop. A project adds its
own commands by subclassing it (see projects/minilm/tools/device.py).
"""
import time

import serial


class BoardError(RuntimeError):
    """The board is connected but replied ERR/FATAL (for example a bad command)."""


class Board:
    def __init__(self, port="/dev/ttyACM0", ready_timeout=15, baudrate=115200):
        self.last_rtt_s = None        # seconds from sending the last command to its final OK
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, baudrate, 1
        # Opening the port would pulse DTR/RTS, which can reset the board into the
        # bootloader. Holding both low keeps it in the normal program.
        self.s.dtr = False
        self.s.rts = False
        # Exclusive lock: two of OUR tools (REPL, CLI, validate) can never use the port at
        # once. The second fails immediately with a clear message instead of getting
        # garbled or empty replies.
        self.s.exclusive = True
        self.s.open()
        # Opening the port resets the board, so wait for its "READY" line.
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if self.s.readline().strip() == b"READY":
                return
        raise ConnectionError("board did not say READY - port busy (another program using it?), "
                              "firmware or data not flashed, or board not connected")

    def command(self, line, timeout=60):
        """Send one command and return its reply lines (without the final OK).

        Raises BoardError if the board answers ERR/FATAL, TimeoutError if it goes quiet.
        Sets self.last_rtt_s, the PC's wall-clock time from send to the final OK."""
        self.s.reset_input_buffer()
        t0 = time.perf_counter()
        self.s.write((line + "\n").encode("ascii", "ignore"))
        lines, deadline = [], time.time() + timeout
        while time.time() < deadline:
            raw = self.s.readline().decode(errors="replace").strip()
            if raw == "OK":
                self.last_rtt_s = time.perf_counter() - t0
                self.on_reply(lines)
                return lines
            if raw.startswith("ERR") or raw.startswith("FATAL"):
                raise BoardError(raw)
            if raw:
                lines.append(raw)
        raise TimeoutError(f"no reply to {line!r}")

    def on_reply(self, lines):
        """Hook: called with the reply lines of every successful command, before they are
        returned. A project overrides this to pick out its own lines (timings and so on)."""

    def close(self):
        self.s.close()
