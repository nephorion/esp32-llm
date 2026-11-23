"""
device.py - talk to the ESP32 running the minilm firmware over the serial port.

The protocol is deliberately simple text:
    PC  -> board :  one command per line            e.g.  "embed hello world"
    board -> PC  :  some reply lines, then "OK"     (or one line starting "ERR")
"""
import time

import serial


class BoardError(RuntimeError):
    """The board understood the connection but replied with ERR/FATAL (e.g. a bad command)."""


def describe_similarity(sim):
    """Rule-of-thumb meaning of a similarity score for this kind of model.
    These are guidelines, not hard limits: scores depend on the text."""
    if sim >= 0.80: return "almost the same meaning"
    if sim >= 0.60: return "closely related"
    if sim >= 0.40: return "somewhat related"
    if sim >= 0.20: return "weakly related"
    return "unrelated"


# How to label each phase the firmware measures, in the order we explain them.
# (key in the firmware's TIMING lines, human-readable label)
PHASES = [
    ("dot",          "arithmetic: multiply-adds"),
    ("weight_fetch", "weight fetch: flash -> float"),
    ("attention",    "attention (tokens mixing)"),
    ("layer_norm",   "layer norm"),
    ("gelu",         "GELU activation"),
    ("residual",     "residual additions"),
    ("lookup",       "embedding lookup"),
    ("tokenize",     "tokenizer"),
    ("pool",         "pooling + normalise"),
]


def timing_summary(timing):
    """One line: round trip = inference + serial/other. `timing` comes from snapshot_timing()."""
    rtt, inf = timing["round_trip_s"], timing["inference_s"]
    return (f"round trip {rtt:.2f} s = inference {inf:.2f} s ({100 * inf / rtt:.0f}%)"
            f" + serial/other {rtt - inf:.2f} s ({100 * (rtt - inf) / rtt:.0f}%)")


def phase_table(timing, width=28, bars=True):
    """Text table of where the inference time went.

    Percentages are of the board's own total for the command. The last row,
    'untimed', is whatever no stopwatch covered: loop overhead plus the cost of
    the stopwatch calls themselves."""
    ph = timing.get("phases_s") or {}
    if not ph.get("total"):
        return ""
    total = ph["total"]
    rows = [(label, ph.get(key, 0.0)) for key, label in PHASES]
    rows.append(("untimed (overhead)", max(0.0, total - sum(v for _, v in rows))))
    out = [f"where the {total:.2f} s went:"]
    for label, secs in rows:
        share = secs / total
        bar = f"  {'█' * round(share * width)}" if bars else ""
        out.append(f"  {label:<30}{secs:6.3f} s {share * 100:5.1f}%{bar}")
    return "\n".join(out)


class Device:
    def __init__(self, port="/dev/ttyACM0"):
        self.last_rtt_s = self.last_infer_s = None
        self.last_phases = {}
        self.s = serial.Serial()
        self.s.port, self.s.baudrate, self.s.timeout = port, 115200, 1
        self.s.dtr = False
        self.s.rts = False
        # Take an exclusive lock so two of OUR tools (REPL, CLI, validate) can never
        # use the port at once: the second one fails immediately with a clear message
        # instead of getting garbled or empty replies.
        self.s.exclusive = True
        self.s.open()
        # Opening the port resets the board, so wait for its "READY" line.
        deadline = time.time() + 15
        while time.time() < deadline:
            if self.s.readline().strip() == b"READY":
                return
        raise ConnectionError("board did not say READY - port busy (another program using it?), "
                              "model not flashed (make flash-model), or board not connected")

    def command(self, line, timeout=60):
        """Send one command, return the list of reply lines (without the final OK).

        Also records timing in  self.last_rtt_s    (send -> final OK, wall clock on the PC)
                                self.last_infer_s  (the board's own TIME_MS: time spent in
                                                    bert_embed, or None if the command has none)
        The difference is serial transfer, command parsing and PC overhead."""
        self.s.reset_input_buffer()
        t0 = time.perf_counter()
        self.s.write((line + "\n").encode("ascii", "ignore"))
        lines, deadline = [], time.time() + timeout
        while time.time() < deadline:
            raw = self.s.readline().decode(errors="replace").strip()
            if raw == "OK":
                self.last_rtt_s = time.perf_counter() - t0
                ms = [float(l.split()[1]) for l in lines if l.startswith("TIME_MS")]
                self.last_infer_s = ms[0] / 1000 if ms else None
                # Per-phase stopwatch readings from the firmware: "TIMING <name> <microseconds>"
                self.last_phases = {l.split()[1]: int(l.split()[2]) / 1e6
                                    for l in lines if l.startswith("TIMING ")}
                return lines
            if raw.startswith("ERR") or raw.startswith("FATAL"):
                raise BoardError(raw)
            if raw:
                lines.append(raw)
        raise TimeoutError(f"no reply to {line!r}")

    def embed(self, text):
        """Return (384-float vector as a list, milliseconds taken)."""
        reply = self.command("embed " + text)
        vec = next(l for l in reply if l.startswith("VEC")).split()[1:]
        ms = float(next(l for l in reply if l.startswith("TIME_MS")).split()[1])
        return [float(v) for v in vec], ms

    def snapshot_timing(self):
        """Timing of the LAST command as a plain dict, or None if it had no inference.

        Take this straight after the command you care about: any further command
        (even a quick `tokens`) overwrites the "last" values."""
        if self.last_infer_s is None:
            return None
        return {
            "round_trip_s": round(self.last_rtt_s, 4),
            "inference_s": round(self.last_infer_s, 4),
            "phases_s": {k: round(v, 5) for k, v in self.last_phases.items()},
        }

    def tokens(self, text):
        """How the text is split into word pieces, e.g. ['[CLS]', 'cats', '!', '[SEP]']."""
        reply = next(l for l in self.command("tokens " + text) if l.startswith("TOKENS"))
        return reply.split(":", 1)[1].split()

    def dist(self, a, b):
        """Compare two texts. Returns (similarity, distance). Texts must not contain '|'."""
        reply = self.command(f"dist {a.replace('|', ' ')} | {b.replace('|', ' ')}")
        sim = float(next(l for l in reply if l.startswith("SIM ")).split()[1])
        return sim, 1.0 - sim

    def info(self):
        """Model and memory facts: a list of text lines."""
        return self.command("info")

    def close(self):
        self.s.close()
