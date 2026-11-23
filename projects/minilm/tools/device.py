"""
device.py - talk to the ESP32 running the minilm firmware over the serial port.

The generic part (opening the port, waiting for READY, sending a line and collecting the
reply) lives in the shared esp32tools.Board. This file adds what is specific to MiniLM:
the embed / dist / tokens / info commands and the timing the firmware reports.

The protocol is deliberately simple text:
    PC  -> board :  one command per line            e.g.  "embed hello world"
    board -> PC  :  some reply lines, then "OK"     (or one line starting "ERR")
"""
from esp32tools import Board, BoardError  # noqa: F401  (BoardError is re-exported for the other tools)


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


class Device(Board):
    def __init__(self, port="/dev/ttyACM0"):
        self.last_infer_s = None
        self.last_phases = {}
        super().__init__(port)

    def on_reply(self, lines):
        """Pick the MiniLM timing out of every reply.

        last_rtt_s    (set by Board) send -> final OK, wall clock on the PC
        last_infer_s  the board's own TIME_MS: time spent in bert_embed, or None if the
                      command has none. The difference between the two is serial transfer,
                      command parsing and PC overhead.
        last_phases   per-phase stopwatch readings: "TIMING <name> <microseconds>" lines."""
        ms = [float(l.split()[1]) for l in lines if l.startswith("TIME_MS")]
        self.last_infer_s = ms[0] / 1000 if ms else None
        self.last_phases = {l.split()[1]: int(l.split()[2]) / 1e6
                            for l in lines if l.startswith("TIMING ")}

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
