"""
charts.py - ASCII-art charts in the terminal, drawn with plotext.

Every function RETURNS A STRING (it does not print), so the CLI, the REPL and the
tests can all decide what to do with it. Colours are only included when writing to a
real terminal; piped or captured output is plain text.

A NOTE ABOUT PLOTEXT 6
----------------------
This project uses plotext 6, which replaced the classic `plt.bar(...)`, `plt.scatter(...)`
style with an explicit figure object. The pattern is:

    f = plt.figure                          # the one shared figure
    f.clear.all()                           # start from a blank page
    f.theme("simple"); f.plot_size(w, h)    # look and size
    f.title("..."); f.label("x name", "x")  # titles and axis names
    f.draw(f.bar(labels, values, orientation="h"))    # build a signal, then DRAW it
    text = f.build().string()               # render to a string (with colour codes)

Two details worth knowing: a bar/line/scatter is first BUILT (f.bar / f.signal) and only
then DRAWN (f.draw); and a line signal needs `.lines(True)` to join its points, while a
scatter keeps the default (points only). Everything plotext-specific lives in this file.
"""
import os
import shutil
import sys

import plotext as plt

import history
from device import PHASES

# Short names for chart labels (the long ones in device.PHASES suit a table, not a chart).
SHORT = {"dot": "arithmetic", "weight_fetch": "weight fetch", "attention": "attention",
         "layer_norm": "layer norm", "gelu": "GELU", "residual": "residual",
         "lookup": "lookup", "tokenize": "tokenizer", "pool": "pooling"}


# ---------------------------------------------------------------------------
# Plumbing shared by every chart
# ---------------------------------------------------------------------------
def _width():
    """Chart width: the terminal's width, but never wider than 100 or narrower than 50."""
    return max(50, min(shutil.get_terminal_size((100, 30)).columns, 100))


def _begin(title, height, xlabel=None, ylabel=None):
    """Start a fresh, empty chart and return the figure."""
    f = plt.figure
    f.clear.all()
    f.theme("simple")              # the 'simple' theme paints no background, so it looks
    f.plot_size(_width(), height)  # right on both dark and light terminals
    f.title(title)
    if xlabel:
        f.label(xlabel, "x")
    if ylabel:
        f.label(ylabel, "y")
    return f


def _finish(f):
    """Render the figure to a string, dropping the colour codes unless we are on a terminal."""
    text = f.build().string()
    if not (sys.stdout.isatty() and "NO_COLOR" not in os.environ):
        text = plt.uncolorize(text)
    return text


def _need(what, have, want):
    return f"(not enough data for this chart yet: {have} {what}, need {want}+. Make a few more calls first.)"


def _bars(rows, title):
    """Horizontal bar chart. rows = [(label, value), ...], biggest first, drawn top to bottom."""
    f = _begin(title, max(8, 2 * len(rows) + 4))
    rows = rows[::-1]              # plotext draws the first bar at the BOTTOM, so reverse
    f.draw(f.bar([r[0] for r in rows], [r[1] for r in rows], orientation="h", width=0.6))
    f.ruler("x").lim(0, max(v for _, v in rows) * 1.08)     # without this the scale clips the longest bar
    return _finish(f)


# ---------------------------------------------------------------------------
# 1. The metrics of ONE call
# ---------------------------------------------------------------------------
def call_chart(timing):
    """Where the inference time of a single call went, biggest phase first.

    Each label carries its own numbers ("weight fetch 1.484s 39%"), so no separate
    table is needed. Phases that took under 0.5% are folded into one 'other' bar."""
    ph = timing["phases_s"]
    total = ph["total"]
    rows = [(SHORT[k], ph.get(k, 0.0)) for k, _ in PHASES]
    rows.sort(key=lambda r: -r[1])
    big = [r for r in rows if r[1] / total >= 0.005]
    small = sum(v for _, v in rows) - sum(v for _, v in big)
    other = small + max(0.0, total - sum(v for _, v in rows))      # tiny phases + untimed overhead
    if other / total >= 0.005:
        big.append(("other", other))
    labelled = [(f"{name} {v:.3f}s {100 * v / total:.0f}%", v) for name, v in big]
    return _bars(labelled, f"where the {total:.2f} s went (seconds)")


# ---------------------------------------------------------------------------
# 2. Charts over the whole history
# ---------------------------------------------------------------------------
def phases_chart(records):
    """Average time per phase, per embedding, over every timed call in the history."""
    t = history.timed(records)
    if not t:
        return _need("timed calls", 0, 1)
    mean = history.stats(records)["mean_phase_s_per_embedding"]
    total = mean["total"]
    rows = sorted(((SHORT[k], mean.get(k, 0.0)) for k, _ in PHASES), key=lambda r: -r[1])
    rows = [(f"{n} {v:.3f}s {100 * v / total:.0f}%", v) for n, v in rows if v / total >= 0.005]
    return _bars(rows, f"average time per embedding over {len(t)} calls: {total:.2f} s")


def latency_chart(records):
    """Seconds per embedding against tokens per embedding, with the straight-line fit.

    The line's two numbers are the whole story: the fixed cost of streaming weights
    from flash, and the cost of each extra token."""
    t = history.timed(records)
    f_ = history.fit(records)
    if f_ is None:
        return _need("timed calls with different lengths", len(t), 3)
    pts = [history.per_embedding(r)[:2] for r in t]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    f = _begin("inference time vs text length", 18, "tokens per text", "seconds per embedding")
    x0, x1 = min(xs), max(xs)
    line = f.signal([x0, x1], [f_["fixed_s"] + f_["per_token_s"] * x0, f_["fixed_s"] + f_["per_token_s"] * x1])
    line.lines(True)
    line.label(f"fit: {f_['fixed_s']:.2f} s + {f_['per_token_s']:.3f} s/token  (R2 {f_['r2']:.3f})")
    sc = f.signal(xs, ys, marker="x")                 # a visible marker: the points often sit exactly ON the line
    sc.lines(False)                                   # points only
    sc.label(f"your calls ({len(pts)})")
    f.draw(line)                                      # line first, points on top of it
    f.draw(sc)
    f.legend(True, ha="right", va="bottom")           # the empty corner of an upward-sloping chart
    return _finish(f)


def bottleneck_chart(records):
    """Weight fetch versus arithmetic, by text length: the bottleneck shifts as text grows."""
    groups = {}
    for r in history.timed(records):
        x, _, ph = history.per_embedding(r)
        groups.setdefault(round(x), []).append(ph)
    if len(groups) < 2:
        return _need("different text lengths", len(groups), 2)
    xs = sorted(groups)
    mean = lambda key: [sum(p.get(key, 0.0) for p in groups[x]) / len(groups[x]) for x in xs]
    f = _begin("fixed cost vs growing cost", 18, "tokens per text", "seconds per embedding")
    a = f.signal(xs, mean("weight_fetch")); a.lines(True); a.label("weight fetch (fixed: flash)")
    b = f.signal(xs, mean("dot"));          b.lines(True); b.label("arithmetic (grows with text)")
    f.draw(a)
    f.draw(b)
    f.legend(True)
    return _finish(f)


def timeline_chart(records):
    """Seconds per embedding for each call, in the order the calls were made."""
    t = history.timed(records)
    if len(t) < 2:
        return _need("timed calls", len(t), 2)
    ys = [history.per_embedding(r)[1] for r in t]
    f = _begin(f"inference time per call ({len(t)} calls, oldest to newest)", 16,
               "call number", "seconds per embedding")
    s = f.signal(list(range(1, len(ys) + 1)), ys)
    s.lines(True)
    f.draw(s)
    return _finish(f)


def similarity_chart(records):
    """How the similarity scores of your `dist` calls are spread."""
    sims = [r["similarity"] for r in records if r.get("similarity") is not None]
    if len(sims) < 2:
        return _need("dist calls", len(sims), 2)
    f = _begin(f"similarity of {len(sims)} dist calls", 16, "similarity (1 = same meaning)", "calls")
    f.draw(f.hist(sims, bins=10))
    return _finish(f)


KINDS = {
    "phases": (phases_chart, "average time per phase over the whole history"),
    "latency": (latency_chart, "seconds vs text length, with the fitted line"),
    "bottleneck": (bottleneck_chart, "weight fetch vs arithmetic as text grows"),
    "timeline": (timeline_chart, "inference time of each call in order"),
    "similarity": (similarity_chart, "histogram of dist similarity scores"),
}


def render(kind, records):
    """Draw the named history chart (one of KINDS) as a string."""
    if kind not in KINDS:
        raise ValueError(f"unknown chart {kind!r}; choose from: {', '.join(KINDS)}")
    return KINDS[kind][0](records)
