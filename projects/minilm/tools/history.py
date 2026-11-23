"""
history.py - remember every call made to the board, and learn from the pile.

WHAT IS STORED
--------------
Every call made through the CLI or the REPL appends ONE line of JSON ("JSON Lines")
to a history file. Plain text, one record per line, so it is easy to read, grep,
back up, or load into pandas:

    {"ts": "2026-11-23T19:02:11+02:00", "source": "cli", "command": "dist",
     "texts": ["the cat sat", "a kitten rested"], "tokens": [4, 6], "n_embeddings": 2,
     "similarity": 0.3062, "distance": 0.6938,
     "timing": {"round_trip_s": 3.5, "inference_s": 3.47, "phases_s": {...}}}

Where it lives: projects/minilm/history/calls.jsonl  (override with the MINILM_HISTORY
environment variable). The folder is git-ignored because it contains your own text.

WHY "PER EMBEDDING"?
--------------------
`embed` runs the network once; `dist` runs it twice (once per text). To compare
calls fairly, the statistics divide by `n_embeddings`:

    seconds per embedding  =  inference_s / n_embeddings
    tokens  per embedding  =  sum(tokens) / n_embeddings

THE FIT
-------
Plotting seconds per embedding against tokens per embedding gives a straight line:

    seconds = fixed + per_token * tokens

`fixed` is the cost of streaming the weights from flash (the same whatever the text
length) and `per_token` is the arithmetic that grows with the text. fit() finds both
numbers from YOUR history with a least-squares line, which is a nice way to confirm
what the firmware's own stopwatches say.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PATH = Path(os.environ.get("MINILM_HISTORY", HERE.parent / "history" / "calls.jsonl"))


# ---------------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------------
def record(source, command, texts=(), tokens=None, n_embeddings=0, timing=None, **extra):
    """Append one call to the history file and return the record.

    source        "cli" or "repl"
    command       "dist", "embed", "tokens" or "info"
    texts         the input text(s)
    tokens        token count of each text (None if unknown)
    n_embeddings  how many times the network ran (embed 1, dist 2, others 0)
    timing        the dict from Device.snapshot_timing(), or None
    extra         command-specific results, e.g. similarity=..., distance=...
    """
    rec = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "command": command,
        "texts": list(texts),
        "tokens": tokens,
        "n_embeddings": n_embeddings,
        **extra,
        "timing": timing,
    }
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load():
    """Read every record. A damaged line is skipped rather than ending the whole read."""
    if not PATH.exists():
        return []
    out = []
    with open(PATH, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Turning records into numbers
# ---------------------------------------------------------------------------
def timed(records):
    """Only the calls that ran the network and have usable timing and token counts."""
    return [r for r in records
            if r.get("timing") and r.get("n_embeddings") and r.get("tokens")
            and r["timing"].get("inference_s")]


def per_embedding(r):
    """(tokens per embedding, seconds per embedding, {phase: seconds per embedding})."""
    n = r["n_embeddings"]
    phases = {k: v / n for k, v in (r["timing"].get("phases_s") or {}).items()}
    return sum(r["tokens"]) / n, r["timing"]["inference_s"] / n, phases


def fit(records):
    """Least-squares line  seconds = fixed + per_token * tokens  (both per embedding).

    Returns {"fixed_s", "per_token_s", "r2", "points"} or None if there is not yet
    enough variety (needs at least 3 timed calls with 2+ different lengths)."""
    pts = [per_embedding(r)[:2] for r in timed(records)]
    if len(pts) < 3 or len({round(x, 3) for x, _ in pts}) < 2:
        return None
    x, y = np.array(pts).T
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res, ss_tot = float(np.sum((y - pred) ** 2)), float(np.sum((y - y.mean()) ** 2))
    return {"fixed_s": float(intercept), "per_token_s": float(slope),
            "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0, "points": len(pts)}


def stats(records):
    """A summary of the whole history, as a plain dict (good for --json)."""
    out = {"calls": len(records), "file": str(PATH)}
    if not records:
        return out
    out["first"], out["last"] = records[0]["ts"], records[-1]["ts"]
    for key in ("command", "source"):
        counts = {}
        for r in records:
            counts[r[key]] = counts.get(r[key], 0) + 1
        out["by_" + key] = counts

    t = timed(records)
    out["timed_calls"] = len(t)
    if t:
        per = [per_embedding(r) for r in t]
        secs = [s for _, s, _ in per]
        out["seconds_per_embedding"] = {"mean": float(np.mean(secs)), "min": min(secs), "max": max(secs)}
        out["tokens_per_embedding"] = {"mean": float(np.mean([x for x, _, _ in per])),
                                       "min": min(x for x, _, _ in per), "max": max(x for x, _, _ in per)}
        keys = {k for _, _, ph in per for k in ph}
        out["mean_phase_s_per_embedding"] = {k: float(np.mean([ph.get(k, 0.0) for _, _, ph in per]))
                                             for k in sorted(keys)}
        out["round_trip_vs_inference"] = {
            "mean_round_trip_s": float(np.mean([r["timing"]["round_trip_s"] for r in t])),
            "mean_inference_s": float(np.mean([r["timing"]["inference_s"] for r in t])),
        }
    sims = [r["similarity"] for r in records if r.get("similarity") is not None]
    if sims:
        out["similarity"] = {"calls": len(sims), "mean": float(np.mean(sims)),
                             "min": min(sims), "max": max(sims)}
    f = fit(records)
    if f:
        out["fit"] = f
    return out


def format_stats(s):
    """A human-readable version of stats()."""
    if s["calls"] == 0:
        return f"No calls recorded yet ({s['file']})."
    lines = [f"{s['calls']} calls recorded  ({s['first']}  ->  {s['last']})",
             "by command: " + ", ".join(f"{k} {v}" for k, v in s["by_command"].items()),
             "by source:  " + ", ".join(f"{k} {v}" for k, v in s["by_source"].items())]
    if "seconds_per_embedding" in s:
        e, tk = s["seconds_per_embedding"], s["tokens_per_embedding"]
        lines += [f"\nper embedding ({s['timed_calls']} timed calls):",
                  f"  inference time  mean {e['mean']:.2f} s   min {e['min']:.2f} s   max {e['max']:.2f} s",
                  f"  tokens          mean {tk['mean']:.1f}     min {tk['min']:.0f}     max {tk['max']:.0f}"]
        rt = s["round_trip_vs_inference"]
        lines.append(f"  round trip {rt['mean_round_trip_s']:.2f} s vs inference {rt['mean_inference_s']:.2f} s (mean per call)")
    if "similarity" in s:
        m = s["similarity"]
        lines.append(f"\nsimilarity over {m['calls']} dist calls:  mean {m['mean']:.3f}   "
                     f"min {m['min']:.3f}   max {m['max']:.3f}")
    if "fit" in s:
        f = s["fit"]
        lines.append(f"\nstraight-line fit of your history ({f['points']} points, R^2 = {f['r2']:.3f}):\n"
                     f"  seconds per embedding = {f['fixed_s']:.3f} s fixed  +  {f['per_token_s']:.3f} s x tokens")
    else:
        lines.append("\n(not enough variety in text length yet for a straight-line fit: "
                     "need 3+ timed calls with different token counts)")
    return "\n".join(lines)


def format_list(records, n=15):
    """The last n calls, one per line."""
    if not records:
        return f"No calls recorded yet ({PATH})."
    rows = []
    for r in records[-n:]:
        text = " | ".join(r["texts"]) if r["texts"] else ""
        text = (text[:46] + "...") if len(text) > 48 else text
        t = r.get("timing")
        secs = f"{t['inference_s']:5.2f}s" if t else "   -  "
        sim = f"{r['similarity']:+.3f}" if r.get("similarity") is not None else "  -   "
        rows.append(f"{r['ts'][5:19].replace('T', ' ')}  {r['source']:<4} {r['command']:<6} {secs}  {sim}  {text}")
    return "\n".join(["time (MM-DD HH:MM:SS)  src  cmd    infer   sim     text"] + rows)
