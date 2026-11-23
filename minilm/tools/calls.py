"""
calls.py - run one command on the board, record it, and present the result.

The one-shot CLI (minilm_cli.py) and the interactive prompt (embed_repl.py) both go
through this file, so they always behave the same way, record the same history, and
draw the same charts.

    result = run_call(dev, "cli", "dist", ["the cat sat", "a kitten rested"])
    print(format_result(result))                # text for a human
    print(json.dumps(to_json(result)))          # or JSON for a program
"""
import charts
import history
from device import describe_similarity, phase_table, timing_summary


def run_call(dev, source, command, texts=(), record=True):
    """Run `command` on the board, append it to the history, and return a result dict.

    command is one of "dist", "embed", "tokens", "info". The result holds what the
    command produced plus its timing, token counts and (after recording) its timestamp.
    """
    texts = list(texts)
    res = {"command": command, "texts": texts, "source": source, "n_embeddings": 0}

    if command == "dist":
        res["similarity"], res["distance"] = dev.dist(*texts)
        res["meaning"] = describe_similarity(res["similarity"])
        res["n_embeddings"] = 2
    elif command == "embed":
        res["vector"], _ = dev.embed(texts[0])
        res["n_embeddings"] = 1
    elif command == "tokens":
        res["token_list"] = dev.tokens(texts[0])
    elif command == "info":
        res["info"] = dev.info()
    else:
        raise ValueError(f"unknown command {command!r}")

    # IMPORTANT: grab the timing NOW. Counting tokens below sends more commands to the
    # board, and each one overwrites the device's "last timing".
    res["timing"] = dev.snapshot_timing()

    # Token counts (cheap: the board only tokenises, no inference). They make the history
    # useful: time is plotted against length.
    if command in ("dist", "embed"):
        res["tokens"] = [len(dev.tokens(t)) for t in texts]
    elif command == "tokens":
        res["tokens"] = [len(res["token_list"])]
    else:
        res["tokens"] = None

    if record:
        extra = {k: res[k] for k in ("similarity", "distance") if k in res}
        rec = history.record(source, command, texts, res["tokens"], res["n_embeddings"],
                             res["timing"], **extra)
        res["ts"] = rec["ts"]
    return res


def format_result(res, full=False, chart=True):
    """Human-readable text for a result: the answer, then the timing, then the chart.

    full   (embed) show all 384 numbers instead of the first 8
    chart  draw the plotext chart of where the time went; if False, print a plain table
    """
    cmd, lines = res["command"], []

    if cmd == "dist":
        lines += [f"SIM {res['similarity']:.4f}  ({res['meaning']})",
                  f"DIST {res['distance']:.4f}",
                  f"tokens: {res['tokens'][0]} + {res['tokens'][1]}"]
    elif cmd == "embed":
        v = res["vector"]
        shown = v if full else v[:8]
        norm = sum(x * x for x in v) ** 0.5
        label = "all 384 numbers" if full else f"first 8 of {len(v)} numbers"
        lines += [f"{label}: " + " ".join(f"{x:+.4f}" for x in shown),
                  f"length of vector: {norm:.4f}  (should be 1.0000)",
                  f"tokens: {res['tokens'][0]}"]
    elif cmd == "tokens":
        lines.append(f"{len(res['token_list'])} tokens: " + " ".join(res["token_list"]))
    elif cmd == "info":
        lines += res["info"]

    t = res.get("timing")
    if t:
        lines.append("timing: " + timing_summary(t))
        lines.append(charts.call_chart(t) if chart else phase_table(t, bars=False))
    return "\n".join(lines)


def to_json(res, full=False):
    """The result as plain JSON-friendly data. Vectors are shortened unless full=True."""
    out = dict(res)
    if "vector" in out:
        out["dimensions"] = len(out["vector"])
        out["norm"] = sum(x * x for x in out["vector"]) ** 0.5
        out["complete"] = full
        if not full:
            out["vector"] = out["vector"][:8]
    return out
