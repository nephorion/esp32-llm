"""
ident.py - ask a board which firmware it is running.

Every project answers the serial command `ident` with a line like

    IDENT project=minilm proto=1 build=3f2a91bc model=ok model_id=9c1d0e77

(see common/components/board_ident). parse_ident() turns that line into an Ident, and
identify() sends the command and copes with boards that cannot answer it.
"""
from dataclasses import dataclass, field

from .board import BoardError


@dataclass
class Ident:
    project: str
    proto: int
    build: str                                   # first 8 hex digits of the build's SHA-256
    fields: dict = field(default_factory=dict)   # any extra key=value pairs the project reports

    def describe(self):
        return f"{self.project} (protocol {self.proto}, build {self.build})"


def parse_ident(line):
    """Parse 'IDENT key=value key=value ...'. Raises ValueError if it is not one."""
    parts = line.split()
    if not parts or parts[0] != "IDENT":
        raise ValueError(f"not an IDENT line: {line!r}")
    kv = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
    try:
        return Ident(kv.pop("project"), int(kv.pop("proto")), kv.pop("build"), kv)
    except (KeyError, ValueError) as e:
        raise ValueError(f"malformed IDENT line: {line!r}") from e


def identify(board, timeout=3):
    """Send `ident`. Returns (Ident, None) on success, or (None, why) when the board cannot say.

    A board that cannot say is running a firmware without the command (an old build, or
    another project), or is not answering at all."""
    try:
        lines = board.command("ident", timeout=timeout)
    except BoardError as e:
        return None, f"its firmware has no `ident` command (it replied: {e})"
    except TimeoutError:
        return None, "it did not answer `ident` (unknown firmware, or the board is stuck)"
    for line in lines:
        if line.startswith("IDENT "):
            try:
                return parse_ident(line), None
            except ValueError as e:
                return None, str(e)
    return None, "its reply to `ident` had no IDENT line"
