"""
ensure.py - connect to a board and make sure it runs the firmware (and data) the tools need.

Host tools such as the MiniLM CLI only work against one specific firmware. Instead of
failing with a confusing timeout when the board has something else on it, they call

    dev = connect(Device, "/dev/ttyACM0", REQUIREMENT, mode="ask")

which does this:

    open the port  ->  ask `ident`  ->  compare with the Requirement
         |                                   |
         |                          matches: return the connection
         |                          differs: explain, ask, then
         |                             1. write the firmware (build first if build/ is stale)
         |                             2. write the data image (e.g. the MiniLM model), if wrong
         +-------------------------------- and check again

What counts as "compatible" is deliberately loose: the same project name and the same protocol
version. A rebuilt firmware with an unchanged protocol is accepted as it is. The data image
(the model) is compared by its content id, because a different file would give different answers.

`mode` says what to do on a mismatch:
    "ask"    ask y/N on the terminal; without a terminal behave like "never"
    "yes"    flash without asking (the user passed --yes)
    "never"  never flash: raise Mismatch with a message that says how to fix it
"""
import sys
from dataclasses import dataclass
from typing import Optional

from . import firmware as fw
from .firmware import DataImage, FlashError, Requirement  # noqa: F401  (re-exported)
from .ident import identify


class Mismatch(RuntimeError):
    """The board does not run what the tool needs, and it was not (or could not be) fixed.
    str(e) is a complete message for the user."""


@dataclass
class Status:
    firmware: Optional[str] = None    # why the firmware is unsuitable, or None if it is fine
    data: Optional[str] = None        # why the data image is unsuitable, or None
    data_fixable: bool = True         # False when the file to write is not on this PC

    @property
    def ok(self):
        return not self.firmware and not self.data


def check(ident, why, req):
    """Compare what the board reported with the requirement. Returns a Status."""
    if ident is None:
        return Status(firmware=f"the board is not running {req.project}: {why}")
    if ident.project != req.project:
        return Status(firmware=f"the board is running '{ident.project}', not '{req.project}'")
    if ident.proto != req.proto:
        return Status(firmware=f"the board runs {req.project} protocol {ident.proto}, "
                               f"but these tools need protocol {req.proto}")
    d = req.data
    if d is None:
        return Status()
    state, got, want = ident.fields.get(d.state_key), ident.fields.get(d.ident_key), d.expected_id()
    if state != "ok":
        if want is None:
            return Status(data=f"the {d.name} on the board is {state or 'not reported'}, and {d.file} "
                               f"is not on this PC with an id (make it with: {d.how_to_make})",
                          data_fixable=False)
        return Status(data=f"the {d.name} on the board is {state or 'not reported'}")
    if want is not None and got != want:
        return Status(data=f"the {d.name} on the board (id {got}) is not the one in {d.file.name} (id {want})")
    return Status()          # (if the file is absent or has no id there is nothing to compare: accept)


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def _interactive(mode):
    """Will we explain and (maybe) write? Not for "never", and not for "ask" without a terminal."""
    return mode == "yes" or (mode == "ask" and sys.stdin.isatty())


def _approved(mode, question):
    """True if we may go ahead. 'ask' prompts on stderr/stdin so stdout stays clean for JSON."""
    if mode == "yes":
        return True
    if mode == "ask" and sys.stdin.isatty():
        sys.stderr.write(f"{question} [y/N] ")
        sys.stderr.flush()
        return sys.stdin.readline().strip().lower() in ("y", "yes")
    return False


def _declined(problem, fix):
    return Mismatch(f"{problem}. Nothing was changed. To fix it: {fix}")


def connect(device_cls, port, req, mode="ask", attempts=3):
    """Return an open device_cls connected to a board that runs `req`, flashing if needed."""
    flashed_fw = flashed_data = False
    last = None
    for _ in range(attempts):
        dev = device_cls(port, ready_timeout=6, require_ready=False)
        ident, why = identify(dev)
        st = check(ident, why, req)
        if flashed_fw and ident is not None and not st.firmware:
            built = fw.app_build_id(req.image)
            if built and ident.build != built:
                st.firmware = (f"the board reports build {ident.build} but build {built} was just written: "
                               f"is PROTO_VERSION the same in the firmware and the host tools?")
        if st.ok:
            return dev
        dev.close()
        last = st

        if st.firmware:
            if flashed_fw:
                raise Mismatch(f"{st.firmware}, even after writing the {req.project} firmware")
            stale = fw.build_is_stale(req)
            fix = f"cd {req.project_dir} && make build flash   (or re-run with --yes)"
            if _interactive(mode):
                _say(f"\n{st.firmware}.")
            if not _approved(mode, f"Write the {req.project} firmware now"
                                   f"{' (rebuilding it first)' if stale else ''}? This replaces what is on the "
                                   f"board and takes under a minute."):
                raise _declined(st.firmware, fix)
            if stale:
                fw.build(req)
            fw.flash_firmware(req, port)
            flashed_fw = True
            continue

        d = req.data
        if not st.data_fixable:
            raise Mismatch(st.data)
        if flashed_data:
            raise Mismatch(f"{st.data}, even after writing it")
        if _interactive(mode):
            _say(f"\n{st.data}.")
        if not _approved(mode, f"Write {d.file.name} now? It takes about {d.minutes} minutes."):
            raise _declined(st.data, f"cd {req.project_dir} && make flash-model   (or re-run with --yes)")
        fw.flash_data(d, port)
        flashed_data = True
    raise Mismatch(f"could not get the board into the required state ({last.firmware or last.data})")
