"""
firmware.py - what the PC knows about a project's firmware, and how to put it on a board.

A project describes what its host tools need as a Requirement. This file can then
  * read the build id out of a compiled firmware image (no ESP-IDF needed),
  * tell whether the project's build/ folder is older than its source,
  * build it (needs ESP-IDF exported), and
  * write it, or a data image such as the MiniLM model, to the board with esptool.

All progress output goes to stderr, so a tool that prints JSON on stdout stays parseable.
"""
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

APP_DESC_MAGIC = 0xABCD5432      # marks esp_app_desc_t inside a firmware image


class FlashError(RuntimeError):
    """Building or flashing failed, with a message that says what to do next."""


@dataclass
class DataImage:
    """A separate blob in flash that the firmware needs, for example the MiniLM model.

    ident_key / state_key are the key=value names the firmware reports in `ident`
    (model_id / model). `expected_id()` reads the id of the file on the PC (or None when the
    file is missing or has no id)."""
    name: str
    offset: int
    file: Path
    ident_key: str
    state_key: str
    expected_id: Callable[[], Optional[str]]
    how_to_make: str             # what to run when the file does not exist
    minutes: int = 3             # shown to the user: how long writing takes


@dataclass
class Requirement:
    """What a project's host tools need on the board."""
    project: str
    proto: int
    project_dir: Path
    data: Optional[DataImage] = None

    @property
    def build_dir(self):
        return self.project_dir / "build"

    @property
    def image(self):
        return self.build_dir / f"{self.project}.bin"


def app_build_id(image):
    """The build id (first 8 hex digits of the ELF SHA-256) stored inside a firmware .bin,
    or None if the file is missing or is not an ESP-IDF app image. It equals the `build=`
    value the running firmware reports in `ident`.

    Layout: 24-byte image header, 8-byte segment header, then esp_app_desc_t (magic 4,
    secure_version 4, reserved 8, version 32, project_name 32, time 16, date 16, idf_ver 32,
    app_elf_sha256 32 ...)."""
    try:
        with open(image, "rb") as f:
            f.seek(32)
            head = f.read(176 - 32 + 32)
    except OSError:
        return None
    if len(head) < 176 - 32 + 32 or struct.unpack_from("<I", head, 0)[0] != APP_DESC_MAGIC:
        return None
    return head[144:144 + 32].hex()[:8]


def _source_files(req):
    root = req.project_dir.parent.parent        # the repository root: projects/<name>/ -> ../..
    patterns = [(req.project_dir, "main/**/*"), (req.project_dir, "components/**/*"),
                (req.project_dir, "CMakeLists.txt"),
                (req.project_dir, "sdkconfig.defaults"), (req.project_dir, "partitions.csv"),
                (root / "common", "components/**/*"), (root / "common", "sdkconfig.*"),
                (root / "common", "partitions/*")]
    for base, pat in patterns:
        for p in base.glob(pat):
            if p.is_file() and "__pycache__" not in p.parts:
                yield p


def build_is_stale(req):
    """True if there is no build, or some source file is newer than the firmware image."""
    if not req.image.exists() or not (req.build_dir / "flash_args").exists():
        return True
    built = req.image.stat().st_mtime
    return any(p.stat().st_mtime > built for p in _source_files(req))


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def build(req):
    """Run `idf.py build` in the project folder. Needs ESP-IDF exported in the shell."""
    if not (os.environ.get("IDF_PATH") and shutil.which("idf.py")):
        raise FlashError(
            f"the {req.project} build is missing or older than its source, and ESP-IDF is not "
            f"available to rebuild it. Run `. ~/esp/esp-idf/export.sh` and try again, or build it "
            f"yourself: cd {req.project_dir} && make build")
    # `uv run` puts the repo's Python environment first on PATH, and idf.py would pick that
    # python (which lacks ESP-IDF's packages). Take it off the PATH for the build.
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    if venv:
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if not p.startswith(os.path.join(venv, "")))
    _say(f"building {req.project} ...")
    r = subprocess.run(["idf.py", "build"], cwd=req.project_dir, env=env, stdout=sys.stderr, stderr=sys.stderr)
    if r.returncode:
        raise FlashError(f"the build failed (see the output above); fix it with: cd {req.project_dir} && make build")


def _esptool(args, cwd=None):
    r = subprocess.run([sys.executable, "-m", "esptool", *args], cwd=cwd, stdout=sys.stderr, stderr=sys.stderr)
    if r.returncode:
        raise FlashError("esptool failed (see the output above). Check the cable and that no other "
                         "program (monitor, REPL) is using the serial port")


def flash_firmware(req, port):
    """Write bootloader, partition table and app from the build folder (a few seconds).
    Never touches the data in flash above 0x90000, so the MiniLM model survives."""
    _say(f"writing the {req.project} firmware to {port} ...")
    # ESP-IDF writes its options with underscores (--flash_mode); esptool 5 wants hyphens and
    # warns about the old spelling, so translate the option names (not the file paths).
    words = (req.build_dir / "flash_args").read_text().split()
    args = [w.replace("_", "-") if w.startswith("--") else w for w in words]
    _esptool(["--port", port, "--chip", "esp32s3", "write-flash", *args], cwd=req.build_dir)


def flash_data(data, port):
    """Write a data image, for example the MiniLM model (slow: minutes)."""
    _say(f"writing {data.file.name} to {port} at 0x{data.offset:x} (about {data.minutes} minutes) ...")
    _esptool(["--port", port, "-b", "921600", "write-flash", hex(data.offset), str(data.file)])
