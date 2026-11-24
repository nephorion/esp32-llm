# Setting up the board and toolchain

C firmware (ESP-IDF) for an **ESP32-S3**, flashed over USB serial. Python (pyserial) is used on the host to talk to the board over the serial terminal. No MicroPython.

## Board capabilities (read from the chip with esptool)

| Item        | Value                                                                    |
| ----------- | ------------------------------------------------------------------------ |
| Chip        | ESP32-S3 (QFN56), revision v0.2, chip ID 9                               |
| CPU         | Dual-core Xtensa LX7 @ up to 240 MHz, plus a low-power (LP) core        |
| Radio       | Wi-Fi (2.4 GHz, 802.11 b/g/n), Bluetooth 5 (LE)                          |
| Flash       | 16 MB, quad (4 data lines), 3.3 V                                        |
| PSRAM       | 8 MB embedded (vendor AP Memory, 3.3 V flag in eFuse)                    |
| Crystal     | 40 MHz                                                                   |
| SRAM        | 512 KB internal (from the S3 datasheet)                                  |
| Vector unit | S3 CPU has SIMD/vector instructions for NN and DSP workloads             |
| Security    | Secure Boot: disabled. Flash Encryption: disabled. eFuse key blocks empty. |
| Debug       | JTAG not disabled in eFuse; USB serial/JTAG available on the native USB port |

Notes:
- Fuse-derived values come from `espefuse summary` and `esptool flash-id`. SRAM size and SIMD come from the ESP32-S3 datasheet, not from the chip query.
- The PSRAM is **octal** (AP Memory, 64 Mbit, 40 MHz), confirmed by booting with `CONFIG_SPIRAM_MODE_OCT=y` in `projects/bench/sdkconfig.defaults`. The eFuses alone don't say.
- Boot reports about 386 KB of internal heap free and 8 MB of PSRAM free.
- The board enumerates as a QinHeng `1a86:55d3` USB serial device on `/dev/ttyACM0`.
- Re-run the checks any time with `make chip-info`.

## Prerequisites

- Linux with a USB port and a data-capable USB cable
- Python 3.11+, [`uv`](https://docs.astral.sh/uv/), `make`, `git`
- ESP-IDF build dependencies (Ubuntu/Debian):

  ```bash
  sudo apt install git wget flex bison gperf python3 python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0
  ```

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Make sure `~/.local/bin` is on your `PATH`.

### 2. Serial port access (groups)

The board is owned by `root:dialout` with mode `660`. Add your user to `dialout`:

```bash
sudo usermod -aG dialout $USER
```

**Log out and back in** (or reboot). A running shell does not pick up new groups. `newgrp dialout` is a stopgap.

Verify:

```bash
id | grep dialout
test -r /dev/ttyACM0 && test -w /dev/ttyACM0 && echo OK
```

If the port appears and then vanishes, `brltty` may be grabbing it: `sudo apt remove brltty`.

### 3. Python tools (esptool, pyserial)

```bash
uv sync
uv run esptool version
```

Optionally install esptool globally: `uv tool install esptool`.

### 4. Install ESP-IDF (C toolchain)

```bash
mkdir -p ~/esp && cd ~/esp
git clone -b v5.4 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf && ./install.sh esp32s3
```

Then, in every new shell before building:

```bash
. ~/esp/esp-idf/export.sh
```

(`idf.py` is not on the `PATH` until you do.)

### 5. Check the board

```bash
make chip-info
```

Expected: `Connected to ESP32-S3`. If it fails, hold **BOOT**, tap **RESET**, release **BOOT**, then retry.

## Usage

Every project under `projects/` has its own Makefile. Run `make` inside the project folder, or
from the repository root with `p=<project>`:

| Command | What it does |
| ------- | ------------ |
| `make p=minilm build` | Sets the target to esp32s3 and builds (needs ESP-IDF exported) |
| `make p=minilm flash` | Flashes the firmware to the board |
| `make p=minilm monitor` | Serial monitor (exit with `Ctrl-]`) |
| `make p=minilm menuconfig` | Change settings, for example the LED pin |
| `make chip-info` | Reads chip ID and flash info with esptool |
| `make list` | Lists the projects |
| `make new NAME=foo` | Creates `projects/foo` from `projects/_template` |

Override the port with `make flash PORT=/dev/ttyUSB0`.

### Talking to the board from Python

```python
from esp32tools import Board          # shared helper, see common/python/
b = Board("/dev/ttyACM0")             # opens the port and waits for the board's READY
print(b.command("ping"))              # sends a line, returns the reply lines before OK
b.close()
```

Only one program can hold the port at a time, so close `make monitor` first.

## Troubleshooting

| Symptom                           | Fix                                                                 |
| --------------------------------- | ------------------------------------------------------------------- |
| `Permission denied: /dev/ttyACM0` | Not in `dialout`, or you haven't logged out and in since adding it. |
| No `/dev/tty*` device appears     | Charge-only cable, or `brltty` is installed.                        |
| `Failed to connect` / timeout     | Enter the bootloader manually (BOOT + RESET).                       |
| `Device busy`                     | Another program has the port open.                                  |
| `idf.py: command not found`       | Run `. ~/esp/esp-idf/export.sh` in this shell.                      |

## Repository layout

```
common/                 shared by every project
  components/board_led/   on-board LED + the "busy" blinker (ESP-IDF component)
  mk/esp.mk               shared build / flash / monitor rules
  partitions/             standard flash layout (512 KB app, nothing above 0x90000)
  sdkconfig.common        settings every project shares
  python/esp32tools/      generic serial helper (Board)
projects/               one self-contained ESP-IDF project per folder
  minilm/                 MiniLM-L6 on the chip (start with its README)
  bench/                  memory and GPIO speed benchmark
  led/                    LED on/off over serial
  _template/              what `make new NAME=foo` copies
```

A project pulls in shared code by adding `set(EXTRA_COMPONENT_DIRS ../../common/components)` to
its `CMakeLists.txt` and listing the component in `REQUIRES`. See `projects/_template` for a
complete minimal example.

**Checking the board from the host tools.** Every project answers a serial `ident` command with
its project name, a protocol version and a build id (`common/components/board_ident`). The
generic `esp32tools.connect()` asks it before a tool does anything, compares the answer with the
project's `Requirement`, and offers to build and flash the right firmware if it is wrong. The
LED REPL (`projects/led/tools/led_repl.py`) and the MiniLM tools use it, and it is the model
for a new project's tools. Compatible means the same project and the same protocol version, so a
rebuild that keeps the protocol is accepted. Without a terminal, or with `--no-flash`, the
tools never write to the board: they exit with code 4 and say how to fix it. `projects/bench/`
only prints at boot and takes no commands, so it is not covered.

**Flash layout.** `led`, `bench` and new projects use the standard table in
`common/partitions/`, whose app ends at `0x90000`, so flashing them never overwrites the
MiniLM model. They do replace the partition table, though. After flashing another project, run
`make flash` in `projects/minilm/` to get MiniLM's table back. The model does not need to be
written again.
