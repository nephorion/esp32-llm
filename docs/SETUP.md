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
- The PSRAM is **octal** (AP Memory, 64 Mbit, 40 MHz), confirmed by booting with `CONFIG_SPIRAM_MODE_OCT=y` in `sdkconfig.defaults`. The eFuses alone don't say.
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

| Command        | What it does                                              |
| -------------- | --------------------------------------------------------- |
| `make build`   | Sets target to esp32s3 and builds (needs ESP-IDF exported) |
| `make flash`   | Flashes the build to the board                            |
| `make monitor` | Serial monitor (exit with `Ctrl-]`)                       |
| `make chip-info` | Reads chip ID and flash info with esptool               |

Override the port with `make flash PORT=/dev/ttyUSB0`.

### Talking to the board from Python

```python
import serial
with serial.Serial("/dev/ttyACM0", 115200, timeout=1) as s:
    s.write(b"hello\n")
    print(s.readline())
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

## Projects in this repo (details)

| Folder | What it is |
|---|---|
| `main/` | LED on/off over serial (`tools/led_repl.py`) |
| `bench/` | Memory and GPIO speed benchmark (`tools/run_bench.py`, results in `results/`) |
| `minilm/` | **MiniLM-L6 sentence embeddings on the chip. Start with [`minilm/README.md`](minilm/README.md)** |

Each folder is its own ESP-IDF project, so build and flash from inside it.

**Warning:** `main/` and `bench/` use the default partition table, which puts a 1 MB app at
`0x10000`. That overlaps the start of the `minilm` model (`0x90000`). If you flash `main/` or
`bench/`, run `make flash-model` in `minilm/` again afterwards. Flashing `minilm` itself only
replaces its small firmware and leaves the model alone.
