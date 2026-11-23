# Build rules shared by every project. A project's Makefile starts with
#     include ../../common/mk/esp.mk
# and then adds its own targets. Needs ESP-IDF exported:  . ~/esp/esp-idf/export.sh
#
#   make build       compile the firmware
#   make flash       write the firmware to the board (never touches data partitions)
#   make monitor     serial monitor (exit with Ctrl-])
#   make menuconfig  change settings, for example the LED pin ("Board LED")
#   make clean       delete this project's build output
#   make chip-info   read the chip and flash details with esptool
PORT ?= /dev/ttyACM0

# The repository root, wherever make is run from, and the Python command that uses the
# repo's shared environment (esptool, pyserial, esp32tools ...).
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
PY   := uv run --project $(ROOT) python

build:
	idf.py set-target esp32s3
	idf.py build

flash:
	idf.py -p $(PORT) flash

monitor:
	idf.py -p $(PORT) monitor

menuconfig:
	idf.py menuconfig

clean:
	idf.py fullclean

chip-info:
	uv run --project $(ROOT) esptool --port $(PORT) chip-id
	uv run --project $(ROOT) esptool --port $(PORT) flash-id

.PHONY: build flash monitor menuconfig clean chip-info
