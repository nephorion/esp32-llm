# ESP-IDF (C) workflow. Requires ESP-IDF exported in the shell: . ~/esp/esp-idf/export.sh
# Usage: make build | make flash | make monitor | make chip-info
PORT ?= /dev/ttyACM0

build:
	idf.py set-target esp32s3
	idf.py build

flash:
	idf.py -p $(PORT) flash

monitor:
	idf.py -p $(PORT) monitor

chip-info:
	uv run esptool --port $(PORT) chip-id
	uv run esptool --port $(PORT) flash-id
