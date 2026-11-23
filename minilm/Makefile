# MiniLM-L6 on the ESP32-S3. Run from the minilm/ directory.
#
#   make model        download + shrink the model       (PC only, ~1 min)
#   make build        compile the firmware              (needs: . ~/esp/esp-idf/export.sh)
#   make flash        write the firmware (small, quick)
#   make flash-model  write the 16 MB model (slow, ~3 min, only needed once)
#   make repl         interactive prompt: embed / dist
#   make validate     compare the board against the numpy reference
#   make stats | history | plot KIND=latency    look at the recorded call history
#
PORT ?= /dev/ttyACM0
PY   := uv run --project .. python

download:
	$(PY) tools/download_model.py

model: download
	$(PY) tools/export_model.py

build:
	idf.py set-target esp32s3
	idf.py build

flash:
	idf.py -p $(PORT) flash

# 0x90000 must match the "model" row in partitions.csv
flash-model:
	uv run --project .. esptool --port $(PORT) -b 921600 write-flash 0x90000 model_build/model.bin

repl:
	$(PY) tools/embed_repl.py $(PORT)

validate:
	$(PY) tools/validate_device.py $(PORT)

# Read the call history (no board needed)
history:
	$(PY) tools/minilm_cli.py history

stats:
	$(PY) tools/minilm_cli.py stats

# usage: make plot KIND=latency   (phases | latency | bottleneck | timeline | similarity | all)
KIND ?= all
plot:
	$(PY) tools/minilm_cli.py plot $(KIND)

reference:
	@echo 'usage: uv run --project .. python tools/minilm_ref.py dist "text a" "text b"'

.PHONY: download model build flash flash-model repl validate history stats plot reference
