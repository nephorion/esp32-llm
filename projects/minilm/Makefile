# MiniLM-L6 on the ESP32-S3. Run from the projects/minilm/ directory (or: make p=minilm <target> from the root).
#
#   make model        download + shrink the model       (PC only, ~1 min)
#   make build        compile the firmware              (needs: . ~/esp/esp-idf/export.sh)
#   make flash        write the firmware (small, quick)
#   make flash-model  write the 16 MB model (slow, ~3 min, only needed once)
#   make repl         interactive prompt: embed / dist
#   make validate     compare the board against the numpy reference
#   make stats | history | plot KIND=latency    look at the recorded call history
#
include ../../common/mk/esp.mk      # build, flash, monitor, menuconfig, clean, chip-info

download:
	$(PY) tools/download_model.py

model: download
	$(PY) tools/export_model.py

# 0x90000 must match the "model" row in partitions.csv
flash-model:
	uv run --project $(ROOT) esptool --port $(PORT) -b 921600 write-flash 0x90000 model_build/model.bin

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
	@echo 'usage: uv run python projects/minilm/tools/minilm_ref.py dist "text a" "text b"'

.PHONY: download model flash-model repl validate history stats plot reference
