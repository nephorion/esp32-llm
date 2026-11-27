# MiniLM with layer weights in PSRAM. Run from projects/minilm-psram/ (or: make p=minilm-psram <target>).
#   make run           checks the board runs this firmware (offers to write it), measures, saves a CSV
#   make build flash   rebuild and write it yourself (needed after changing the code)
# The model is the one projects/minilm puts in flash (make model flash-model there).
include ../../common/mk/esp.mk

run:
	$(PY) tools/compare.py $(PORT)

.PHONY: run
