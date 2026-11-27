# Matrix kernel benchmark (C against assembly). Run from projects/matbench/ (or: make p=matbench <target>).
#   make run           checks the board runs this firmware (offers to write it), runs, saves a CSV
#   make build flash   rebuild and write it yourself (needed after changing the code)
include ../../common/mk/esp.mk

run:
	$(PY) tools/run_matbench.py $(PORT)

.PHONY: run
