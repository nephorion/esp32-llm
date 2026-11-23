# Memory and GPIO speed benchmark. Run from projects/bench/ (or: make p=bench <target> from the root).
#   make build flash   then   make run   (prints a table, saves a CSV in results/)
include ../../common/mk/esp.mk

run:
	$(PY) tools/run_bench.py $(PORT)

.PHONY: run
