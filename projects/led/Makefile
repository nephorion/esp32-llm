# LED on/off over serial. Run from projects/led/ (or: make p=led <target> from the root).
#   make build flash   then   make repl   and type: on, off, exit
include ../../common/mk/esp.mk

repl:
	$(PY) tools/led_repl.py $(PORT)

.PHONY: repl
