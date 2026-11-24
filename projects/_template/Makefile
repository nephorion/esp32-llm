# __NAME__. Run from projects/__NAME__/ (or: make p=__NAME__ <target> from the root).
include ../../common/mk/esp.mk

# Talk to the board from Python with the shared helper:
#   uv run python -c "from esp32tools import Board; b = Board(); print(b.command('ping'))"
#
# Host tools should check the board runs YOUR firmware before using it. See
# projects/led/tools/led_repl.py for a complete small example:
#   REQUIREMENT = Requirement(project="__NAME__", proto=PROTO_VERSION, project_dir=...)
#   board = connect(Board, port, REQUIREMENT, mode)     # asks `ident`, offers to flash
# PROTO_VERSION in main/main.c and in the host tools must be the same number.
