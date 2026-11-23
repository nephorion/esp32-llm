# Repository front door. Each folder under projects/ is a self-contained ESP-IDF project with
# its own Makefile; this file just forwards to it so you can stay in the repository root.
#
#   make list                       show the projects
#   make p=minilm build flash       run targets in one project (needs ESP-IDF exported)
#   make p=minilm repl
#   make new NAME=myproject         start a new project from projects/_template
#   make chip-info                  read the chip and flash details
#
# Running `make` inside a project folder works too, and is the same thing.
PORT ?= /dev/ttyACM0
PROJECTS := $(filter-out _template,$(notdir $(wildcard projects/*)))

# Anything we do not define here is forwarded to the project named by p=
%:
	@test -n "$(p)" || { echo "usage: make p=<project> $@   (projects: $(PROJECTS))"; exit 2; }
	@test -d projects/$(p) || { echo "no such project: $(p)   (projects: $(PROJECTS))"; exit 2; }
	$(MAKE) -C projects/$(p) $@ PORT=$(PORT)

list:
	@printf '%s\n' $(PROJECTS)

new:
	@test -n "$(NAME)" || { echo "usage: make new NAME=myproject"; exit 2; }
	@test ! -e projects/$(NAME) || { echo "projects/$(NAME) already exists"; exit 2; }
	cp -r projects/_template projects/$(NAME)
	grep -rl __NAME__ projects/$(NAME) | xargs sed -i 's/__NAME__/$(NAME)/g'
	@echo "created projects/$(NAME). Next: cd projects/$(NAME) && make build flash"

chip-info:
	uv run esptool --port $(PORT) chip-id
	uv run esptool --port $(PORT) flash-id

.PHONY: list new chip-info
