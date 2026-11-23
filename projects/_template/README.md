# __NAME__

Describe what this project does here.

```bash
. ~/esp/esp-idf/export.sh
make build flash        # from this folder
uv run python -c "from esp32tools import Board; b = Board(); print(b.command('echo hi'))"
```
