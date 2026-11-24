/*
 * board_ident.h - let the host ask "which firmware are you?"
 *
 * Host tools that talk to a project need to know the board is running the right firmware,
 * and not some other project or an older version. Every project therefore answers the
 * serial command `ident` with one line:
 *
 *     IDENT project=minilm proto=1 build=3f2a91bc model=ok model_id=9c1d0e77
 *
 *   project   the name given to project(...) in the project's CMakeLists.txt
 *   proto     the PROTOCOL VERSION: bump it whenever a change would break the host tools
 *             (renamed or removed commands, changed reply format). The host requires an
 *             exact match, so an old build with the same protocol is still accepted.
 *   build     the first 8 hex digits of the SHA-256 of this exact build. It changes with
 *             every rebuild, so it says WHICH build is running, not whether it is compatible.
 *   (more)    anything the project wants to add, as more key=value pairs (MiniLM reports
 *             the state and id of the model data in flash)
 *
 * The host side is common/python/esp32tools (ensure.py, firmware.py).
 */
#ifndef BOARD_IDENT_H
#define BOARD_IDENT_H

/* Print the IDENT line. `extra` is optional ("key=value key=value", or NULL / ""). It does not
 * print the closing "OK": that is the caller's job, like for every other command. */
void board_ident_print(int proto, const char *extra);

#endif /* BOARD_IDENT_H */
