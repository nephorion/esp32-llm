/* board_ident.c - see board_ident.h */
#include <stdio.h>
#include "esp_app_desc.h"
#include "board_ident.h"

void board_ident_print(int proto, const char *extra)
{
    /* ESP-IDF stores a description of the build inside the firmware image itself,
     * including the project name and a SHA-256 fingerprint of the compiled program. */
    const esp_app_desc_t *d = esp_app_get_description();
    char sha[65];                                  /* 64 hex digits + the end-of-string zero */
    esp_app_get_elf_sha256(sha, sizeof sha);

    printf("IDENT project=%s proto=%d build=%.8s%s%s\n",
           d->project_name, proto, sha,
           (extra && *extra) ? " " : "", (extra && *extra) ? extra : "");
}
