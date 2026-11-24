# Shared by every project that answers the serial `ident` command (see board_ident.h).
idf_component_register(SRCS "board_ident.c" INCLUDE_DIRS "." REQUIRES esp_app_format)
