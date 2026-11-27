idf_component_register(SRCS "main.c" INCLUDE_DIRS "."
                       REQUIRES driver board_led board_ident matmath esp_partition esp_timer esp_psram)
