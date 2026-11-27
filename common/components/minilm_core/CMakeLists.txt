# The MiniLM model code, shared by projects/minilm and projects/minilm-psram.
idf_component_register(SRCS "shell.c" "model.c" "tokenizer.c" "bert.c"
                       INCLUDE_DIRS "."
                       REQUIRES driver esp_partition esp_timer board_led board_ident)
