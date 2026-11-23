idf_component_register(SRCS "main.c" "model.c" "tokenizer.c" "bert.c" "activity_led.c"
                       INCLUDE_DIRS "."
                       REQUIRES driver esp_partition esp_timer)
