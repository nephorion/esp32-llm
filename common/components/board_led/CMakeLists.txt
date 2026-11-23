# Shared by every project: the on-board LED (board_led) and the "busy" blinker built on it
# (activity_led). A project uses it by listing `board_led` in REQUIRES.
idf_component_register(SRCS "board_led.c" "activity_led.c"
                       INCLUDE_DIRS "."
                       REQUIRES driver)
