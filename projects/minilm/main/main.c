/*
 * minilm - MiniLM-L6 with all its weights read from flash.
 *
 * The program itself (serial commands, model loading, the neural network) is the shared
 * component common/components/minilm_core: start with shell.c there, then bert.c.
 */
#include "minilm_shell.h"

void app_main(void)
{
    minilm_shell_run(NULL);
}
