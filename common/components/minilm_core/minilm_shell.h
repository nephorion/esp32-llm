/*
 * minilm_shell.h - the whole MiniLM program (load the model, answer serial commands), with
 * two optional hooks so that another project can reuse it and change a detail.
 */
#ifndef MINILM_SHELL_H
#define MINILM_SHELL_H

#include "model.h"

typedef struct {
    /* Called once, after the model is mapped and the working buffers are allocated, and
     * before READY. It may repoint weight pointers in *m (for example to copies in RAM). */
    void (*model_ready)(model_t *m);

    /* Called for a command line the shell does not know. Return 0 if it is not yours,
     * 1 if you handled it (the shell then prints OK), or -1 if you handled it and have
     * already printed an "ERR ..." line. */
    int (*command)(model_t *m, const char *line);
} minilm_hooks_t;

/* Run the program. Never returns. `hooks` may be NULL (the original MiniLM project). */
void minilm_shell_run(const minilm_hooks_t *hooks);

#endif /* MINILM_SHELL_H */
