/*  fuzz_weights.c – p2_load_weights on a (mutated) weights.json / weights.bin pair, and p2_tokens_load on a
 *  tokens.txt: each must accept or reject cleanly. Built with the sanitizers by `make -C csrc fuzz`. */
#include "phonon2.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: fuzz_weights <weights.bin> <weights.json> [tokens.txt]\n"); return 2; }
    p2_verbose = 0;
    P2Model m;
    int rc = p2_load_weights(&m, argv[1], argv[2]);
    printf("weights: %s\n", rc == 0 ? "accepted" : "rejected");
    if (rc == 0) p2_free_weights(&m);
    if (argc > 3) {
        P2Tokens *t = p2_tokens_load(argv[3]);
        printf("tokens: %s\n", t ? "accepted" : "rejected");
        p2_tokens_free(t);
    }
    return 0;
}
