/*  fuzz_wav.c – p2_wav_read on one file: the reader must return 0 or -1 and never crash, read out of bounds or
 *  leak (built with -fsanitize=address,undefined by `make -C csrc fuzz`). Exit 0 either way; a sanitizer report is
 *  the failure. With a second argument the samples are also resampled to it, exercising p2_resample on odd
 *  lengths and rates. */
#include "phonon2.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: fuzz_wav <file.wav> [resample_to_hz]\n"); return 2; }
    float *samples;
    int n, sr, channels;
    int rc = p2_wav_read(argv[1], &samples, &n, &sr, &channels);
    if (rc != 0) return 0;   /* rejected cleanly */
    if (argc > 2) {
        float *y;
        int ny;
        if (p2_resample(samples, n, sr, atoi(argv[2]), &y, &ny) == 0) free(y);
    }
    printf("%d samples, %d Hz, %d ch\n", n, sr, channels);
    free(samples);
    return 0;
}
