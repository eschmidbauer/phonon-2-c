/*  examples/transcribe.c – The library API end to end: open the weights, read a WAV, print the text and word times.
 *
 *      make -C csrc example && ./examples/transcribe c_weights recording.wav
 *      cmake -S . -B build && cmake --build build && ./build/transcribe c_weights recording.wav
 */

#include "phonon2.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <weights_dir> <audio.wav>\n", argv[0]);
        return 1;
    }
    p2_verbose = 0;   /* no encoder progress on stderr */

    P2Engine *engine = p2_engine_open(argv[1]);
    if (!engine) return 1;

    float *audio;
    int n_samples, sample_rate, channels;
    if (p2_wav_read(argv[2], &audio, &n_samples, &sample_rate, &channels) != 0) {
        p2_engine_close(engine);
        return 1;
    }

    P2Result r;
    if (p2_transcribe(engine, audio, n_samples, sample_rate, &r) != 0) {
        fprintf(stderr, "transcription failed\n");
        free(audio);
        p2_engine_close(engine);
        return 1;
    }

    printf("%s\n\n", r.text);
    for (int i = 0; i < r.n_words; i++)
        printf("%7.2f %7.2f  %s\n", r.words[i].start, r.words[i].end, r.words[i].text);
    fprintf(stderr, "%.2f s of audio: mel %.3f s, encoder %.3f s, decoder %.3f s\n",
            r.duration_seconds, r.t_mel, r.t_encoder, r.t_decoder);

    p2_result_free(&r);
    free(audio);
    p2_engine_close(engine);
    return 0;
}
