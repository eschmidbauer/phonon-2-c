/*  main.c – Phonon-2 transcription from WAV file(s), on the engine API in phonon2.h.
 *
 *  Usage:  phonon2 [options] <weights_dir> <audio.wav> [audio2.wav ...]
 *
 *  weights_dir holds weights.bin, weights.json and tokens.txt from convert_weights.py.
 *  Several WAV files are transcribed concurrently, one thread each, sharing the loaded model.
 *
 *  Options:
 *    --json          text plus one {text, start, end} per word (seconds, from the decoder's emission frames)
 *    --tokens        print token ids instead of text
 *    --mel-in FILE   read log-mel features (raw float32, [frames × 128] time-major) instead of computing them
 *    --mel-out FILE  write the computed features in that layout
 *    --threads N     thread-pool size (default: online CPUs; also P2_THREADS)
 *    --exact         fp32 expansion for the quantised GEMMs instead of int8 dot products (also P2_EXACT=1)
 *    --cache-codes   pre-pack the int8 weight codes once (576 MB of RAM) so short utterances skip the per-pass
 *                    expansion (also P2_CACHE_CODES=1)
 *    --repeat N      run the pipeline N times on the (single) file and report the fastest; for latency work
 *    --quiet         no encoder progress on stderr
 *    --stream        no audio files: read 16-bit little-endian mono PCM from stdin and print each finished 25–35 s
 *                    window as it completes ("[start – end] text", or a JSON object per line with --json)
 *    --rate R        with --stream: the PCM's sample rate (default 16000; anything else is resampled on the way)
 *    --partial S     with --stream: print a provisional transcript of the pending audio to stderr every S seconds
 */

#include "phonon2.h"
#include "blas.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    int         json;
    int         tokens;
    int         repeat;
    const char *mel_in;
    const char *mel_out;
    int         stream;    /* --stream: PCM on stdin, a line per finished window */
    int         rate;      /* --rate R: the stdin PCM's sample rate */
    double      partial;   /* --partial S: provisional text every S seconds of pending audio */
} Options;

#define now_sec p2_now_sec

/* ── Output ──────────────────────────────────────────────────────────────── */

/* --json is the library's p2_result_json (json.c); --tokens is the ids, space-separated. */

static char *result_token_ids(const P2Result *r)
{
    size_t cap = (size_t)r->n_tokens * 6 + 1;
    char *s = (char *)malloc(cap);
    size_t off = 0;
    s[0] = '\0';
    for (int i = 0; i < r->n_tokens; i++)
        off += snprintf(s + off, cap - off, "%d%s", r->tokens[i].id, i < r->n_tokens - 1 ? " " : "");
    return s;
}

/* ── Feature files for verification ─────────────────────────────────────── */

static float *read_mel_file(const char *path, int *n_frames)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t row = P2_N_MELS * sizeof(float);
    if (bytes <= 0 || bytes % row != 0) {
        fprintf(stderr, "%s: size %ld is not a whole number of %d-float frames\n", path, bytes, P2_N_MELS);
        fclose(f);
        return NULL;
    }
    float *mel = (float *)malloc(bytes);
    if (fread(mel, 1, bytes, f) != (size_t)bytes) { perror(path); fclose(f); free(mel); return NULL; }
    fclose(f);
    *n_frames = (int)(bytes / row);
    return mel;
}

static int write_mel_file(const char *path, const float *mel, int n_frames)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    fwrite(mel, sizeof(float), (size_t)n_frames * P2_N_MELS, f);
    fclose(f);
    return 0;
}

/* ── Per-file job ────────────────────────────────────────────────────────── */

typedef struct {
    int              id;
    const char      *wav_path;
    const P2Engine  *engine;
    const Options   *opt;
    char            *result;
    float            audio_dur;
    double           elapsed, t_mel, t_enc, t_dec;
    int              error;     /* set by the job's own thread; read only after the join */
    int              started;   /* its thread exists and must be joined */
} TranscribeJob;

/* One pass over 16 kHz audio, through the feature-file options when given. Returns 0 and fills *r. */
static int run_once(const TranscribeJob *job, const float *audio, int n_samples, int first, P2Result *r)
{
    const Options *opt = job->opt;
    if (!opt->mel_in && !opt->mel_out)
        return p2_transcribe(job->engine, audio, n_samples, P2_SAMPLE_RATE, r);

    double t0 = now_sec();
    float *mel;
    int n_frames;
    if (opt->mel_in) {
        mel = read_mel_file(opt->mel_in, &n_frames);
        if (!mel) return -1;
        if (first) fprintf(stderr, "[%d] features from %s: %d frames\n", job->id, opt->mel_in, n_frames);
    } else if (p2_mel_spectrogram(audio, n_samples, &mel, &n_frames) != 0) {
        return -1;
    }
    if (first && opt->mel_out && write_mel_file(opt->mel_out, mel, n_frames) == 0)
        fprintf(stderr, "[%d] features written to %s: %d frames\n", job->id, opt->mel_out, n_frames);
    double t_mel = now_sec() - t0;

    int rc = p2_transcribe_mel(job->engine, mel, n_frames, r);
    free(mel);
    if (rc != 0) return -1;
    r->t_mel = t_mel;
    r->duration_seconds = (float)n_samples / P2_SAMPLE_RATE;
    return 0;
}

static void *transcribe_thread(void *arg)
{
    TranscribeJob *job = (TranscribeJob *)arg;
    const Options *opt = job->opt;
    double t0 = now_sec();

    float *audio;
    int n_samples, sr, channels;
    if (p2_wav_read(job->wav_path, &audio, &n_samples, &sr, &channels) != 0) {
        fprintf(stderr, "[%d] failed to read WAV: %s\n", job->id, job->wav_path);
        job->error = 1;
        return NULL;
    }
    job->audio_dur = (float)n_samples / sr;
    fprintf(stderr, "[%d] %s: %d samples, %d Hz, %d channel%s, %.2fs\n", job->id, job->wav_path, n_samples, sr,
            channels, channels == 1 ? "" : "s (mixed to mono)", job->audio_dur);
    /* p2_transcribe would resample itself; doing it here keeps it out of the repeated timings and reports it. */
    if (sr != P2_SAMPLE_RATE) {
        float *resampled;
        int n_resampled;
        double tr = now_sec();
        if (p2_resample(audio, n_samples, sr, P2_SAMPLE_RATE, &resampled, &n_resampled) != 0) {
            fprintf(stderr, "[%d] cannot resample %d Hz to %d Hz\n", job->id, sr, P2_SAMPLE_RATE);
            free(audio);
            job->error = 1;
            return NULL;
        }
        fprintf(stderr, "[%d] resampled %d Hz → %d Hz (%.3fs)\n", job->id, sr, P2_SAMPLE_RATE, now_sec() - tr);
        free(audio);
        audio = resampled;
        n_samples = n_resampled;
    }

    /* --repeat runs the whole pipeline several times and keeps the fastest iteration's timings and result. */
    int reps = opt->repeat > 1 ? opt->repeat : 1;
    double best = 1e30;
    P2Result kept = {0};
    for (int rep = 0; rep < reps; rep++) {
        P2Result r;
        if (run_once(job, audio, n_samples, rep == 0, &r) != 0) {
            free(audio);
            p2_result_free(&kept);
            job->error = 1;
            return NULL;
        }
        if (r.t_mel + r.t_encoder + r.t_decoder < best) {
            best = r.t_mel + r.t_encoder + r.t_decoder;
            p2_result_free(&kept);
            kept = r;
        } else {
            p2_result_free(&r);
        }
        if (reps > 1) p2_verbose = 0;
    }
    free(audio);
    if (reps > 1)
        fprintf(stderr, "[%d] %d runs, fastest %.3fs\n", job->id, reps, best);

    job->t_mel = kept.t_mel; job->t_enc = kept.t_encoder; job->t_dec = kept.t_decoder;
    if (opt->tokens)    job->result = result_token_ids(&kept);
    else if (opt->json) job->result = p2_result_json(&kept);
    else              { job->result = kept.text; kept.text = NULL; }
    p2_result_free(&kept);

    job->elapsed = reps > 1 ? best : now_sec() - t0;
    return NULL;
}

/* ── Streaming from stdin ────────────────────────────────────────────────── */

/* Every segment not yet printed: "[start – end] text", or with --json a segment object carrying its words. */
static void print_new_segments(P2Stream *s, const Options *opt)
{
    const P2Segment *seg;
    while ((seg = p2_stream_next(s)) != NULL) {
        if (opt->json) {
            char *json = p2_segment_json(seg, p2_stream_result(s));
            printf("%s\n", json);
            free(json);
        } else {
            printf("[%7.2f – %7.2f] %s\n", seg->start, seg->end, seg->text);
        }
    }
    fflush(stdout);
}

/* Release the stream and its buffers; the exit status passes through. */
static int stream_done(P2Stream *s, unsigned char *pcm, float *f, int rc)
{
    free(pcm);
    free(f);
    p2_stream_close(s);
    return rc;
}

/* 16-bit little-endian mono PCM at opt->rate on stdin, 100 ms at a time, until EOF. */
static int stream_stdin(const P2Engine *engine, const Options *opt)
{
    P2Stream *s = p2_stream_open(engine, opt->rate);
    if (!s) { fprintf(stderr, "stream: cannot open a stream at %d Hz\n", opt->rate); return 1; }
    int chunk = opt->rate / 10;
    unsigned char *pcm = (unsigned char *)malloc((size_t)chunk * 2);
    float *f = (float *)malloc((size_t)chunk * sizeof(float));
    if (!pcm || !f) { free(pcm); free(f); p2_stream_close(s); return 1; }
    double partial_at = 0.0, t0 = now_sec();
    size_t got;
    while ((got = fread(pcm, 2, (size_t)chunk, stdin)) > 0) {
        p2_pcm16le_to_float(pcm, (int)got, 1, f);
        if (p2_stream_feed(s, f, (int)got) < 0) {
            fprintf(stderr, "stream: decoding failed\n");
            return stream_done(s, pcm, f, 1);
        }
        print_new_segments(s, opt);
        if (opt->partial > 0.0) {
            double pending = p2_stream_pending_seconds(s);
            if (pending < partial_at) partial_at = 0.0;   /* a window was cut */
            if (pending - partial_at >= opt->partial) {
                char *text = p2_stream_partial(s);
                if (!text) {
                    fprintf(stderr, "stream: decoding failed\n");
                    return stream_done(s, pcm, f, 1);
                }
                fprintf(stderr, "  … %s\n", text);
                free(text);
                partial_at = pending;
            }
        }
    }
    if (ferror(stdin)) {
        perror("stream: stdin");
        return stream_done(s, pcm, f, 1);
    }
    if (p2_stream_finish(s) < 0) {
        fprintf(stderr, "stream: decoding failed\n");
        return stream_done(s, pcm, f, 1);
    }
    print_new_segments(s, opt);
    const P2Result *r = p2_stream_result(s);
    fprintf(stderr, "stream: %.1f s of audio in %d window%s, %d words, %.2fs wall\n", r->duration_seconds,
            r->n_segments, r->n_segments == 1 ? "" : "s", r->n_words, now_sec() - t0);
    return stream_done(s, pcm, f, 0);
}

/* ── CLI ─────────────────────────────────────────────────────────────────── */

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [options] <weights_dir> <audio.wav> [audio2.wav ...]\n"
            "  --json          text plus word times\n"
            "  --tokens        token ids instead of text\n"
            "  --mel-in FILE   read features (raw float32 [frames x 128]) instead of computing them (one file only)\n"
            "  --mel-out FILE  write the computed features in that layout (one file only)\n"
            "  --threads N     thread-pool size (default: online CPUs)\n"
            "  --exact         fp32 expansion for the quantised GEMMs instead of int8 dot products\n"
            "  --cache-codes   pre-pack the int8 weight codes (576 MB RAM): faster short utterances\n"
            "  --repeat N      run the pipeline N times on one file, report the fastest\n"
            "  --quiet         no encoder progress\n"
            "  --stream        16-bit mono PCM on stdin → a line per finished window (no audio files)\n"
            "  --rate R        with --stream: the PCM's sample rate (default 16000)\n"
            "  --partial S     with --stream: provisional text of the pending audio on stderr every S seconds\n"
            "  --version\n", argv0);
}

int main(int argc, char **argv)
{
    Options opt = {0, 0, 1, NULL, NULL, 0, P2_SAMPLE_RATE, 0.0};
    const char *positional[1024];
    int n_pos = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--json") == 0) opt.json = 1;
        else if (strcmp(a, "--tokens") == 0) opt.tokens = 1;
        else if (strcmp(a, "--quiet") == 0) p2_verbose = 0;
        else if (strcmp(a, "--mel-in") == 0 && i + 1 < argc) opt.mel_in = argv[++i];
        else if (strcmp(a, "--mel-out") == 0 && i + 1 < argc) opt.mel_out = argv[++i];
        else if (strcmp(a, "--threads") == 0 && i + 1 < argc) setenv("P2_THREADS", argv[++i], 1);
        else if (strcmp(a, "--exact") == 0) setenv("P2_EXACT", "1", 1);
        else if (strcmp(a, "--cache-codes") == 0) setenv("P2_CACHE_CODES", "1", 1);
        else if (strcmp(a, "--repeat") == 0 && i + 1 < argc) opt.repeat = atoi(argv[++i]);
        else if (strcmp(a, "--version") == 0) { printf("phonon2 %s\n", P2_VERSION); return 0; }
        else if (strcmp(a, "--stream") == 0) opt.stream = 1;
        else if (strcmp(a, "--rate") == 0 && i + 1 < argc) opt.rate = atoi(argv[++i]);
        else if (strcmp(a, "--partial") == 0 && i + 1 < argc) opt.partial = atof(argv[++i]);
        else if (a[0] == '-' && a[1] == '-') { usage(argv[0]); return 1; }
        else if (n_pos < 1024) positional[n_pos++] = a;
    }
    if (n_pos < (opt.stream ? 1 : 2)) { usage(argv[0]); return 1; }
    const char *weights_dir = positional[0];
    int n_jobs = n_pos - 1;
    if (opt.stream && (n_jobs > 0 || opt.tokens || opt.repeat > 1 || opt.mel_in || opt.mel_out)) {
        fprintf(stderr, "--stream reads stdin and prints text: no audio files, --tokens, --repeat, --mel-in or --mel-out\n");
        return 1;
    }
    if (!opt.stream && (opt.partial > 0.0 || opt.rate != P2_SAMPLE_RATE)) {
        fprintf(stderr, "--partial and --rate need --stream (files carry their own rate)\n");
        return 1;
    }
    if (opt.stream && opt.rate <= 0) {
        fprintf(stderr, "--rate must be positive\n");
        return 1;
    }
    if (!opt.stream && (opt.mel_in || opt.mel_out || opt.repeat > 1) && n_jobs != 1) {
        fprintf(stderr, "--mel-in / --mel-out / --repeat apply to a single audio file\n");
        return 1;
    }

    fprintf(stderr, "loading model …\n");
    double t0 = now_sec();
    P2Engine *engine = p2_engine_open(weights_dir);
    if (!engine) {
        fprintf(stderr, "failed to load %s\n", weights_dir);
        return 1;
    }
    fprintf(stderr, "  loaded in %.2fs, %d threads, quantised GEMMs: %s\n",
            now_sec() - t0, p2_pool_nthreads(p2_pool()), p2_gemm_mode());

    if (opt.stream) {
        p2_verbose = 0;
        int rc = stream_stdin(engine, &opt);
        p2_engine_close(engine);
        return rc;
    }

    TranscribeJob *jobs = (TranscribeJob *)calloc(n_jobs, sizeof(TranscribeJob));
    pthread_t     *tids = (pthread_t *)malloc(n_jobs * sizeof(pthread_t));
    if (n_jobs > 1) p2_verbose = 0;

    fprintf(stderr, "transcribing %d file%s …\n", n_jobs, n_jobs > 1 ? "s" : "");
    t0 = now_sec();
    for (int i = 0; i < n_jobs; i++) {
        jobs[i].id = i;
        jobs[i].wav_path = positional[1 + i];
        jobs[i].engine = engine;
        jobs[i].opt = &opt;
        if (n_jobs == 1) {
            transcribe_thread(&jobs[i]);
        } else if (pthread_create(&tids[i], NULL, transcribe_thread, &jobs[i]) == 0) {
            jobs[i].started = 1;
        } else {
            fprintf(stderr, "[%d] pthread_create failed\n", i);
            jobs[i].error = 1;
        }
    }
    for (int i = 0; i < n_jobs; i++)
        if (jobs[i].started) pthread_join(tids[i], NULL);
    double total = now_sec() - t0;

    int failed = 0;
    for (int i = 0; i < n_jobs; i++) {
        if (jobs[i].error) { failed = 1; continue; }
        double rtf = jobs[i].audio_dur > 0 ? jobs[i].elapsed / jobs[i].audio_dur : 0.0;
        fprintf(stderr, "[%d] %s  mel=%.2fs  enc=%.2fs  dec=%.2fs  total=%.2fs  audio=%.2fs  RTF=%.4f (%.0fx realtime)\n",
                i, jobs[i].wav_path, jobs[i].t_mel, jobs[i].t_enc, jobs[i].t_dec,
                jobs[i].elapsed, jobs[i].audio_dur, rtf, rtf > 0 ? 1.0 / rtf : 0.0);
        printf("%s%s%s\n", n_jobs > 1 ? jobs[i].wav_path : "", n_jobs > 1 ? ": " : "", jobs[i].result);
        free(jobs[i].result);
    }
    if (n_jobs > 1) fprintf(stderr, "total: %.2fs\n", total);

    free(jobs);
    free(tids);
    p2_engine_close(engine);
    return failed;
}
