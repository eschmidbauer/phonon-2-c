/*  engine.c – The one-call API: a loaded model with its tokens, and samples → transcript with word times.
 *
 *  A thin layer over the component API (p2_resample → p2_mel_spectrogram → p2_encoder → p2_greedy_decode →
 *  p2_tokens_text / p2_tokens_words). main.c is a consumer of it like any other; examples/transcribe.c is the
 *  shortest one.
 */

#include "phonon2.h"
#include "threadpool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct P2Engine {
    P2Model   model;
    P2Tokens *tokens;
};

double p2_now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define now_sec p2_now_sec

P2Engine *p2_engine_open(const char *weights_dir)
{
    char bin[1024], json[1024], tok[1024];
    snprintf(bin,  sizeof(bin),  "%s/weights.bin",  weights_dir);
    snprintf(json, sizeof(json), "%s/weights.json", weights_dir);
    snprintf(tok,  sizeof(tok),  "%s/tokens.txt",   weights_dir);

    P2Engine *e = (P2Engine *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    if (p2_load_weights(&e->model, bin, json) != 0) {
        free(e);
        return NULL;
    }
    e->tokens = p2_tokens_load(tok);
    if (!e->tokens) {
        p2_free_weights(&e->model);
        free(e);
        return NULL;
    }
    p2_pool();   /* start the workers here rather than inside the first transcription */
    return e;
}

P2Engine *p2_engine_open_mem(const void *bin, size_t bin_len, const char *json, size_t json_len,
                             const char *tokens, size_t tokens_len)
{
    P2Engine *e = (P2Engine *)calloc(1, sizeof(*e));
    if (!e) return NULL;
    if (p2_load_weights_mem(&e->model, json, json_len, bin, bin_len) != 0) {
        free(e);
        return NULL;
    }
    e->tokens = p2_tokens_parse(tokens, tokens_len);
    if (!e->tokens) {
        p2_free_weights(&e->model);
        free(e);
        return NULL;
    }
    p2_pool();
    return e;
}

void p2_engine_close(P2Engine *e)
{
    if (!e) return;
    p2_tokens_free(e->tokens);
    p2_free_weights(&e->model);
    free(e);
}

const P2Model  *p2_engine_model(const P2Engine *e)  { return &e->model; }
const P2Tokens *p2_engine_tokens(const P2Engine *e) { return e->tokens; }

void p2_result_free(P2Result *r)
{
    if (!r) return;
    free(r->text);
    p2_words_free(r->words, r->n_words);
    free(r->tokens);
    for (int i = 0; i < r->n_segments; i++) free(r->segments[i].text);
    free(r->segments);
    memset(r, 0, sizeof(*r));
}

/* Grow `*arr` of `*cap` elements of `size` bytes to hold `need`. Returns 0 on success. */
static int reserve(void **arr, int *cap, int need, size_t size)
{
    if (need <= *cap) return 0;
    int cap2 = *cap ? *cap : 64;
    while (cap2 < need) cap2 *= 2;
    void *p = realloc(*arr, (size_t)cap2 * size);
    if (!p) return -1;
    *arr = p;
    *cap = cap2;
    return 0;
}

typedef struct { int tok, word, seg; } Caps;   /* allocated sizes of the result's arrays */

/* Encoder and decoder over one block of features → tokens (frames relative to the block). Timings accumulate. */
static int decode_features(const P2Engine *e, const float *mel, int n_frames, P2Token **toks, int *n_toks, int *T,
                           P2Result *out)
{
    *toks = NULL; *n_toks = 0; *T = 0;
    if (n_frames < P2_SUB_FACTOR) return 0;   /* fewer frames than one encoder step is silence */
    double t = now_sec();
    float *enc_out;
    if (p2_encoder(&e->model, mel, n_frames, &enc_out, T) != 0) return -1;
    out->t_encoder += now_sec() - t;
    t = now_sec();
    *n_toks = p2_greedy_decode(&e->model, enc_out, *T, toks);
    free(enc_out);
    out->t_decoder += now_sec() - t;
    out->encoder_frames += *T;
    return 0;
}

/* Append a decoded block to the result: its tokens (frames offset by `frame_offset`), its words and its text as
 * one segment. Takes ownership of `toks`. Returns 0, or -1 on failure. */
static int append_window(const P2Engine *e, P2Token *toks, int n_toks, int frame_offset, float start_s, float end_s,
                         P2Result *out, Caps *caps)
{
    int *tok_cap = &caps->tok, *word_cap = &caps->word, *seg_cap = &caps->seg;
    P2Word *words = NULL;
    int n_words = p2_tokens_words(e->tokens, toks, n_toks, &words);
    char *text = p2_tokens_text(e->tokens, toks, n_toks);

    if (reserve((void **)&out->tokens, tok_cap, out->n_tokens + n_toks, sizeof(P2Token)) ||
        reserve((void **)&out->words, word_cap, out->n_words + n_words, sizeof(P2Word)) ||
        reserve((void **)&out->segments, seg_cap, out->n_segments + 1, sizeof(P2Segment))) {
        free(toks); p2_words_free(words, n_words); free(text);
        return -1;
    }
    for (int i = 0; i < n_toks; i++) {
        toks[i].frame += frame_offset;
        out->tokens[out->n_tokens++] = toks[i];
    }
    for (int i = 0; i < n_words; i++) {
        words[i].start += frame_offset * P2_FRAME_SECONDS;
        words[i].end   += frame_offset * P2_FRAME_SECONDS;
        out->words[out->n_words++] = words[i];
    }
    P2Segment *seg = &out->segments[out->n_segments++];
    seg->start = start_s;
    seg->end   = end_s;
    seg->text  = text;
    seg->first_word  = out->n_words - n_words;   seg->n_words  = n_words;
    seg->first_token = out->n_tokens - n_toks;   seg->n_tokens = n_toks;
    free(toks);
    free(words);   /* the word texts now belong to out->words */
    return 0;
}

/* The result's text: the segments' texts joined with single spaces, empty ones skipped (Fermion's shape). */
static int join_segments(P2Result *out)
{
    size_t total = 1;
    for (int i = 0; i < out->n_segments; i++) total += strlen(out->segments[i].text) + 1;
    char *text = (char *)malloc(total);
    if (!text) return -1;
    size_t len = 0;
    for (int i = 0; i < out->n_segments; i++) {
        const char *s = out->segments[i].text;
        if (!*s) continue;
        if (len) text[len++] = ' ';
        size_t n = strlen(s);
        memcpy(text + len, s, n);
        len += n;
    }
    text[len] = '\0';
    out->text = text;
    return 0;
}

int p2_transcribe_mel(const P2Engine *e, const float *mel, int n_frames, P2Result *out)
{
    memset(out, 0, sizeof(*out));
    if (n_frames < 0 || (n_frames > 0 && !mel)) return -1;
    out->duration_seconds = (float)n_frames * P2_HOP_LEN / P2_SAMPLE_RATE;
    Caps caps = {0, 0, 0};
    P2Token *toks;
    int n_toks, T;
    if (decode_features(e, mel, n_frames, &toks, &n_toks, &T, out) != 0 ||
        append_window(e, toks, n_toks, 0, 0.0f, out->duration_seconds, out, &caps) != 0 ||
        join_segments(out) != 0) {
        p2_result_free(out);
        return -1;
    }
    return 0;
}

/* "0" / "off" in the environment variable `name` turns a feature off. */
static int env_enabled(const char *name)
{
    const char *env = getenv(name);
    return !(env && (env[0] == '0' || strcmp(env, "off") == 0));
}

/* P2_SEGMENT=0 hands long audio to the encoder whole (which chunks it beyond 40 s). */
static int segmentation_enabled(void) { return env_enabled("P2_SEGMENT"); }

/* P2_REDECODE=0 leaves a window that went quiet as it came out (for measuring the dropout rate). */
static int redecode_enabled(void) { return env_enabled("P2_REDECODE"); }

/* A dropout: the model emitted nothing for P2_DROPOUT_S seconds of a window while at least P2_DROPOUT_SPEECH of
 * that stretch's blocks were above the window's noise floor (segment.c: ≥ 0.42 in every skipped stretch measured,
 * ≈ 0 in a real pause of read speech). Normal decoding never leaves a 2 s gap over speech – tokens come every
 * 0.1–0.5 s and the first within a second; the longest word gap over 4 hours of read speech was 1.7 s. */
#define P2_DROPOUT_S        2.0
#define P2_DROPOUT_SPEECH   0.40f
#define P2_DROPOUT_TONE     0.90f /* above this share the stretch is a steady tone (hold music, a dial tone), not speech:
                                     speech over 2 s always has blocks under the gate; the dropouts measured ran ≤ 0.76 */
#define P2_SPLIT_MIN_S      6.0   /* a window shorter than this is left as it came out */
#define P2_SPLIT_MAX_DEPTH  2     /* 35 s → 17 → 9 s at most: every cure observed came by the second split, and two
                                     levels bound what a false alarm on telephone noise costs to about 1.5 windows */

static int window_dropped_out(const float *x, int len, const P2Token *toks, int n_toks, int *gap_from, int *gap_to,
                              float *speech)
{
    const int frame = P2_HOP_LEN * P2_SUB_FACTOR, min_gap = (int)(P2_DROPOUT_S * P2_SAMPLE_RATE);
    float *rms = NULL, gate = 0.0f;   /* the window's energy profile, computed on the first long gap */
    int nb = 0, found = 0, prev = 0;
    for (int i = 0; i <= n_toks && !found; i++) {
        int cur = i < n_toks ? toks[i].frame * frame : len;
        if (cur > len) cur = len;
        if (cur - prev >= min_gap) {
            if (!rms) {
                rms = p2_block_rms(x, len, P2_SAMPLE_RATE, &nb);
                if (!rms) break;
                gate = p2_speech_gate(rms, nb);
                if (gate < 0.0f) break;   /* out of memory: no check rather than a check with a wrong gate */
            }
            float s = p2_speech_share(rms, nb, P2_SAMPLE_RATE, prev, cur, gate);
            if (s >= P2_DROPOUT_SPEECH && s <= P2_DROPOUT_TONE) {
                *gap_from = prev; *gap_to = cur; *speech = s;
                found = 1;
            }
        }
        if (i < n_toks) prev = cur + (toks[i].duration > 0 ? toks[i].duration : 1) * frame;
    }
    free(rms);
    return found;
}

/* Transcribe x[start, end): features, encoder, decoder; on a dropout, re-decode as two windows split at the
 * longest pause in the middle third (a different start point almost always decodes normally). `base` is how many
 * samples of the whole recording precede x[0] (a stream's consumed audio), for the frames and times reported. */
static int transcribe_window(const P2Engine *e, const float *x, int start, int end, int depth, long long base,
                             P2Result *out, Caps *caps)
{
    int len = end - start;
    float *mel = NULL;
    int n_frames = 0;
    if (len >= P2_SUB_FACTOR * P2_HOP_LEN) {   /* under 80 ms: nothing the encoder could frame */
        double t = now_sec();
        if (p2_mel_spectrogram(x + start, len, &mel, &n_frames) != 0) return -1;
        out->t_mel += now_sec() - t;
    }
    P2Token *toks;
    int n_toks, T;
    int rc = decode_features(e, mel, n_frames, &toks, &n_toks, &T, out);
    free(mel);
    if (rc != 0) return -1;

    double t_start = (double)(base + start) / P2_SAMPLE_RATE, t_end = (double)(base + end) / P2_SAMPLE_RATE;
    int gap_from, gap_to;
    float speech;
    if (depth < P2_SPLIT_MAX_DEPTH && len >= 2 * P2_SPLIT_MIN_S * P2_SAMPLE_RATE && redecode_enabled() &&
        window_dropped_out(x + start, len, toks, n_toks, &gap_from, &gap_to, &speech)) {
        int mid = start + p2_split_point(x + start, len, P2_SAMPLE_RATE);
        if (p2_verbose)
            fprintf(stderr, "  window %.1f–%.1f s went quiet: no token for %.1f–%.1f s (%d tokens, speech share %.2f); "
                    "re-decoding as %.1f–%.1f and %.1f–%.1f\n", t_start, t_end,
                    (double)(base + start + gap_from) / P2_SAMPLE_RATE, (double)(base + start + gap_to) / P2_SAMPLE_RATE,
                    n_toks, speech, t_start, (double)(base + mid) / P2_SAMPLE_RATE,
                    (double)(base + mid) / P2_SAMPLE_RATE, t_end);
        free(toks);
        if (transcribe_window(e, x, start, mid, depth + 1, base, out, caps) != 0) return -1;
        return transcribe_window(e, x, mid, end, depth + 1, base, out, caps);
    }

    int frame_offset = (int)((double)(base + start) / (P2_HOP_LEN * P2_SUB_FACTOR) + 0.5);
    return append_window(e, toks, n_toks, frame_offset, (float)t_start, (float)t_end, out, caps);
}

/* Samples with every NaN and infinity replaced by zero, when there are any: one bad sample would otherwise spread
 * through the STFT frame and the per-feature mel normalisation and silence the whole window (a float WAV with a
 * NaN every thousand samples transcribed to nothing). Returns `x` itself when it is all finite, else a copy the
 * caller frees through *owned. NULL on allocation failure. */
static const float *finite_samples(const float *x, int n, float **owned)
{
    *owned = NULL;
    int bad = 0;
    for (int i = 0; i < n && !bad; i++) bad = !p2_is_finite(x[i]);
    if (!bad) return x;
    float *y = (float *)malloc((size_t)(n > 0 ? n : 1) * sizeof(float));
    if (!y) return NULL;
    for (int i = 0; i < n; i++) y[i] = p2_is_finite(x[i]) ? x[i] : 0.0f;
    *owned = y;
    return y;
}

int p2_transcribe(const P2Engine *e, const float *audio, int n_samples, int sample_rate, P2Result *out)
{
    memset(out, 0, sizeof(*out));
    if (n_samples < 0 || sample_rate <= 0 || (n_samples > 0 && !audio)) return -1;
    out->duration_seconds = (float)n_samples / sample_rate;

    double t0 = now_sec();
    float *cleaned = NULL;
    const float *x = finite_samples(audio, n_samples, &cleaned);
    if (!x) return -1;
    float *resampled = NULL;
    int n = n_samples;
    if (sample_rate != P2_SAMPLE_RATE) {
        int rc = p2_resample(x, n_samples, sample_rate, P2_SAMPLE_RATE, &resampled, &n);
        free(cleaned);
        cleaned = NULL;
        if (rc != 0) return -1;
        x = resampled;
    }

    P2Window *win = NULL;
    int n_win, depth0 = 0;
    if (segmentation_enabled()) {
        n_win = p2_plan_windows(x, n, P2_SAMPLE_RATE, &win);
    } else {   /* the pre-window path, whole: one window and no re-decode either */
        win = (P2Window *)malloc(sizeof(P2Window));
        if (win) { win[0].start = 0; win[0].end = n; }
        n_win = win ? 1 : 0;
        depth0 = P2_SPLIT_MAX_DEPTH;
    }
    if (n > 0 && n_win == 0) { free(resampled); free(cleaned); return -1; }
    if (p2_verbose && n_win > 1)
        fprintf(stderr, "  %d windows of %.1f–%.1f s\n", n_win, (win[0].end - win[0].start) / (float)P2_SAMPLE_RATE,
                (win[n_win - 1].end - win[n_win - 1].start) / (float)P2_SAMPLE_RATE);
    out->t_mel += now_sec() - t0;

    Caps caps = {0, 0, 0};
    int rc = 0;
    for (int i = 0; i < n_win && rc == 0; i++)
        rc = transcribe_window(e, x, win[i].start, win[i].end, depth0, 0, out, &caps);
    free(win);
    free(resampled);
    free(cleaned);
    if (rc == 0) rc = join_segments(out);
    if (rc != 0) { p2_result_free(out); return -1; }
    return 0;
}

/* ── Streaming ───────────────────────────────────────────────────────────── */

/* The audio since the last cut sits in `buf`; `consumed` samples precede it. A window is cut exactly as
 * p2_transcribe's planner cuts it (p2_first_cut on the pending audio, which is why the planner aligns its blocks to
 * the window start), decoded with the same transcribe_window and appended to `res`. */
struct P2Stream {
    const P2Engine *e;
    P2Resampler *rs;        /* input rate → 16 kHz, NULL when the input is 16 kHz already */
    float     *buf;
    int        n, cap;
    long long  consumed;
    P2Result   res;
    Caps       caps;
    int        delivered;   /* segments handed out by p2_stream_next */
    int        finished;    /* finish ran: no more feeding, a second finish is a no-op */
};

P2Stream *p2_stream_open(const P2Engine *e, int sample_rate)
{
    if (sample_rate <= 0) return NULL;
    P2Stream *s = (P2Stream *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->e = e;
    s->cap = p2_window_max_samples(P2_SAMPLE_RATE) + P2_SAMPLE_RATE;
    s->buf = (float *)malloc((size_t)s->cap * sizeof(float));
    if (sample_rate != P2_SAMPLE_RATE) s->rs = p2_resampler_open(sample_rate, P2_SAMPLE_RATE);
    if (!s->buf || (sample_rate != P2_SAMPLE_RATE && !s->rs)) { p2_stream_close(s); return NULL; }
    return s;
}

void p2_stream_close(P2Stream *s)
{
    if (!s) return;
    p2_resampler_close(s->rs);
    p2_result_free(&s->res);
    free(s->buf);
    free(s);
}

/* Decode buf[0, cut) as the next window and drop it from the buffer. */
static int stream_cut(P2Stream *s, int cut)
{
    if (transcribe_window(s->e, s->buf, 0, cut, 0, s->consumed, &s->res, &s->caps) != 0) return -1;
    memmove(s->buf, s->buf + cut, (size_t)(s->n - cut) * sizeof(float));
    s->n -= cut;
    s->consumed += cut;
    return 0;
}

/* Append 16 kHz samples to the pending audio, cutting a window whenever the band is complete. */
static int stream_append(P2Stream *s, const float *audio, int n_samples)
{
    int single = p2_window_max_samples(P2_SAMPLE_RATE), i = 0;
    while (i < n_samples) {
        int take = n_samples - i;
        if (take > s->cap - s->n) take = s->cap - s->n;
        memcpy(s->buf + s->n, audio + i, (size_t)take * sizeof(float));
        s->n += take;
        i += take;
        while (s->n > single) {   /* the band is complete: cut at its best pause */
            int cut = p2_first_cut(s->buf, s->n, P2_SAMPLE_RATE);
            if (cut < 0 || stream_cut(s, cut) != 0) return -1;
        }
    }
    return 0;
}

int p2_stream_feed(P2Stream *s, const float *audio, int n_samples)
{
    if (n_samples < 0 || (n_samples > 0 && !audio) || s->finished) return -1;
    int before = s->res.n_segments, rc;
    float *cleaned = NULL;
    const float *x = finite_samples(audio, n_samples, &cleaned);
    if (!x) return -1;
    if (s->rs) {
        float *y;
        int ny;
        if (p2_resampler_feed(s->rs, x, n_samples, &y, &ny) != 0) { free(cleaned); return -1; }
        rc = stream_append(s, y, ny);
        free(y);
    } else {
        rc = stream_append(s, x, n_samples);
    }
    free(cleaned);
    return rc == 0 ? s->res.n_segments - before : -1;
}

const P2Segment *p2_stream_next(P2Stream *s)
{
    if (s->delivered >= s->res.n_segments) return NULL;
    return &s->res.segments[s->delivered++];
}

double p2_stream_pending_seconds(const P2Stream *s)
{
    return (double)s->n / P2_SAMPLE_RATE;
}

char *p2_stream_partial(P2Stream *s)
{
    if (s->n == 0) return (char *)calloc(1, 1);
    P2Result tmp;
    Caps caps = {0, 0, 0};
    char *text = NULL;
    memset(&tmp, 0, sizeof(tmp));
    if (transcribe_window(s->e, s->buf, 0, s->n, P2_SPLIT_MAX_DEPTH, s->consumed, &tmp, &caps) == 0 &&
        join_segments(&tmp) == 0) {
        text = tmp.text;
        tmp.text = NULL;
    }
    p2_result_free(&tmp);
    return text;   /* NULL when the decode failed */
}

int p2_stream_finish(P2Stream *s)
{
    if (s->finished) return 0;
    s->finished = 1;
    int before = s->res.n_segments;
    if (s->rs) {   /* the resampler's tail */
        float *y;
        int ny;
        if (p2_resampler_finish(s->rs, &y, &ny) != 0) return -1;
        int rc = stream_append(s, y, ny);
        free(y);
        if (rc != 0) return -1;
    }
    if (s->n > 0 && stream_cut(s, s->n) != 0) return -1;
    free(s->res.text);
    s->res.text = NULL;
    if (join_segments(&s->res) != 0) return -1;
    s->res.duration_seconds = (float)((double)s->consumed / P2_SAMPLE_RATE);
    return s->res.n_segments - before;
}

const P2Result *p2_stream_result(const P2Stream *s)
{
    return &s->res;
}
