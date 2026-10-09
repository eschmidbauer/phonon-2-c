/*  phonon2.h – Pure-C inference for Phonon-2 (huggingface.co/FermionResearch/Phonon-2).
 *
 *  Phonon-2 is NVIDIA's Parakeet-TDT 0.6B v3 re-trained so that every encoder linear weight takes one of five values
 *  per output row, {0, ±lo, ±hi}, at about 2.1 bits each.  This engine keeps that representation (FiveValueB in blas.h)
 *  and expands it to fp32 inside the GEMM's pack step, so the whole model runs from ~230 MB of weights and the result
 *  equals an fp32 GEMM on the expanded weights.
 *
 *  Encoder  : FastConformer – 24 blocks, d=1024, 8 heads, ff=4096, relative-position attention, no linear biases
 *  Decoder  : 2-layer LSTM (h=640) prediction network + joint network with a 5-way duration head
 *  Decoding : greedy TDT (token-and-duration transducer)
 *
 *  Using it:  P2Engine *e = p2_engine_open("c_weights");
 *             p2_wav_read(path, &audio, &n, &rate, &channels);
 *             P2Result r;  p2_transcribe(e, audio, n, rate, &r);   →  r.text, r.words[i].{text,start,end}
 *             p2_result_free(&r);  p2_engine_close(e);
 *  The stages behind that (resample, mel, encoder, decoder, detokeniser) are exported too, further down.
 */
#ifndef PHONON2_H
#define PHONON2_H

#define P2_VERSION "0.1.0"   /* keep in step with CMakeLists.txt and the README */
/* The weights layout this loader reads: convert_weights.py writes it into weights.json under "_meta" (its FORMAT
 * constant; tests/check.py keeps the two equal) and names the published archive after it. */
#define P2_WEIGHTS_FORMAT "phonon-2-c-weights-v1"

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

/* ── Numerics shared by every file ───────────────────────────────────────── */

/* Scalar sigmoid with the exponent clamped as the SIMD exps (p2_exp_* in tensor_ops.c) clamp theirs. Without the
 * clamp expf(-x) is +inf below x ≈ −88.7; IEEE still gives 1 / (1 + inf) = 0, but under -ffast-math the compiler
 * assumes no infinities and may take the reciprocal through an approximation whose refinement computes inf · 0 =
 * NaN – the x86 SSE build did, and transcribed nonsense (verification.md, *Portability*). Every scalar sigmoid in
 * the engine goes through this; a new scalar transcendental gets the same treatment. */
static inline float p2_expneg(float x)   /* expf(-x), finite: the argument is clamped at 87 */
{
    return expf(x < -87.0f ? 87.0f : -x);
}

static inline float p2_sigmoid(float x)
{
    return 1.0f / (1.0f + p2_expneg(x));
}

/* Finiteness decided on the bits, behind a barrier the optimiser cannot see through. Neither isfinite() nor a plain
 * bit test survives -ffast-math: GCC 13 folds isfinite() to true under -ffinite-math-only, so the NaN gates in
 * engine.c and wav.c compiled to nothing on Linux and a float WAV with a NaN every thousand samples transcribed to
 * "tamb tamb tamb …"; Apple clang 21 keeps isfinite() but recognises the exponent-field test as a class check and
 * folds that instead, and the same file transcribed to nothing on macOS. The empty asm turns the bits into an
 * opaque integer that no fast-math assumption reaches (a float is NaN or infinite exactly when its exponent field
 * is all ones). Never call isfinite / isnan / isinf in this tree; use this. */
static inline int p2_is_finite(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
#if defined(__GNUC__) || defined(__clang__)
    __asm__("" : "+r"(bits));
#else
    volatile uint32_t opaque = bits;
    bits = opaque;
#endif
    return (bits & 0x7f800000u) != 0x7f800000u;
}

/* Wall-clock seconds (CLOCK_MONOTONIC), for the stage timings. */
double p2_now_sec(void);

/* ── Model constants ──────────────────────────────────────────────────────── */

#define P2_D_MODEL      1024
#define P2_N_HEADS      8
#define P2_HEAD_DIM     (P2_D_MODEL / P2_N_HEADS)   /* 128 */
#define P2_FF_DIM       4096
#define P2_N_BLOCKS     24
#define P2_CONV_KERNEL  9

#define P2_N_MELS       128
#define P2_SUB_CHANNELS 256
#define P2_SUB_FACTOR   8     /* three stride-2 convolutions */

#define P2_LSTM_HIDDEN  640
#define P2_LSTM_LAYERS  2
#define P2_EMBED_DIM    640
#define P2_JOINT_DIM    640
#define P2_VOCAB        8193  /* 8192 SentencePiece pieces + blank */
#define P2_BLANK        8192
#define P2_N_DURATIONS  5     /* the joint predicts a duration in {0, 1, 2, 3, 4} frames next to the token */
#define P2_N_LOGITS     (P2_VOCAB + P2_N_DURATIONS)
#define P2_MAX_SYMBOLS  10    /* NeMo's guard: tokens at one frame before the pointer is forced on */

/* Token ids the detokeniser drops (the tokenizer's special ids: <unk>, <pad>, <|endoftext|>). */
#define P2_UNK_ID       0
#define P2_PAD_ID       2
#define P2_EOS_ID       3

/* Audio front end (transformers' ParakeetFeatureExtractor) */
#define P2_SAMPLE_RATE  16000
#define P2_N_FFT        512
#define P2_WIN_LEN      400
#define P2_HOP_LEN      160
#define P2_PREEMPHASIS  0.97f
#define P2_FRAME_SECONDS ((float)(P2_SUB_FACTOR * P2_HOP_LEN) / (float)P2_SAMPLE_RATE)   /* 0.08 s per encoder frame */

/* ── Weight structures ────────────────────────────────────────────────────── */

/* A GEMM weight, stored as op(B) = [K, N]: K input features, N output features, row-major. */
typedef enum {
    P2_W_F32 = 0,   /* fp32 */
    P2_W_I8  = 8,   /* int8 with one fp32 scale per output column */
    P2_W_FV  = 5    /* five-value: per column n the weights are {0, ±lo[n], ±hi[n]} */
} P2WeightKind;

typedef struct {
    P2WeightKind kind;
    int K, N;
    const float         *f32;      /* P2_W_F32: [K, N] */
    const signed char   *q8;       /* P2_W_I8:  [K, N] */
    const float         *scales;   /* P2_W_I8:  [N] */
    const unsigned char *signs;    /* P2_W_FV:  [K, N/4]  2-bit codes, 0 → -1, 1 → 0, 2 → +1; weight n at bits 2*(n&3) */
    const unsigned char *flags;    /* P2_W_FV:  [K, N/8]  bit (n&7) set → magnitude hi[n], clear → lo[n] */
    const float         *lo, *hi;  /* P2_W_FV:  [N] */
    signed char         *codes;    /* P2_W_FV:  the int8w codes pre-packed for the kernel (p2_cache_codes), or NULL */
} Weight;

typedef struct {
    /* Conv2d(1→256, k3, s2, p1) → ReLU → [dw Conv2d(256, k3, s2, p1) → pw Conv2d(256→256) → ReLU] × 2 */
    const float *conv0_w, *conv0_b;   /* [256, 1, 3, 3], [256] */
    const float *conv2_w, *conv2_b;   /* depthwise [256, 1, 3, 3] */
    const float *conv3_w, *conv3_b;   /* pointwise [256, 256] */
    const float *conv5_w, *conv5_b;
    const float *conv6_w, *conv6_b;
    Weight       linear_w;            /* [4096, 1024], int8 per column */
    const float *linear_b;            /* [1024] */
} SubsamplingWeights;

typedef struct {
    Weight linear1_w;   /* [1024, 4096] five-value, no bias */
    Weight linear2_w;   /* [4096, 1024] */
} FFWeights;

typedef struct {
    Weight       qkv_w;     /* [1024, 3072]: q_proj | k_proj | v_proj, fused by the converter */
    Weight       pos_w;     /* [1024, 1024] relative_k_proj */
    Weight       out_w;     /* [1024, 1024] o_proj */
    const float *bias_u;    /* [8, 128] */
    const float *bias_v;    /* [8, 128] */
} MHAWeights;

typedef struct {
    Weight       pw1_w;     /* [1024, 2048] pointwise_conv1 (then GLU) */
    const float *dw_w;      /* [1024, 9] depthwise, with the BatchNorm folded in by the converter */
    const float *dw_b;      /* [1024] */
    Weight       pw2_w;     /* [1024, 1024] pointwise_conv2 */
} ConvModWeights;

typedef struct {
    const float *norm_ff1_w, *norm_ff1_b;       /* [1024] each */
    FFWeights ff1;
    const float *norm_sa_w, *norm_sa_b;
    MHAWeights mha;
    const float *norm_conv_w, *norm_conv_b;
    ConvModWeights conv;
    const float *norm_ff2_w, *norm_ff2_b;
    FFWeights ff2;
    const float *norm_out_w, *norm_out_b;
} ConformerBlock;

/* Chunked encoder inference – splits long sequences into fixed-size chunks to bound the O(T²) attention workspace.
 * Chunk boundaries change the result slightly; P2_CHUNK=0 in the environment forces one full-length pass. */
#define P2_CHUNK_SIZE      400  /* encoder frames per chunk (32 s of audio) */
#define P2_CHUNK_OVERLAP   8    /* context frames on each side */
#define P2_CHUNK_THRESHOLD 500  /* chunk automatically beyond this many frames */

typedef struct {
    SubsamplingWeights subsampling;
    ConformerBlock     blocks[P2_N_BLOCKS];
    Weight             projector_w;   /* [1024, 640] encoder_projector, int8 per column */
    const float       *projector_b;   /* [640] */
} EncoderWeights;

/* The decoder tables are int8 with one fp32 scale per row on disk (exact: the container holds them as int6 with a
 * per-row fp16 scale) and are expanded to fp32 when loaded. Streaming them as int8 was measured slower than fp32 on
 * Apple M-series – the per-element widen/convert/multiply outweighs the bytes saved – so the expansion happens once. */
#define P2_DEC_OWNED 8   /* fp32 buffers the loader allocates for the decoder */

typedef struct {
    const float *embed_w;                        /* [8193, 640]; row 8192 (blank) is all zeros */
    const float *lstm_w_ih[P2_LSTM_LAYERS];      /* [2560, 640]  PyTorch layout [4H, in], gate order i, f, g, o */
    const float *lstm_w_hh[P2_LSTM_LAYERS];      /* [2560, 640] */
    const float *lstm_b_ih[P2_LSTM_LAYERS];      /* [2560] */
    const float *lstm_b_hh[P2_LSTM_LAYERS];      /* [2560] */
    const float *proj_w, *proj_b;                /* decoder_projector [640, 640] ([out, in]), [640] */
    const float *head_w, *head_b;                /* joint head [8198, 640] ([out, in]), [8198] */
    float       *owned[P2_DEC_OWNED];            /* the expanded tables, freed by p2_free_weights */
    int          n_owned;
} DecoderWeights;

typedef struct {
    EncoderWeights enc;
    DecoderWeights dec;
    void          *map;       /* weights.bin, mapped read-only for this model's lifetime */
    size_t         map_len;
} P2Model;

/* ── Decoder state and output ─────────────────────────────────────────────── */

typedef struct {
    float h[P2_LSTM_LAYERS][P2_LSTM_HIDDEN];
    float c[P2_LSTM_LAYERS][P2_LSTM_HIDDEN];
} LSTMState;

typedef struct {
    int id;        /* token id, never blank */
    int frame;     /* encoder frame it was emitted at (× P2_FRAME_SECONDS for seconds) */
    int duration;  /* frames the decoder then advanced, 0..4 */
} P2Token;

/* ── Verbosity ───────────────────────────────────────────────────────────── */

/* Encoder progress on stderr. Default 1. */
extern int p2_verbose;

/* ── Engine (engine.c): the one-call API ─────────────────────────────────── */

typedef struct P2Engine P2Engine;
typedef struct P2Tokens P2Tokens;   /* the detokeniser, further down */

/* weights_dir holds weights.bin, weights.json and tokens.txt from convert_weights.py. Maps the weights, expands the
 * decoder tables (0.02 s) and starts the thread pool. NULL on failure, with the reason on stderr. */
P2Engine *p2_engine_open(const char *weights_dir);
void      p2_engine_close(P2Engine *e);

/* The same from the three files' bytes, for a host without a file system (the wasm build, web/). `bin` is
 * weights.bin and must stay allocated and unchanged for the engine's life – the model reads its tensors from it and
 * never frees it; the manifest and tokens text are parsed and may go afterwards. */
P2Engine *p2_engine_open_mem(const void *bin, size_t bin_len, const char *json, size_t json_len,
                             const char *tokens, size_t tokens_len);

/* A word with its times: pieces starting with ▁ open a word; special ids and blank are dropped. */
typedef struct {
    char *text;    /* UTF-8, owned by the array it sits in */
    float start;   /* seconds: the emission frame of the word's first piece */
    float end;     /* seconds: the end of its last piece's duration (at least one frame) */
} P2Word;

/* One decoded window of a long recording (segment.c): audio over 35 s is cut at pauses into 25–35 s windows that are
 * encoded and decoded on their own, because the model goes silent for tens of seconds when a window begins
 * mid-speech. Audio up to 35 s is one window. */
typedef struct {
    float start, end;            /* seconds in the audio given */
    char *text;                  /* this window's transcript; the result's text is the windows' joined with spaces */
    int   first_word, n_words;   /* its words: P2Result.words[first_word .. first_word + n_words) */
    int   first_token, n_tokens; /* its tokens, likewise */
} P2Segment;

typedef struct {
    char      *text;              /* the transcript, UTF-8 */
    P2Word    *words;
    int        n_words;
    P2Token   *tokens;            /* the decoder's emissions with their frames (in the whole audio) and durations */
    int        n_tokens;
    P2Segment *segments;
    int        n_segments;
    int        encoder_frames;    /* 80 ms each */
    float      duration_seconds;  /* of the audio given */
    double     t_mel, t_encoder, t_decoder;   /* wall seconds spent in each stage */
} P2Result;

/* Transcribe mono float32 samples at any sample rate: resampled to 16 kHz when needed, cut into pause-aligned
 * windows beyond 35 s, each decoded on its own and re-decoded as two when the model went silent in it
 * (P2_SEGMENT=0 turns both off and hands the whole file to the encoder, which then chunks it beyond 40 s – the old
 * behaviour; P2_REDECODE=0 turns off only the re-decode), then per window log-mel, encoder, greedy TDT and
 * detokenisation. Fills *out,
 * freed with p2_result_free. Returns 0, or -1 for an unusable input. Any number of threads may transcribe through
 * one engine at once; each call's GEMMs run on the process pool, or on the calling thread's own pool when it set one
 * (threadpool.h). Thread count, GEMM mode and chunking come from the environment (P2_THREADS, P2_GEMM, P2_CHUNK…)
 * read on first use. */
int  p2_transcribe(const P2Engine *e, const float *audio, int n_samples, int sample_rate, P2Result *out);

/* The same from log-mel features [n_frames × 128] in p2_mel_spectrogram's layout, as one window whatever its
 * length (the encoder chunks beyond 40 s), for feature-level tests. */
int  p2_transcribe_mel(const P2Engine *e, const float *mel, int n_frames, P2Result *out);

void p2_result_free(P2Result *r);

/* The result as JSON (json.c), the one serialisation the CLI's --json and the wasm bindings share: text, words
 * with times, segments, duration_seconds, encoder_frames, tokens, and the stage seconds. A segment with its own
 * words (`words_from` the result holding them, or NULL for the segment alone) for --stream --json. Caller frees. */
char *p2_result_json(const P2Result *r);
char *p2_segment_json(const P2Segment *seg, const P2Result *words_from);

/* The parts, for callers that want to run the stages themselves. */
const P2Model  *p2_engine_model(const P2Engine *e);
const P2Tokens *p2_engine_tokens(const P2Engine *e);

/* ── Streaming (engine.c) ─────────────────────────────────────────────────── */

/* Live transcription at window granularity, the only kind this full-context model allows. Feed mono float32 at the
 * rate given to open (resampled to 16 kHz on the way, chunk by chunk, exactly as p2_transcribe would resample the
 * whole) as it arrives; whenever 35 s have accumulated since the last cut, the window up to the best pause in its 25–35 s
 * band is decoded – the same windows and the same dropout re-decode as p2_transcribe with its default settings, so a
 * stream fed a whole file gives that file's p2_transcribe result (the stream always windows: P2_SEGMENT=0, the
 * whole-file comparison mode, does not apply to it; P2_REDECODE and P2_WINDOW_MAX do) – and its segments become
 * available through p2_stream_next. p2_stream_partial decodes the audio pending since the last cut for a
 * provisional transcript at any time (a full decode of up to 35 s: cheap at 150× realtime, so every second or two
 * is fine). p2_stream_finish decodes what remains. A stream is used from one thread at a time; several streams may
 * run on one engine. */
typedef struct P2Stream P2Stream;

P2Stream *p2_stream_open(const P2Engine *e, int sample_rate);
void      p2_stream_close(P2Stream *s);
/* Append samples; returns how many segments this call added (a window that went quiet is re-decoded as two or more),
 * or -1 on failure. */
int       p2_stream_feed(P2Stream *s, const float *audio, int n_samples);
/* The next segment not yet returned, or NULL. It points into the stream's result and stays valid until the next
 * p2_stream_feed / p2_stream_finish (which may grow the array), so take what you need before feeding again. */
const P2Segment *p2_stream_next(P2Stream *s);
/* Provisional transcript of the pending audio (no re-decode); caller frees. "" when nothing is pending, NULL on
 * failure. */
char     *p2_stream_partial(P2Stream *s);
/* Seconds of audio pending since the last cut. */
double    p2_stream_pending_seconds(const P2Stream *s);
/* Decode the pending audio as the last window; returns how many segments that added, or -1. Ends the stream: a
 * second finish adds nothing, a feed after it fails. */
int       p2_stream_finish(P2Stream *s);
/* Everything so far: segments, tokens and words always current, text and duration complete after finish. */
const P2Result *p2_stream_result(const P2Stream *s);

/* ── Audio input ──────────────────────────────────────────────────────────── */

/* WAV reader (wav.c): PCM 8/16/24/32-bit or 32-bit float, plain or WAVE_FORMAT_EXTENSIBLE, any channel count,
 * averaged to mono. Returns float32 in [-1, 1) at the file's own rate, which p2_transcribe accepts as is. Caller
 * frees *out_samples. Returns 0, or -1 with a message on stderr. */
int  p2_wav_read(const char *path, float **out_samples, int *out_count, int *out_sr, int *out_channels);

/* Raw 16-bit little-endian PCM (what a WAV's data chunk, a socket or `--stream` carries) → float32 in [-1, 1), on
 * any host. `channels` interleaved channels are averaged to mono; out holds n_frames floats. */
void p2_pcm16le_to_float(const unsigned char *bytes, int n_frames, int channels, float *out);

/* Sample-rate conversion (resample.c): mono float32 at sr_in → sr_out, windowed-sinc polyphase. Caller frees *out.
 * Returns 0, or -1 for an unusable ratio. */
int  p2_resample(const float *in, int n_in, int sr_in, int sr_out, float **out, int *n_out);

/* The same conversion for audio that arrives in pieces: feed any chunk sizes, get the output that is complete so
 * far (caller frees each *out), finish for the tail. The pieces concatenated are bit for bit what p2_resample gives
 * for the whole – p2_resample is open → feed → finish. NULL from open for an unusable ratio. */
typedef struct P2Resampler P2Resampler;
P2Resampler *p2_resampler_open(int sr_in, int sr_out);
void         p2_resampler_close(P2Resampler *r);
int          p2_resampler_feed(P2Resampler *r, const float *in, int n_in, float **out, int *n_out);
int          p2_resampler_finish(P2Resampler *r, float **out, int *n_out);

/* Window plan (segment.c): contiguous [start, end) sample ranges covering the audio – one for anything up to 35 s,
 * otherwise 25–35 s windows cut in the longest pause of each band (Fermion's gate), or at 30 s without one.
 * Returns the count; caller frees *out. */
typedef struct { int start, end; } P2Window;
int  p2_plan_windows(const float *audio, int n_samples, int sample_rate, P2Window **out);
/* The first window's end for audio starting at a window start: n_samples itself when that is at most the maximum
 * window (35 s, P2_WINDOW_MAX), else the cut inside the band; -1 when out of memory (never a fixed cut). The
 * planner and the stream both build on this. */
int  p2_first_cut(const float *audio, int n_samples, int sample_rate);
int  p2_window_max_samples(int sample_rate);
/* Where to split a window that went quiet: the middle of the longest pause in its middle third, else its midpoint. */
int  p2_split_point(const float *audio, int n_samples, int sample_rate);
/* Energy measures for the dropout check: 50 ms block RMS of the audio (caller frees; NULL when out of memory), the
 * window's speech gate (3 × its 10th-percentile block RMS, at least the room-tone floor; -1 when out of memory),
 * and the share of the blocks covering samples [from, to) that are above a gate. */
float *p2_block_rms(const float *audio, int n_samples, int sample_rate, int *n_blocks);
float  p2_speech_gate(const float *rms, int n_blocks);
float  p2_speech_share(const float *rms, int n_blocks, int sample_rate, int from, int to, float gate);

/* ── Component API ───────────────────────────────────────────────────────── */

/* Load weights.bin + weights.json (from convert_weights.py). Returns 0 on success. With P2_CACHE_CODES=1 in the
 * environment the loader also calls p2_cache_codes. */
int  p2_load_weights(P2Model *m, const char *bin_path, const char *json_path);
/* The same from the manifest text and the bytes of weights.bin, which must outlive the model (never freed by it). */
int  p2_load_weights_mem(P2Model *m, const char *json, size_t json_len, const void *bin, size_t bin_len);
void p2_free_weights(P2Model *m);

/* Pre-pack every five-value weight's int8w codes in the GEMM kernel's layout (K × N bytes each, 576 MB in all), so
 * that the default mode skips the per-GEMM expansion of the sign and flag planes – most of a short utterance's
 * encoder time, nothing on long audio. Returns the bytes allocated, 0 when there is no int8 path or memory ran
 * out (the planes are then used as before). Idempotent; p2_free_weights frees the cache. */
size_t p2_cache_codes(P2Model *m);

/* Log-mel features: mono float32 16 kHz audio → mel [n_frames × 128], time-major, normalised per feature.
 * n_frames = n_samples / 160, exactly the frames transformers' extractor marks valid. Caller frees *out_mel. */
int  p2_mel_spectrogram(const float *audio, int n_samples, float **out_mel, int *out_frames);

/* Encoder: mel [T × 128] → projected frames [T' × 640] ready for the joint network. Caller frees *out. */
int  p2_encoder(const P2Model *m, const float *mel, int n_frames, float **out, int *out_len);

/* Force the chunked encoder regardless of length. Same contract. */
int  p2_encoder_chunked(const P2Model *m, const float *mel, int n_frames, float **out, int *out_len);

/* One prediction-network step: token → projected decoder output [640]. Mutates state. */
void p2_decoder_step(const P2Model *m, int token_id, LSTMState *state, float *dec_out);

/* Joint: one encoder frame [640] + decoder output [640] → logits [8198] (8193 tokens, then 5 durations). */
void p2_joint(const P2Model *m, const float *enc_t, const float *dec_out, float *logits);

/* Greedy TDT decode over enc_out [T × 640]. Returns the token count; caller frees *out_tokens. */
int  p2_greedy_decode(const P2Model *m, const float *enc_out, int T, P2Token **out_tokens);

/* ── Detokeniser (tokens.c) ───────────────────────────────────────────────── */

/* tokens.txt: one SentencePiece piece per line, the line number is the id. NULL on failure. */
P2Tokens *p2_tokens_load(const char *path);
P2Tokens *p2_tokens_parse(const char *text, size_t len);   /* the same from the file's bytes */
void      p2_tokens_free(P2Tokens *t);
/* The piece for an id, or NULL. */
const char *p2_tokens_piece(const P2Tokens *t, int id);
/* Text for a token sequence: special ids dropped, ▁ → space, leading space stripped. Caller frees. */
char *p2_tokens_text(const P2Tokens *t, const P2Token *tokens, int n);
/* Words with times (P2Word above). Returns the count; *out is freed with p2_words_free. The times are emission
 * frames, which run a few frames behind the acoustics. */
int   p2_tokens_words(const P2Tokens *t, const P2Token *tokens, int n, P2Word **out);
void  p2_words_free(P2Word *words, int n);

/* ── Tensor ops (tensor_ops.c) ────────────────────────────────────────────── */

/* C[M, N] = alpha · A[M, K] @ W + beta · C, for any weight kind. lda = K, ldc = N. */
void p2_wmatmul(const float *A, const Weight *W, float *C, int M, float alpha, float beta);

/* Activations already quantised for the int8 GEMMs: int8 rows with one scale each, in the kernel's tile layout, in a
 * buffer that belongs to the calling thread. p2_quant_prepare (blas.h) hands one out; the next prepare on the same
 * thread may reuse the buffer, so a QuantA is consumed by one GEMM before another is prepared. */
typedef struct {
    signed char *q;
    float       *scales;
    int          M, K;
} QuantA;

/* C[M, N] = alpha · A @ W + beta · C from pre-quantised A; W must be a five-value or int8 table. */
void p2_wmatmul_q8a(const QuantA *A, const Weight *W, float *C, float alpha, float beta);

/* LayerNorm of each row straight into a QuantA (no fp32 copy of the normalised rows). D ≤ P2_D_MODEL. */
void p2_layer_norm_quant_rows(const float *x, const float *gamma, const float *beta_, const QuantA *qa, int n_rows, int D);

/* SiLU of each row straight into a QuantA. len ≤ P2_FF_DIM. */
void p2_swish_quant_rows(const float *h, const QuantA *qa, int n_rows, int len);

/* Dot product of two fp32 vectors (SIMD). */
float p2_dot_f32(const float *a, const float *b, int n);

/* y[M] = W[M, K] @ x[K] for a row-major fp32 matrix; parallel across rows when the matrix is large. */
void p2_gemv_rows(const float *W, const float *x, float *y, int M, int K);

void p2_layer_norm(const float *x, const float *gamma, const float *beta_, float *out, int len);
void p2_layer_norm_rows(const float *x, const float *gamma, const float *beta_, float *out, int n_rows, int D);
void p2_layer_norm_rows_inplace(float *x, const float *gamma, const float *beta_, int n_rows, int D);

void p2_swish(float *x, int len);            /* x · sigmoid(x) */
void p2_relu(float *x, int len);
void p2_softmax(float *x, int len);
void p2_softmax_rows(float *x, int n_rows, int len);
void p2_bias_add(float *x, const float *b, int len);
void p2_bias_add_rows(float *x, const float *b, int n_rows, int len);

/* Depthwise 1-D convolution, symmetric padding (K-1)/2. x, out: [C × T]; w: [C × K]; b: [C] or NULL. */
void p2_depthwise_conv1d(const float *x, const float *w, const float *b, float *out, int C, int T, int K);

#endif /* PHONON2_H */
