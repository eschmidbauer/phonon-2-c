/*  weights.c – Load weights.bin + weights.json into a P2Model.
 *
 *  convert_weights.py writes both files. weights.json maps a tensor name to an object with a "dtype" and byte
 *  offsets into weights.bin (every tensor 64-byte aligned):
 *    "float32"     offset, shape, numel
 *    "int8"        offset, scales_offset, shape – one fp32 scale per output feature: GEMM tables are stored
 *                  [in, out] (scale per column) and used as they are; decoder tables are stored [out, in] (scale
 *                  per row) and expanded to fp32 here
 *    "five_value"  signs_offset ([K, N/4]), flags_offset ([K, N/8]), lo_offset, hi_offset ([N] fp32 each), shape
 *  The names looked up below are the ones the converter emits; change them in both places.
 *
 *  Loading is strict: a missing tensor or a shape that does not match the model constants fails the load.
 */

#include "phonon2.h"
#include "blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ── Manifest entries ────────────────────────────────────────────────────── */

typedef enum { DT_F32 = 0, DT_I8 = 8, DT_FV = 5 } DType;

typedef struct {
    char     *name;
    char     *format;   /* the "_meta" entry's "format" string; NULL elsewhere */
    DType     dtype;
    long long offset, numel;
    long long scales_offset, signs_offset, flags_offset, lo_offset, hi_offset;
    long long shape[4];
    int       ndim;
} WEntry;

typedef struct {
    WEntry     *entries;
    int         n;
    const char *base;
    size_t      size;
    int         errors;
} Manifest;

/* ── Minimal JSON parser for the manifest's shape ────────────────────────── */

static int skip_ws(const char *s, int i)
{
    while (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t') i++;
    return i;
}

/* "…" → pointer and length of the raw contents (no escape processing: tensor names carry none). -1 when there is no
 * string at i: a manifest cut off right after `"dtype":` once sent this one byte past the buffer. */
static int parse_string(const char *s, int i, const char **start, int *len)
{
    if (s[i] != '"') { *start = s + i; *len = 0; return -1; }
    i++;
    *start = s + i;
    int j = i;
    while (s[j] && s[j] != '"') {
        if (s[j] == '\\' && s[j + 1]) j++;
        j++;
    }
    *len = j - i;
    return s[j] ? j + 1 : j;
}

/* Returns the index after the number, or -1 when there is no number at i – every caller must stop then: a loop that
 * kept going on a non-number spun forever and overran the entry (found by tests/fuzz). */
static int parse_number(const char *s, int i, double *val)
{
    char *end;
    *val = strtod(s + i, &end);
    return end == s + i ? -1 : (int)(end - s);
}

/* A manifest integer: anything a 64-bit offset or shape cannot hold becomes -1, which every range check rejects
 * (the plain cast of an out-of-range double is undefined behaviour). */
static long long to_ll(double v)
{
    return (v >= -9.0e18 && v <= 9.0e18) ? (long long)v : -1;
}

static int skip_value(const char *s, int i);

static int skip_container(const char *s, int i, char open, char close)
{
    int depth = 0;
    do {
        if (s[i] == '"') {
            const char *d; int dl;
            i = parse_string(s, i, &d, &dl);
            continue;
        }
        if (s[i] == open) depth++;
        else if (s[i] == close) depth--;
        if (!s[i]) break;
        i++;
    } while (depth > 0);
    return i;
}

static int skip_value(const char *s, int i)
{
    i = skip_ws(s, i);
    if (s[i] == '"') { const char *d; int dl; return parse_string(s, i, &d, &dl); }
    if (s[i] == '[') return skip_container(s, i, '[', ']');
    if (s[i] == '{') return skip_container(s, i, '{', '}');
    while (s[i] && s[i] != ',' && s[i] != '}' && s[i] != ']') i++;
    return i;
}

static int key_is(const char *k, int kl, const char *lit)
{
    return (int)strlen(lit) == kl && strncmp(k, lit, kl) == 0;
}

/* Parse one tensor object { "key": value, ... } into *e. */
static int parse_entry(const char *s, int i, WEntry *e)
{
    i = skip_ws(s, i);
    if (s[i] != '{') return -1;
    i++;
    for (;;) {
        i = skip_ws(s, i);
        if (s[i] == '}') return i + 1;
        if (s[i] == ',') { i++; continue; }
        if (s[i] != '"') return -1;
        const char *k; int kl;
        i = parse_string(s, i, &k, &kl);
        i = skip_ws(s, i);
        if (s[i] != ':') return -1;
        i = skip_ws(s, i + 1);

        if (key_is(k, kl, "dtype")) {
            const char *v; int vl;
            i = parse_string(s, i, &v, &vl);
            if (i < 0) return -1;
            if (key_is(v, vl, "int8")) e->dtype = DT_I8;
            else if (key_is(v, vl, "five_value")) e->dtype = DT_FV;
            else e->dtype = DT_F32;
        } else if (key_is(k, kl, "format")) {   /* "_meta": the layout the converter wrote, checked after parsing */
            const char *v; int vl;
            i = parse_string(s, i, &v, &vl);
            if (i < 0) return -1;
            if (vl > 63) vl = 63;
            free(e->format);
            e->format = (char *)malloc(vl + 1);
            if (!e->format) return -1;
            memcpy(e->format, v, vl);
            e->format[vl] = '\0';
        } else if (key_is(k, kl, "shape")) {
            if (s[i] != '[') return -1;
            i++;
            e->ndim = 0;
            for (;;) {
                i = skip_ws(s, i);
                if (s[i] == ']') { i++; break; }
                if (s[i] == ',') { i++; continue; }
                double v;
                i = parse_number(s, i, &v);
                if (i < 0 || e->ndim >= 4) return -1;   /* not a number, or more dimensions than any tensor has */
                e->shape[e->ndim++] = to_ll(v);
            }
        } else {
            double v = 0;
            int is_num = (s[i] == '-' || (s[i] >= '0' && s[i] <= '9'));
            if (is_num) i = parse_number(s, i, &v); else i = skip_value(s, i);
            if (i < 0) return -1;
            long long ll = to_ll(v);
            if      (key_is(k, kl, "offset"))        e->offset = ll;
            else if (key_is(k, kl, "numel"))         e->numel = ll;
            else if (key_is(k, kl, "scales_offset")) e->scales_offset = ll;
            else if (key_is(k, kl, "signs_offset"))  e->signs_offset = ll;
            else if (key_is(k, kl, "flags_offset"))  e->flags_offset = ll;
            else if (key_is(k, kl, "lo_offset"))     e->lo_offset = ll;
            else if (key_is(k, kl, "hi_offset"))     e->hi_offset = ll;
        }
    }
}

static WEntry *parse_manifest(const char *json, int *count)
{
    int cap = 1024, n = 0;
    WEntry *entries = (WEntry *)calloc(cap, sizeof(WEntry));
    if (!entries) return NULL;

    int i = skip_ws(json, 0);
    if (json[i] != '{') { free(entries); return NULL; }
    i++;
    for (;;) {
        i = skip_ws(json, i);
        if (json[i] == '}' || !json[i]) break;
        if (json[i] == ',') { i++; continue; }
        if (json[i] != '"') { free(entries); return NULL; }

        const char *name_start; int name_len;
        i = parse_string(json, i, &name_start, &name_len);
        i = skip_ws(json, i);
        if (json[i] != ':') { free(entries); return NULL; }
        i = skip_ws(json, i + 1);

        if (n >= cap) {
            WEntry *grown = (WEntry *)realloc(entries, (size_t)cap * 2 * sizeof(WEntry));
            if (!grown) { *count = n; return entries; }   /* out of memory: what was parsed; the lookups report the rest */
            entries = grown;
            cap *= 2;
            memset(entries + n, 0, (size_t)(cap - n) * sizeof(WEntry));
        }
        WEntry *e = &entries[n];
        memset(e, 0, sizeof(*e));
        e->name = (char *)malloc(name_len + 1);
        if (!e->name) { *count = n; return entries; }
        memcpy(e->name, name_start, name_len);
        e->name[name_len] = '\0';
        e->offset = -1;

        int j = parse_entry(json, i, e);
        if (j < 0) {   /* not a tensor object: skip the value */
            free(e->name);
            free(e->format);
            e->name = e->format = NULL;
            i = skip_value(json, i);
            continue;
        }
        i = j;
        n++;
    }
    *count = n;
    return entries;
}

/* ── Lookups ─────────────────────────────────────────────────────────────── */

static void free_entries(Manifest *mf)
{
    for (int i = 0; i < mf->n; i++) { free(mf->entries[i].name); free(mf->entries[i].format); }
    free(mf->entries);
    mf->entries = NULL;
    mf->n = 0;
}

static const WEntry *find_entry(const Manifest *mf, const char *name)
{
    for (int i = 0; i < mf->n; i++)
        if (strcmp(mf->entries[i].name, name) == 0) return &mf->entries[i];
    return NULL;
}

static int in_range(const Manifest *mf, long long off, size_t bytes)
{
    return off >= 0 && (size_t)off + bytes <= mf->size;
}

/* Float arrays are read as float: the offset must be 4-byte aligned as well (the converter aligns every tensor to
 * 64 bytes; a mutated manifest once pointed a float array at an odd byte – undefined behaviour, found by tests/fuzz). */
static int in_range_f32(const Manifest *mf, long long off, size_t bytes)
{
    return in_range(mf, off, bytes) && (off % 4) == 0;
}

static const char *dtype_name(DType d)
{
    return d == DT_I8 ? "int8" : d == DT_FV ? "five_value" : "float32";
}

/* fp32 tensor with exactly `numel` elements. */
static const float *lookup_f32(Manifest *mf, const char *name, long long numel)
{
    const WEntry *e = find_entry(mf, name);
    if (!e) { fprintf(stderr, "weights: '%s' missing\n", name); mf->errors++; return NULL; }
    if (e->dtype != DT_F32) {
        fprintf(stderr, "weights: '%s' is %s, float32 expected\n", name, dtype_name(e->dtype));
        mf->errors++; return NULL;
    }
    if (e->numel != numel || !in_range_f32(mf, e->offset, (size_t)numel * sizeof(float))) {
        fprintf(stderr, "weights: '%s' has %lld elements, %lld expected\n", name, e->numel, numel);
        mf->errors++; return NULL;
    }
    return (const float *)(mf->base + e->offset);
}

/* GEMM weight of logical shape [K, N] in any kind. */
static void lookup_weight(Manifest *mf, const char *name, Weight *w, int K, int N)
{
    memset(w, 0, sizeof(*w));
    w->K = K; w->N = N;
    const WEntry *e = find_entry(mf, name);
    if (!e) { fprintf(stderr, "weights: '%s' missing\n", name); mf->errors++; return; }
    if (e->ndim != 2 || e->shape[0] != K || e->shape[1] != N) {
        fprintf(stderr, "weights: '%s' has shape [%lld, %lld], [%d, %d] expected\n",
                name, e->ndim > 0 ? e->shape[0] : -1, e->ndim > 1 ? e->shape[1] : -1, K, N);
        mf->errors++; return;
    }
    size_t KN = (size_t)K * N;
    switch (e->dtype) {
    case DT_FV:
        if (N % 16 != 0 ||
            !in_range(mf, e->signs_offset, KN / 4) || !in_range(mf, e->flags_offset, KN / 8) ||
            !in_range_f32(mf, e->lo_offset, N * sizeof(float)) || !in_range_f32(mf, e->hi_offset, N * sizeof(float))) {
            fprintf(stderr, "weights: '%s' five-value planes out of range\n", name);
            mf->errors++; return;
        }
        w->kind  = P2_W_FV;
        w->signs = (const unsigned char *)(mf->base + e->signs_offset);
        w->flags = (const unsigned char *)(mf->base + e->flags_offset);
        w->lo    = (const float *)(mf->base + e->lo_offset);
        w->hi    = (const float *)(mf->base + e->hi_offset);
        break;
    case DT_I8:
        if (N % 16 != 0 || !in_range(mf, e->offset, KN) || !in_range_f32(mf, e->scales_offset, N * sizeof(float))) {
            fprintf(stderr, "weights: '%s' int8 data out of range\n", name);
            mf->errors++; return;
        }
        w->kind   = P2_W_I8;
        w->q8     = (const signed char *)(mf->base + e->offset);
        w->scales = (const float *)(mf->base + e->scales_offset);
        break;
    default:
        if (!in_range_f32(mf, e->offset, KN * sizeof(float))) {
            fprintf(stderr, "weights: '%s' float32 data out of range\n", name);
            mf->errors++; return;
        }
        w->kind = P2_W_F32;
        w->f32  = (const float *)(mf->base + e->offset);
        break;
    }
}

/* int8 table [rows, cols] with one scale per row, expanded to an fp32 buffer that the model owns: w = q · scale, the
 * same fp32 values the converter would have written. */
static const float *expand_q8_rows(Manifest *mf, DecoderWeights *dec, const char *name, int rows, int cols)
{
    const WEntry *e = find_entry(mf, name);
    if (!e) { fprintf(stderr, "weights: '%s' missing\n", name); mf->errors++; return NULL; }
    if (e->dtype != DT_I8 || e->ndim != 2 || e->shape[0] != rows || e->shape[1] != cols) {
        fprintf(stderr, "weights: '%s' is %s [%lld, %lld], int8 [%d, %d] expected\n", name, dtype_name(e->dtype),
                e->ndim > 0 ? e->shape[0] : -1, e->ndim > 1 ? e->shape[1] : -1, rows, cols);
        mf->errors++; return NULL;
    }
    if (!in_range(mf, e->offset, (size_t)rows * cols) || !in_range_f32(mf, e->scales_offset, (size_t)rows * sizeof(float))) {
        fprintf(stderr, "weights: '%s' int8 data out of range\n", name);
        mf->errors++; return NULL;
    }
    if (dec->n_owned >= P2_DEC_OWNED) {
        fprintf(stderr, "weights: too many expanded decoder tables\n");
        mf->errors++; return NULL;
    }
    const signed char *q = (const signed char *)(mf->base + e->offset);
    const float *scales = (const float *)(mf->base + e->scales_offset);
    float *w = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!w) { fprintf(stderr, "weights: out of memory expanding '%s'\n", name); mf->errors++; return NULL; }
    for (int r = 0; r < rows; r++) {
        float s = scales[r];
        const signed char *qr = q + (size_t)r * cols;
        float *wr = w + (size_t)r * cols;
        for (int c = 0; c < cols; c++) wr[c] = (float)qr[c] * s;
    }
    dec->owned[dec->n_owned++] = w;
    return w;
}

/* ── Loading ─────────────────────────────────────────────────────────────── */

/* Pointers to the model's five-value weights: nine per conformer block. */
#define P2_N_FV (P2_N_BLOCKS * 9)

static void fv_weights(P2Model *m, Weight *out[P2_N_FV])
{
    for (int i = 0; i < P2_N_BLOCKS; i++) {
        ConformerBlock *b = &m->enc.blocks[i];
        Weight *w[9] = { &b->ff1.linear1_w, &b->ff1.linear2_w, &b->mha.qkv_w, &b->mha.pos_w, &b->mha.out_w,
                         &b->conv.pw1_w, &b->conv.pw2_w, &b->ff2.linear1_w, &b->ff2.linear2_w };
        for (int j = 0; j < 9; j++) out[9 * i + j] = w[j];
    }
}

size_t p2_cache_codes(P2Model *m)
{
    Weight *fv[P2_N_FV];
    fv_weights(m, fv);
    size_t bytes = 0;
    for (int i = 0; i < P2_N_FV; i++) {
        Weight *w = fv[i];
        if (w->kind != P2_W_FV || w->codes) continue;
        FiveValueB b = { w->signs, w->flags, w->lo, w->hi, NULL };
        w->codes = p2_fv_pack_codes(&b, w->K, w->N);
        if (!w->codes) break;   /* no int8 path, or out of memory: the rest keeps expanding the planes */
        bytes += (size_t)w->K * w->N;
    }
    return bytes;
}

static void free_codes(P2Model *m)
{
    Weight *fv[P2_N_FV];
    fv_weights(m, fv);
    for (int i = 0; i < P2_N_FV; i++) {
        free(fv[i]->codes);
        fv[i]->codes = NULL;
    }
}

/* The loader proper, over the manifest text (NUL-terminated) and the bytes of weights.bin; `label` names the
 * manifest in messages. The model's tensor pointers point into `bin`, which must outlive it: p2_load_weights hands
 * its own mapping (released by p2_free_weights), p2_load_weights_mem a caller's buffer that the model never
 * frees. On failure whatever the loader allocated is released; the map, if any, is the wrapper's to drop. */
static int load_from(P2Model *m, const char *json, const char *label, const void *bin, size_t bin_len)
{
    Manifest mf;
    memset(&mf, 0, sizeof(mf));
    mf.entries = parse_manifest(json, &mf.n);
    if (!mf.entries) { fprintf(stderr, "%s: cannot parse\n", label); return -1; }

    /* The layout must be the one this loader was written for: a weights.json from another converter version, or
     * one without "_meta" at all, is refused here rather than read as garbage. */
    const WEntry *meta = find_entry(&mf, "_meta");
    const char *format = meta && meta->format ? meta->format : "(none)";
    if (strcmp(format, P2_WEIGHTS_FORMAT) != 0) {
        fprintf(stderr, "%s: weights format \"%s\", this engine reads \"%s\"; fetch the matching archive "
                        "(scripts/get_weights.sh) or rebuild with convert_weights.py\n", label, format, P2_WEIGHTS_FORMAT);
        free_entries(&mf);
        return -1;
    }

    mf.base = (const char *)bin;
    mf.size = bin_len;

    int n_fv = 0, n_i8 = 0;
    for (int i = 0; i < mf.n; i++) {
        if (mf.entries[i].dtype == DT_FV) n_fv++;
        else if (mf.entries[i].dtype == DT_I8) n_i8++;
    }
    if (p2_verbose)
        fprintf(stderr, "weights: %d tensors (%d five-value, %d int8, %d fp32), %.0f MB\n",
                mf.n, n_fv, n_i8, mf.n - n_fv - n_i8, bin_len / 1e6);

    const int D = P2_D_MODEL, FF = P2_FF_DIM, C = P2_SUB_CHANNELS, H = P2_LSTM_HIDDEN;

#define F(field, name, numel) m->field = lookup_f32(&mf, name, numel)
#define W(field, name, K, N)  lookup_weight(&mf, name, &m->field, K, N)

    F(enc.subsampling.conv0_w, "subsampling.conv.0.weight", (long long)C * 9);
    F(enc.subsampling.conv0_b, "subsampling.conv.0.bias",   C);
    F(enc.subsampling.conv2_w, "subsampling.conv.2.weight", (long long)C * 9);
    F(enc.subsampling.conv2_b, "subsampling.conv.2.bias",   C);
    F(enc.subsampling.conv3_w, "subsampling.conv.3.weight", (long long)C * C);
    F(enc.subsampling.conv3_b, "subsampling.conv.3.bias",   C);
    F(enc.subsampling.conv5_w, "subsampling.conv.5.weight", (long long)C * 9);
    F(enc.subsampling.conv5_b, "subsampling.conv.5.bias",   C);
    F(enc.subsampling.conv6_w, "subsampling.conv.6.weight", (long long)C * C);
    F(enc.subsampling.conv6_b, "subsampling.conv.6.bias",   C);
    W(enc.subsampling.linear_w, "subsampling.linear.weight", C * (P2_N_MELS / P2_SUB_FACTOR), D);
    F(enc.subsampling.linear_b, "subsampling.linear.bias", D);

    for (int i = 0; i < P2_N_BLOCKS; i++) {
        ConformerBlock *b = &m->enc.blocks[i];
        char buf[256];
#define BF(field, fmt, numel) do { snprintf(buf, sizeof(buf), fmt, i); b->field = lookup_f32(&mf, buf, numel); } while (0)
#define BW(field, fmt, K, N)  do { snprintf(buf, sizeof(buf), fmt, i); lookup_weight(&mf, buf, &b->field, K, N); } while (0)

        BF(norm_ff1_w, "layers.%d.norm_feed_forward1.weight", D);
        BF(norm_ff1_b, "layers.%d.norm_feed_forward1.bias",   D);
        BW(ff1.linear1_w, "layers.%d.feed_forward1.linear1.weight", D, FF);
        BW(ff1.linear2_w, "layers.%d.feed_forward1.linear2.weight", FF, D);

        BF(norm_sa_w, "layers.%d.norm_self_att.weight", D);
        BF(norm_sa_b, "layers.%d.norm_self_att.bias",   D);
        BW(mha.qkv_w, "layers.%d.self_attn.qkv_proj.weight",        D, 3 * D);
        BW(mha.pos_w, "layers.%d.self_attn.relative_k_proj.weight", D, D);
        BW(mha.out_w, "layers.%d.self_attn.o_proj.weight",          D, D);
        BF(mha.bias_u, "layers.%d.self_attn.bias_u", D);
        BF(mha.bias_v, "layers.%d.self_attn.bias_v", D);

        BF(norm_conv_w, "layers.%d.norm_conv.weight", D);
        BF(norm_conv_b, "layers.%d.norm_conv.bias",   D);
        BW(conv.pw1_w, "layers.%d.conv.pointwise_conv1.weight", D, 2 * D);
        BF(conv.dw_w,  "layers.%d.conv.depthwise_conv.weight",  (long long)D * P2_CONV_KERNEL);
        BF(conv.dw_b,  "layers.%d.conv.depthwise_conv.bias",    D);
        BW(conv.pw2_w, "layers.%d.conv.pointwise_conv2.weight", D, D);

        BF(norm_ff2_w, "layers.%d.norm_feed_forward2.weight", D);
        BF(norm_ff2_b, "layers.%d.norm_feed_forward2.bias",   D);
        BW(ff2.linear1_w, "layers.%d.feed_forward2.linear1.weight", D, FF);
        BW(ff2.linear2_w, "layers.%d.feed_forward2.linear2.weight", FF, D);

        BF(norm_out_w, "layers.%d.norm_out.weight", D);
        BF(norm_out_b, "layers.%d.norm_out.bias",   D);
#undef BF
#undef BW
    }

    W(enc.projector_w, "encoder_projector.weight", D, P2_JOINT_DIM);
    F(enc.projector_b, "encoder_projector.bias", P2_JOINT_DIM);

    DecoderWeights *dec = &m->dec;
    dec->embed_w = expand_q8_rows(&mf, dec, "decoder.embedding.weight", P2_VOCAB, P2_EMBED_DIM);
    for (int l = 0; l < P2_LSTM_LAYERS; l++) {
        char buf[128];
        int in = l == 0 ? P2_EMBED_DIM : H;
        snprintf(buf, sizeof(buf), "decoder.lstm.weight_ih_l%d", l); dec->lstm_w_ih[l] = expand_q8_rows(&mf, dec, buf, 4 * H, in);
        snprintf(buf, sizeof(buf), "decoder.lstm.weight_hh_l%d", l); dec->lstm_w_hh[l] = expand_q8_rows(&mf, dec, buf, 4 * H, H);
        snprintf(buf, sizeof(buf), "decoder.lstm.bias_ih_l%d",   l); dec->lstm_b_ih[l] = lookup_f32(&mf, buf, 4 * H);
        snprintf(buf, sizeof(buf), "decoder.lstm.bias_hh_l%d",   l); dec->lstm_b_hh[l] = lookup_f32(&mf, buf, 4 * H);
    }
    dec->proj_w = expand_q8_rows(&mf, dec, "decoder.decoder_projector.weight", P2_JOINT_DIM, H);
    F(dec.proj_b, "decoder.decoder_projector.bias", P2_JOINT_DIM);
    dec->head_w = expand_q8_rows(&mf, dec, "joint.head.weight", P2_N_LOGITS, P2_JOINT_DIM);
    F(dec.head_b, "joint.head.bias", P2_N_LOGITS);

#undef F
#undef W

    free_entries(&mf);

    if (mf.errors) {
        fprintf(stderr, "weights: %d problems, refusing to run\n", mf.errors);
        p2_free_weights(m);
        return -1;
    }

    const char *cache = getenv("P2_CACHE_CODES");
    if (cache && cache[0] && cache[0] != '0') {
        size_t bytes = p2_cache_codes(m);
        if (p2_verbose) fprintf(stderr, "weights: int8 codes pre-packed, %.0f MB\n", bytes / 1e6);
    }
    return 0;
}

int p2_load_weights(P2Model *m, const char *bin_path, const char *json_path)
{
    memset(m, 0, sizeof(*m));
    FILE *jf = fopen(json_path, "rb");
    if (!jf) { perror(json_path); return -1; }
    fseek(jf, 0, SEEK_END);
    long jlen = ftell(jf);
    fseek(jf, 0, SEEK_SET);
    char *json = (char *)malloc(jlen + 1);
    if (fread(json, 1, jlen, jf) != (size_t)jlen) { perror(json_path); fclose(jf); free(json); return -1; }
    json[jlen] = '\0';
    fclose(jf);

    int fd = open(bin_path, O_RDONLY);
    if (fd < 0) { perror(bin_path); free(json); return -1; }
    struct stat st;
    fstat(fd, &st);
    m->map_len = (size_t)st.st_size;
    m->map = mmap(NULL, m->map_len, PROT_READ, MAP_PRIVATE, fd, 0);   /* this model's own mapping */
    close(fd);
    if (m->map == MAP_FAILED) { perror("mmap"); m->map = NULL; m->map_len = 0; free(json); return -1; }

    int rc = load_from(m, json, json_path, m->map, m->map_len);
    free(json);
    if (rc != 0) p2_free_weights(m);   /* drops the mapping; load_from released the rest */
    return rc;
}

int p2_load_weights_mem(P2Model *m, const char *json, size_t json_len, const void *bin, size_t bin_len)
{
    memset(m, 0, sizeof(*m));
    char *text = (char *)malloc(json_len + 1);   /* the manifest parser wants a NUL-terminated string */
    if (!text) return -1;
    memcpy(text, json, json_len);
    text[json_len] = '\0';
    int rc = load_from(m, text, "weights.json", bin, bin_len);
    free(text);
    return rc;
}

void p2_free_weights(P2Model *m)
{
    free_codes(m);
    for (int i = 0; i < m->dec.n_owned; i++) free(m->dec.owned[i]);
    m->dec.n_owned = 0;
    if (m->map) {
        munmap(m->map, m->map_len);
        m->map = NULL;
    }
}
