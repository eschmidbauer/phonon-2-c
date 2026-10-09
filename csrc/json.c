/*  json.c – JSON for the result types: the one serialisation the CLI's --json and --stream --json and the wasm
 *  bindings (web/wasm/bindings.c) share, so a consumer that cannot read a C struct gets the same fields as one that
 *  can. {"text":…,"words":[{"text":…,"start":…,"end":…},…],"segments":[{"start":…,"end":…,"text":…},…],
 *  "duration_seconds":…,"encoder_frames":…,"tokens":…,"seconds":{"mel":…,"encoder":…,"decoder":…}}. Strings are
 *  escaped as RFC 8259 asks (quote, backslash, controls); the text is UTF-8 and passes through byte for byte.
 */

#include "phonon2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Small growable string ───────────────────────────────────────────────── */

typedef struct { char *s; size_t len, cap; } Buf;

static void buf_put(Buf *b, const char *s)
{
    size_t n = strlen(s);
    if (b->len + n + 1 > b->cap) {
        b->cap = (b->len + n + 1) * 2;
        b->s = (char *)realloc(b->s, b->cap);
    }
    memcpy(b->s + b->len, s, n + 1);
    b->len += n;
}

static void buf_put_json_string(Buf *b, const char *s)
{
    buf_put(b, "\"");
    char tmp[8];
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { tmp[0] = '\\'; tmp[1] = (char)c; tmp[2] = 0; buf_put(b, tmp); }
        else if (c < 0x20) { snprintf(tmp, sizeof(tmp), "\\u%04x", c); buf_put(b, tmp); }
        else { tmp[0] = (char)c; tmp[1] = 0; buf_put(b, tmp); }
    }
    buf_put(b, "\"");
}

/* [{"text":…,"start":…,"end":…}, …] for words[first .. first + count). */
static void buf_put_words_json(Buf *b, const P2Word *words, int first, int count)
{
    char num[96];
    buf_put(b, "[");
    for (int i = first; i < first + count; i++) {
        buf_put(b, i > first ? ",{\"text\":" : "{\"text\":");
        buf_put_json_string(b, words[i].text);
        snprintf(num, sizeof(num), ",\"start\":%.3f,\"end\":%.3f}", words[i].start, words[i].end);
        buf_put(b, num);
    }
    buf_put(b, "]");
}

/* {"start":…,"end":…,"text":…[,"words":[…]]}; the words are the segment's own range of the result's word list. */
static void buf_put_segment_json(Buf *b, const P2Segment *seg, const P2Result *words_from)
{
    char num[96];
    snprintf(num, sizeof(num), "{\"start\":%.3f,\"end\":%.3f,\"text\":", seg->start, seg->end);
    buf_put(b, num);
    buf_put_json_string(b, seg->text);
    if (words_from) {
        buf_put(b, ",\"words\":");
        buf_put_words_json(b, words_from->words, seg->first_word, seg->n_words);
    }
    buf_put(b, "}");
}

/* ── The API ─────────────────────────────────────────────────────────────── */

char *p2_segment_json(const P2Segment *seg, const P2Result *words_from)
{
    Buf b = {0, 0, 0};
    buf_put_segment_json(&b, seg, words_from);
    return b.s;
}

char *p2_result_json(const P2Result *r)
{
    Buf b = {0, 0, 0};
    char num[192];
    buf_put(&b, "{\"text\":");
    buf_put_json_string(&b, r->text);
    buf_put(&b, ",\"words\":");
    buf_put_words_json(&b, r->words, 0, r->n_words);
    buf_put(&b, ",\"segments\":[");
    for (int i = 0; i < r->n_segments; i++) {
        if (i) buf_put(&b, ",");
        buf_put_segment_json(&b, &r->segments[i], NULL);
    }
    snprintf(num, sizeof(num), "],\"duration_seconds\":%.3f,\"encoder_frames\":%d,\"tokens\":%d,"
                               "\"seconds\":{\"mel\":%.4f,\"encoder\":%.4f,\"decoder\":%.4f}}",
             r->duration_seconds, r->encoder_frames, r->n_tokens, r->t_mel, r->t_encoder, r->t_decoder);
    buf_put(&b, num);
    return b.s;
}
