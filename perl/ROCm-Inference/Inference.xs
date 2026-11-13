/* Inference.xs -- thin XS mapping of the inference C library (src/inference.h) for ROCm::Inference.
 *
 * One Perl function per C function, the model handle first; no classes. Conventions:
 *   - errors croak with the library's message (inference_get_last_error);
 *   - text goes in and out as Perl character strings (UTF-8 in the library);
 *   - token lists are array references.
 * The handle is a blessed scalar reference only so that a model is freed when its last reference goes away
 * (DESTROY); free_model frees it explicitly. Everything else (chat templates, sessions, files) is plain Perl
 * in lib/ROCm/Inference.pm. */

#define PERL_NO_GET_CONTEXT
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

#include "inference.h"
#include "vision.h"
#include "image.h"
#include "mrope.h"
#include "whisper.h"
#include "qwen3tts_speech.h"
#include "audio.h"
#include <stdlib.h>
#include <string.h>

/* a speech handle: the talker and the codec of one voice-capable model pair */
typedef struct {
    qwen3tts_model_t* m;
    qwen3tts_codec_t* c;
} speech_handle;

#define HANDLE_CLASS "ROCm::Inference::Model"

static inference_model_t* handle_ptr(pTHX_ SV* sv, int allow_freed) {
    if (!SvROK(sv) || !sv_derived_from(sv, HANDLE_CLASS)) croak("ROCm::Inference: not a model handle");
    inference_model_t* m = INT2PTR(inference_model_t*, SvIV(SvRV(sv)));
    if (!m && !allow_freed) croak("ROCm::Inference: model handle already freed");
    return m;
}

static void croak_last(pTHX_ const char* what, int rc) {
    const char* e = inference_get_last_error();
    croak("ROCm::Inference::%s: %s (%d)", what, e && e[0] ? e : inference_strerror((inference_error_t)rc), rc);
}

/* Perl array ref of token ids -> malloc'ed int32 array (caller frees with Safefree) */
static int32_t* ids_from_av(pTHX_ SV* ref, int* n) {
    if (!SvROK(ref) || SvTYPE(SvRV(ref)) != SVt_PVAV) croak("ROCm::Inference: expected an array reference of token ids");
    AV* av = (AV*)SvRV(ref);
    const SSize_t len = av_len(av) + 1;
    int32_t* ids;
    Newx(ids, len > 0 ? len : 1, int32_t);
    for (SSize_t i = 0; i < len; ++i) {
        SV** e = av_fetch(av, i, 0);
        ids[i] = e ? (int32_t)SvIV(*e) : 0;
    }
    *n = (int)len;
    return ids;
}

static SV* text_sv(pTHX_ const char* s, STRLEN len) {
    SV* sv = newSVpvn(s, len);
    if (is_utf8_string((const U8*)s, len)) SvUTF8_on(sv);
    return sv;
}

/* on_token callback: calls the Perl sub with (token, text); a true return value stops generation. A die in the
 * sub stops it too and is rethrown after the C call returns. */
struct cb_ctx {
#ifdef PERL_IMPLICIT_CONTEXT
    PerlInterpreter* perl;
#endif
    SV* code;
    SV* error;
};

static int on_token_cb(void* user, int32_t token, const char* text, int len) {
    struct cb_ctx* c = (struct cb_ctx*)user;
#ifdef PERL_IMPLICIT_CONTEXT
    dTHXa(c->perl);
#endif
    dSP;
    int stop = 0;
    ENTER;
    SAVETMPS;
    PUSHMARK(SP);
    XPUSHs(sv_2mortal(newSViv(token)));
    XPUSHs(sv_2mortal(text_sv(aTHX_ text, (STRLEN)len)));
    PUTBACK;
    const int count = call_sv(c->code, G_SCALAR | G_EVAL);
    SPAGAIN;
    if (SvTRUE(ERRSV)) {
        c->error = newSVsv(ERRSV);
        stop = 1;
        if (count > 0) (void)POPs;
    } else if (count > 0) {
        stop = SvTRUE(POPs) ? 1 : 0;
    }
    PUTBACK;
    FREETMPS;
    LEAVE;
    return stop;
}

static SV* opt(pTHX_ HV* h, const char* key) {
    SV** v = h ? hv_fetch(h, key, (I32)strlen(key), 0) : NULL;
    return v && SvOK(*v) ? *v : NULL;
}

/* generate / stream_begin options (all optional): max_new_tokens, temperature, top_k, top_p, min_p,
 * repeat_penalty, presence_penalty, repeat_last_n, seed, stop => [strings], ignore_eog, reuse_prefix, checkpoint_tail. Returns the
 * stop strings' array (Safefree it after the call that uses g), NULL when none. */
static const char** gen_options(pTHX_ HV* o, inference_gen_options* g, const char* who) {
    *g = inference_gen_default_options();
    SV* v;
    if ((v = opt(aTHX_ o, "max_new_tokens"))) g->max_new_tokens = (int)SvIV(v);
    if ((v = opt(aTHX_ o, "temperature"))) g->sampler.temperature = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "top_k"))) g->sampler.top_k = (int)SvIV(v);
    if ((v = opt(aTHX_ o, "top_p"))) g->sampler.top_p = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "min_p"))) g->sampler.min_p = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "repeat_penalty"))) g->sampler.repeat_penalty = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "presence_penalty"))) g->sampler.presence_penalty = (float)SvNV(v);
    if ((v = opt(aTHX_ o, "repeat_last_n"))) g->sampler.repeat_last_n = (int)SvIV(v);
    if ((v = opt(aTHX_ o, "seed"))) g->sampler.seed = (uint64_t)SvUV(v);
    if ((v = opt(aTHX_ o, "ignore_eog"))) g->ignore_eog = SvTRUE(v) ? 1 : 0;
    if ((v = opt(aTHX_ o, "reuse_prefix"))) g->reuse_prefix = SvTRUE(v) ? 1 : 0;
    if ((v = opt(aTHX_ o, "checkpoint_tail"))) g->checkpoint_tail = (int)SvIV(v);
    if ((v = opt(aTHX_ o, "prefill_chunk"))) g->prefill_chunk = (int)SvIV(v);
    const char** stops = NULL;
    if ((v = opt(aTHX_ o, "stop"))) {
        if (!SvROK(v) || SvTYPE(SvRV(v)) != SVt_PVAV) croak("ROCm::Inference::%s: stop must be an array reference", who);
        AV* av = (AV*)SvRV(v);
        const SSize_t ns = av_len(av) + 1;
        Newx(stops, ns > 0 ? ns : 1, const char*);
        for (SSize_t i = 0; i < ns; ++i) {
            SV** e = av_fetch(av, i, 0);
            stops[i] = e && SvOK(*e) ? SvPVutf8_nolen(*e) : "";
        }
        g->stop_strings = stops;
        g->n_stop_strings = (int)ns;
    }
    return stops;
}

/* Engine events (inference_set_event_cb): the sub gets (event name, { seq, n0, n1, ms }). On prefill_progress a true
 * return, or a die, cancels the prefill (the engine stops after that step); on the other events it is ignored. One
 * handler per model, kept in g_events until it is replaced or the model is freed. */
static const char* const EVENT_NAMES[] = {"unknown", "prefill_begin", "prefill_progress", "prefill_end",
                                          "checkpoint", "store", "draft", "context_full"};
static HV* g_events = NULL;  /* "%p" of the model => IV of its struct event_ctx */

struct event_ctx {
#ifdef PERL_IMPLICIT_CONTEXT
    PerlInterpreter* perl;
#endif
    SV* code;
};

static int on_event_cb(void* user, const inference_event* ev) {
    struct event_ctx* c = (struct event_ctx*)user;
#ifdef PERL_IMPLICIT_CONTEXT
    dTHXa(c->perl);
#endif
    dSP;
    const int t = ev->type >= 1 && ev->type <= 7 ? ev->type : 0;
    HV* info = newHV();
    (void)hv_stores(info, "seq", newSViv(ev->seq));
    (void)hv_stores(info, "n0", newSViv(ev->n0));
    (void)hv_stores(info, "n1", newSViv(ev->n1));
    (void)hv_stores(info, "ms", newSVnv(ev->ms));
    ENTER;
    SAVETMPS;
    PUSHMARK(SP);
    XPUSHs(sv_2mortal(newSVpv(EVENT_NAMES[t], 0)));
    XPUSHs(sv_2mortal(newRV_noinc((SV*)info)));
    PUTBACK;
    /* the sub's true return, or its death, cancels the work the engine is doing (PREFILL_PROGRESS); the others ignore it */
    int stop = 0;
    const int count = call_sv(c->code, G_SCALAR | G_EVAL);
    SPAGAIN;
    if (SvTRUE(ERRSV)) {
        stop = 1;
        Perl_warn(aTHX_ "ROCm::Inference: event handler died, the work is cancelled: %s", SvPV_nolen(ERRSV));
    } else if (count > 0) {
        stop = SvTRUE(POPs) ? 1 : 0;
    }
    PUTBACK;
    FREETMPS;
    LEAVE;
    return stop;
}

/* Drops a model's event handler (before the model is freed) */
static void drop_event_handler(pTHX_ inference_model_t* m) {
    char key[32];
    if (!g_events) return;
    snprintf(key, sizeof key, "%p", (void*)m);
    SV** old = hv_fetch(g_events, key, (I32)strlen(key), 0);
    if (!old) return;
    struct event_ctx* c = INT2PTR(struct event_ctx*, SvIV(*old));
    inference_set_event_cb(m, NULL, NULL);
    SvREFCNT_dec(c->code);
    Safefree(c);
    hv_delete(g_events, key, (I32)strlen(key), G_DISCARD);
}

/* ---------------------------------------------------------------- JSON (for the converter: config / tokenizer files)
   A strict RFC 8259 decoder to Perl data: objects -> hash refs, arrays -> array refs, strings -> UTF-8 strings,
   integers -> IV (NV past 64 bits), other numbers -> NV, true / false -> copies of PL_sv_yes / PL_sv_no (booleans to
   builtin::is_bool), null -> undef. JSON::PP takes minutes on a 20 MB tokenizer.json. */
typedef struct { const char* p; const char* end; const char* err; } jsn_t;

static void jsn_ws(jsn_t* j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) ++j->p;
}

static int jsn_hex4(jsn_t* j, unsigned* out) {
    unsigned v = 0;
    if (j->end - j->p < 4) return 0;
    for (int i = 0; i < 4; ++i) {
        const char c = *j->p++;
        v <<= 4;
        if (c >= '0' && c <= '9') v |= (unsigned)(c - '0');
        else if (c >= 'a' && c <= 'f') v |= (unsigned)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') v |= (unsigned)(c - 'A' + 10);
        else return 0;
    }
    *out = v;
    return 1;
}

static SV* jsn_string(pTHX_ jsn_t* j) {
    ++j->p; /* the opening quote */
    const char* q = j->p;
    while (q < j->end && *q != '"' && *q != '\\') ++q;
    SV* sv = newSVpvn(j->p, (STRLEN)(q - j->p));
    j->p = q;
    while (j->p < j->end && *j->p != '"') {
        if (*j->p != '\\') {
            const char* r = j->p;
            while (r < j->end && *r != '"' && *r != '\\') ++r;
            sv_catpvn(sv, j->p, (STRLEN)(r - j->p));
            j->p = r;
            continue;
        }
        if (++j->p >= j->end) break;
        char c = *j->p++, buf[4];
        switch (c) {
        case '"': case '\\': case '/': sv_catpvn(sv, &c, 1); break;
        case 'b': sv_catpvn(sv, "\b", 1); break;
        case 'f': sv_catpvn(sv, "\f", 1); break;
        case 'n': sv_catpvn(sv, "\n", 1); break;
        case 'r': sv_catpvn(sv, "\r", 1); break;
        case 't': sv_catpvn(sv, "\t", 1); break;
        case 'u': {
            unsigned u, lo;
            if (!jsn_hex4(j, &u)) { j->err = "bad \\u escape"; SvREFCNT_dec(sv); return NULL; }
            if (u >= 0xD800 && u < 0xDC00 && j->end - j->p >= 6 && j->p[0] == '\\' && j->p[1] == 'u') {
                const char* save = j->p;
                j->p += 2;
                if (jsn_hex4(j, &lo) && lo >= 0xDC00 && lo < 0xE000) u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                else j->p = save;
            }
            int n;
            if (u < 0x80) { buf[0] = (char)u; n = 1; }
            else if (u < 0x800) { buf[0] = (char)(0xC0 | (u >> 6)); buf[1] = (char)(0x80 | (u & 63)); n = 2; }
            else if (u < 0x10000) {
                buf[0] = (char)(0xE0 | (u >> 12)); buf[1] = (char)(0x80 | ((u >> 6) & 63));
                buf[2] = (char)(0x80 | (u & 63)); n = 3;
            } else {
                buf[0] = (char)(0xF0 | (u >> 18)); buf[1] = (char)(0x80 | ((u >> 12) & 63));
                buf[2] = (char)(0x80 | ((u >> 6) & 63)); buf[3] = (char)(0x80 | (u & 63)); n = 4;
            }
            sv_catpvn(sv, buf, (STRLEN)n);
            break;
        }
        default: j->err = "bad escape"; SvREFCNT_dec(sv); return NULL;
        }
    }
    if (j->p >= j->end) { j->err = "unterminated string"; SvREFCNT_dec(sv); return NULL; }
    ++j->p; /* the closing quote */
    SvUTF8_on(sv);
    return sv;
}

static SV* jsn_value(pTHX_ jsn_t* j, int depth) {
    jsn_ws(j);
    if (j->p >= j->end) { j->err = "unexpected end"; return NULL; }
    if (depth > 512) { j->err = "nested too deeply"; return NULL; }
    const char c = *j->p;
    if (c == '"') return jsn_string(aTHX_ j);
    if (c == '{') {
        HV* hv = newHV();
        ++j->p;
        jsn_ws(j);
        if (j->p < j->end && *j->p == '}') { ++j->p; return newRV_noinc((SV*)hv); }
        for (;;) {
            jsn_ws(j);
            if (j->p >= j->end || *j->p != '"') { j->err = "expected a key"; break; }
            SV* k = jsn_string(aTHX_ j);
            if (!k) break;
            jsn_ws(j);
            if (j->p >= j->end || *j->p != ':') { j->err = "expected ':'"; SvREFCNT_dec(k); break; }
            ++j->p;
            SV* v = jsn_value(aTHX_ j, depth + 1);
            if (!v) { SvREFCNT_dec(k); break; }
            (void)hv_store_ent(hv, k, v, 0);
            SvREFCNT_dec(k);
            jsn_ws(j);
            if (j->p < j->end && *j->p == ',') { ++j->p; continue; }
            if (j->p < j->end && *j->p == '}') { ++j->p; return newRV_noinc((SV*)hv); }
            j->err = "expected ',' or '}'";
            break;
        }
        SvREFCNT_dec((SV*)hv);
        return NULL;
    }
    if (c == '[') {
        AV* av = newAV();
        ++j->p;
        jsn_ws(j);
        if (j->p < j->end && *j->p == ']') { ++j->p; return newRV_noinc((SV*)av); }
        for (;;) {
            SV* v = jsn_value(aTHX_ j, depth + 1);
            if (!v) break;
            av_push(av, v);
            jsn_ws(j);
            if (j->p < j->end && *j->p == ',') { ++j->p; continue; }
            if (j->p < j->end && *j->p == ']') { ++j->p; return newRV_noinc((SV*)av); }
            j->err = "expected ',' or ']'";
            break;
        }
        SvREFCNT_dec((SV*)av);
        return NULL;
    }
    if (j->end - j->p >= 4 && !memcmp(j->p, "true", 4)) { j->p += 4; return newSVsv(&PL_sv_yes); }
    if (j->end - j->p >= 5 && !memcmp(j->p, "false", 5)) { j->p += 5; return newSVsv(&PL_sv_no); }
    if (j->end - j->p >= 4 && !memcmp(j->p, "null", 4)) { j->p += 4; return newSV(0); }
    if (c == '-' || (c >= '0' && c <= '9')) {
        const char* s = j->p;
        int is_int = 1;
        if (*j->p == '-') ++j->p;
        while (j->p < j->end && *j->p >= '0' && *j->p <= '9') ++j->p;
        if (j->p < j->end && *j->p == '.') { is_int = 0; ++j->p; while (j->p < j->end && *j->p >= '0' && *j->p <= '9') ++j->p; }
        if (j->p < j->end && (*j->p == 'e' || *j->p == 'E')) {
            is_int = 0;
            ++j->p;
            if (j->p < j->end && (*j->p == '+' || *j->p == '-')) ++j->p;
            while (j->p < j->end && *j->p >= '0' && *j->p <= '9') ++j->p;
        }
        char tmp[64];
        const size_t n = (size_t)(j->p - s);
        if (n == 0 || n >= sizeof tmp) { j->err = "bad number"; return NULL; }
        memcpy(tmp, s, n);
        tmp[n] = 0;
        if (is_int && n < 19) return newSViv((IV)strtoll(tmp, NULL, 10));
        return newSVnv(strtod(tmp, NULL));
    }
    j->err = "unexpected character";
    return NULL;
}

MODULE = ROCm::Inference    PACKAGE = ROCm::Inference

PROTOTYPES: DISABLE

SV*
load_model(path, opts = NULL)
    const char* path
    SV* opts
  CODE:
    /* opts: { stage => '/dev/shm' } (inference_load_opts) */
    inference_load_opts lo = {0};
    if (opts && SvOK(opts)) {
        if (!SvROK(opts) || SvTYPE(SvRV(opts)) != SVt_PVHV) croak("load_model: options must be a hash reference");
        SV** sv = hv_fetchs((HV*)SvRV(opts), "stage", 0);
        if (sv && SvOK(*sv)) lo.stage_dir = SvPV_nolen(*sv);
    }
    inference_model_t* m = inference_load_model_ex(path, &lo);
    if (!m) croak_last(aTHX_ "load_model", INFERENCE_ERR_MODEL_LOAD);
    RETVAL = sv_setref_pv(newSV(0), HANDLE_CLASS, (void*)m);
  OUTPUT:
    RETVAL

SV*
model_info(path)
    const char* path
  CODE:
    /* What a GGUF is, from its header (inference_model_info): nothing loaded */
    inference_model_info_t mi;
    if (inference_model_info(path, &mi) != 0) croak_last(aTHX_ "model_info", INFERENCE_ERR_MODEL_LOAD);
    static const char* const kinds[] = {"unsupported", "chat", "embedding", "draft"};
    HV* hv = newHV();
    (void)hv_stores(hv, "kind", newSVpv(kinds[mi.kind + 1], 0));
    (void)hv_stores(hv, "arch", newSVpv(mi.arch, 0));
    if (mi.kind == INFERENCE_MODEL_UNSUPPORTED) (void)hv_stores(hv, "why", newSVpv(mi.why, 0));
    (void)hv_stores(hv, "n_ctx_train", newSViv((IV)mi.n_ctx_train));
    (void)hv_stores(hv, "files", newSViv(mi.n_files));
    (void)hv_stores(hv, "bytes", newSVuv((UV)mi.bytes));
    (void)hv_stores(hv, "kv_bytes_per_token", newSVuv((UV)mi.kv_bytes_per_token));
    RETVAL = newRV_noinc((SV*)hv);
  OUTPUT:
    RETVAL

SV*
quantize(dst_type, src_type, src, nrows, ncols, n_threads = 0)
    int dst_type
    int src_type
    SV* src
    IV nrows
    IV ncols
    int n_threads
  CODE:
    /* nrows x ncols values of src_type (a byte string) re-encoded as dst_type: a byte string (inference_quantize) */
    const size_t srs = inference_quant_row_size(src_type, ncols), drs = inference_quant_row_size(dst_type, ncols);
    if (!srs || !drs) croak("quantize: type %d or %d cannot hold rows of %ld values", src_type, dst_type, (long)ncols);
    STRLEN len;
    const char* p = SvPVbyte(src, len);
    if (nrows < 0 || len < srs * (size_t)nrows)
        croak("quantize: %lu bytes of input, %lu needed", (unsigned long)len, (unsigned long)(srs * (size_t)nrows));
    RETVAL = newSV(drs * (size_t)nrows + 1);
    SvPOK_only(RETVAL);
    if (inference_quantize(dst_type, SvPVX(RETVAL), src_type, p, nrows, ncols, n_threads) != 0) {
        SvREFCNT_dec(RETVAL);
        croak_last(aTHX_ "quantize", INFERENCE_ERR_INVALID_ARG);
    }
    SvCUR_set(RETVAL, drs * (size_t)nrows);
  OUTPUT:
    RETVAL

SV*
quantize_fp8_blocks(dst_type, src, scale, nrows, ncols, block_rows, block_cols, scale_cols, n_threads = 0)
    int dst_type
    SV* src
    SV* scale
    IV nrows
    IV ncols
    int block_rows
    int block_cols
    IV scale_cols
    int n_threads
  CODE:
    /* FP8 e4m3fn rows (a byte string) with float32 block scales (a byte string, scale_cols a row) re-encoded as
       dst_type (inference_quantize_fp8_blocks) */
    const size_t drs = inference_quant_row_size(dst_type, ncols);
    if (!drs) croak("quantize_fp8_blocks: type %d cannot hold rows of %ld values", dst_type, (long)ncols);
    STRLEN len, slen;
    const char* p = SvPVbyte(src, len);
    const char* sp = SvPVbyte(scale, slen);
    const size_t srows = block_rows > 0 ? (size_t)((nrows + block_rows - 1) / block_rows) : 0;
    if (nrows < 0 || len < (size_t)ncols * (size_t)nrows || slen < 4 * srows * (size_t)scale_cols)
        croak("quantize_fp8_blocks: %lu bytes of rows and %lu of scales for %ld rows of %ld", (unsigned long)len,
              (unsigned long)slen, (long)nrows, (long)ncols);
    RETVAL = newSV(drs * (size_t)nrows + 1);
    SvPOK_only(RETVAL);
    if (inference_quantize_fp8_blocks(dst_type, SvPVX(RETVAL), p, (const float*)sp, nrows, ncols, block_rows,
                                      block_cols, scale_cols, n_threads) != 0) {
        SvREFCNT_dec(RETVAL);
        croak_last(aTHX_ "quantize_fp8_blocks", INFERENCE_ERR_INVALID_ARG);
    }
    SvCUR_set(RETVAL, drs * (size_t)nrows);
  OUTPUT:
    RETVAL

SV*
json_decode(text)
    SV* text
  CODE:
    /* JSON (UTF-8 bytes) to Perl data, for the converter's config and tokenizer files */
    STRLEN len;
    const char* p = SvPVbyte(text, len);
    jsn_t j = {p, p + len, NULL};
    RETVAL = jsn_value(aTHX_ &j, 0);
    if (RETVAL) {
        jsn_ws(&j);
        if (j.p != j.end) { SvREFCNT_dec(RETVAL); RETVAL = NULL; j.err = "trailing characters"; }
    }
    if (!RETVAL) croak("json_decode: %s at byte %ld", j.err ? j.err : "?", (long)(j.p - p));
  OUTPUT:
    RETVAL

SV*
exp_neg_f32(src)
    SV* src
  CODE:
    /* -expf(x) for each float32 of a byte string (the converter's ssm_a = -exp(A_log), in float32 as torch) */
    STRLEN len;
    const char* p = SvPVbyte(src, len);
    const size_t n = len / 4;
    RETVAL = newSV(len + 1);
    SvPOK_only(RETVAL);
    float* o = (float*)SvPVX(RETVAL);
    for (size_t i = 0; i < n; ++i) {
        float x;
        memcpy(&x, p + 4 * i, 4);
        o[i] = -expf(x);
    }
    SvCUR_set(RETVAL, n * 4);
  OUTPUT:
    RETVAL

int
drop_cache(fd, offset, len, sync = 0)
    int fd
    IV offset
    IV len
    int sync
  CODE:
    /* The page cache's copy of [offset, offset + len) of an open file dropped (written back first with sync): a
       converter streaming hundreds of GB leaves the cache to the models being served. Pages other processes map
       stay. Returns posix_fadvise's result. */
    if (sync) (void)fdatasync(fd);
    RETVAL = posix_fadvise(fd, (off_t)offset, (off_t)len, POSIX_FADV_DONTNEED);
  OUTPUT:
    RETVAL

SV*
repack_mxfp4(packed, scale, rows, cols)
    SV* packed
    SV* scale
    IV rows
    IV cols
  CODE:
    /* FP4 weights as DeepSeek / compressed-tensors store them (rows x cols/2 bytes: value 2i in the low nibble, 2i+1
       in the high; an E8M0 scale byte per 32 values) as ggml's MXFP4 blocks (a scale byte, then 16 bytes: value j
       low, j+16 high). Bits are moved, not changed: both index ggml's kvalues_mxfp4. */
    STRLEN lp, ls;
    const unsigned char* p = (const unsigned char*)SvPVbyte(packed, lp);
    const unsigned char* sc = (const unsigned char*)SvPVbyte(scale, ls);
    if (cols % 32 || rows < 0 || lp < (STRLEN)(rows * cols / 2) || ls < (STRLEN)(rows * cols / 32))
        croak("repack_mxfp4: %lu packed and %lu scale bytes for %ld x %ld", (unsigned long)lp, (unsigned long)ls,
              (long)rows, (long)cols);
    const IV nb = cols / 32;
    RETVAL = newSV((STRLEN)(rows * nb * 17) + 1);
    SvPOK_only(RETVAL);
    unsigned char* o = (unsigned char*)SvPVX(RETVAL);
    for (IV r = 0; r < rows; ++r) {
        for (IV b = 0; b < nb; ++b) {
            const unsigned char* src = p + r * (cols / 2) + b * 16;
            unsigned char v[32];
            for (int i = 0; i < 16; ++i) { v[2 * i] = src[i] & 15; v[2 * i + 1] = src[i] >> 4; }
            *o++ = sc[r * nb + b];
            for (int j = 0; j < 16; ++j) *o++ = (unsigned char)(v[j] | (v[j + 16] << 4));
        }
    }
    SvCUR_set(RETVAL, (STRLEN)(rows * nb * 17));
  OUTPUT:
    RETVAL

UV
quant_row_size(type, ncols)
    int type
    IV ncols
  CODE:
    RETVAL = inference_quant_row_size(type, ncols);
  OUTPUT:
    RETVAL

int
quant_can_encode(type)
    int type
  CODE:
    RETVAL = inference_quant_can_encode(type);
  OUTPUT:
    RETVAL

const char*
stage_info(h)
    SV* h
  CODE:
    RETVAL = inference_stage_info(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

void
free_model(h)
    SV* h
  CODE:
    inference_model_t* m = handle_ptr(aTHX_ h, 1);
    if (m) {
        drop_event_handler(aTHX_ m);
        inference_free_model(m);  /* also frees the KV cache */
        sv_setiv(SvRV(h), 0);
    }

void
set_event_handler(h, cb = NULL)
    SV* h
    SV* cb
  CODE:
    /* The engine's events go to cb as (name, { seq, n0, n1, ms }); cb undef removes the handler */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    if (!g_events) g_events = newHV();
    drop_event_handler(aTHX_ m);
    if (cb && SvOK(cb)) {
        if (!SvROK(cb) || SvTYPE(SvRV(cb)) != SVt_PVCV) croak("set_event_handler: a code reference (or undef)");
        struct event_ctx* c;
        Newxz(c, 1, struct event_ctx);
#ifdef PERL_IMPLICIT_CONTEXT
        c->perl = aTHX;
#endif
        c->code = newSVsv(cb);
        char key[32];
        snprintf(key, sizeof key, "%p", (void*)m);
        (void)hv_store(g_events, key, (I32)strlen(key), newSViv(PTR2IV(c)), 0);
        inference_set_event_cb(m, on_event_cb, c);
    }

void
kv_cache_init(h, n_ctx)
    SV* h
    int n_ctx
  CODE:
    const int rc = inference_kv_cache_init(handle_ptr(aTHX_ h, 0), n_ctx);
    if (rc != 0) croak_last(aTHX_ "kv_cache_init", rc);

int
set_lm_head_q8(h, enable)
    SV* h
    int enable
  CODE:
    RETVAL = inference_set_lm_head_q8(handle_ptr(aTHX_ h, 0), enable);
    if (RETVAL < 0) croak_last(aTHX_ "set_lm_head_q8", RETVAL);
  OUTPUT:
    RETVAL

int
set_draft_model(h, path)
    SV* h
    SV* path
  CODE:
    RETVAL = inference_set_draft_model(handle_ptr(aTHX_ h, 0), SvOK(path) ? SvPV_nolen(path) : NULL);
    if (RETVAL < 0) croak_last(aTHX_ "set_draft_model", RETVAL);
  OUTPUT:
    RETVAL

int
set_prefill_batch(h, n)
    SV* h
    int n
  CODE:
    RETVAL = inference_set_prefill_batch(handle_ptr(aTHX_ h, 0), n);
    if (RETVAL < 0) croak_last(aTHX_ "set_prefill_batch", RETVAL);
  OUTPUT:
    RETVAL

int
set_swa_ring(h, n)
    SV* h
    int n
  CODE:
    RETVAL = inference_set_swa_ring(handle_ptr(aTHX_ h, 0), n);
    if (RETVAL < 0) croak_last(aTHX_ "set_swa_ring", RETVAL);
  OUTPUT:
    RETVAL

int
set_kv_type(h, type)
    SV* h
    const char* type
  CODE:
    /* 'f16' or 'q8_0' */
    int t = !strcmp(type, "f16") ? INFERENCE_KV_F16 : !strcmp(type, "q8_0") ? INFERENCE_KV_Q8_0
          : !strcmp(type, "q4_0") ? INFERENCE_KV_Q4_0 : -1;
    if (t < 0) croak("set_kv_type: 'f16', 'q8_0' or 'q4_0', not '%s'", type);
    RETVAL = inference_set_kv_type(handle_ptr(aTHX_ h, 0), t);
    if (RETVAL < 0) croak_last(aTHX_ "set_kv_type", RETVAL);
  OUTPUT:
    RETVAL

int
swa_ring(h)
    SV* h
  CODE:
    RETVAL = inference_swa_ring(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
set_draft_tokens(h, n)
    SV* h
    int n
  CODE:
    RETVAL = inference_set_draft_tokens(handle_ptr(aTHX_ h, 0), n);
    if (RETVAL < 0) croak_last(aTHX_ "set_draft_tokens", RETVAL);
  OUTPUT:
    RETVAL

SV*
tokenize(h, text)
    SV* h
    SV* text
  CODE:
    STRLEN len;
    const char* s = SvPVutf8(text, len);
    int n = 0;
    int32_t* ids = inference_tokenize(handle_ptr(aTHX_ h, 0), s, &n);
    if (!ids && len > 0) croak_last(aTHX_ "tokenize", -1);
    AV* av = newAV();
    av_extend(av, n);
    for (int i = 0; i < n; ++i) av_push(av, newSViv(ids[i]));
    if (ids) inference_free_tokens(ids, n);
    RETVAL = newRV_noinc((SV*)av);
  OUTPUT:
    RETVAL

SV*
detokenize(h, ids_ref)
    SV* h
    SV* ids_ref
  CODE:
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    int n = 0;
    int32_t* ids = ids_from_av(aTHX_ ids_ref, &n);
    const int len = inference_detokenize(m, ids, n, NULL, 0);
    if (len < 0) { Safefree(ids); croak_last(aTHX_ "detokenize", len); }
    char* buf;
    Newx(buf, len + 1, char);
    inference_detokenize(m, ids, n, buf, len + 1);
    RETVAL = text_sv(aTHX_ buf, (STRLEN)len);
    Safefree(buf);
    Safefree(ids);
  OUTPUT:
    RETVAL

SV*
chat_template(h, name = NULL)
    SV* h
    const char* name
  CODE:
    const char* t = inference_chat_template(handle_ptr(aTHX_ h, 0), name);
    RETVAL = t ? text_sv(aTHX_ t, strlen(t)) : &PL_sv_undef;
  OUTPUT:
    RETVAL

int
bos_id(h)
    SV* h
  CODE:
    RETVAL = inference_bos_id(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
eos_id(h)
    SV* h
  CODE:
    RETVAL = inference_eos_id(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
token_is_eog(h, token)
    SV* h
    int token
  CODE:
    RETVAL = inference_token_is_eog(handle_ptr(aTHX_ h, 0), token);
  OUTPUT:
    RETVAL

int
vocab_size(h)
    SV* h
  CODE:
    RETVAL = inference_vocab_size(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
hidden_size(h)
    SV* h
  CODE:
    RETVAL = inference_hidden_size(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
embedding_dim(h)
    SV* h
  CODE:
    RETVAL = inference_embedding_dim(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

SV*
embed(h, ids_ref, normalize = 1)
    SV* h
    SV* ids_ref
    int normalize
  CODE:
    /* the embedding of one input's token ids, as an array ref of floats */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    const int d = inference_embedding_dim(m);
    if (d <= 0) croak("ROCm::Inference::embed: not an embedding model");
    int n = 0;
    int32_t* ids = ids_from_av(aTHX_ ids_ref, &n);
    float* out;
    Newx(out, d, float);
    const int rc = inference_embed(m, ids, n, out, normalize);
    Safefree(ids);
    if (rc != 0) { Safefree(out); croak_last(aTHX_ "embed", rc); }
    AV* av = newAV();
    av_extend(av, d - 1);
    for (int i = 0; i < d; ++i) av_push(av, newSVnv(out[i]));
    Safefree(out);
    RETVAL = newRV_noinc((SV*)av);
  OUTPUT:
    RETVAL

int
n_ctx(h)
    SV* h
  CODE:
    RETVAL = inference_n_ctx(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

int
n_past(h)
    SV* h
  CODE:
    RETVAL = inference_n_past(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

void
reset(h)
    SV* h
  CODE:
    inference_reset(handle_ptr(aTHX_ h, 0));

void
prefill(h, ids_ref)
    SV* h
    SV* ids_ref
  CODE:
    int n = 0;
    int32_t* ids = ids_from_av(aTHX_ ids_ref, &n);
    const int rc = inference_prefill(handle_ptr(aTHX_ h, 0), ids, n);
    Safefree(ids);
    if (rc != 0) croak_last(aTHX_ "prefill", rc);

int
decode_token(h, token)
    SV* h
    int token
  CODE:
    const int rc = inference_decode_token(handle_ptr(aTHX_ h, 0), token, &RETVAL);
    if (rc != 0) croak_last(aTHX_ "decode_token", rc);
  OUTPUT:
    RETVAL

SV*
get_logits(h)
    SV* h
  CODE:
    /* The last token's logits as packed native floats: unpack("f*", $logits) */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    const int v = inference_vocab_size(m);
    int* ids;
    Newx(ids, v, int);
    RETVAL = newSV((STRLEN)v * sizeof(float) + 1);
    SvPOK_only(RETVAL);
    const int rc = inference_get_logits(m, ids, (float*)SvPVX(RETVAL));
    Safefree(ids);
    if (rc != 0) { SvREFCNT_dec(RETVAL); croak_last(aTHX_ "get_logits", rc); }
    SvCUR_set(RETVAL, (STRLEN)v * sizeof(float));
  OUTPUT:
    RETVAL

void
generate(h, prompt_ref, opts_ref = NULL)
    SV* h
    SV* prompt_ref
    SV* opts_ref
  PPCODE:
    /* Returns (\@token_ids, $stop_reason). Options (all optional): max_new_tokens, temperature, top_k, top_p,
     * min_p, repeat_penalty, presence_penalty, repeat_last_n, seed, stop => [strings], ignore_eog, reuse_prefix, checkpoint_tail,
     * on_token => sub { my ($token, $text) = @_; return $stop } */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    HV* o = NULL;
    if (opts_ref && SvOK(opts_ref)) {
        if (!SvROK(opts_ref) || SvTYPE(SvRV(opts_ref)) != SVt_PVHV) croak("ROCm::Inference::generate: options must be a hash reference");
        o = (HV*)SvRV(opts_ref);
    }
    inference_gen_options g;
    const char** stops = gen_options(aTHX_ o, &g, "generate");
    SV* v;
    struct cb_ctx cb;
#ifdef PERL_IMPLICIT_CONTEXT
    cb.perl = aTHX;
#endif
    cb.code = NULL;
    cb.error = NULL;
    if ((v = opt(aTHX_ o, "on_token"))) {
        if (!SvROK(v) || SvTYPE(SvRV(v)) != SVt_PVCV) croak("ROCm::Inference::generate: on_token must be a code reference");
        cb.code = v;
        g.on_token = on_token_cb;
        g.user = &cb;
    }
    int n = 0;
    int32_t* prompt = ids_from_av(aTHX_ prompt_ref, &n);
    const int cap = g.max_new_tokens > 0 ? g.max_new_tokens : 1;
    int32_t* out;
    Newx(out, cap, int32_t);
    int n_out = 0;
    inference_stop_reason reason = INFERENCE_STOP_NONE;
    const int rc = inference_generate_stream(m, prompt, n, &g, out, cap, &n_out, &reason);
    Safefree(prompt);
    Safefree(stops);
    if (cb.error) {
        Safefree(out);
        croak_sv(sv_2mortal(cb.error));
    }
    if (rc != 0) {
        Safefree(out);
        croak_last(aTHX_ "generate", rc);
    }
    AV* av = newAV();
    for (int i = 0; i < n_out && i < cap; ++i) av_push(av, newSViv(out[i]));
    Safefree(out);
    EXTEND(SP, 2);
    PUSHs(sv_2mortal(newRV_noinc((SV*)av)));
    PUSHs(sv_2mortal(newSViv((IV)reason)));

int
seq_new(h)
    SV* h
  CODE:
    RETVAL = inference_seq_new(handle_ptr(aTHX_ h, 0));
    if (RETVAL < 0) croak_last(aTHX_ "seq_new", INFERENCE_ERR_OUT_OF_MEMORY);
  OUTPUT:
    RETVAL

void
seq_select(h, seq)
    SV* h
    int seq
  CODE:
    const int rc = inference_seq_select(handle_ptr(aTHX_ h, 0), seq);
    if (rc != 0) croak_last(aTHX_ "seq_select", rc);

int
seq_current(h)
    SV* h
  CODE:
    RETVAL = inference_seq_current(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

void
seq_free(h, seq)
    SV* h
    int seq
  CODE:
    const int rc = inference_seq_free(handle_ptr(aTHX_ h, 0), seq);
    if (rc != 0) croak_last(aTHX_ "seq_free", rc);

UV
_stream_begin(h, prompt_ref, opts_ref = NULL)
    SV* h
    SV* prompt_ref
    SV* opts_ref
  CODE:
    /* the C stream as a number; ROCm::Inference::Stream (Inference.pm) wraps it */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    HV* o = NULL;
    if (opts_ref && SvOK(opts_ref)) {
        if (!SvROK(opts_ref) || SvTYPE(SvRV(opts_ref)) != SVt_PVHV) croak("ROCm::Inference::stream_begin: options must be a hash reference");
        o = (HV*)SvRV(opts_ref);
    }
    if (opt(aTHX_ o, "on_token")) croak("ROCm::Inference::stream_begin: no on_token (step returns each token)");
    inference_gen_options g;
    const char** stops = gen_options(aTHX_ o, &g, "stream_begin");
    int n = 0;
    int32_t* prompt = ids_from_av(aTHX_ prompt_ref, &n);
    inference_stream_t* st = inference_stream_begin(m, prompt, n, &g);
    Safefree(prompt);
    Safefree(stops);
    if (!st) croak_last(aTHX_ "stream_begin", INFERENCE_ERR_INVALID_ARG);
    RETVAL = PTR2UV(st);
  OUTPUT:
    RETVAL

void
_stream_step(p)
    UV p
  PPCODE:
    /* (status 1 going / 0 done, token or -1, text, stop reason) */
    int32_t tok = -1;
    const char* text = "";
    int len = 0;
    inference_stop_reason reason = INFERENCE_STOP_NONE;
    const int rc = inference_stream_step(INT2PTR(inference_stream_t*, p), &tok, &text, &len, &reason);
    if (rc < 0) croak_last(aTHX_ "Stream::step", rc);
    SV* t = newSVpvn(text, (STRLEN)len);
    SvUTF8_on(t);  /* whole UTF-8 characters only (the stream holds a split one back) */
    EXTEND(SP, 4);
    PUSHs(sv_2mortal(newSViv(rc)));
    PUSHs(sv_2mortal(newSViv(tok)));
    PUSHs(sv_2mortal(t));
    PUSHs(sv_2mortal(newSViv((IV)reason)));

void
_stream_step_batch(ptrs)
    SV* ptrs
  PPCODE:
    /* one step of each stream, one forward pass: per stream [status, token, text, stop reason] */
    if (!SvROK(ptrs) || SvTYPE(SvRV(ptrs)) != SVt_PVAV) croak("ROCm::Inference: expected an array reference of streams");
    AV* av = (AV*)SvRV(ptrs);
    const SSize_t n = av_len(av) + 1;
    if (n <= 0) XSRETURN_EMPTY;
    inference_stream_t** st;
    inference_step_result* r;
    Newx(st, n, inference_stream_t*);
    Newx(r, n, inference_step_result);
    for (SSize_t i = 0; i < n; ++i) {
        SV** e = av_fetch(av, i, 0);
        st[i] = INT2PTR(inference_stream_t*, e ? SvUV(*e) : 0);
    }
    int rc = inference_stream_step_batch(st, (int)n, r);
    for (SSize_t i = 0; rc == 0 && i < n; ++i)
        if (r[i].rc < 0) rc = r[i].rc;
    if (rc < 0) {
        Safefree(st);
        Safefree(r);
        croak_last(aTHX_ "step_streams", rc);
    }
    EXTEND(SP, n);
    for (SSize_t i = 0; i < n; ++i) {
        AV* o = newAV();
        SV* t = newSVpvn(r[i].text, (STRLEN)r[i].text_len);
        SvUTF8_on(t);
        av_push(o, newSViv(r[i].rc));
        av_push(o, newSViv(r[i].token));
        av_push(o, t);
        av_push(o, newSViv((IV)r[i].reason));
        PUSHs(sv_2mortal(newRV_noinc((SV*)o)));
    }
    Safefree(st);
    Safefree(r);

int
_stream_tokens(p)
    UV p
  CODE:
    RETVAL = inference_stream_tokens(INT2PTR(inference_stream_t*, p));
  OUTPUT:
    RETVAL

int
_stream_seq(p)
    UV p
  CODE:
    RETVAL = inference_stream_seq(INT2PTR(inference_stream_t*, p));
  OUTPUT:
    RETVAL

void
_stream_free(p)
    UV p
  CODE:
    inference_stream_free(INT2PTR(inference_stream_t*, p));

void
last_prompt_stats(h)
    SV* h
  PPCODE:
    /* (tokens reused, tokens prefilled) by the last generate */
    int reused = 0, prefilled = 0;
    inference_last_prompt_stats(handle_ptr(aTHX_ h, 0), &reused, &prefilled);
    EXTEND(SP, 2);
    PUSHs(sv_2mortal(newSViv(reused)));
    PUSHs(sv_2mortal(newSViv(prefilled)));

void
store_open(h, dir, snapshot_every = 4)
    SV* h
    const char* dir
    int snapshot_every
  CODE:
    const int rc = inference_store_open(handle_ptr(aTHX_ h, 0), dir, snapshot_every);
    if (rc != 0) croak_last(aTHX_ "store_open", rc);

void
store_close(h)
    SV* h
  CODE:
    inference_store_close(handle_ptr(aTHX_ h, 0));

void
store_set_lazy(h, lazy)
    SV* h
    int lazy
  CODE:
    const int rc = inference_store_set_lazy(handle_ptr(aTHX_ h, 0), lazy);
    if (rc != 0) croak_last(aTHX_ "store_set_lazy", rc);

SV*
store_stats(h)
    SV* h
  CODE:
    inference_store_stats st;
    if (inference_store_get_stats(handle_ptr(aTHX_ h, 0), &st) != 0) croak("ROCm::Inference::store_stats failed");
    HV* hv = newHV();
    (void)hv_stores(hv, "blocks_published", newSViv((IV)st.blocks_published));
    (void)hv_stores(hv, "blocks_attached", newSViv((IV)st.blocks_attached));
    (void)hv_stores(hv, "snapshots_saved", newSViv((IV)st.snapshots_saved));
    (void)hv_stores(hv, "snapshots_restored", newSViv((IV)st.snapshots_restored));
    (void)hv_stores(hv, "snapshots_dropped", newSViv((IV)st.snapshots_dropped));
    (void)hv_stores(hv, "snapshots_held", newSViv((IV)st.snapshots_held));
    (void)hv_stores(hv, "on_disk", newSViv((IV)st.on_disk));
    (void)hv_stores(hv, "tokens_from_store", newSViv((IV)st.tokens_from_store));
    (void)hv_stores(hv, "block_tokens", newSViv(st.block_tokens));
    (void)hv_stores(hv, "block_bytes", newSViv((IV)st.block_bytes));
    (void)hv_stores(hv, "snapshot_bytes", newSViv((IV)st.snapshot_bytes));
    RETVAL = newRV_noinc((SV*)hv);
  OUTPUT:
    RETVAL

void
store_block_names(h, ids_ref)
    SV* h
    SV* ids_ref
  PPCODE:
    /* The store files (<hash>.kvb) of the token list's full blocks, in order */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    int n = 0;
    int32_t* ids = ids_from_av(aTHX_ ids_ref, &n);
    const int len = inference_store_block_names(m, ids, n, NULL, 0);
    if (len < 0) { Safefree(ids); croak("ROCm::Inference::store_block_names: no store open"); }
    char* buf;
    Newx(buf, len + 1, char);
    inference_store_block_names(m, ids, n, buf, len + 1);
    Safefree(ids);
    for (char *p = buf, *e; *p; p = e + 1) {
        e = strchr(p, '\n');
        if (!e) break;
        XPUSHs(sv_2mortal(newSVpvn(p, (STRLEN)(e - p))));
    }
    Safefree(buf);

SV*
last_error()
  CODE:
    const char* e = inference_get_last_error();
    RETVAL = e && e[0] ? newSVpv(e, 0) : &PL_sv_undef;
  OUTPUT:
    RETVAL

int
_constant_stop(which)
    int which
  CODE:
    /* stop reasons, for the constants in Inference.pm */
    switch (which) {
    case 0: RETVAL = INFERENCE_STOP_NONE; break;
    case 1: RETVAL = INFERENCE_STOP_EOG; break;
    case 2: RETVAL = INFERENCE_STOP_MAX_TOKENS; break;
    case 3: RETVAL = INFERENCE_STOP_STRING; break;
    case 4: RETVAL = INFERENCE_STOP_CALLBACK; break;
    default: RETVAL = INFERENCE_STOP_CONTEXT_FULL; break;
    }
  OUTPUT:
    RETVAL


# ---- Images (phase V5): the vision tower, the image processor, the M-RoPE positions, the prompt's overrides

IV
_vision_open(path)
    const char* path
  CODE:
    vision_model_t* vm = vision_open(path);
    if (!vm) croak("ROCm::Inference: vision_open: %s", vision_last_error());
    RETVAL = PTR2IV(vm);
  OUTPUT:
    RETVAL

void
_vision_close(vm)
    IV vm
  CODE:
    vision_close(INT2PTR(vision_model_t*, vm));

void
_vision_info(vm)
    IV vm
  PPCODE:
    /* (n_out, merge, patch, temporal, mean, std) */
    vision_model_t* v = INT2PTR(vision_model_t*, vm);
    float mean = 0.5f, sd = 0.5f;
    vision_image_norm(v, &mean, &sd);
    EXTEND(SP, 6);
    PUSHs(sv_2mortal(newSViv(vision_n_out(v))));
    PUSHs(sv_2mortal(newSViv(vision_merge(v))));
    PUSHs(sv_2mortal(newSViv(vision_patch(v))));
    PUSHs(sv_2mortal(newSViv(vision_temporal(v))));
    PUSHs(sv_2mortal(newSVnv(mean)));
    PUSHs(sv_2mortal(newSVnv(sd)));

SV*
_vision_encode(vm, pixels, h, w)
    IV vm
    SV* pixels
    int h
    int w
  CODE:
    /* pixels: packed floats ([h * w][3 * temporal * patch * patch]); the result: packed floats [(h / m) * (w / m)][n_out] */
    vision_model_t* v = INT2PTR(vision_model_t*, vm);
    STRLEN len;
    const char* px = SvPV(pixels, len);
    const int m = vision_merge(v), p = vision_patch(v), t = vision_temporal(v), n_out = vision_n_out(v);
    if (h <= 0 || w <= 0 || m <= 0 || h % m || w % m) croak("ROCm::Inference: vision_encode: grid %d x %d", h, w);
    const size_t need = (size_t)h * w * 3 * t * p * p * sizeof(float);
    if (len != need) croak("ROCm::Inference: vision_encode: %lu bytes of pixels, expected %lu", (unsigned long)len, (unsigned long)need);
    const size_t rows = (size_t)(h / m) * (w / m);
    SV* out = newSV(rows * n_out * sizeof(float));
    SvPOK_on(out);
    if (vision_encode(v, (const float*)px, h, w, (float*)SvPVX(out), NULL, NULL) != 0) {
        SvREFCNT_dec(out);
        croak("ROCm::Inference: vision_encode: %s", vision_last_error());
    }
    SvCUR_set(out, rows * n_out * sizeof(float));
    RETVAL = out;
  OUTPUT:
    RETVAL

void
_image_pixels(bytes, patch, merge, temporal, mean, sd, min_px, max_px)
    SV* bytes
    int patch
    int merge
    int temporal
    double mean
    double sd
    double min_px
    double max_px
  PPCODE:
    /* an image file's bytes -> (packed float pixel rows, grid_h, grid_w) as the processor gives them */
    STRLEN len;
    const unsigned char* data = (const unsigned char*)SvPV(bytes, len);
    unsigned char* rgb = NULL;
    int w = 0, h = 0, gh = 0, gw = 0;
    if (image_decode_rgb(data, len, &rgb, &w, &h) != 0) croak("ROCm::Inference: cannot decode the image");
    float* rows = NULL;
    const int rc = image_qwen_pixels(rgb, w, h, patch, merge, temporal, (float)mean, (float)sd, (long long)min_px,
                                     (long long)max_px, &rows, &gh, &gw);
    image_free(rgb);
    if (rc != 0) croak("ROCm::Inference: the image (%d x %d) cannot be processed", w, h);
    const size_t n = (size_t)gh * gw * 3 * temporal * patch * patch * sizeof(float);
    SV* out = newSVpvn((const char*)rows, n);
    free(rows);
    EXTEND(SP, 3);
    PUSHs(sv_2mortal(out));
    PUSHs(sv_2mortal(newSViv(gh)));
    PUSHs(sv_2mortal(newSViv(gw)));

void
_mrope_positions(ids_ref, hw_ref, merge, image_token)
    SV* ids_ref
    SV* hw_ref
    int merge
    int image_token
  PPCODE:
    /* the prompt's M-RoPE positions: (packed int64 [n][3], the decode delta) */
    if (!SvROK(ids_ref) || SvTYPE(SvRV(ids_ref)) != SVt_PVAV || !SvROK(hw_ref) || SvTYPE(SvRV(hw_ref)) != SVt_PVAV)
        croak("ROCm::Inference: mrope_positions: array references expected");
    AV* ids_av = (AV*)SvRV(ids_ref);
    AV* hw_av = (AV*)SvRV(hw_ref);
    const int n = (int)(av_len(ids_av) + 1), nhw = (int)(av_len(hw_av) + 1);
    if (n <= 0 || nhw % 2) croak("ROCm::Inference: mrope_positions: empty prompt or odd grid list");
    int32_t* ids;
    int32_t* hw;
    int64_t* pos;
    Newx(ids, n, int32_t);
    Newx(hw, nhw ? nhw : 1, int32_t);
    Newx(pos, 3 * (size_t)n, int64_t);
    for (int i = 0; i < n; ++i) { SV** e = av_fetch(ids_av, i, 0); ids[i] = e ? (int32_t)SvIV(*e) : 0; }
    for (int i = 0; i < nhw; ++i) { SV** e = av_fetch(hw_av, i, 0); hw[i] = e ? (int32_t)SvIV(*e) : 0; }
    int64_t delta = 0;
    const int rc = mrope_positions(ids, n, hw, nhw / 2, merge, image_token, pos, &delta);
    Safefree(ids);
    Safefree(hw);
    if (rc != 0) { Safefree(pos); croak("ROCm::Inference: mrope_positions: the image pad runs do not match the grids"); }
    SV* out = newSVpvn((const char*)pos, 3 * (size_t)n * sizeof(int64_t));
    Safefree(pos);
    EXTEND(SP, 2);
    PUSHs(sv_2mortal(out));
    PUSHs(sv_2mortal(newSViv((IV)delta)));

void
set_mm(h, base, n, emb, use, pos3)
    SV* h
    int base
    int n
    SV* emb
    SV* use
    SV* pos3
  CODE:
    /* the prompt's overrides for its next prefill (inference_set_mm; the engine copies the buffers); emb undef clears */
    inference_model_t* m = handle_ptr(aTHX_ h, 0);
    if (!SvOK(emb)) {
        if (inference_set_mm(m, 0, 0, NULL, NULL, NULL) != 0) croak_last(aTHX_ "set_mm", -1);
        XSRETURN_EMPTY;
    }
    STRLEN el, ul, pl = 0;
    const char* ep = SvPV(emb, el);
    const char* up = SvPV(use, ul);
    const char* pp = SvOK(pos3) ? SvPV(pos3, pl) : NULL;
    if (n <= 0 || ul != (STRLEN)n || el % ((size_t)n * sizeof(float)) != 0 || (pp && pl != (STRLEN)n * 3 * sizeof(int64_t)))
        croak("ROCm::Inference: set_mm: %d rows, %lu embedding bytes, %lu use bytes", n, (unsigned long)el, (unsigned long)ul);
    if (inference_set_mm(m, base, n, (const float*)ep, (const unsigned char*)up, pp ? (const int64_t*)pp : NULL) != 0)
        croak_last(aTHX_ "set_mm", -1);

int
mm_rope_delta(h)
    SV* h
  CODE:
    RETVAL = inference_mm_rope_delta(handle_ptr(aTHX_ h, 0));
  OUTPUT:
    RETVAL

MODULE = ROCm::Inference    PACKAGE = ROCm::Inference::Model

void
DESTROY(h)
    SV* h
  CODE:
    inference_model_t* m = handle_ptr(aTHX_ h, 1);
    if (m) {
        inference_free_model(m);  /* also frees the KV cache */
        sv_setiv(SvRV(h), 0);
    }

MODULE = ROCm::Inference    PACKAGE = ROCm::Inference

IV
_whisper_open(path)
    const char* path
  CODE:
    whisper_model_t* wm = whisper_open(path);
    if (!wm) croak("ROCm::Inference: whisper_open: %s", whisper_last_error());
    RETVAL = PTR2IV(wm);
  OUTPUT:
    RETVAL

void
_whisper_close(wm)
    IV wm
  CODE:
    whisper_close(INT2PTR(whisper_model_t*, wm));

void
_whisper_transcribe(wm, wav, language, task)
    IV wm
    SV* wav
    SV* language
    SV* task
  PPCODE:
    /* (text, tokens): a WAV file's bytes (mono float32 at 16 kHz after the decode; any length: long-form), the language's
     * code or undef, the task (transcribe or translate; undef: transcribe) */
    STRLEN len = 0;
    const unsigned char* data = (const unsigned char*)SvPV(wav, len);
    float* samples = NULL;
    size_t n = 0;
    if (whisper_decode_wav(data, len, &samples, &n) != 0) croak("ROCm::Inference: %s", whisper_last_error());
    /* the text: long-form clips are joined from their windows; a buffer of 64 KiB holds an hour of speech's text */
    char* text = (char*)malloc(65536);
    if (!text) { whisper_free_samples(samples); croak("ROCm::Inference: out of memory"); }
    const char* lang = SvOK(language) ? SvPV_nolen(language) : NULL;
    const char* tsk = SvOK(task) ? SvPV_nolen(task) : NULL;
    const int k = whisper_transcribe_task(INT2PTR(whisper_model_t*, wm), samples, n, lang, tsk, text, 65536);
    whisper_free_samples(samples);
    if (k < 0) { free(text); croak("ROCm::Inference: whisper_transcribe: %s", whisper_last_error()); }
    SV* out = newSVpv(text, 0);
    free(text);
    SvUTF8_on(out);
    EXTEND(SP, 2);
    PUSHs(sv_2mortal(out));
    PUSHs(sv_2mortal(newSViv(k)));

# ---- Speech (Qwen3-TTS: the talker, the codec; src/qwen3tts_speech.{h,cpp}). A handle is the talker and the codec.

IV
_speech_open(talker, codec)
    const char* talker
    const char* codec
  CODE:
    qwen3tts_model_t* m = qwen3tts_open(talker);
    if (!m) croak("ROCm::Inference: qwen3tts_open: %s", qwen3tts_last_error());
    qwen3tts_codec_t* c = qwen3tts_codec_open(codec);
    if (!c) {
        qwen3tts_close(m);
        croak("ROCm::Inference: qwen3tts_codec_open: %s", qwen3tts_codec_last_error());
    }
    speech_handle* h = (speech_handle*)malloc(sizeof(speech_handle));
    if (!h) { qwen3tts_codec_close(c); qwen3tts_close(m); croak("ROCm::Inference: out of memory"); }
    h->m = m;
    h->c = c;
    RETVAL = PTR2IV(h);
  OUTPUT:
    RETVAL

void
_speech_close(h)
    IV h
  CODE:
    speech_handle* s = INT2PTR(speech_handle*, h);
    if (s) {
        qwen3tts_codec_close(s->c);
        qwen3tts_close(s->m);
        free(s);
    }

int
_speech_n_embd(h)
    IV h
  CODE:
    speech_handle* s = INT2PTR(speech_handle*, h);
    RETVAL = qwen3tts_n_embd(s->m);
  OUTPUT:
    RETVAL

void
_speech_speak(h, text, ref_text, language, codes, spk, R, temperature, top_k, repetition_penalty, sub_temperature, sub_top_k, seed, max_frames)
    IV h
    const char* text
    const char* ref_text
    const char* language
    SV* codes
    SV* spk
    int R
    double temperature
    int top_k
    double repetition_penalty
    double sub_temperature
    int sub_top_k
    UV seed
    int max_frames
  PPCODE:
    /* (the samples as float32 bytes, the talker's frames): the voice's codes (R x 16 int32) and speaker embedding */
    speech_handle* s = INT2PTR(speech_handle*, h);
    STRLEN cl = 0, sl = 0;
    const char* cb = SvPV(codes, cl);
    const char* sb = SvPV(spk, sl);
    const int ncb = qwen3tts_n_codebooks(s->m), C = qwen3tts_n_embd(s->m);
    if (R < 1 || cl != (STRLEN)R * ncb * 4) croak("ROCm::Inference: the voice's codes: %d frames of %d codebooks, %lu bytes", R, ncb, (unsigned long)cl);
    if (sl != (STRLEN)C * 4) croak("ROCm::Inference: the voice's speaker embedding: %d floats, %lu bytes", C, (unsigned long)sl);
    const int lang = qwen3tts_language_id(s->m, language);
    if (lang < 0) croak("ROCm::Inference: no language '%s'", language);
    int* ids = (int*)malloc(sizeof(int) * (size_t)R * ncb);
    float* spkf = (float*)malloc(sizeof(float) * (size_t)C);
    if (!ids || !spkf) { free(ids); free(spkf); croak("ROCm::Inference: out of memory"); }
    /* little-endian int32 and float32 (the voice files): memcpy on this host's order */
    memcpy(ids, cb, cl);
    memcpy(spkf, sb, sl);
    qwen3tts_sampling_t smp;
    smp.temperature = (float)temperature;
    smp.top_k = top_k;
    smp.repetition_penalty = (float)repetition_penalty;
    smp.sub_temperature = (float)sub_temperature;
    smp.sub_top_k = sub_top_k;
    smp.seed = (uint64_t)seed;
    float* wav = NULL;
    size_t n = 0;
    int frames = 0;
    const int rc = qwen3tts_speak(s->m, s->c, text, ref_text, ids, R, spkf, lang, &smp, max_frames, &wav, &n, &frames);
    free(ids);
    free(spkf);
    if (rc != 0) croak("ROCm::Inference: qwen3tts_speak: %s", qwen3tts_last_error());
    SV* out = newSVpvn((const char*)wav, (STRLEN)(n * sizeof(float)));
    qwen3tts_speech_free_wav(wav);
    EXTEND(SP, 2);
    PUSHs(sv_2mortal(out));
    PUSHs(sv_2mortal(newSViv(frames)));

void
_speech_make_voice(h, samples)
    IV h
    SV* samples
  PPCODE:
    /* (codes packed int32 [R][16], R, speaker embedding packed float32): a voice from the clip's samples at 24 kHz */
    speech_handle* s = INT2PTR(speech_handle*, h);
    STRLEN sl = 0;
    const char* sb = SvPV(samples, sl);
    if (sl % 4) croak("ROCm::Inference: the samples: float32 bytes");
    const size_t n = sl / 4;
    const int C = qwen3tts_speaker_dim(s->m);
    if (C <= 0) croak("ROCm::Inference: the talker GGUF has no speaker encoder (spk.* tensors)");
    float* x = (float*)malloc(sl ? sl : 4);
    float* spk = (float*)malloc(sizeof(float) * (size_t)C);
    if (!x || !spk) { free(x); free(spk); croak("ROCm::Inference: out of memory"); }
    memcpy(x, sb, sl);
    int* codes = NULL;
    int R = 0;
    const int rc = qwen3tts_make_voice(s->m, s->c, x, n, &codes, &R, spk);
    free(x);
    if (rc != 0) { free(spk); croak("ROCm::Inference: qwen3tts_make_voice: %s", qwen3tts_last_error()); }
    const int ncb = qwen3tts_codec_n_quantizers(s->c);
    SV* cs = newSVpvn((const char*)codes, (STRLEN)R * ncb * 4);
    SV* ss = newSVpvn((const char*)spk, (STRLEN)C * 4);
    qwen3tts_codec_free_codes(codes);
    free(spk);
    EXTEND(SP, 3);
    PUSHs(sv_2mortal(cs));
    PUSHs(sv_2mortal(newSViv(R)));
    PUSHs(sv_2mortal(ss));

SV*
_audio_wav_samples(wav, rate)
    SV* wav
    int rate
  CODE:
    /* a WAV file's bytes as mono float32 at rate (the band-limited resampler), packed */
    STRLEN len = 0;
    const unsigned char* data = (const unsigned char*)SvPV(wav, len);
    float* out = NULL;
    size_t n = 0;
    if (audio_decode_wav(data, len, rate, &out, &n) != 0) croak("ROCm::Inference: %s", audio_last_error());
    RETVAL = newSVpvn((const char*)out, (STRLEN)(n * sizeof(float)));
    audio_free(out);
  OUTPUT:
    RETVAL
