// Whisper on the host in float32 (whisper.h): the encoder (the conv stem, pre-norm blocks with non-causal attention,
// the final norm) and the decoder (causal self-attention with a cache, cross-attention over the encoder output, the
// tied LM head), with the order of operations of transformers' WhisperForConditionalGeneration.
#include "whisper.h"

#include "audio.h"

#include "gguf_mmap.h"
#include "mel.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>
#include <zlib.h>

namespace {

std::string g_error;

struct Lin {
    const float* w = nullptr;  // [out][in], row-major (the checkpoint's layout)
    const float* b = nullptr;  // [out], or NULL (k_proj has none)
    int in = 0, out = 0;
};

struct Norm {
    const float* w = nullptr;
    const float* b = nullptr;
};

struct EncLayer {
    Norm attn_ln;
    Lin q, k, v, o;
    Norm mlp_ln;
    Lin fc1, fc2;
};

struct DecLayer {
    Norm attn_ln;
    Lin q, k, v, o;
    Norm cross_ln;
    Lin cq, ck, cv, co;
    Norm mlp_ln;
    Lin fc1, fc2;
};

// WHISPER_PROFILE=1: the wall time per stage of a transcription, printed on stderr at its end. Without the variable a
// timer is one flag test.
enum ProfKind { P_MEL, P_ENC, P_KV, P_DEC, P_LIN, P_SELF, P_CROSS, P_LN, P_LOGITS, P_CHOOSE, P_N };
double g_prof_ms[P_N];
long g_prof_steps = 0;  // decoder steps (the prompt's included)

bool prof_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("WHISPER_PROFILE");
        return e && *e && *e != '0';
    }();
    return on;
}

struct ProfTimer {
    ProfKind k;
    bool on;
    std::chrono::steady_clock::time_point t0;
    explicit ProfTimer(ProfKind kind) : k(kind), on(prof_enabled()) {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~ProfTimer() {
        if (on)
            g_prof_ms[k] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

void prof_reset() {
    std::memset(g_prof_ms, 0, sizeof g_prof_ms);
    g_prof_steps = 0;
}

// One line on stderr: the stages' milliseconds, the decoder's steps and ms per step, and the split of the decoder's time
void prof_report() {
    if (!prof_enabled()) return;
    const double dec = g_prof_ms[P_DEC];
    const double inner = g_prof_ms[P_LIN] + g_prof_ms[P_SELF] + g_prof_ms[P_CROSS] + g_prof_ms[P_LN] + g_prof_ms[P_LOGITS];
    std::fprintf(stderr,
                 "whisper profile: threads=%d mel=%.1f ms encoder=%.1f ms cross_kv=%.1f ms | decoder=%.1f ms over %ld "
                 "steps (%.3f ms/token): linear=%.1f self_attn=%.1f cross_attn=%.1f layer_norm=%.1f logits=%.1f "
                 "other=%.1f | choose=%.1f ms\n",
                 omp_get_max_threads(), g_prof_ms[P_MEL], g_prof_ms[P_ENC], g_prof_ms[P_KV], dec, g_prof_steps,
                 g_prof_steps ? dec / (double)g_prof_steps : 0.0, g_prof_ms[P_LIN], g_prof_ms[P_SELF],
                 g_prof_ms[P_CROSS], g_prof_ms[P_LN], g_prof_ms[P_LOGITS], dec - inner, g_prof_ms[P_CHOOSE]);
}

// The float32 tensor of an exact element count (a missing one is an error unless optional)
const float* tensor(const gguf_model_t* g, const std::string& name, size_t count, bool optional = false) {
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(g, name.c_str(), &bytes);
    if (!p) {
        if (optional) return nullptr;
        throw std::runtime_error("whisper: no tensor " + name);
    }
    if (bytes != count * sizeof(float))
        throw std::runtime_error("whisper: " + name + " has " + std::to_string(bytes) + " bytes, expected " +
                                 std::to_string(count * sizeof(float)));
    return static_cast<const float*>(p);
}

}  // namespace

struct whisper_model {
    gguf_model_t* g = nullptr;
    int n_mels = 0, n_actx = 0, n_astate = 0, n_ahead = 0, n_alayer = 0, n_affn = 0;
    int n_tctx = 0, n_tstate = 0, n_thead = 0, n_tlayer = 0, n_tffn = 0, n_vocab = 0;
    int sot = 0, eot = 0;
    float eps = 1e-5f;
    std::vector<int> suppress, begin_suppress;
    tokenizer_t* tok = nullptr;      // the GGUF's vocabulary (tokenizer.ggml.*), decodes the text
    std::unordered_map<std::string, int> special;  // the vocabulary's control markers ("<|en|>", ...) by text
    const float *conv1_w = nullptr, *conv1_b = nullptr, *conv2_w = nullptr, *conv2_b = nullptr;
    const float* enc_pos = nullptr;  // [n_actx][n_astate]
    Norm enc_post;
    std::vector<EncLayer> enc;
    const float* tok_emb = nullptr;  // [n_vocab][n_tstate], tied to the LM head
    const float* dec_pos = nullptr;  // [n_tctx][n_tstate]
    Norm dec_post;
    std::vector<DecLayer> dec;
    float fallback_cr_max = 2.4f, fallback_lp_min = -1.0f;  // the long form's temperature fallback (whisper_set_fallback)
    int fallback_keep_last = 0;                              // 1: keep the last attempt when all fail (the reference's rule)
    int attempts = 0;                                        // the decoder passes of the last long-form call
    ~whisper_model() {
        if (tok) tokenizer_free(tok);
        if (g) gguf_mmap_close(g);
    }
};

namespace {

// The multiply-adds from which a loop is shared out over the threads (below it the calling thread runs it alone)
#ifndef WHISPER_PAR_MACS
#define WHISPER_PAR_MACS 50000
#endif
constexpr long LINEAR_PAR_MACS = WHISPER_PAR_MACS;

// LayerNorm over each row of x [T][C] (population variance, eps inside the root, as torch's)
void layer_norm(const float* x, int T, int C, float eps, const Norm& n, float* y) {
#pragma omp parallel for schedule(static) if ((long)T * C >= WHISPER_PAR_MACS)
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t)t * C;
        float* yr = y + (size_t)t * C;
        double mean = 0.0, var = 0.0;
        for (int c = 0; c < C; ++c) mean += xr[c];
        mean /= C;
        for (int c = 0; c < C; ++c) var += (xr[c] - mean) * (xr[c] - mean);
        var /= C;
        const float inv = 1.0f / std::sqrt((float)var + eps);
        for (int c = 0; c < C; ++c) yr[c] = ((float)(xr[c] - mean) * inv) * n.w[c] + n.b[c];
    }
}

// The dot product of a and b over n floats, summed in SIMD lanes (the order of the additions differs from a serial sum
// in the last bits only), as the talker's dot_f32
inline float dot_f32(const float* a, const float* b, int n) {
    float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

// y[t][o] = x[t][:] . w[o][:] + b[o] for the T rows of x [T][in]. Each (t, o) is one independent dot product, so the
// pairs are shared out over the threads: a decoder step (T = 1) has only out of them. The bias is added after the dot.
void linear(const float* x, int T, const Lin& L, float* y) {
    const long n = (long)T * L.out;
#pragma omp parallel for schedule(static) if (n * L.in >= LINEAR_PAR_MACS)
    for (long r = 0; r < n; ++r) {
        const int t = (int)(r / L.out), o = (int)(r % L.out);
        const float* xr = x + (size_t)t * L.in;
        const float* wr = L.w + (size_t)o * L.in;
        const float acc = dot_f32(xr, wr, L.in) + (L.b ? L.b[o] : 0.0f);
        y[(size_t)t * L.out + o] = acc;
    }
}

// The exact (erf) GELU, as nn.functional.gelu
inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); }

// A 1-D convolution over x [C_in][T_in] (channel-major, as the mel) with w [C_out][C_in][K] and zero padding
void conv1d(const float* x, int C_in, int T_in, const float* w, const float* b, int C_out, int K, int stride, int pad,
            float* y, int T_out) {
#pragma omp parallel for schedule(static)
    for (int co = 0; co < C_out; ++co) {
        for (int t = 0; t < T_out; ++t) {
            float acc = b ? b[co] : 0.0f;
            for (int k = 0; k < K; ++k) {
                const int i = t * stride + k - pad;
                if (i < 0 || i >= T_in) continue;
                const float* wk = w + (size_t)co * C_in * K + k;
                const float* xi = x + i;
                float s = 0.0f;
#pragma omp simd reduction(+ : s)
                for (int ci = 0; ci < C_in; ++ci) s += wk[(size_t)ci * K] * xi[(size_t)ci * T_in];
                acc += s;
            }
            y[(size_t)co * T_out + t] = acc;
        }
    }
}

// Attention of the queries q [Tq][C] over the keys k and values v [Tk][C], H heads of C / H: out [Tq][C]. With
// causal, query i sees the keys 0 .. i + causal_offset (a decoder's own positions); without, all of them.
void attend(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int H, bool causal, int causal_offset,
            float* out) {
    const int d = C / H;
    const float scale = 1.0f / std::sqrt((float)d);
    // the (query, head) pairs are independent, shared out over the threads (a decoder step has H of them)
    const long n = (long)Tq * H;
#pragma omp parallel for schedule(static) if (n * Tk * d >= WHISPER_PAR_MACS)
    for (long r = 0; r < n; ++r) {
        const int i = (int)(r / H), h = (int)(r % H);
        std::vector<float> s(Tk);
        const float* qr = q + (size_t)i * C + (size_t)h * d;
        const int nk = causal ? std::min(Tk, i + causal_offset + 1) : Tk;
        float mx = -INFINITY;
        for (int j = 0; j < nk; ++j) {
            const float* kr = k + (size_t)j * C + (size_t)h * d;
            s[j] = dot_f32(qr, kr, d) * scale;
            mx = std::max(mx, s[j]);
        }
        double sum = 0.0;
        for (int j = 0; j < nk; ++j) {
            s[j] = std::exp(s[j] - mx);
            sum += s[j];
        }
        float* orow = out + (size_t)i * C + (size_t)h * d;
        for (int c = 0; c < d; ++c) orow[c] = 0.0f;
        for (int j = 0; j < nk; ++j) {
            const float p = (float)(s[j] / sum);
            const float* vr = v + (size_t)j * C + (size_t)h * d;
            for (int c = 0; c < d; ++c) orow[c] += p * vr[c];
        }
    }
}

// The token's logits for the last state x [n_tstate]: the tied embedding's dot products
void logits_of(const whisper_model* m, const float* x, float* logits) {
#pragma omp parallel for schedule(static)
    for (int v = 0; v < m->n_vocab; ++v) {
        const float* er = m->tok_emb + (size_t)v * m->n_tstate;
        logits[v] = dot_f32(x, er, m->n_tstate);
    }
}

// The decoder's stages with their profile timers (the same calls)
void linear_p(const float* x, int T, const Lin& L, float* y) {
    ProfTimer t(P_LIN);
    linear(x, T, L, y);
}
void layer_norm_p(const float* x, int T, int C, float eps, const Norm& n, float* y) {
    ProfTimer t(P_LN);
    layer_norm(x, T, C, eps, n, y);
}
void attend_p(ProfKind k, const float* q, int Tq, const float* kk, const float* vv, int Tk, int C, int H, bool causal,
              int causal_offset, float* out) {
    ProfTimer t(k);
    attend(q, Tq, kk, vv, Tk, C, H, causal, causal_offset, out);
}

// The decoder's state for one sequence: the self-attention cache and the cross-attention keys and values
struct DecState {
    std::vector<std::vector<float>> sk, sv;  // per layer [n_tctx][C]
    std::vector<std::vector<float>> ck, cv;  // per layer [n_actx][C], from the encoder output
};

// One decoder step: the token at position pos, its logits into logits [n_vocab]
void decoder_step(const whisper_model* m, DecState& st, int token, int pos, float* logits) {
    ProfTimer total(P_DEC);
    if (prof_enabled()) ++g_prof_steps;
    const int C = m->n_tstate;
    std::vector<float> x(C), h(C), q(C), kk(C), vv(C), a(C), o(C), f(m->n_tffn);
    for (int c = 0; c < C; ++c) x[c] = m->tok_emb[(size_t)token * C + c] + m->dec_pos[(size_t)pos * C + c];
    for (int l = 0; l < m->n_tlayer; ++l) {
        const DecLayer& L = m->dec[l];
        // self-attention over the positions so far (causal)
        layer_norm_p(x.data(), 1, C, m->eps, L.attn_ln, h.data());
        linear_p(h.data(), 1, L.q, q.data());
        linear_p(h.data(), 1, L.k, kk.data());
        linear_p(h.data(), 1, L.v, vv.data());
        std::memcpy(&st.sk[l][(size_t)pos * C], kk.data(), sizeof(float) * C);
        std::memcpy(&st.sv[l][(size_t)pos * C], vv.data(), sizeof(float) * C);
        attend_p(P_SELF, q.data(), 1, st.sk[l].data(), st.sv[l].data(), pos + 1, C, m->n_thead, true, pos, a.data());
        linear_p(a.data(), 1, L.o, o.data());
        for (int c = 0; c < C; ++c) x[c] += o[c];
        // cross-attention over the encoder output
        layer_norm_p(x.data(), 1, C, m->eps, L.cross_ln, h.data());
        linear_p(h.data(), 1, L.cq, q.data());
        attend_p(P_CROSS, q.data(), 1, st.ck[l].data(), st.cv[l].data(), m->n_actx, C, m->n_thead, false, 0, a.data());
        linear_p(a.data(), 1, L.co, o.data());
        for (int c = 0; c < C; ++c) x[c] += o[c];
        // the feed-forward
        layer_norm_p(x.data(), 1, C, m->eps, L.mlp_ln, h.data());
        linear_p(h.data(), 1, L.fc1, f.data());
        for (int c = 0; c < m->n_tffn; ++c) f[c] = gelu(f[c]);
        linear_p(f.data(), 1, L.fc2, o.data());
        for (int c = 0; c < C; ++c) x[c] += o[c];
    }
    layer_norm_p(x.data(), 1, C, m->eps, m->dec_post, h.data());
    ProfTimer lt(P_LOGITS);
    logits_of(m, h.data(), logits);
}

// splitmix64's next draw (the state advances), as src/qwen3tts.cpp's
uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// The next token from the logits [n_vocab] after the suppressions (begin: the begin_suppress ones too). temperature <= 0:
// the first maximum; else a token drawn from softmax(logits / temperature) with the splitmix state rng. lp (may be NULL)
// receives the chosen token's log-probability: log_softmax of the suppressed logits, not scaled by the temperature.
int choose_token(const whisper_model* m, const float* logits, bool begin, float temperature, uint64_t* rng, double* lp) {
    const int V = m->n_vocab;
    std::vector<float> l(logits, logits + V);
    for (int t : m->suppress)
        if (t >= 0 && t < V) l[t] = -INFINITY;
    if (begin)
        for (int t : m->begin_suppress)
            if (t >= 0 && t < V) l[t] = -INFINITY;
    const float mx = *std::max_element(l.begin(), l.end());  // finite: the vocabulary has unsuppressed tokens
    int tok = 0;
    if (temperature <= 0.0f) {
        for (int v = 1; v < V; ++v)
            if (l[v] > l[tok]) tok = v;
    } else {
        std::vector<double> w(V);
        double total = 0.0;
        for (int v = 0; v < V; ++v) {
            w[v] = std::exp(((double)l[v] - mx) / temperature);
            total += w[v];
        }
        const double u = (double)(splitmix64_next(rng) >> 11) * (1.0 / 9007199254740992.0);
        const double target = u * total;
        double cum = 0.0;
        for (int v = 0; v < V; ++v) {
            if (w[v] == 0.0) continue;
            tok = v;
            cum += w[v];
            if (cum > target) break;
        }
    }
    if (lp) {
        double z = 0.0;
        for (int v = 0; v < V; ++v) z += std::exp((double)l[v] - mx);
        *lp = (double)l[tok] - ((double)mx + std::log(z));
    }
    return tok;
}

// The decoding of one window's prompt and tokens (the greedy one at temperature 0): out_tokens receives the tokens after the
// prompt, without the end-of-transcript token; the count is returned. With sum_lp, *sum_lp is the sum of the log-probabilities
// of the chosen tokens (the end-of-transcript token included when it was chosen). Throws on failure.
int decode_tokens(whisper_model_t* m, const float* enc, const int* prompt, int n_prompt, int max_new, float temperature,
                  uint64_t seed, int* out_tokens, double* sum_lp) {
    const int C = m->n_tstate, T = m->n_actx;
    DecState st;
    st.sk.assign(m->n_tlayer, std::vector<float>((size_t)m->n_tctx * C));
    st.sv.assign(m->n_tlayer, std::vector<float>((size_t)m->n_tctx * C));
    st.ck.assign(m->n_tlayer, std::vector<float>((size_t)T * C));
    st.cv.assign(m->n_tlayer, std::vector<float>((size_t)T * C));
    // the cross-attention keys and values of the encoder output, per layer
    {
        ProfTimer t(P_KV);
        for (int l = 0; l < m->n_tlayer; ++l) {
            const DecLayer& L = m->dec[l];
            linear(enc, T, L.ck, st.ck[l].data());
            linear(enc, T, L.cv, st.cv[l].data());
        }
    }
    uint64_t rng = seed;
    std::vector<float> logits(m->n_vocab);
    for (int i = 0; i < n_prompt; ++i) decoder_step(m, st, prompt[i], i, logits.data());
    double lp = 0.0, sum = 0.0;
    int next = 0;
    {
        ProfTimer t(P_CHOOSE);
        next = choose_token(m, logits.data(), true, temperature, &rng, sum_lp ? &lp : nullptr);
    }
    int n_out = 0, pos = n_prompt;
    while (n_out < max_new && next != m->eot && pos < m->n_tctx) {
        out_tokens[n_out++] = next;
        sum += lp;
        decoder_step(m, st, next, pos, logits.data());
        ++pos;
        ProfTimer t(P_CHOOSE);
        next = choose_token(m, logits.data(), false, temperature, &rng, sum_lp ? &lp : nullptr);
    }
    if (sum_lp) {
        if (next == m->eot) sum += lp;
        *sum_lp = sum;
    }
    return n_out;
}

// The fallback's seed of a window (its seek position) and attempt: fixed constants mixed with both
uint64_t fallback_seed(size_t seek, int attempt) {
    return 0x5EED7E3F00DF00D5ull ^ ((uint64_t)seek * 0x9E3779B97F4A7C15ull) ^ ((uint64_t)(attempt + 1) << 48);
}

}  // namespace

extern "C" {

whisper_model_t* whisper_open(const char* gguf_path) {
    try {
        std::unique_ptr<whisper_model> m(new whisper_model());
        m->g = gguf_mmap_open(gguf_path);
        if (!m->g) {
            g_error = std::string("cannot open ") + (gguf_path ? gguf_path : "(null)");
            return nullptr;
        }
        if (std::string(gguf_arch(m->g)) != "whisper") throw std::runtime_error("not a whisper GGUF (architecture " +
                                                                              std::string(gguf_arch(m->g)) + ")");
        auto ar = [&](const char* k) { return (int)gguf_arch_int(m->g, k, 0); };
        m->n_mels = ar("n_mels");
        m->n_actx = ar("n_audio_ctx");
        m->n_astate = ar("n_audio_state");
        m->n_ahead = ar("n_audio_head");
        m->n_alayer = ar("n_audio_layer");
        m->n_affn = ar("n_audio_ffn");
        m->n_tctx = ar("n_text_ctx");
        m->n_tstate = ar("n_text_state");
        m->n_thead = ar("n_text_head");
        m->n_tlayer = ar("n_text_layer");
        m->n_tffn = ar("n_text_ffn");
        m->n_vocab = ar("n_vocab");
        m->sot = ar("decoder_start_token");
        m->eot = ar("eot_token");
        m->eps = gguf_arch_float(m->g, "layer_norm_epsilon", 1e-5f);
        if (m->n_mels != 80 || m->n_actx <= 0 || m->n_astate <= 0 || m->n_ahead <= 0 || m->n_alayer <= 0 || m->n_affn <= 0 ||
            m->n_tctx <= 0 || m->n_tstate <= 0 || m->n_thead <= 0 || m->n_tlayer <= 0 || m->n_tffn <= 0 || m->n_vocab <= 0 ||
            m->n_astate % m->n_ahead != 0 || m->n_tstate % m->n_thead != 0)
            throw std::runtime_error("whisper: missing or bad whisper.* hyperparameters");
        // the suppression lists (int arrays of the GGUF)
        auto ints = [&](const char* key, std::vector<int>& out) {
            int32_t type = 0;
            const void* data = nullptr;
            size_t count = 0;
            if (gguf_kv_get_array(m->g, key, &type, &data, &count) != 0) return;  // absent: none
            if (type != 4 && type != 5) throw std::runtime_error(std::string("bad ") + key);
            out.resize(count);
            for (size_t i = 0; i < count; ++i) {
                out[i] = type == 4 ? (int)static_cast<const uint32_t*>(data)[i] : static_cast<const int32_t*>(data)[i];
            }
        };
        ints("whisper.suppress_tokens", m->suppress);
        ints("whisper.begin_suppress_tokens", m->begin_suppress);

        m->tok = tokenizer_load_from_gguf(m->g);
        if (!m->tok) throw std::runtime_error("whisper: no tokenizer.ggml.* vocabulary in the GGUF");
        // the control markers by their text (the language, task and timestamp tokens are whole strings in the vocabulary)
        {
            const char** vocab = nullptr;
            size_t n_vocab = 0;
            if (gguf_kv_get_string_array(m->g, "tokenizer.ggml.tokens", &vocab, &n_vocab) != 0)
                throw std::runtime_error("whisper: no tokenizer.ggml.tokens");
            for (size_t i = 0; i < n_vocab; ++i)
                if (vocab[i] && std::strncmp(vocab[i], "<|", 2) == 0) m->special.emplace(vocab[i], (int)i);
        }

        const int C = m->n_astate, T = m->n_actx, Ct = m->n_tstate, F = m->n_affn, FT = m->n_tffn;
        m->conv1_w = tensor(m->g, "encoder.conv1.weight", (size_t)C * m->n_mels * 3);
        m->conv1_b = tensor(m->g, "encoder.conv1.bias", C);
        m->conv2_w = tensor(m->g, "encoder.conv2.weight", (size_t)C * C * 3);
        m->conv2_b = tensor(m->g, "encoder.conv2.bias", C);
        m->enc_pos = tensor(m->g, "encoder.positional_embedding", (size_t)T * C);
        m->enc_post = {tensor(m->g, "encoder.ln_post.weight", C), tensor(m->g, "encoder.ln_post.bias", C)};
        auto lin = [&](const std::string& base, int in, int out) {
            Lin L;
            L.in = in;
            L.out = out;
            L.w = tensor(m->g, base + ".weight", (size_t)in * out);
            L.b = tensor(m->g, base + ".bias", out, true);
            return L;
        };
        auto norm = [&](const std::string& base) {
            Norm n;
            n.w = tensor(m->g, base + ".weight", C);
            n.b = tensor(m->g, base + ".bias", C);
            return n;
        };
        for (int l = 0; l < m->n_alayer; ++l) {
            const std::string p = "encoder.blk." + std::to_string(l) + ".";
            EncLayer L;
            L.attn_ln = norm(p + "attn_ln");
            L.q = lin(p + "attn_q", C, C);
            L.k = lin(p + "attn_k", C, C);
            L.v = lin(p + "attn_v", C, C);
            L.o = lin(p + "attn_out", C, C);
            L.mlp_ln = norm(p + "mlp_ln");
            L.fc1 = lin(p + "mlp_fc1", C, F);
            L.fc2 = lin(p + "mlp_fc2", F, C);
            m->enc.push_back(L);
        }
        m->tok_emb = tensor(m->g, "decoder.token_embd.weight", (size_t)m->n_vocab * Ct);
        m->dec_pos = tensor(m->g, "decoder.positional_embedding", (size_t)m->n_tctx * Ct);
        m->dec_post = norm("decoder.ln");
        for (int l = 0; l < m->n_tlayer; ++l) {
            const std::string p = "decoder.blk." + std::to_string(l) + ".";
            DecLayer L;
            L.attn_ln = norm(p + "attn_ln");
            L.q = lin(p + "attn_q", Ct, Ct);
            L.k = lin(p + "attn_k", Ct, Ct);
            L.v = lin(p + "attn_v", Ct, Ct);
            L.o = lin(p + "attn_out", Ct, Ct);
            L.cross_ln = norm(p + "cross_attn_ln");
            L.cq = lin(p + "cross_attn_q", Ct, Ct);
            L.ck = lin(p + "cross_attn_k", Ct, Ct);
            L.cv = lin(p + "cross_attn_v", Ct, Ct);
            L.co = lin(p + "cross_attn_out", Ct, Ct);
            L.mlp_ln = norm(p + "mlp_ln");
            L.fc1 = lin(p + "mlp_fc1", Ct, FT);
            L.fc2 = lin(p + "mlp_fc2", FT, Ct);
            m->dec.push_back(L);
        }
        return m.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void whisper_close(whisper_model_t* model) { delete model; }

int whisper_decode_wav(const unsigned char* d, size_t n, float** samples_out, size_t* count_out) {
    // (16 kHz mono: the audio module's WAV decode through its band-limited resampler, audio.h)
    if (audio_decode_wav(d, n, 16000, samples_out, count_out) != 0) {
        g_error = audio_last_error();
        return -1;
    }
    return 0;
}

void whisper_free_samples(float* samples) { audio_free(samples); }

void whisper_set_fallback(whisper_model_t* m, float compression_ratio_max, float avg_logprob_min, int keep_last) {
    if (!m) return;
    m->fallback_cr_max = compression_ratio_max;
    m->fallback_lp_min = avg_logprob_min;
    m->fallback_keep_last = keep_last != 0;
}

int whisper_fallback_attempts(const whisper_model_t* m) { return m ? m->attempts : 0; }

float whisper_compression_ratio(const char* text, size_t len) {
    if (!text || len == 0) return 0.0f;
    uLongf out_len = compressBound((uLong)len);
    std::vector<Bytef> out(out_len);
    if (compress2(out.data(), &out_len, reinterpret_cast<const Bytef*>(text), (uLong)len, Z_DEFAULT_COMPRESSION) != Z_OK)
        return 0.0f;
    return (float)len / (float)out_len;
}

int whisper_token_id(const whisper_model_t* m, const char* text) {
    if (!m || !text) return -1;
    const auto it = m->special.find(text);
    return it == m->special.end() ? -1 : it->second;
}

const char* whisper_last_error(void) { return g_error.c_str(); }

int whisper_n_mels(const whisper_model_t* m) { return m ? m->n_mels : 0; }
int whisper_n_audio_ctx(const whisper_model_t* m) { return m ? m->n_actx : 0; }
int whisper_n_audio_state(const whisper_model_t* m) { return m ? m->n_astate : 0; }
int whisper_n_vocab(const whisper_model_t* m) { return m ? m->n_vocab : 0; }
int whisper_eot(const whisper_model_t* m) { return m ? m->eot : 0; }

int whisper_encode(whisper_model_t* m, const float* mel, float* out) {
    if (!m || !mel || !out) { g_error = "whisper_encode: null argument"; return -1; }
    ProfTimer prof_t(P_ENC);
    try {
        const int C = m->n_astate, T = m->n_actx, F = WHISPER_N_FRAMES;
        // the conv stem: conv1 (stride 1), conv2 (stride 2), both with GELU; then [T][C] with the positions
        std::vector<float> x1((size_t)C * F), x2((size_t)C * T);
        conv1d(mel, m->n_mels, F, m->conv1_w, m->conv1_b, C, 3, 1, 1, x1.data(), F);
        for (float& v : x1) v = gelu(v);
        conv1d(x1.data(), C, F, m->conv2_w, m->conv2_b, C, 3, 2, 1, x2.data(), T);
        for (float& v : x2) v = gelu(v);
        std::vector<float> x((size_t)T * C), h((size_t)T * C), q((size_t)T * C), k((size_t)T * C), v((size_t)T * C),
            a((size_t)T * C), o((size_t)T * C), f((size_t)T * m->n_affn);
        for (int t = 0; t < T; ++t)
            for (int c = 0; c < C; ++c) x[(size_t)t * C + c] = x2[(size_t)c * T + t] + m->enc_pos[(size_t)t * C + c];
        for (const EncLayer& L : m->enc) {
            layer_norm(x.data(), T, C, m->eps, L.attn_ln, h.data());
            linear(h.data(), T, L.q, q.data());
            linear(h.data(), T, L.k, k.data());
            linear(h.data(), T, L.v, v.data());
            attend(q.data(), T, k.data(), v.data(), T, C, m->n_ahead, false, 0, a.data());
            linear(a.data(), T, L.o, o.data());
            for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
            layer_norm(x.data(), T, C, m->eps, L.mlp_ln, h.data());
            linear(h.data(), T, L.fc1, f.data());
            for (float& u : f) u = gelu(u);
            linear(f.data(), T, L.fc2, o.data());
            for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        }
        layer_norm(x.data(), T, C, m->eps, m->enc_post, out);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int whisper_greedy(whisper_model_t* m, const float* enc, const int* prompt, int n_prompt, int max_new, int* out_tokens) {
    if (!m || !enc || !prompt || n_prompt <= 0 || max_new < 0 || (max_new > 0 && !out_tokens)) {
        g_error = "whisper_greedy: bad arguments";
        return -1;
    }
    if (n_prompt >= m->n_tctx) { g_error = "whisper_greedy: the prompt fills the context"; return -1; }
    try {
        return decode_tokens(m, enc, prompt, n_prompt, max_new, 0.0f, 0, out_tokens, nullptr);
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

// One window of at most 30 s: the prompt (<|startoftranscript|> <|language|> <|transcribe|> or <|translate|>,
// <|notimestamps|>), the encoder, the greedy decoding, the text. Returns the tokens, or -1.
static int transcribe_window(whisper_model_t* m, const float* samples, size_t n, const std::string& lang, bool translate,
                             std::string& out) {
    const int lang_id = whisper_token_id(m, lang.c_str());
    const int task_id = whisper_token_id(m, translate ? "<|translate|>" : "<|transcribe|>");
    const int notimestamps = whisper_token_id(m, "<|notimestamps|>");
    if (lang_id < 0) { g_error = "whisper_transcribe: unknown language " + lang; return -1; }
    if (task_id < 0 || notimestamps < 0) throw std::runtime_error("whisper: no task or timestamp tokens in the vocabulary");
    const int prompt[4] = {m->sot, lang_id, task_id, notimestamps};

    std::vector<float> mel((size_t)80 * WHISPER_N_FRAMES);
    {
        ProfTimer t(P_MEL);
        if (whisper_log_mel(samples, (int)n, 80, mel.data()) != 0) throw std::runtime_error("whisper_log_mel failed");
    }
    std::vector<float> enc((size_t)m->n_actx * m->n_astate);
    if (whisper_encode(m, mel.data(), enc.data()) != 0) throw std::runtime_error(g_error);
    std::vector<int> ids(std::max(1, m->n_tctx - 4));
    const int k = whisper_greedy(m, enc.data(), prompt, 4, (int)ids.size(), ids.data());
    if (k < 0) throw std::runtime_error(g_error);

    // the text: the tokenizer's detokenization (its length is the full one; retry with a buffer that holds it)
    std::vector<char> buf(64 + 8 * (size_t)k);
    int len = tokenizer_detokenize(m->tok, ids.data(), k, buf.data(), (int)buf.size());
    if (len < 0) throw std::runtime_error("whisper: detokenize failed");
    if ((size_t)len >= buf.size()) {
        buf.assign((size_t)len + 1, 0);
        len = tokenizer_detokenize(m->tok, ids.data(), k, buf.data(), (int)buf.size());
    }
    if (len < 0) throw std::runtime_error("whisper: detokenize failed");
    out.assign(buf.data(), (size_t)len);
    return k;
}

// The text of token ids (the tokenizer's detokenization; the buffer grows to fit)
static std::string detok(whisper_model_t* m, const std::vector<int>& ids) {
    if (ids.empty()) return std::string();
    std::vector<char> buf(64 + 8 * ids.size());
    int len = tokenizer_detokenize(m->tok, const_cast<int*>(ids.data()), (int)ids.size(), buf.data(), (int)buf.size());
    if (len < 0) throw std::runtime_error("whisper: detokenize failed");
    if ((size_t)len >= buf.size()) {
        buf.assign((size_t)len + 1, 0);
        len = tokenizer_detokenize(m->tok, const_cast<int*>(ids.data()), (int)ids.size(), buf.data(), (int)buf.size());
    }
    if (len < 0) throw std::runtime_error("whisper: detokenize failed");
    return std::string(buf.data(), (size_t)len);
}

// The long form (a clip over 30 s), as Whisper's sequential decoding: each window (at most 30 s from seek) is decoded WITH
// timestamps, conditioned on the previous text (<|startofprev|> and its last tokens, up to 224), and seek moves to the end
// of the last complete segment of the window: a pair of timestamps (the consecutive ones) closes a segment. A window whose
// output ends in an open segment moves seek to its last timestamp; a window with no pair of timestamps moves it by the
// whole window. The last window (up to 30 s) is taken whole.
struct WindowOut {
    std::vector<int> text;  // the accepted text tokens (no timestamps)
    size_t advance = 0;     // samples the seek moves by
};

// The window's segments from its tokens ids (timestamps included): the text, and how far seek moves (see above)
static void segment_window(whisper_model_t* m, const std::vector<int>& ids, int ts_begin, bool final_window, size_t window_n,
                           WindowOut& out) {
    const size_t k = ids.size();
    auto is_ts = [&](int i) { return ids[(size_t)i] >= ts_begin; };
    // the text: the vocabulary's tokens, not the control markers (the end of text and above) and not the timestamps
    auto is_text = [&](int i) { return ids[(size_t)i] < m->eot; };
    // the segments: the consecutive timestamps close one (openai's rule), and a window ending on a timestamp closes its last one
    std::vector<size_t> cuts;  // the token index after each pair of consecutive timestamps
    for (int i = 1; i < (int)k; ++i)
        if (is_ts(i - 1) && is_ts(i)) cuts.push_back((size_t)i);
    const bool single_end = k >= 2 && !is_ts((int)k - 2) && is_ts((int)k - 1);
    out.text.clear();
    if (!cuts.empty()) {
        std::vector<size_t> slices = cuts;
        if (single_end) slices.push_back(k);
        size_t last = 0;
        for (size_t cur : slices) {
            for (size_t i = last; i < cur; ++i)
                if (is_text((int)i)) out.text.push_back(ids[i]);
            last = cur;
        }
        if (final_window) {  // the clip's end: an open tail is text too
            for (size_t i = last; i < k; ++i)
                if (is_text((int)i)) out.text.push_back(ids[i]);
            out.advance = window_n;
        } else if (single_end) {
            out.advance = window_n;
        } else {
            const size_t last_ts = (size_t)(ids[last - 1] - ts_begin);  // the window's last closed segment ends here (20 ms units)
            out.advance = last_ts * 320;  // 0.02 s at 16 kHz
        }
    } else {
        for (size_t i = 0; i < k; ++i)
            if (is_text((int)i)) out.text.push_back(ids[i]);
        out.advance = window_n;
    }
    if (out.advance == 0) out.advance = window_n;  // (a seek always moves)
}

static void transcribe_window_ts(whisper_model_t* m, const float* samples, size_t n, const std::string& lang, bool translate,
                                 const std::vector<int>& prev, bool final_window, size_t seek, WindowOut& out) {
    const int lang_id = whisper_token_id(m, lang.c_str());
    const int task_id = whisper_token_id(m, translate ? "<|translate|>" : "<|transcribe|>");
    const int startofprev = whisper_token_id(m, "<|startofprev|>");
    const int ts_begin = whisper_token_id(m, "<|0.00|>");
    if (lang_id < 0) throw std::runtime_error("whisper_transcribe: unknown language " + lang);
    if (task_id < 0 || startofprev < 0 || ts_begin < 0) throw std::runtime_error("whisper: no timestamp or prev tokens in the vocabulary");
    std::vector<int> prompt;
    if (!prev.empty()) {
        const size_t keep = std::min<size_t>(prev.size(), 224);
        prompt.push_back(startofprev);
        prompt.insert(prompt.end(), prev.end() - keep, prev.end());
    }
    prompt.push_back(m->sot);
    prompt.push_back(lang_id);
    prompt.push_back(task_id);

    std::vector<float> mel((size_t)80 * WHISPER_N_FRAMES);
    {
        ProfTimer t(P_MEL);
        if (whisper_log_mel(samples, (int)n, 80, mel.data()) != 0) throw std::runtime_error("whisper_log_mel failed");
    }
    std::vector<float> enc((size_t)m->n_actx * m->n_astate);
    if (whisper_encode(m, mel.data(), enc.data()) != 0) throw std::runtime_error(g_error);
    // Whisper's temperature fallback: greedy first; a window whose text compresses above compression_ratio_max or whose
    // average log-probability (the sum over the chosen tokens and the end-of-transcript token, divided by their count + 1)
    // is below avg_logprob_min is decoded again at the next temperature; the first attempt that passes is taken.
    // When every attempt fails, the default (keep_last 0) keeps the attempt with the highest average log-probability (the
    // lowest temperature on a tie). This differs from the reference, which keeps the last attempt (keep_last 1): audio
    // that is genuinely repetitive or hard keeps its greedy text, while a real loop still gets the higher temperatures to
    // escape it.
    static const float temps[6] = {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f};
    const int room = m->n_tctx - (int)prompt.size() - 1;
    std::vector<int> ids, best_ids;
    double best_avg = -INFINITY;
    for (int a = 0; a < 6; ++a) {
        ids.assign((size_t)std::max(1, room), 0);
        double sum_lp = 0.0;
        const int k = decode_tokens(m, enc.data(), prompt.data(), (int)prompt.size(), (int)ids.size(), temps[a],
                                    fallback_seed(seek, a), ids.data(), &sum_lp);
        ids.resize((size_t)k);
        ++m->attempts;
        std::vector<int> text_ids;
        for (int id : ids)
            if (id < m->eot) text_ids.push_back(id);
        const std::string text = detok(m, text_ids);
        const float ratio = whisper_compression_ratio(text.data(), text.size());
        const double avg_lp = sum_lp / (k + 1.0);
        const bool needs_fallback = ratio > m->fallback_cr_max || avg_lp < (double)m->fallback_lp_min;
        if (!needs_fallback || (m->fallback_keep_last && a == 5)) {
            segment_window(m, ids, ts_begin, final_window, n, out);
            return;
        }
        if (avg_lp > best_avg) {  // (strictly greater: the lowest temperature wins a tie)
            best_avg = avg_lp;
            best_ids = ids;
        }
    }
    segment_window(m, best_ids, ts_begin, final_window, n, out);
}

int whisper_transcribe_task(whisper_model_t* m, const float* samples, size_t n, const char* language, const char* task,
                            char* text, size_t text_len) {
    if (!m || (!samples && n > 0) || !text || text_len == 0) { g_error = "whisper_transcribe: bad arguments"; return -1; }
    const bool translate = task && std::string(task) == "translate";
    if (task && !translate && std::string(task) != "transcribe") { g_error = "whisper_transcribe: task transcribe or translate"; return -1; }
    try {
        const std::string lang = std::string("<|") + (language && *language ? language : "en") + "|>";
        std::string all;
        int tokens = 0;
        m->attempts = 0;
        prof_reset();
        if (n <= (size_t)WHISPER_CHUNK_SAMPLES) {  // one window: the prompt without timestamps (as before)
            std::string part;
            const int k = transcribe_window(m, samples, n, lang, translate, part);
            if (k < 0) return -1;
            all = part;
            tokens = k;
        } else {
            // the long form: seek through the clip, each window conditioned on the text so far
            std::vector<int> prev;
            size_t seek = 0;
            while (seek < n) {
                const size_t remain = n - seek;
                const bool last = remain <= (size_t)WHISPER_CHUNK_SAMPLES;
                const size_t len = last ? remain : (size_t)WHISPER_CHUNK_SAMPLES;
                WindowOut w;
                transcribe_window_ts(m, samples + seek, len, lang, translate, prev, last, seek, w);
                all += detok(m, w.text);
                tokens += (int)w.text.size();
                prev.insert(prev.end(), w.text.begin(), w.text.end());
                if (prev.size() > 224) prev.erase(prev.begin(), prev.end() - 224);
                seek += std::min(w.advance, remain);
            }
        }
        if (all.size() + 1 > text_len) { g_error = "whisper_transcribe: the text does not fit"; return -1; }
        std::memcpy(text, all.data(), all.size());
        text[all.size()] = 0;
        prof_report();
        return tokens;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int whisper_transcribe(whisper_model_t* m, const float* samples, size_t n, const char* language, char* text,
                       size_t text_len) {
    return whisper_transcribe_task(m, samples, n, language, "transcribe", text, text_len);
}

}  // extern "C"
