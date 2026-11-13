// Whisper on the host in float32 (whisper.h): the encoder (the conv stem, pre-norm blocks with non-causal attention,
// the final norm) and the decoder (causal self-attention with a cache, cross-attention over the encoder output, the
// tied LM head), with the order of operations of transformers' WhisperForConditionalGeneration.
#include "whisper.h"

#include "audio.h"

#include "gguf_mmap.h"
#include "mel.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    ~whisper_model() {
        if (tok) tokenizer_free(tok);
        if (g) gguf_mmap_close(g);
    }
};

namespace {

// LayerNorm over each row of x [T][C] (population variance, eps inside the root, as torch's)
void layer_norm(const float* x, int T, int C, float eps, const Norm& n, float* y) {
#pragma omp parallel for schedule(static)
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

// y[t][o] = x[t][:] . w[o][:] + b[o] for the T rows of x [T][in]
void linear(const float* x, int T, const Lin& L, float* y) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t)t * L.in;
        float* yr = y + (size_t)t * L.out;
        for (int o = 0; o < L.out; ++o) {
            const float* wr = L.w + (size_t)o * L.in;
            float acc = L.b ? L.b[o] : 0.0f;
            for (int i = 0; i < L.in; ++i) acc += xr[i] * wr[i];
            yr[o] = acc;
        }
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
            for (int ci = 0; ci < C_in; ++ci) {
                const float* wr = w + ((size_t)co * C_in + ci) * K;
                const float* xr = x + (size_t)ci * T_in;
                for (int k = 0; k < K; ++k) {
                    const int i = t * stride + k - pad;
                    if (i >= 0 && i < T_in) acc += wr[k] * xr[i];
                }
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
#pragma omp parallel for schedule(static)
    for (int i = 0; i < Tq; ++i) {
        std::vector<float> s(Tk);
        for (int h = 0; h < H; ++h) {
            const float* qr = q + (size_t)i * C + (size_t)h * d;
            const int n = causal ? std::min(Tk, i + causal_offset + 1) : Tk;
            float mx = -INFINITY;
            for (int j = 0; j < n; ++j) {
                const float* kr = k + (size_t)j * C + (size_t)h * d;
                float dot = 0.0f;
                for (int c = 0; c < d; ++c) dot += qr[c] * kr[c];
                s[j] = dot * scale;
                mx = std::max(mx, s[j]);
            }
            double sum = 0.0;
            for (int j = 0; j < n; ++j) {
                s[j] = std::exp(s[j] - mx);
                sum += s[j];
            }
            float* orow = out + (size_t)i * C + (size_t)h * d;
            for (int c = 0; c < d; ++c) orow[c] = 0.0f;
            for (int j = 0; j < n; ++j) {
                const float p = (float)(s[j] / sum);
                const float* vr = v + (size_t)j * C + (size_t)h * d;
                for (int c = 0; c < d; ++c) orow[c] += p * vr[c];
            }
        }
    }
}

// The token's logits for the last state x [n_tstate]: the tied embedding's dot products
void logits_of(const whisper_model* m, const float* x, float* logits) {
#pragma omp parallel for schedule(static)
    for (int v = 0; v < m->n_vocab; ++v) {
        const float* er = m->tok_emb + (size_t)v * m->n_tstate;
        float acc = 0.0f;
        for (int c = 0; c < m->n_tstate; ++c) acc += x[c] * er[c];
        logits[v] = acc;
    }
}

// The decoder's state for one sequence: the self-attention cache and the cross-attention keys and values
struct DecState {
    std::vector<std::vector<float>> sk, sv;  // per layer [n_tctx][C]
    std::vector<std::vector<float>> ck, cv;  // per layer [n_actx][C], from the encoder output
};

// One decoder step: the token at position pos, its logits into logits [n_vocab]
void decoder_step(const whisper_model* m, DecState& st, int token, int pos, float* logits) {
    const int C = m->n_tstate;
    std::vector<float> x(C), h(C), q(C), kk(C), vv(C), a(C), o(C), f(m->n_tffn);
    for (int c = 0; c < C; ++c) x[c] = m->tok_emb[(size_t)token * C + c] + m->dec_pos[(size_t)pos * C + c];
    for (int l = 0; l < m->n_tlayer; ++l) {
        const DecLayer& L = m->dec[l];
        // self-attention over the positions so far (causal)
        layer_norm(x.data(), 1, C, m->eps, L.attn_ln, h.data());
        linear(h.data(), 1, L.q, q.data());
        linear(h.data(), 1, L.k, kk.data());
        linear(h.data(), 1, L.v, vv.data());
        std::memcpy(&st.sk[l][(size_t)pos * C], kk.data(), sizeof(float) * C);
        std::memcpy(&st.sv[l][(size_t)pos * C], vv.data(), sizeof(float) * C);
        attend(q.data(), 1, st.sk[l].data(), st.sv[l].data(), pos + 1, C, m->n_thead, true, pos, a.data());
        linear(a.data(), 1, L.o, o.data());
        for (int c = 0; c < C; ++c) x[c] += o[c];
        // cross-attention over the encoder output
        layer_norm(x.data(), 1, C, m->eps, L.cross_ln, h.data());
        linear(h.data(), 1, L.cq, q.data());
        attend(q.data(), 1, st.ck[l].data(), st.cv[l].data(), m->n_actx, C, m->n_thead, false, 0, a.data());
        linear(a.data(), 1, L.co, o.data());
        for (int c = 0; c < C; ++c) x[c] += o[c];
        // the feed-forward
        layer_norm(x.data(), 1, C, m->eps, L.mlp_ln, h.data());
        linear(h.data(), 1, L.fc1, f.data());
        for (int c = 0; c < m->n_tffn; ++c) f[c] = gelu(f[c]);
        linear(f.data(), 1, L.fc2, o.data());
        for (int c = 0; c < C; ++c) x[c] += o[c];
    }
    layer_norm(x.data(), 1, C, m->eps, m->dec_post, h.data());
    logits_of(m, h.data(), logits);
}

int argmax_suppressed(const whisper_model* m, const float* logits, bool begin) {
    std::vector<float> l(logits, logits + m->n_vocab);
    for (int t : m->suppress)
        if (t >= 0 && t < m->n_vocab) l[t] = -INFINITY;
    if (begin)
        for (int t : m->begin_suppress)
            if (t >= 0 && t < m->n_vocab) l[t] = -INFINITY;
    int best = 0;
    for (int v = 1; v < m->n_vocab; ++v)
        if (l[v] > l[best]) best = v;
    return best;
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
        const int C = m->n_tstate, T = m->n_actx;
        DecState st;
        st.sk.assign(m->n_tlayer, std::vector<float>((size_t)m->n_tctx * C));
        st.sv.assign(m->n_tlayer, std::vector<float>((size_t)m->n_tctx * C));
        st.ck.assign(m->n_tlayer, std::vector<float>((size_t)T * C));
        st.cv.assign(m->n_tlayer, std::vector<float>((size_t)T * C));
        // the cross-attention keys and values of the encoder output, per layer
        std::vector<float> h((size_t)T * C);
        for (int l = 0; l < m->n_tlayer; ++l) {
            const DecLayer& L = m->dec[l];
            linear(enc, T, L.ck, st.ck[l].data());
            linear(enc, T, L.cv, st.cv[l].data());
        }
        std::vector<float> logits(m->n_vocab);
        for (int i = 0; i < n_prompt; ++i) decoder_step(m, st, prompt[i], i, logits.data());
        int next = argmax_suppressed(m, logits.data(), true);
        int n_out = 0, pos = n_prompt;
        while (n_out < max_new && next != m->eot && pos < m->n_tctx) {
            out_tokens[n_out++] = next;
            decoder_step(m, st, next, pos, logits.data());
            ++pos;
            next = argmax_suppressed(m, logits.data(), false);
        }
        return n_out;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int whisper_transcribe(whisper_model_t* m, const float* samples, size_t n, const char* language, char* text,
                       size_t text_len) {
    if (!m || (!samples && n > 0) || !text || text_len == 0) { g_error = "whisper_transcribe: bad arguments"; return -1; }
    if (n > (size_t)WHISPER_CHUNK_SAMPLES) { g_error = "whisper_transcribe: longer than 30 s (long-form is not supported)"; return -1; }
    try {
        const std::string lang = std::string("<|") + (language && *language ? language : "en") + "|>";
        const int lang_id = whisper_token_id(m, lang.c_str());
        const int transcribe = whisper_token_id(m, "<|transcribe|>"), notimestamps = whisper_token_id(m, "<|notimestamps|>");
        if (lang_id < 0) { g_error = "whisper_transcribe: unknown language " + lang; return -1; }
        if (transcribe < 0 || notimestamps < 0) throw std::runtime_error("whisper: no task or timestamp tokens in the vocabulary");
        const int prompt[4] = {m->sot, lang_id, transcribe, notimestamps};

        std::vector<float> mel((size_t)80 * WHISPER_N_FRAMES);
        if (whisper_log_mel(samples, (int)n, 80, mel.data()) != 0) throw std::runtime_error("whisper_log_mel failed");
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
        if (len < 0 || (size_t)len + 1 > text_len) { g_error = "whisper_transcribe: the text does not fit"; return -1; }
        std::memcpy(text, buf.data(), (size_t)len);
        text[len] = 0;
        return k;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

}  // extern "C"
