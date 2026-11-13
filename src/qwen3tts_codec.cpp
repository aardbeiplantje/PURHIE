// Qwen3-TTS-Tokenizer-12Hz's decoder on the host in float32 (qwen3tts_codec.h), the order of operations of qwen-tts 0.1.1
// (Qwen3TTSTokenizerV2Decoder). Channel-major tensors [C][T] for the convolutions, [T][C] for the transformer and the
// ConvNeXt's channel-wise steps. The transposed convolutions are written as gathers (each output sample sums the inputs
// that reach it), so every output is independent and the loops parallelize without races.
#include "qwen3tts_codec.h"

#include "gguf_mmap.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string g_error;

struct Lin {
    const float* w = nullptr;  // [out][in]
    const float* b = nullptr;  // [out] or NULL
    int in = 0, out = 0;
};

struct Conv {  // causal, stride 1: weight [cout][cin / groups][k]
    const float* w = nullptr;
    const float* b = nullptr;
    int cin = 0, cout = 0, k = 1, dil = 1, groups = 1;
};

struct Tconv {  // weight [cin][cout][k], stride s; the output is trimmed to T * s
    const float* w = nullptr;
    const float* b = nullptr;
    int cin = 0, cout = 0, k = 1, stride = 1;
};

struct Snake {
    const float* alpha = nullptr;  // log-scale parameters (exp applied)
    const float* beta = nullptr;
    int c = 0;
};

struct ResUnit {
    Snake act1, act2;
    Conv conv1, conv2;
};

struct DecBlock {
    Snake snake;
    Tconv up;
    ResUnit res[3];
    int rate = 1;
};

struct CNX {  // ConvNeXt: depthwise causal conv (k 7), channel LayerNorm, pw 4x, GELU, pw back, gamma, residual
    Conv dw;
    const float *ln_w = nullptr, *ln_b = nullptr, *gamma = nullptr;
    Lin pw1, pw2;
    int dim = 0;
};

struct Up {
    Tconv conv;
    CNX cnx;
};

struct TfBlock {
    Lin q, k, v, o, gate, up, down;
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *attn_scale = nullptr, *ffn_scale = nullptr;
};

const float* tensor(const gguf_model_t* g, const std::string& name, size_t count, bool optional = false) {
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(g, name.c_str(), &bytes);
    if (!p) {
        if (optional) return nullptr;
        throw std::runtime_error("qwen3tts_codec: no tensor " + name);
    }
    if (bytes != count * sizeof(float))
        throw std::runtime_error("qwen3tts_codec: " + name + " has " + std::to_string(bytes) + " bytes, expected " +
                                 std::to_string(count * sizeof(float)));
    return static_cast<const float*>(p);
}

// y[t][o] = x[t][:] . w[o][:] + b[o] for T rows of x [T][in]
void linear(const float* x, int T, const Lin& L, float* y) {
    const long long n = (long long)T * L.out;
#pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < n; ++idx) {
        const int t = (int)(idx / L.out), o = (int)(idx % L.out);
        const float* xr = x + (size_t)t * L.in;
        const float* wr = L.w + (size_t)o * L.in;
        float acc = L.b ? L.b[o] : 0.0f;
        for (int i = 0; i < L.in; ++i) acc += xr[i] * wr[i];
        y[(size_t)t * L.out + o] = acc;
    }
}

// The causal convolution of x [cin][T] (left padding (k - 1) * dil): y [cout][T]; groups == cin == cout is depthwise
void conv_causal(const float* x, int T, const Conv& C, float* y) {
    const int pad = (C.k - 1) * C.dil;
    const long long n = (long long)C.cout * T;
    if (C.groups == 1) {
#pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < n; ++idx) {
            const int o = (int)(idx / T), t = (int)(idx % T);
            float acc = C.b ? C.b[o] : 0.0f;
            for (int c = 0; c < C.cin; ++c) {
                const float* wr = C.w + ((size_t)o * C.cin + c) * C.k;
                const float* xr = x + (size_t)c * T;
                for (int k = 0; k < C.k; ++k) {
                    const int i = t + k * C.dil - pad;
                    if (i >= 0) acc += wr[k] * xr[i];
                }
            }
            y[(size_t)o * T + t] = acc;
        }
    } else {  // depthwise: one input channel per output channel
#pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < n; ++idx) {
            const int o = (int)(idx / T), t = (int)(idx % T);
            float acc = C.b ? C.b[o] : 0.0f;
            const float* wr = C.w + (size_t)o * C.k;
            const float* xr = x + (size_t)o * T;
            for (int k = 0; k < C.k; ++k) {
                const int i = t + k * C.dil - pad;
                if (i >= 0) acc += wr[k] * xr[i];
            }
            y[(size_t)o * T + t] = acc;
        }
    }
}

// The transposed convolution of x [cin][T] (weight [cin][cout][k], stride s), trimmed to T * s: y [cout][T * s].
// Output u sums x[t] * w[j] over the (t, j) with t * s + j = u.
void conv_transpose(const float* x, int T, const Tconv& C, float* y) {
    const int L = T * C.stride;
    const long long n = (long long)C.cout * L;
#pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < n; ++idx) {
        const int o = (int)(idx / L), u = (int)(idx % L);
        float acc = C.b ? C.b[o] : 0.0f;
        for (int c = 0; c < C.cin; ++c) {
            const float* wr = C.w + ((size_t)c * C.cout + o) * C.k;
            const float* xr = x + (size_t)c * T;
            for (int j = 0; j < C.k; ++j) {
                const int d = u - j;
                if (d < 0 || d % C.stride != 0) continue;
                const int t = d / C.stride;
                if (t < T) acc += xr[t] * wr[j];
            }
        }
        y[(size_t)o * L + u] = acc;
    }
}

// SnakeBeta in place on x [c][T]: x + sin^2(x * exp(alpha)) / (exp(beta) + 1e-9)
void snake(float* x, int C, int T, const Snake& s) {
#pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        const float a = std::exp(s.alpha[c]), b = std::exp(s.beta[c]);
        float* xr = x + (size_t)c * T;
        for (int t = 0; t < T; ++t) {
            const float v = std::sin(xr[t] * a);
            xr[t] = xr[t] + (1.0f / (b + 1e-9f)) * v * v;
        }
    }
}

// The ConvNeXt block on x [dim][T] in place
void convnext(float* x, int T, const CNX& B) {
    const int D = B.dim;
    std::vector<float> dw((size_t)D * T), h((size_t)T * D), u((size_t)T * 4 * D);
    conv_causal(x, T, B.dw, dw.data());
    // [dim][T] -> [T][dim]
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < D; ++c) h[(size_t)t * D + c] = dw[(size_t)c * T + t];
    // the channel LayerNorm (eps 1e-6)
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        float* r = &h[(size_t)t * D];
        double mean = 0.0, var = 0.0;
        for (int c = 0; c < D; ++c) mean += r[c];
        mean /= D;
        for (int c = 0; c < D; ++c) var += (r[c] - mean) * (r[c] - mean);
        var /= D;
        const float inv = 1.0f / std::sqrt((float)var + 1e-6f);
        for (int c = 0; c < D; ++c) r[c] = (r[c] - (float)mean) * inv * B.ln_w[c] + B.ln_b[c];
    }
    linear(h.data(), T, B.pw1, u.data());
    for (float& v : u) v = 0.5f * v * (1.0f + std::erf(v * 0.70710678118654752f));  // GELU
    std::vector<float> o((size_t)T * D);
    linear(u.data(), T, B.pw2, o.data());
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < D; ++c) x[(size_t)c * T + t] += o[(size_t)t * D + c] * B.gamma[c];
}

}  // namespace

struct qwen3tts_codec {
    gguf_model_t* g = nullptr;
    int nq = 0, cb = 0, emb = 0, latent = 0, tfL = 0, tfC = 0, tfH = 0, tfD = 0, tfFF = 0, sw = 0, dec_dim = 0, sr = 24000;
    float tf_theta = 1e4f, tf_eps = 1e-5f;
    std::vector<std::vector<float>> emb_tables;  // nq tables [cb][emb]: embedding_sum / cluster_usage
    Lin out_first, out_rest;                     // [512][256], no bias
    Conv pre_conv;
    Lin tf_in, tf_out;
    const float* tf_norm = nullptr;
    std::vector<TfBlock> tf;
    Up up[2];
    Conv dec_in;
    DecBlock dec[4];
    Snake out_snake;
    Conv out_conv;
    ~qwen3tts_codec() {
        if (g) gguf_mmap_close(g);
    }
};

namespace {

// The frames' samples: the quantizer, the pre-conv, the transformer, the upsamplers, the decoder, the clamp. codes [n][nq]
void decode_frames(const qwen3tts_codec* m, const int* codes, int n, float* wav) {
    const int E = m->emb, C = m->latent, TF = m->tfC;
    // the residual quantizer: the first codebook's embeddings through its projection, plus the others' sum through theirs
    std::vector<float> first((size_t)n * E, 0.0f), rest((size_t)n * E, 0.0f);
    for (int t = 0; t < n; ++t) {
        for (int q = 0; q < m->nq; ++q) {
            const int id = codes[(size_t)t * m->nq + q];
            if (id < 0 || id >= m->cb) throw std::runtime_error("qwen3tts_codec: code out of range");
            const float* row = m->emb_tables[q].data() + (size_t)id * E;
            float* dst = q == 0 ? &first[(size_t)t * E] : &rest[(size_t)t * E];
            for (int e = 0; e < E; ++e) dst[e] += row[e];
        }
    }
    std::vector<float> qf((size_t)n * TF), qr((size_t)n * TF), qsum((size_t)n * TF);
    linear(first.data(), n, m->out_first, qf.data());
    linear(rest.data(), n, m->out_rest, qr.data());
    for (size_t i = 0; i < qsum.size(); ++i) qsum[i] = qf[i] + qr[i];
    // [T][512] -> [512][T] for the causal convolution
    std::vector<float> xc((size_t)TF * n), pre((size_t)C * n);
    for (int t = 0; t < n; ++t)
        for (int c = 0; c < TF; ++c) xc[(size_t)c * n + t] = qsum[(size_t)t * TF + c];
    conv_causal(xc.data(), n, m->pre_conv, pre.data());
    // the transformer: [T][1024] -> in -> 8 blocks -> norm -> out, then [1024][T]
    std::vector<float> x((size_t)n * TF), h((size_t)n * TF), tmp((size_t)n * TF);
    {
        std::vector<float> xt((size_t)n * C);
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < C; ++c) xt[(size_t)t * C + c] = pre[(size_t)c * n + t];
        linear(xt.data(), n, m->tf_in, x.data());
    }
    const int D = m->tfD, H = m->tfH, FF = m->tfFF;
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> q((size_t)n * H * D), k((size_t)n * H * D), v((size_t)n * H * D), a((size_t)n * H * D),
        o((size_t)n * TF), g((size_t)n * FF), u((size_t)n * FF);
    for (const TfBlock& B : m->tf) {
        // the rms norm of each row (eps 1e-5), with the weight
        auto rms = [&](const float* in, const float* w, float* out) {
#pragma omp parallel for schedule(static)
            for (int t = 0; t < n; ++t) {
                const float* r = in + (size_t)t * TF;
                double ss = 0.0;
                for (int c = 0; c < TF; ++c) ss += (double)r[c] * r[c];
                const float inv = 1.0f / std::sqrt((float)(ss / TF) + m->tf_eps);
                for (int c = 0; c < TF; ++c) out[(size_t)t * TF + c] = r[c] * inv * w[c];
            }
        };
        rms(x.data(), B.attn_norm, h.data());
        linear(h.data(), n, B.q, q.data());
        linear(h.data(), n, B.k, k.data());
        linear(h.data(), n, B.v, v.data());
        // rope (standard, theta 1e4) on q and k: pairs (i, i + D / 2)
#pragma omp parallel for schedule(static)
        for (int t = 0; t < n; ++t)
            for (int hh = 0; hh < H; ++hh)
                for (int hk = 0; hk < 2; ++hk) {
                    float* r = (hk == 0 ? q.data() : k.data()) + ((size_t)t * H + hh) * D;
                    for (int i = 0; i < D / 2; ++i) {
                        const double f = (double)t * std::pow((double)m->tf_theta, -2.0 * i / D);
                        const float c = (float)std::cos(f), s = (float)std::sin(f);
                        const float x1 = r[i], x2 = r[i + D / 2];
                        r[i] = x1 * c - x2 * s;
                        r[i + D / 2] = x2 * c + x1 * s;
                    }
                }
        // causal attention with the sliding window (key j visible to query t when t - j < window)
#pragma omp parallel for schedule(static)
        for (int t = 0; t < n; ++t) {
            std::vector<float> sc(t + 1);
            const int j0 = std::max(0, t - m->sw + 1);
            for (int hh = 0; hh < H; ++hh) {
                const float* qr = q.data() + ((size_t)t * H + hh) * D;
                float mx = -INFINITY;
                for (int j = j0; j <= t; ++j) {
                    const float* kr = k.data() + ((size_t)j * H + hh) * D;
                    float dot = 0.0f;
                    for (int c = 0; c < D; ++c) dot += qr[c] * kr[c];
                    sc[j] = dot * scale;
                    mx = std::max(mx, sc[j]);
                }
                double sum = 0.0;
                for (int j = j0; j <= t; ++j) {
                    sc[j] = std::exp(sc[j] - mx);
                    sum += sc[j];
                }
                float* orow = a.data() + ((size_t)t * H + hh) * D;
                for (int c = 0; c < D; ++c) orow[c] = 0.0f;
                for (int j = j0; j <= t; ++j) {
                    const float p = (float)(sc[j] / sum);
                    const float* vr = v.data() + ((size_t)j * H + hh) * D;
                    for (int c = 0; c < D; ++c) orow[c] += p * vr[c];
                }
            }
        }
        linear(a.data(), n, B.o, o.data());
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < TF; ++c) x[(size_t)t * TF + c] += B.attn_scale[c] * o[(size_t)t * TF + c];
        rms(x.data(), B.ffn_norm, h.data());
        linear(h.data(), n, B.gate, g.data());
        linear(h.data(), n, B.up, u.data());
        for (size_t i = 0; i < g.size(); ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];  // SiLU(gate) * up
        linear(g.data(), n, B.down, o.data());
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < TF; ++c) x[(size_t)t * TF + c] += B.ffn_scale[c] * o[(size_t)t * TF + c];
    }
    // the final norm and the output projection: [T][512] -> [T][1024] -> [1024][T]
    std::vector<float> cur((size_t)C * n);
    {
        std::vector<float> nrm((size_t)n * TF);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < n; ++t) {
            const float* r = &x[(size_t)t * TF];
            double ss = 0.0;
            for (int c = 0; c < TF; ++c) ss += (double)r[c] * r[c];
            const float inv = 1.0f / std::sqrt((float)(ss / TF) + m->tf_eps);
            for (int c = 0; c < TF; ++c) nrm[(size_t)t * TF + c] = r[c] * inv * m->tf_norm[c];
        }
        std::vector<float> outp((size_t)n * C);
        linear(nrm.data(), n, m->tf_out, outp.data());
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < C; ++c) cur[(size_t)c * n + t] = outp[(size_t)t * C + c];
    }
    // the upsamplers (ratio 2 each): transposed convolution, then the ConvNeXt block
    int L = n;
    for (int s = 0; s < 2; ++s) {
        std::vector<float> nxt((size_t)C * L * 2);
        conv_transpose(cur.data(), L, m->up[s].conv, nxt.data());
        L *= 2;
        convnext(nxt.data(), L, m->up[s].cnx);
        cur.swap(nxt);
    }
    // the decoder: the input convolution, four blocks (snake, transposed convolution, three residual units), the output
    std::vector<float> dcur((size_t)m->dec_dim * L);
    conv_causal(cur.data(), L, m->dec_in, dcur.data());
    int ch = m->dec_dim;
    for (int b = 0; b < 4; ++b) {
        const DecBlock& B = m->dec[b];
        snake(dcur.data(), ch, L, B.snake);
        const int out = B.up.cout;
        std::vector<float> up((size_t)out * L * B.rate);
        conv_transpose(dcur.data(), L, B.up, up.data());
        L *= B.rate;
        dcur.swap(up);
        ch = out;
        for (int r = 0; r < 3; ++r) {
            const ResUnit& R = B.res[r];
            std::vector<float> y = dcur, z((size_t)ch * L);
            snake(y.data(), ch, L, R.act1);
            conv_causal(y.data(), L, R.conv1, z.data());
            snake(z.data(), ch, L, R.act2);
            conv_causal(z.data(), L, R.conv2, y.data());
            for (size_t i = 0; i < dcur.size(); ++i) dcur[i] += y[i];
        }
    }
    snake(dcur.data(), ch, L, m->out_snake);
    std::vector<float> outw((size_t)L);
    conv_causal(dcur.data(), L, m->out_conv, outw.data());
    for (int i = 0; i < L; ++i) wav[i] = std::min(1.0f, std::max(-1.0f, outw[i]));
}

}  // namespace

extern "C" {

qwen3tts_codec_t* qwen3tts_codec_open(const char* gguf_path) {
    try {
        std::unique_ptr<qwen3tts_codec> m(new qwen3tts_codec());
        m->g = gguf_mmap_open(gguf_path);
        if (!m->g) {
            g_error = std::string("cannot open ") + (gguf_path ? gguf_path : "(null)");
            return nullptr;
        }
        if (std::string(gguf_arch(m->g)) != "qwen3tts_codec") throw std::runtime_error("not a qwen3tts codec GGUF");
        auto kv = [&](const char* key) { return (int)gguf_arch_int(m->g, key, -1); };
        m->nq = kv("n_quantizers");
        m->cb = kv("codebook_size");
        m->emb = kv("codebook_dim");
        m->latent = kv("latent_dim");
        m->tfL = kv("tf.n_layer");
        m->tfC = kv("tf.n_embd");
        m->tfH = kv("tf.n_head");
        m->tfD = kv("tf.head_dim");
        m->tfFF = kv("tf.n_ff");
        m->sw = kv("tf.sliding_window");
        m->dec_dim = kv("decoder_dim");
        m->sr = kv("sample_rate");
        m->tf_theta = gguf_arch_float(m->g, "tf.rope_theta", 1e4f);
        m->tf_eps = gguf_arch_float(m->g, "tf.rms_eps", 1e-5f);
        if (m->nq <= 1 || m->cb <= 0 || m->emb <= 0 || m->latent <= 0 || m->tfL <= 0 || m->tfC <= 0 || m->tfH <= 0 ||
            m->tfD <= 0 || m->tfFF <= 0 || m->sw <= 0 || m->dec_dim <= 0)
            throw std::runtime_error("qwen3tts_codec: missing or bad hyperparameters");
        const int E = m->emb, C = m->latent, TF = m->tfC;
        // the codebooks: embedding_sum / max(cluster_usage, 1e-5), per quantizer
        for (int q = 0; q < m->nq; ++q) {
            const std::string base = q == 0 ? "codec.rvq_first.0." : "codec.rvq_rest." + std::to_string(q - 1) + ".";
            const float* es = tensor(m->g, base + "embedding_sum", (size_t)m->cb * E);
            const float* cu = tensor(m->g, base + "cluster_usage", m->cb);
            std::vector<float> table((size_t)m->cb * E);
            for (int i = 0; i < m->cb; ++i) {
                const float u = std::max(cu[i], 1e-5f);
                for (int e = 0; e < E; ++e) table[(size_t)i * E + e] = es[(size_t)i * E + e] / u;
            }
            m->emb_tables.push_back(std::move(table));
        }
        m->out_first = {tensor(m->g, "codec.rvq_first.out.weight", (size_t)TF * E), nullptr, E, TF};
        m->out_rest = {tensor(m->g, "codec.rvq_rest.out.weight", (size_t)TF * E), nullptr, E, TF};
        auto conv = [&](const std::string& base, int cin, int cout, int k, int dil, int groups, bool bias) {
            Conv c;
            c.cin = cin;
            c.cout = cout;
            c.k = k;
            c.dil = dil;
            c.groups = groups;
            c.w = tensor(m->g, base + ".weight", (size_t)cout * (cin / groups) * k);
            c.b = bias ? tensor(m->g, base + ".bias", cout) : nullptr;
            return c;
        };
        auto lin = [&](const std::string& base, int in, int out) {
            Lin L;
            L.in = in;
            L.out = out;
            L.w = tensor(m->g, base + ".weight", (size_t)in * out);
            L.b = tensor(m->g, base + ".bias", out);
            return L;
        };
        m->pre_conv = conv("codec.pre_conv", TF, C, 3, 1, 1, true);
        m->tf_in = lin("codec.tf.in", C, TF);
        m->tf_out = lin("codec.tf.out", TF, C);
        m->tf_norm = tensor(m->g, "codec.tf.norm.weight", TF);
        for (int l = 0; l < m->tfL; ++l) {
            const std::string p = "codec.tf.blk." + std::to_string(l) + ".";
            TfBlock B;
            B.attn_norm = tensor(m->g, p + "attn_norm.weight", TF);
            B.ffn_norm = tensor(m->g, p + "ffn_norm.weight", TF);
            B.attn_scale = tensor(m->g, p + "attn_scale", TF);
            B.ffn_scale = tensor(m->g, p + "ffn_scale", TF);
            auto bl = [&](const std::string& base, int in, int out, bool bias) {
                Lin L;
                L.in = in;
                L.out = out;
                L.w = tensor(m->g, base + ".weight", (size_t)in * out);
                L.b = bias ? tensor(m->g, base + ".bias", out) : nullptr;
                return L;
            };
            const int AD = m->tfH * m->tfD;  // the attention's width (16 heads of 64: wider than the 512 of the hidden state)
            B.q = bl(p + "attn_q", TF, AD, false);
            B.k = bl(p + "attn_k", TF, AD, false);
            B.v = bl(p + "attn_v", TF, AD, false);
            B.o = bl(p + "attn_o", AD, TF, false);
            B.gate = bl(p + "ffn_gate", TF, m->tfFF, false);
            B.up = bl(p + "ffn_up", TF, m->tfFF, false);
            B.down = bl(p + "ffn_down", m->tfFF, TF, false);
            m->tf.push_back(B);
        }
        // the upsamplers: ratio 2 each; the transposed convolution C -> C, the ConvNeXt over C channels
        const int ups[2] = {2, 2};
        for (int s = 0; s < 2; ++s) {
            const std::string p = "codec.up." + std::to_string(s) + ".";
            Up& U = m->up[s];
            U.conv.cin = C;
            U.conv.cout = C;
            U.conv.k = ups[s];
            U.conv.stride = ups[s];
            U.conv.w = tensor(m->g, p + "conv.weight", (size_t)C * C * ups[s]);
            U.conv.b = tensor(m->g, p + "conv.bias", C);
            U.cnx.dim = C;
            U.cnx.dw = conv(p + "cnx.dwconv", C, C, 7, 1, C, true);
            U.cnx.ln_w = tensor(m->g, p + "cnx.norm.weight", C);
            U.cnx.ln_b = tensor(m->g, p + "cnx.norm.bias", C);
            U.cnx.gamma = tensor(m->g, p + "cnx.gamma", C);
            U.cnx.pw1 = lin(p + "cnx.pwconv1", C, 4 * C);
            U.cnx.pw2 = lin(p + "cnx.pwconv2", 4 * C, C);
        }
        // the decoder
        m->dec_in = conv("codec.dec.in", C, m->dec_dim, 7, 1, 1, true);
        const int rates[4] = {8, 5, 4, 3};
        int in_ch = m->dec_dim;
        for (int b = 0; b < 4; ++b) {
            const std::string p = "codec.dec." + std::to_string(b + 1) + ".";
            DecBlock& B = m->dec[b];
            const int out_ch = in_ch / 2;
            B.rate = rates[b];
            B.snake.c = in_ch;
            B.snake.alpha = tensor(m->g, p + "snake.alpha", in_ch);
            B.snake.beta = tensor(m->g, p + "snake.beta", in_ch);
            B.up.cin = in_ch;
            B.up.cout = out_ch;
            B.up.k = 2 * B.rate;
            B.up.stride = B.rate;
            B.up.w = tensor(m->g, p + "up.weight", (size_t)in_ch * out_ch * B.up.k);
            B.up.b = tensor(m->g, p + "up.bias", out_ch);
            const int dil[3] = {1, 3, 9};
            for (int r = 0; r < 3; ++r) {
                const std::string rp = p + "res." + std::to_string(r) + ".";
                ResUnit& R = B.res[r];
                R.act1.c = out_ch;
                R.act1.alpha = tensor(m->g, rp + "act1.alpha", out_ch);
                R.act1.beta = tensor(m->g, rp + "act1.beta", out_ch);
                R.act2.c = out_ch;
                R.act2.alpha = tensor(m->g, rp + "act2.alpha", out_ch);
                R.act2.beta = tensor(m->g, rp + "act2.beta", out_ch);
                R.conv1 = conv(rp + "conv1", out_ch, out_ch, 7, dil[r], 1, true);
                R.conv2 = conv(rp + "conv2", out_ch, out_ch, 1, 1, 1, true);
            }
            in_ch = out_ch;
        }
        m->out_snake.c = in_ch;
        m->out_snake.alpha = tensor(m->g, "codec.dec.out_snake.alpha", in_ch);
        m->out_snake.beta = tensor(m->g, "codec.dec.out_snake.beta", in_ch);
        m->out_conv = conv("codec.dec.out", in_ch, 1, 7, 1, 1, true);
        return m.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void qwen3tts_codec_close(qwen3tts_codec_t* codec) { delete codec; }

const char* qwen3tts_codec_last_error(void) { return g_error.c_str(); }

int qwen3tts_codec_n_quantizers(const qwen3tts_codec_t* c) { return c ? c->nq : 0; }
int qwen3tts_codec_samples_per_frame(const qwen3tts_codec_t* c) {
    if (!c) return 0;
    int s = 1;
    for (int r : {8, 5, 4, 3, 2, 2}) s *= r;
    return s;
}
int qwen3tts_codec_sample_rate(const qwen3tts_codec_t* c) { return c ? c->sr : 0; }

int qwen3tts_codec_decode(qwen3tts_codec_t* m, const int* codes, int T, float** wav, size_t* n_out) {
    if (!m || !codes || T <= 0 || !wav || !n_out) { g_error = "qwen3tts_codec_decode: bad arguments"; return -1; }
    try {
        const int S = qwen3tts_codec_samples_per_frame(m);
        float* out = static_cast<float*>(std::malloc(sizeof(float) * (size_t)T * S));
        if (!out) throw std::runtime_error("qwen3tts_codec_decode: no memory");
        // qwen-tts's chunked_decode: chunks of 300 frames, each with up to 25 frames of left context, the context's samples cut
        const int chunk = 300, context_max = 25;
        int start = 0;
        size_t written = 0;
        while (start < T) {
            const int end = std::min(start + chunk, T);
            const int context = start - context_max > 0 ? context_max : start;
            const int n = end - (start - context);
            std::vector<float> w((size_t)n * S);
            decode_frames(m, codes + (size_t)(start - context) * m->nq, n, w.data());
            std::memcpy(out + written, w.data() + (size_t)context * S, sizeof(float) * (size_t)(end - start) * S);
            written += (size_t)(end - start) * S;
            start = end;
        }
        *wav = out;
        *n_out = written;
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

void qwen3tts_codec_free_wav(float* wav) { std::free(wav); }

}  // extern "C"
