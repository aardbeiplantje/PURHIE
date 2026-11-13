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
#include <functional>
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

// y[t][o] = x[t][:] . w[o][:] + b[o] for T rows of x [T][in]: parallel over the output rows, each row read once for the T
// inputs
void linear(const float* x, int T, const Lin& L, float* y) {
#pragma omp parallel for schedule(static)
    for (int o = 0; o < L.out; ++o) {
        const float* wr = L.w + (size_t)o * L.in;
        const float bias = L.b ? L.b[o] : 0.0f;
        for (int t = 0; t < T; ++t) {
            const float* xr = x + (size_t)t * L.in;
            float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
            for (int i = 0; i < L.in; ++i) acc += xr[i] * wr[i];
            y[(size_t)t * L.out + o] = bias + acc;
        }
    }
}

// The causal convolution of x [cin][T] (left padding (k - 1) * dil): y [cout][T]; groups == cin == cout is depthwise.
// Each output row is built from shifted input rows (the inner loop runs along time: contiguous, vectorized); the sums
// run in the same order as the per-sample form (bias, then the input channels, then the taps).
void conv_causal(const float* x, int T, const Conv& C, float* y) {
    const int pad = (C.k - 1) * C.dil;
    if (C.groups == 1) {
#pragma omp parallel for schedule(static)
        for (int o = 0; o < C.cout; ++o) {
            float* yr = y + (size_t)o * T;
            const float bias = C.b ? C.b[o] : 0.0f;
            for (int t = 0; t < T; ++t) yr[t] = bias;
            for (int c = 0; c < C.cin; ++c) {
                const float* xr = x + (size_t)c * T;
                const float* wr = C.w + ((size_t)o * C.cin + c) * C.k;
                for (int k = 0; k < C.k; ++k) {
                    const int sh = k * C.dil - pad;  // y[t] += w * x[t + sh] where 0 <= t + sh < T
                    const float w = wr[k];
                    const int t0 = std::max(0, -sh), t1 = std::min(T, T - sh);
#pragma omp simd
                    for (int t = t0; t < t1; ++t) yr[t] += w * xr[t + sh];
                }
            }
        }
    } else {  // depthwise: one input channel per output channel
#pragma omp parallel for schedule(static)
        for (int o = 0; o < C.cout; ++o) {
            float* yr = y + (size_t)o * T;
            const float* xr = x + (size_t)o * T;
            const float* wr = C.w + (size_t)o * C.k;
            const float bias = C.b ? C.b[o] : 0.0f;
            for (int t = 0; t < T; ++t) yr[t] = bias;
            for (int k = 0; k < C.k; ++k) {
                const int sh = k * C.dil - pad;
                const float w = wr[k];
                const int t0 = std::max(0, -sh), t1 = std::min(T, T - sh);
#pragma omp simd
                for (int t = t0; t < t1; ++t) yr[t] += w * xr[t + sh];
            }
        }
    }
}

// The transposed convolution of x [cin][T] (weight [cin][cout][k], stride s), trimmed to T * s: y [cout][T * s].
// Output u sums x[t] * w[j] over the (t, j) with t * s + j = u; per output channel, the inputs' rows are scattered in
// (the inner loop over t).
void conv_transpose(const float* x, int T, const Tconv& C, float* y) {
    const int L = T * C.stride;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < C.cout; ++o) {
        float* yr = y + (size_t)o * L;
        const float bias = C.b ? C.b[o] : 0.0f;
        for (int u = 0; u < L; ++u) yr[u] = bias;
        for (int c = 0; c < C.cin; ++c) {
            const float* xr = x + (size_t)c * T;
            const float* wr = C.w + ((size_t)c * C.cout + o) * C.k;
            for (int j = 0; j < C.k; ++j) {
                const float w = wr[j];
                // (the outputs past T * stride are trimmed: the taps that land there are dropped)
                const int tmax = std::min(T, (L - j + C.stride - 1) / C.stride);
                for (int t = 0; t < tmax; ++t) yr[(size_t)t * C.stride + j] += w * xr[t];
            }
        }
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

// The encoder (T6 of PLAN.md; qwen-tts's Qwen3TTSTokenizerV2Encoder, a MimiModel with the 12 Hz config): the SEANet (causal
// convolutions, zero-padded; the downsample replicate-padded), the 8-layer causal transformer with a sliding window and
// layer scales, the downsample to the codec's frame rate, and the split residual quantizer's encode (nearest codebook
// entry, residual by residual; the semantic codebook first, then the acoustic ones).
struct EConv {  // weight [cout][cin][k], bias [cout] or NULL
    const float *w = nullptr, *b = nullptr;
    int cin = 0, cout = 0, k = 1, stride = 1, dil = 1;
    bool replicate = false;  // the padding repeats the edge samples (the downsample); zero otherwise
};

struct ERes {  // the SEANet residual block: ELU, a k-conv (dilated), ELU, a 1x1 conv, the residual added
    EConv c1, c2;
};

struct ETf {
    const float *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr;
    const float *attn_scale = nullptr, *ffn_scale = nullptr;
    Lin q, k, v, o, fc1, fc2;
};

struct Enc {
    int filters = 0, kernel = 0, last_kernel = 0, res_kernel = 0, C = 0, L = 0, H = 0, D = 0, FF = 0, sw = 0;
    int valid_q = 0, sem_q = 0, cb = 0, cb_dim = 0, spf = 0, down_k = 0;
    float theta = 1e4f, eps = 1e-5f;
    std::vector<int> ratios;                        // the SEANet's strides, as the config lists them (encoding runs them reversed)
    EConv sea0;                                     // the first convolution (1 -> filters, k 7)
    std::vector<ERes> res;                          // one per stage (the first residual of each stage)
    std::vector<EConv> down;                        // one per stage (strided)
    EConv sea_last;                                 // the last convolution (-> C)
    std::vector<ETf> tf;
    EConv downsample;                               // C -> C, k down_k, stride 2, replicate padding
    Lin proj_sem, proj_ac;                          // the quantizers' input projections (C -> cb_dim), no bias
    std::vector<std::vector<float>> emb_sem, emb_ac;  // per codebook: [cb][cb_dim] = embed_sum / max(usage, 1e-5)
    std::vector<std::vector<float>> emb_sem_norm, emb_ac_norm;  // |e|^2 per entry
};

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
    std::unique_ptr<Enc> enc;  // the encoder (absent in a GGUF without enc.* tensors)
    ~qwen3tts_codec() {
        if (g) gguf_mmap_close(g);
    }
};

namespace {

EConv conv_of(const gguf_model_t* g, const std::string& base, int cin, int cout, int k, int stride, int dil, bool replicate) {
    EConv c;
    c.cin = cin;
    c.cout = cout;
    c.k = k;
    c.stride = stride;
    c.dil = dil;
    c.replicate = replicate;
    c.w = tensor(g, base + ".weight", (size_t)cout * cin * k);
    c.b = tensor(g, base + ".bias", (size_t)cout, true);  // (the downsample has none)
    return c;
}

Lin lin_of(const gguf_model_t* g, const std::string& base, int in, int out, bool bias = false) {
    Lin L;
    L.in = in;
    L.out = out;
    L.w = tensor(g, base + ".weight", (size_t)out * in);
    L.b = bias ? tensor(g, base + ".bias", (size_t)out) : nullptr;
    return L;
}

// Mimi's MimiConv1d arithmetic: the effective kernel, the total padding (kernel - stride), the extra padding on the right
// that makes the frames whole. Returns the output length; left and right receive the padding.
long conv_geometry(const EConv& c, long L, long* left, long* right) {
    const long keff = (long)(c.k - 1) * c.dil + 1, pt = keff - c.stride;
    const long num = L - keff + pt;
    const long nf = num >= 0 ? (num + c.stride - 1) / c.stride : -((-num) / c.stride);  // ceil(num / stride)
    const long ideal = nf * c.stride + keff - pt;
    *left = pt;
    *right = ideal - L;
    return (L + *left + *right - keff) / c.stride + 1;
}

// the convolution of x [cin][L] with the padding of MimiConv1d (causal: left the total, right the extra): [cout][Lout]
std::vector<float> econv(const EConv& c, const std::vector<float>& x, long L, long* Lout) {
    long left = 0, right = 0;
    const long lo = conv_geometry(c, L, &left, &right);
    *Lout = lo;
    const long Lp = L + left + right;
    std::vector<float> xp((size_t)c.cin * Lp);
    for (int i = 0; i < c.cin; ++i) {
        const float* xr = x.data() + (size_t)i * L;
        float* pr = xp.data() + (size_t)i * Lp;
        for (long t = 0; t < Lp; ++t) {
            long src = t - left;
            if (src < 0) src = c.replicate ? 0 : -1;
            else if (src >= L) src = c.replicate ? L - 1 : -1;
            pr[t] = src < 0 ? 0.0f : xr[src];
        }
    }
    std::vector<float> y((size_t)c.cout * lo);
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < c.cout; ++o) {
        for (long t = 0; t < lo; ++t) {
            float acc = c.b ? c.b[o] : 0.0f;
            for (int i = 0; i < c.cin; ++i) {
                const float* wr = c.w + ((size_t)o * c.cin + i) * c.k;
                const float* pr = xp.data() + (size_t)i * Lp + t * c.stride;
                for (int j = 0; j < c.k; ++j) acc += wr[j] * pr[(long)j * c.dil];
            }
            y[(size_t)o * lo + t] = acc;
        }
    }
    return y;
}

void elu_in(std::vector<float>& x) {
    for (float& v : x) v = v > 0.0f ? v : std::exp(v) - 1.0f;
}

// the encoder's transformer over h [L][C] (time-major), causal with the sliding window, RoPE (half rotation), GELU MLPs
void enc_transformer(const Enc& e, std::vector<float>& h, long L) {
    const int C = e.C, H = e.H, D = e.D, half = D / 2;
    std::vector<float> cosv((size_t)L * half), sinv((size_t)L * half);
    for (long p = 0; p < L; ++p)
        for (int i = 0; i < half; ++i) {
            const double f = (double)p * std::pow((double)e.theta, -2.0 * i / D);
            cosv[(size_t)p * half + i] = (float)std::cos(f);
            sinv[(size_t)p * half + i] = (float)std::sin(f);
        }
    auto layer_norm = [&](const float* x, const float* w, const float* b, float* y) {
        double m = 0.0, v = 0.0;
        for (int i = 0; i < C; ++i) m += x[i];
        m /= C;
        for (int i = 0; i < C; ++i) v += (x[i] - m) * (x[i] - m);
        v /= C;
        const double inv = 1.0 / std::sqrt(v + e.eps);
        for (int i = 0; i < C; ++i) y[i] = (float)((x[i] - m) * inv * w[i] + b[i]);
    };
    std::vector<float> n((size_t)L * C), q((size_t)L * C), k((size_t)L * C), v((size_t)L * C), att((size_t)L * C);
    std::vector<float> ff((size_t)L * e.FF);
    for (const ETf& B : e.tf) {
        #pragma omp parallel for schedule(static)
        for (long t = 0; t < L; ++t) layer_norm(&h[(size_t)t * C], B.ln1_w, B.ln1_b, &n[(size_t)t * C]);
        linear(n.data(), (int)L, B.q, q.data());
        linear(n.data(), (int)L, B.k, k.data());
        linear(n.data(), (int)L, B.v, v.data());
        // rotary on q and k: (x1, x2) = (first half, second half) of each head, rotated by the position's angle
        #pragma omp parallel for schedule(static)
        for (long t = 0; t < L; ++t)
            for (int hd = 0; hd < H; ++hd)
                for (int i = 0; i < half; ++i) {
                    const float c = cosv[(size_t)t * half + i], sn = sinv[(size_t)t * half + i];
                    float* qa = &q[(size_t)t * C + hd * D];
                    float* ka = &k[(size_t)t * C + hd * D];
                    const float q1 = qa[i], q2 = qa[i + half], k1 = ka[i], k2 = ka[i + half];
                    qa[i] = q1 * c - q2 * sn;
                    qa[i + half] = q2 * c + q1 * sn;
                    ka[i] = k1 * c - k2 * sn;
                    ka[i + half] = k2 * c + k1 * sn;
                }
        // causal attention over the last sw positions (the sliding window: t - s < sw)
        #pragma omp parallel for schedule(static)
        for (long t = 0; t < L; ++t) {
            const long s0 = std::max<long>(0, t - e.sw + 1);
            std::vector<float> sc((size_t)(t - s0 + 1));
            for (int hd = 0; hd < H; ++hd) {
                const float* qr = &q[(size_t)t * C + hd * D];
                float mx = -INFINITY;
                for (long s = s0; s <= t; ++s) {
                    const float* kr = &k[(size_t)s * C + hd * D];
                    float d = 0.0f;
                    for (int i = 0; i < D; ++i) d += qr[i] * kr[i];
                    sc[(size_t)(s - s0)] = d / std::sqrt((float)D);
                    mx = std::max(mx, sc[(size_t)(s - s0)]);
                }
                double z = 0.0;
                for (float& x : sc) {
                    x = std::exp(x - mx);
                    z += x;
                }
                float* out = &att[(size_t)t * C + hd * D];
                for (int i = 0; i < D; ++i) out[i] = 0.0f;
                for (long s = s0; s <= t; ++s) {
                    const float pw = (float)(sc[(size_t)(s - s0)] / z);
                    const float* vr = &v[(size_t)s * C + hd * D];
                    for (int i = 0; i < D; ++i) out[i] += pw * vr[i];
                }
            }
        }
        std::vector<float> o((size_t)L * C);
        linear(att.data(), (int)L, B.o, o.data());
        for (long t = 0; t < L; ++t)
            for (int c = 0; c < C; ++c) h[(size_t)t * C + c] += B.attn_scale[c] * o[(size_t)t * C + c];
        #pragma omp parallel for schedule(static)
        for (long t = 0; t < L; ++t) layer_norm(&h[(size_t)t * C], B.ln2_w, B.ln2_b, &n[(size_t)t * C]);
        linear(n.data(), (int)L, B.fc1, ff.data());
        for (float& x : ff) x = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));  // GELU (exact)
        std::vector<float> m2((size_t)L * C);
        linear(ff.data(), (int)L, B.fc2, m2.data());
        for (long t = 0; t < L; ++t)
            for (int c = 0; c < C; ++c) h[(size_t)t * C + c] += B.ffn_scale[c] * m2[(size_t)t * C + c];
    }
}

// the nearest entry of a codebook (L2) to r [cb_dim]: the entries' squared norms precomputed (the same argmin)
int nearest(const std::vector<float>& emb, const std::vector<float>& norms, const float* r, int cb, int dim) {
    int best = 0;
    float bv = INFINITY;
    for (int c = 0; c < cb; ++c) {
        const float* e = &emb[(size_t)c * dim];
        float d = 0.0f;
        for (int i = 0; i < dim; ++i) d += r[i] * e[i];
        const float v = norms[c] - 2.0f * d;
        if (v < bv) {
            bv = v;
            best = c;
        }
    }
    return best;
}

std::unique_ptr<Enc> load_encoder(gguf_model_t* g, const std::function<int(const char*)>& kv) {
    std::unique_ptr<Enc> e(new Enc());
    e->filters = kv("enc.n_filters");
    e->kernel = kv("enc.kernel");
    e->last_kernel = kv("enc.last_kernel");
    e->res_kernel = kv("enc.residual_kernel");
    e->C = kv("enc.n_embd");
    e->H = kv("enc.n_head");
    e->D = kv("enc.n_embd") / e->H;
    e->FF = kv("enc.n_ff");
    e->sw = kv("enc.sliding_window");
    e->theta = gguf_arch_float(g, "enc.rope_theta", 1e4f);
    e->eps = gguf_arch_float(g, "enc.norm_eps", 1e-5f);
    e->valid_q = kv("enc.valid_quantizers");
    e->sem_q = kv("enc.semantic_quantizers");
    e->cb = kv("enc.codebook_size");
    e->cb_dim = kv("enc.codebook_dim");
    e->spf = kv("enc.samples_per_frame");
    e->down_k = kv("enc.downsample_kernel");
    const int n_layer = kv("enc.n_layer"), n_stage = 4;
    const int compress = kv("enc.compress");
    if (e->filters <= 0 || e->C <= 0 || e->H <= 0 || e->FF <= 0 || e->sw <= 0 || e->valid_q <= 0 || e->cb <= 0 || e->cb_dim <= 0 ||
        e->spf <= 0 || e->down_k <= 0 || n_layer <= 0 || compress <= 0)
        throw std::runtime_error("qwen3tts_codec: missing or bad encoder hyperparameters");
    {
        int32_t type = 0;
        const void* data = nullptr;
        size_t count = 0;
        if (gguf_kv_get_array(g, "qwen3tts_codec.enc.ratios", &type, &data, &count) != 0)
            throw std::runtime_error("qwen3tts_codec: no enc.ratios");
        for (size_t i = 0; i < count; ++i) e->ratios.push_back((int)static_cast<const uint32_t*>(data)[i]);
    }
    if (e->ratios.size() != (size_t)n_stage) throw std::runtime_error("qwen3tts_codec: the encoder's stages are not four");
    const int F = e->filters;
    // the SEANet's layers as transformers names them: layers.0 the first convolution; each stage s (encoding runs the
    // strides reversed) has its residual at layers.(3s+1).block.1 / block.3, its strided convolution at layers.(3s+3); the
    // last convolution at layers.14
    e->sea0 = conv_of(g, "enc.sea0", 1, F, 7, 1, 1, false);
    for (int s = 0; s < n_stage; ++s) {
        const int ratio = e->ratios[n_stage - 1 - s];
        const int cur = F << s, nxt = F << (s + 1);
        const std::string base = "enc.res" + std::to_string(s);
        ERes r;
        r.c1 = conv_of(g, base + ".c1", cur, cur / compress, e->res_kernel, 1, 1, false);
        r.c2 = conv_of(g, base + ".c2", cur / compress, cur, 1, 1, 1, false);
        e->res.push_back(r);
        e->down.push_back(conv_of(g, "enc.down" + std::to_string(s), cur, nxt, 2 * ratio, ratio, 1, false));
    }
    e->sea_last = conv_of(g, "enc.sealast", F << n_stage, e->C, e->last_kernel, 1, 1, false);
    for (int l = 0; l < n_layer; ++l) {
        const std::string base = "enc.tf" + std::to_string(l) + ".";
        ETf B;
        B.ln1_w = tensor(g, base + "attn_norm.weight", e->C);
        B.ln1_b = tensor(g, base + "attn_norm.bias", e->C);
        B.ln2_w = tensor(g, base + "ffn_norm.weight", e->C);
        B.ln2_b = tensor(g, base + "ffn_norm.bias", e->C);
        B.attn_scale = tensor(g, base + "attn_scale", e->C);
        B.ffn_scale = tensor(g, base + "ffn_scale", e->C);
        B.q = lin_of(g, base + "attn_q", e->C, e->C);
        B.k = lin_of(g, base + "attn_k", e->C, e->C);
        B.v = lin_of(g, base + "attn_v", e->C, e->C);
        B.o = lin_of(g, base + "attn_o", e->C, e->C);
        B.fc1 = lin_of(g, base + "ffn_up", e->C, e->FF);
        B.fc2 = lin_of(g, base + "ffn_down", e->FF, e->C);
        e->tf.push_back(B);
    }
    e->downsample = conv_of(g, "enc.ds", e->C, e->C, e->down_k, 2, 1, true);
    // the quantizers: the semantic one (input projection, codebooks) and the acoustic one
    e->proj_sem = lin_of(g, "enc.qs.proj", e->C, e->cb_dim);
    e->proj_ac = lin_of(g, "enc.qa.proj", e->C, e->cb_dim);
    const int n_ac = e->valid_q - e->sem_q;
    auto codebooks = [&](const std::string& quant, int n, std::vector<std::vector<float>>& emb, std::vector<std::vector<float>>& norms) {
        for (int q = 0; q < n; ++q) {
            const std::string base = "enc.q" + quant + "." + std::to_string(q) + ".";
            const float* sum = tensor(g, base + "sum", (size_t)e->cb * e->cb_dim);
            const float* usage = tensor(g, base + "usage", (size_t)e->cb);
            std::vector<float> table((size_t)e->cb * e->cb_dim), nn((size_t)e->cb);
            for (int c = 0; c < e->cb; ++c) {
                const float u = std::max(usage[c], 1e-5f);
                float sq = 0.0f;
                for (int d = 0; d < e->cb_dim; ++d) {
                    table[(size_t)c * e->cb_dim + d] = sum[(size_t)c * e->cb_dim + d] / u;
                    sq += table[(size_t)c * e->cb_dim + d] * table[(size_t)c * e->cb_dim + d];
                }
                nn[c] = sq;
            }
            emb.push_back(std::move(table));
            norms.push_back(std::move(nn));
        }
    };
    codebooks("s", e->sem_q, e->emb_sem, e->emb_sem_norm);
    codebooks("a", n_ac, e->emb_ac, e->emb_ac_norm);
    return e;
}

// the codes [T][valid_q] of the samples (24 kHz): the encoder's frames, the first ceil(n / samples_per_frame) of them
std::vector<int> enc_encode(const Enc& e, const float* x, size_t n, long* T_out) {
    long L = (long)n;
    std::vector<float> cur(x, x + n);  // [1][L]
    // the SEANet: the first convolution, then the stages (a residual block, ELU, the strided convolution), ELU, the last one
    cur = econv(e.sea0, cur, L, &L);
    for (size_t st = 0; st < e.res.size(); ++st) {
        const ERes& r = e.res[st];
        std::vector<float> a = cur;
        elu_in(a);
        long La = 0;
        a = econv(r.c1, a, L, &La);
        elu_in(a);
        a = econv(r.c2, a, La, &La);
        for (size_t i = 0; i < cur.size(); ++i) cur[i] += a[i];
        elu_in(cur);
        cur = econv(e.down[st], cur, L, &L);
    }
    elu_in(cur);
    cur = econv(e.sea_last, cur, L, &L);
    // the transformer over the frames (time-major), then the downsample (channel-major again)
    const int C = e.C;
    std::vector<float> tm((size_t)L * C);
    for (int c = 0; c < C; ++c)
        for (long t = 0; t < L; ++t) tm[(size_t)t * C + c] = cur[(size_t)c * L + t];
    enc_transformer(e, tm, L);
    std::vector<float> cm((size_t)C * L);
    for (int c = 0; c < C; ++c)
        for (long t = 0; t < L; ++t) cm[(size_t)c * L + t] = tm[(size_t)t * C + c];
    long Ld = 0;
    cm = econv(e.downsample, cm, L, &Ld);
    // the quantizers, frame by frame: the projected frame, its residual through each codebook's nearest entries
    const long T = std::min<long>(Ld, (long)((n + e.spf - 1) / e.spf));
    if (T <= 0) throw std::runtime_error("qwen3tts_codec_encode: the clip is too short");
    *T_out = T;
    const int V = e.valid_q, n_ac = V - e.sem_q;
    std::vector<int> codes((size_t)T * V);
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < T; ++t) {
        std::vector<float> x0(C), r(e.cb_dim);
        for (int c = 0; c < C; ++c) x0[c] = cm[(size_t)c * Ld + t];
        // the semantic codebook, on its projection of the frame
        for (int d = 0; d < e.cb_dim; ++d) {
            float acc = 0.0f;
            for (int i = 0; i < C; ++i) acc += e.proj_sem.w[(size_t)d * C + i] * x0[i];
            r[d] = acc;
        }
        for (int q = 0; q < e.sem_q; ++q) {
            const int idx = nearest(e.emb_sem[q], e.emb_sem_norm[q], r.data(), e.cb, e.cb_dim);
            codes[(size_t)t * V + q] = idx;
            const float* em = &e.emb_sem[q][(size_t)idx * e.cb_dim];
            for (int d = 0; d < e.cb_dim; ++d) r[d] -= em[d];
        }
        // the acoustic codebooks, on the acoustic projection of the same frame
        for (int d = 0; d < e.cb_dim; ++d) {
            float acc = 0.0f;
            for (int i = 0; i < C; ++i) acc += e.proj_ac.w[(size_t)d * C + i] * x0[i];
            r[d] = acc;
        }
        for (int q = 0; q < n_ac; ++q) {
            const int idx = nearest(e.emb_ac[q], e.emb_ac_norm[q], r.data(), e.cb, e.cb_dim);
            codes[(size_t)t * V + e.sem_q + q] = idx;
            const float* em = &e.emb_ac[q][(size_t)idx * e.cb_dim];
            for (int d = 0; d < e.cb_dim; ++d) r[d] -= em[d];
        }
    }
    return codes;
}

}  // namespace

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
        // the encoder, when the GGUF has it (enc.* tensors; a decoder-only GGUF has none)
        size_t enc_bytes = 0;
        if (gguf_get_tensor_ptr(m->g, "enc.sea0.weight", &enc_bytes)) m->enc = load_encoder(m->g, kv);
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

int qwen3tts_codec_encode_frame_samples(const qwen3tts_codec_t* m) { return m && m->enc ? m->enc->spf : 0; }

int qwen3tts_codec_encode(qwen3tts_codec_t* m, const float* samples, size_t n, int** codes, int* T_out) {
    if (!m || !samples || !codes || !T_out) { g_error = "qwen3tts_codec_encode: bad arguments"; return -1; }
    if (!m->enc) { g_error = "the GGUF has no encoder (enc.* tensors)"; return -1; }
    *codes = nullptr;
    *T_out = 0;
    try {
        long T = 0;
        std::vector<int> c = enc_encode(*m->enc, samples, n, &T);
        const size_t count = (size_t)T * m->enc->valid_q;
        int* out = static_cast<int*>(std::malloc(sizeof(int) * std::max<size_t>(count, 1)));
        if (!out) throw std::runtime_error("qwen3tts_codec_encode: no memory");
        std::memcpy(out, c.data(), sizeof(int) * count);
        *codes = out;
        *T_out = (int)T;
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

void qwen3tts_codec_free_codes(int* codes) { std::free(codes); }

}  // extern "C"
