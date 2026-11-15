// Qwen3-TTS-Tokenizer-12Hz's decoder on the host in float32 (qwen3tts_codec.h), the order of operations of qwen-tts 0.1.1
// (Qwen3TTSTokenizerV2Decoder). Channel-major tensors [C][T] for the convolutions, [T][C] for the transformer and the
// ConvNeXt's channel-wise steps. The transposed convolutions are written as gathers (each output sample sums the inputs
// that reach it), so every output is independent and the loops parallelize without races.
#include "qwen3tts_codec.h"

#include "gguf_mmap.h"
#include "kernel_loader.h"

#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
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
        // blocks of 8 output rows and 16 time steps, the 8 x 16 sums in registers: each input vector is loaded once for the
        // 8 rows (the row-by-row form read and wrote each output row once per tap). The input is copied with its zero left
        // padding, so every tap reads a full vector; the zero products add nothing (the same bits). Every output sums in the
        // order it always did (its bias, then the input channels, then the taps).
        constexpr int MO = 4, TV = 64;
        const int padl = pad, Tp = T + pad + TV;
        std::vector<float> xp((size_t)C.cin * Tp, 0.0f);
#pragma omp parallel for schedule(static)
        for (int c = 0; c < C.cin; ++c)
            std::memcpy(xp.data() + (size_t)c * Tp + padl, x + (size_t)c * T, sizeof(float) * (size_t)T);
        const int nob = (C.cout + MO - 1) / MO, ntv = (T + TV - 1) / TV;
#pragma omp parallel for schedule(static) collapse(2)
        for (int ob = 0; ob < nob; ++ob) {
            for (int tv = 0; tv < ntv; ++tv) {
                const int o0 = ob * MO, no = std::min(MO, C.cout - o0);
                const int t0 = tv * TV;
                float acc[MO][TV];
                for (int j = 0; j < MO; ++j) {
                    const float bias = j < no && C.b ? C.b[o0 + j] : 0.0f;
                    for (int t = 0; t < TV; ++t) acc[j][t] = bias;
                }
                for (int c = 0; c < C.cin; ++c) {
                    const float* xr = xp.data() + (size_t)c * Tp;
                    for (int k = 0; k < C.k; ++k) {
                        const float* xv = xr + t0 + k * C.dil;  // y[t] += w * x[t + k * dil - pad]
                        for (int j = 0; j < MO; ++j) {
                            const float w = j < no ? C.w[((size_t)(o0 + j) * C.cin + c) * C.k + k] : 0.0f;
#pragma omp simd
                            for (int t = 0; t < TV; ++t) acc[j][t] += w * xv[t];
                        }
                    }
                }
                for (int j = 0; j < no; ++j) {
                    float* yr = y + (size_t)(o0 + j) * T;
                    for (int t = 0; t < TV && t0 + t < T; ++t) yr[t0 + t] = acc[j][t];
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
    // y[t * stride + r] for each phase r: the taps j = r + m * stride land on it from the input t - m, so each phase is a causal
    // filter over the input (contiguous, vectorized; the stride-scattered form was not). Every output sums in the order it
    // always did (its bias, then the input channels, then the taps in order), so the result is the same bits.
    const int s = C.stride, L = T * s;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < C.cout; ++o) {
        float* yr = y + (size_t)o * L;
        const float bias = C.b ? C.b[o] : 0.0f;
        std::vector<float> acc((size_t)T);
        for (int r = 0; r < s; ++r) {
            for (int t = 0; t < T; ++t) acc[(size_t)t] = bias;
            for (int c = 0; c < C.cin; ++c) {
                const float* xr = x + (size_t)c * T;
                const float* wr = C.w + ((size_t)c * C.cout + o) * C.k;
                for (int m = 0; r + m * s < C.k; ++m) {
                    const float w = wr[r + m * s];
                    // acc[t] += w * x[t - m] for t >= m (the taps that reach past the output's end are not needed)
                    float* ar = acc.data();
#pragma omp simd
                    for (int t = m; t < T; ++t) ar[t] += w * xr[t - m];
                }
            }
            for (int t = 0; t < T; ++t) yr[(size_t)t * s + r] = acc[(size_t)t];
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

// The decoder as a stream (decode_stream). Every layer is causal, and each output sums its inputs in the same order whatever
// piece of the frames it is computed in, so the frames can be decoded in pieces, the state carrying what the next piece
// reads: for each causal layer the last columns of its input (its receptive field's past), for the transformer the keys and
// values of the last sw - 1 frames of each block, and the index of the next frame. Pieces give the same bits as one piece.
struct DecState {
    long long frame0 = 0;                    // the next frame's index (the rope's position and the window's start)
    int nh = 0;                              // the frames in the transformer's cache (at most sw - 1)
    std::vector<std::vector<float>> tail;    // per causal layer (the L_* ids): its input's last g columns, [ch][g]
    std::vector<std::vector<float>> tk, tv;  // per transformer block: the keys and values of the cache, [nh][H * D]
};

// The device path (PLAN.md item 9; codec_gpu.hip): the decoder's convolutions from its input convolution on, in pieces of
// at most DEV_FRAMES frames (the stream's state makes the pieces exact). The weights are read in place (the GGUF mapping
// registered read-only); the activations are an arena of registered pages (UMA: the host writes the first input and the
// histories, the kernels the rest; one sync per piece). QWEN3TTS_CODEC_DEVICE=cpu keeps the host's decoder.
const int DEV_FRAMES = 32;

struct CodecDev {
    struct SnakeDev {
        float *ea = nullptr, *ib = nullptr;  // exp(alpha) and 1 / (exp(beta) + 1e-9) per channel
    };
    std::mutex mu;  // one decode on the device at a time
    hipStream_t stream = nullptr;
    char* base = nullptr;  // the arena
    size_t bytes = 0, used = 0;
    const void* wbase = nullptr;  // the registered GGUF mapping
    float *xin = nullptr, *u[2] = {nullptr, nullptr}, *p = nullptr, *x1 = nullptr, *z = nullptr, *outw = nullptr;
    std::vector<float*> tin, tout;  // per layer id: its history in and out [ch][g] (null: no history on the device)
    SnakeDev bsnake[4], rsnake[4][3][2], osnake;
    float* alloc(size_t n) {
        used = (used + 255) & ~(size_t)255;
        if (used + n * sizeof(float) > bytes) throw std::runtime_error("qwen3tts_codec device: the arena is too small");
        float* r = reinterpret_cast<float*>(base + used);
        used += n * sizeof(float);
        return r;
    }
    ~CodecDev() {
        if (stream) {
            (void)hipStreamSynchronize(stream);
            (void)hipStreamDestroy(stream);
        }
        if (base) {
            (void)hipHostUnregister(base);
            munmap(base, bytes);
        }
        if (wbase) (void)hipHostUnregister(const_cast<void*>(wbase));
    }
};

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
    std::mutex cache_mu;       // the references' decoder states (qwen3tts_codec_decode_after), the last few
    std::vector<std::pair<unsigned long long, DecState>> cache;
    std::unique_ptr<CodecDev> dev;  // the device path's state (absent: the host's decoder)
    ~qwen3tts_codec() {
        dev.reset();  // (the mapping is unregistered before it is unmapped)
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
// the layers' ids in the state (causal layers; each decoder block's transposed convolution, then its residual units' convs)
enum : int { L_PRE = 0, L_UP0 = 1, L_UP0_DW = 2, L_UP1 = 3, L_UP1_DW = 4, L_DEC_IN = 5, L_DEC_BLOCK0 = 6, L_OUT = 34, L_COUNT = 35 };

// the state of the stream at its start: zero histories (the start of the sequence: the zero left padding of each layer)
void init_state(const qwen3tts_codec* m, DecState& st) {
    st.frame0 = 0;
    st.nh = 0;
    st.tk.assign(m->tf.size(), {});
    st.tv.assign(m->tf.size(), {});
    st.tail.assign(L_COUNT, {});
    auto set = [&](int id, int ch, int g) { st.tail[id].assign((size_t)ch * (size_t)g, 0.0f); };
    set(L_PRE, m->tfC, (m->pre_conv.k - 1) * m->pre_conv.dil);
    for (int s2 = 0; s2 < 2; ++s2) {
        set(L_UP0 + 2 * s2, m->latent, (m->up[s2].conv.k - 1) / m->up[s2].conv.stride);
        set(L_UP0_DW + 2 * s2, m->up[s2].cnx.dim, (m->up[s2].cnx.dw.k - 1) * m->up[s2].cnx.dw.dil);
    }
    set(L_DEC_IN, m->latent, (m->dec_in.k - 1) * m->dec_in.dil);
    int ch_in = m->dec_dim;
    for (int b = 0; b < 4; ++b) {
        const int base = L_DEC_BLOCK0 + 7 * b;
        const DecBlock& B = m->dec[b];
        set(base, ch_in, (B.up.k - 1) / B.up.stride);
        const int ch = B.up.cout;
        for (int r = 0; r < 3; ++r) {
            set(base + 1 + 2 * r, ch, (B.res[r].conv1.k - 1) * B.res[r].conv1.dil);
            set(base + 2 + 2 * r, ch, (B.res[r].conv2.k - 1) * B.res[r].conv2.dil);
        }
        ch_in = ch;
    }
    set(L_OUT, ch_in, (m->out_conv.k - 1) * m->out_conv.dil);
}

// y [cout][T]: the causal convolution of the new columns x [ch][T], with the layer's history in its state
void stage_conv(DecState& st, int id, const float* x, int ch, int T, const Conv& C, float* y) {
    const int g = (C.k - 1) * C.dil;
    std::vector<float>& tl = st.tail[id];
    std::vector<float> X((size_t)ch * (g + T));
    for (int c = 0; c < ch; ++c) {
        std::memcpy(&X[(size_t)c * (g + T)], &tl[(size_t)c * g], sizeof(float) * (size_t)g);
        std::memcpy(&X[(size_t)c * (g + T) + g], x + (size_t)c * T, sizeof(float) * (size_t)T);
    }
    std::vector<float> Y((size_t)C.cout * (g + T));
    conv_causal(X.data(), g + T, C, Y.data());
    for (int o = 0; o < C.cout; ++o)
        std::memcpy(y + (size_t)o * T, &Y[(size_t)o * (g + T) + g], sizeof(float) * (size_t)T);
    for (int c = 0; c < ch; ++c)  // the new history: the input's last g columns
        std::memcpy(&tl[(size_t)c * g], &X[(size_t)c * (g + T) + T], sizeof(float) * (size_t)g);
}

// y [cout][T * stride]: the transposed convolution of the new columns (each output depends on the inputs up to its own)
void stage_tconv(DecState& st, int id, const float* x, int ch, int T, const Tconv& C, float* y) {
    const int s = C.stride, g = (C.k - 1) / s;
    std::vector<float>& tl = st.tail[id];
    std::vector<float> X((size_t)ch * (g + T));
    for (int c = 0; c < ch; ++c) {
        std::memcpy(&X[(size_t)c * (g + T)], &tl[(size_t)c * g], sizeof(float) * (size_t)g);
        std::memcpy(&X[(size_t)c * (g + T) + g], x + (size_t)c * T, sizeof(float) * (size_t)T);
    }
    std::vector<float> Y((size_t)C.cout * (g + T) * s);
    conv_transpose(X.data(), g + T, C, Y.data());
    for (int o = 0; o < C.cout; ++o)
        std::memcpy(y + (size_t)o * T * s, &Y[(size_t)o * (g + T) * s + (size_t)g * s], sizeof(float) * (size_t)T * s);
    for (int c = 0; c < ch; ++c)
        std::memcpy(&tl[(size_t)c * g], &X[(size_t)c * (g + T) + T], sizeof(float) * (size_t)g);
}

// the ConvNeXt block on the new columns x [dim][T] (in place), with its depthwise convolution's history
void stage_convnext(DecState& st, int id, float* x, int T, const CNX& B) {
    const int D = B.dim, g = (B.dw.k - 1) * B.dw.dil;
    std::vector<float>& tl = st.tail[id];
    std::vector<float> X((size_t)D * (g + T));
    for (int c = 0; c < D; ++c) {
        std::memcpy(&X[(size_t)c * (g + T)], &tl[(size_t)c * g], sizeof(float) * (size_t)g);
        std::memcpy(&X[(size_t)c * (g + T) + g], x + (size_t)c * T, sizeof(float) * (size_t)T);
    }
    for (int c = 0; c < D; ++c)
        std::memcpy(&tl[(size_t)c * g], &X[(size_t)c * (g + T) + T], sizeof(float) * (size_t)g);
    convnext(X.data(), g + T, B);
    for (int c = 0; c < D; ++c)
        std::memcpy(x + (size_t)c * T, &X[(size_t)c * (g + T) + g], sizeof(float) * (size_t)T);
}

// the rms norm of each of the n rows [TF] (the transformer's norm: the double sum of squares, eps)
void rms_rows(const float* in, int n, int TF, const float* w, float eps, float* out) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < n; ++t) {
        const float* r = in + (size_t)t * TF;
        double ss = 0.0;
        for (int c = 0; c < TF; ++c) ss += (double)r[c] * r[c];
        const float inv = 1.0f / std::sqrt((float)(ss / TF) + eps);
        for (int c = 0; c < TF; ++c) out[(size_t)t * TF + c] = r[c] * inv * w[c];
    }
}

// the transformer over the n new frames x [n][TF] (in place): each block's keys and values are the cache's (the last
// sw - 1 frames) followed by the new frames'; query t attends to the keys of the last sw frames up to itself
void tf_stream(const qwen3tts_codec* m, DecState& st, std::vector<float>& x, int n) {
    const int TF = m->tfC, D = m->tfD, H = m->tfH, FF = m->tfFF, HD = H * D, sw = m->sw;
    const int nh = st.nh, nc = nh + n, keep = std::min(sw - 1, nc);
    const int base = (int)st.frame0 - nh;  // the absolute index of the cache's first frame
    const float scale = 1.0f / std::sqrt((float)D);
    std::vector<float> h((size_t)n * TF), q((size_t)n * HD), k((size_t)n * HD), v((size_t)n * HD), a((size_t)n * HD),
        o((size_t)n * TF), g((size_t)n * FF), u((size_t)n * FF);
    std::vector<float> Kc((size_t)nc * HD), Vc((size_t)nc * HD);
    for (size_t bi = 0; bi < m->tf.size(); ++bi) {
        const TfBlock& B = m->tf[bi];
        rms_rows(x.data(), n, TF, B.attn_norm, m->tf_eps, h.data());
        linear(h.data(), n, B.q, q.data());
        linear(h.data(), n, B.k, k.data());
        linear(h.data(), n, B.v, v.data());
        // rope (standard, theta 1e4) on q and k at the absolute positions: pairs (i, i + D / 2)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const double tp = (double)(st.frame0 + i);
            for (int hh = 0; hh < H; ++hh)
                for (int hk = 0; hk < 2; ++hk) {
                    float* r = (hk == 0 ? q.data() : k.data()) + ((size_t)i * H + hh) * D;
                    for (int d = 0; d < D / 2; ++d) {
                        const double f = tp * std::pow((double)m->tf_theta, -2.0 * d / D);
                        const float c = (float)std::cos(f), s2 = (float)std::sin(f);
                        const float x1 = r[d], x2 = r[d + D / 2];
                        r[d] = x1 * c - x2 * s2;
                        r[d + D / 2] = x2 * c + x1 * s2;
                    }
                }
        }
        if (nh) {
            std::memcpy(Kc.data(), st.tk[bi].data(), sizeof(float) * (size_t)nh * HD);
            std::memcpy(Vc.data(), st.tv[bi].data(), sizeof(float) * (size_t)nh * HD);
        }
        std::memcpy(Kc.data() + (size_t)nh * HD, k.data(), sizeof(float) * (size_t)n * HD);
        std::memcpy(Vc.data() + (size_t)nh * HD, v.data(), sizeof(float) * (size_t)n * HD);
        // causal attention with the sliding window (key j visible to query t when t - j < window)
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const int tabs = (int)st.frame0 + i;
            const int jc0 = std::max(0, tabs - sw + 1) - base, jc1 = nh + i;  // the keys, as indices of the combined arrays
            std::vector<float> sc((size_t)(jc1 - jc0 + 1));
            for (int hh = 0; hh < H; ++hh) {
                const float* qr = q.data() + ((size_t)i * H + hh) * D;
                float mx = -INFINITY;
                for (int jc = jc0; jc <= jc1; ++jc) {
                    const float* kr = Kc.data() + ((size_t)jc * H + hh) * D;
                    float dot = 0.0f;
                    for (int c = 0; c < D; ++c) dot += qr[c] * kr[c];
                    sc[(size_t)(jc - jc0)] = dot * scale;
                    mx = std::max(mx, sc[(size_t)(jc - jc0)]);
                }
                double sum = 0.0;
                for (int jc = jc0; jc <= jc1; ++jc) {
                    sc[(size_t)(jc - jc0)] = std::exp(sc[(size_t)(jc - jc0)] - mx);
                    sum += sc[(size_t)(jc - jc0)];
                }
                float* orow = a.data() + ((size_t)i * H + hh) * D;
                for (int c = 0; c < D; ++c) orow[c] = 0.0f;
                for (int jc = jc0; jc <= jc1; ++jc) {
                    const float p = (float)(sc[(size_t)(jc - jc0)] / sum);
                    const float* vr = Vc.data() + ((size_t)jc * H + hh) * D;
                    for (int c = 0; c < D; ++c) orow[c] += p * vr[c];
                }
            }
        }
        linear(a.data(), n, B.o, o.data());
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < TF; ++c) x[(size_t)t * TF + c] += B.attn_scale[c] * o[(size_t)t * TF + c];
        rms_rows(x.data(), n, TF, B.ffn_norm, m->tf_eps, h.data());
        linear(h.data(), n, B.gate, g.data());
        linear(h.data(), n, B.up, u.data());
        for (size_t i = 0; i < g.size(); ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];  // SiLU(gate) * up
        linear(g.data(), n, B.down, o.data());
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < TF; ++c) x[(size_t)t * TF + c] += B.ffn_scale[c] * o[(size_t)t * TF + c];
        st.tk[bi].assign(Kc.begin() + (size_t)(nc - keep) * HD, Kc.end());
        st.tv[bi].assign(Vc.begin() + (size_t)(nc - keep) * HD, Vc.end());
    }
    st.nh = keep;
}

// The device path's kernels (codec_gpu.co), loaded once per process
namespace cdev {
std::mutex mu;
int state = -1;  // -1: not tried, 0: the host only, 1: loaded
hipModule_t mod = nullptr;
hipFunction_t f_pad = nullptr, f_conv = nullptr, f_tconv = nullptr;

bool load() {
    std::lock_guard<std::mutex> lk(mu);
    if (state >= 0) return state == 1;
    state = 0;
    const char* e = std::getenv("QWEN3TTS_CODEC_DEVICE");
    if (e && std::string(e) != "gpu") return false;
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) return false;
    if (kernel_module_load(&mod, "codec_gpu.co") != hipSuccess) return false;
    if (hipModuleGetFunction(&f_pad, mod, "cd_pad_act") != hipSuccess || hipModuleGetFunction(&f_conv, mod, "cd_conv") != hipSuccess ||
        hipModuleGetFunction(&f_tconv, mod, "cd_tconv") != hipSuccess)
        return false;
    state = 1;
    return true;
}

void launch(const CodecDev& d, hipFunction_t f, unsigned gx, unsigned gy, unsigned shm, void** args) {
    if (hipModuleLaunchKernel(f, gx, gy, 1, 256, 1, 1, shm, d.stream, args, nullptr) != hipSuccess)
        throw std::runtime_error("qwen3tts_codec device: a kernel launch failed");
}

// dst [ch][g + L] = [tail | act(src [ch][L])], tail_out = its last g columns (cd_pad_act)
void pad_act(const CodecDev& d, const float* src, int lds, const float* tail, float* dst, float* tail_out, int ch, int L, int g,
             const CodecDev::SnakeDev* sn) {
    const int act = sn ? 1 : 0;
    const float *ea = sn ? sn->ea : nullptr, *ib = sn ? sn->ib : nullptr;
    void* args[] = {(void*)&src, (void*)&lds, (void*)&tail, (void*)&dst, (void*)&tail_out, (void*)&ch,
                    (void*)&L,   (void*)&g,   (void*)&act,  (void*)&ea,  (void*)&ib};
    launch(d, f_pad, (unsigned)((g + L + 255) / 256), (unsigned)ch, 0, args);
}

// Y [cout][ldy] (T columns) from the padded X [cin][ldx] (cd_conv); mode 0: store, 1: add into Y, 2: SnakeBeta then store
void conv(const CodecDev& d, const float* X, int ldx, const Conv& C, float* Y, int ldy, int T, int mode,
          const CodecDev::SnakeDev* sn) {
    const int KC = C.k == 1 ? 64 : 16;
    const float *W = C.w, *b = C.b, *ea = sn ? sn->ea : nullptr, *ib = sn ? sn->ib : nullptr;
    const int cin = C.cin, cout = C.cout, k = C.k, dil = C.dil;
    void* args[] = {(void*)&X,   (void*)&ldx, (void*)&W, (void*)&b,  (void*)&Y,    (void*)&ldy, (void*)&cin, (void*)&cout,
                    (void*)&k,   (void*)&dil, (void*)&T, (void*)&KC, (void*)&mode, (void*)&ea,  (void*)&ib};
    const unsigned shm = (unsigned)((32 * KC * k + KC * (128 + (k - 1) * dil)) * sizeof(float));
    launch(d, f_conv, (unsigned)((cout + 31) / 32), (unsigned)((T + 127) / 128), shm, args);
}

// Y [cout][ldy] (T * stride columns) from the padded X [cin][ldx] (cd_tconv)
void tconv(const CodecDev& d, const float* X, int ldx, const Tconv& C, float* Y, int ldy, int T) {
    const int KC = 48;
    const float *W = C.w, *b = C.b;
    const int cin = C.cin, cout = C.cout, k = C.k, s = C.stride, g = k / s - 1;
    void* args[] = {(void*)&X,   (void*)&ldx,  (void*)&W, (void*)&b, (void*)&Y, (void*)&ldy,
                    (void*)&cin, (void*)&cout, (void*)&k, (void*)&s, (void*)&T, (void*)&KC};
    const unsigned shm = (unsigned)(KC * ((32 / s + 2) * k + 128 + g) * sizeof(float));
    launch(d, f_tconv, (unsigned)((cout * s + 31) / 32), (unsigned)((T + 127) / 128), shm, args);
}
}  // namespace cdev

// The device path's state for a model: the arena sized for DEV_FRAMES frames, the SnakeBeta parameters, the weights'
// mapping registered. Leaves m->dev absent when the device or the model's shape does not fit (the host's decoder runs).
void dev_init(qwen3tts_codec* m) {
    if (!cdev::load() || gguf_n_maps(m->g) != 1) return;
    if (m->dec_in.groups != 1 || m->out_conv.groups != 1) return;
    for (int b = 0; b < 4; ++b) {
        const DecBlock& B = m->dec[b];
        if (B.up.k != B.up.stride && B.up.k != 2 * B.up.stride) return;
        for (int r = 0; r < 3; ++r)
            if (B.res[r].conv2.k != 1 || B.res[r].conv1.groups != 1 || B.res[r].conv2.groups != 1) return;
    }
    try {
        std::unique_ptr<CodecDev> d(new CodecDev());
        // the buffers' sizes for a piece of DEV_FRAMES frames
        int L = DEV_FRAMES;
        for (int s2 = 0; s2 < 2; ++s2) L *= m->up[s2].conv.stride;
        const int L0 = L, gin = (m->dec_in.k - 1) * m->dec_in.dil;
        size_t n_xin = (size_t)m->latent * (gin + L0), n_u = (size_t)m->dec_dim * L0, n_p = 0, n_x1 = 0, n_z = 0, n_small = 0;
        int ch = m->dec_dim, allocs = 8;
        for (int b = 0; b < 4; ++b) {
            const DecBlock& B = m->dec[b];
            const int g = (B.up.k - 1) / B.up.stride;
            n_p = std::max(n_p, (size_t)ch * (g + L));
            n_small += (size_t)2 * ch * g + (size_t)2 * ch;
            L *= B.rate;
            const int out = B.up.cout;
            n_u = std::max(n_u, (size_t)out * L);
            n_z = std::max(n_z, (size_t)out * L);
            for (int r = 0; r < 3; ++r) {
                const int g1 = (B.res[r].conv1.k - 1) * B.res[r].conv1.dil;
                n_x1 = std::max(n_x1, (size_t)out * (g1 + L));
                n_small += (size_t)2 * out * g1 + (size_t)4 * out;
            }
            ch = out;
            allocs += 4 + 3 * 6;
        }
        const int gout = (m->out_conv.k - 1) * m->out_conv.dil;
        n_x1 = std::max(n_x1, (size_t)ch * (gout + L));
        n_small += (size_t)2 * ch * gout + (size_t)2 * ch;
        const size_t floats = n_xin + 2 * n_u + n_p + n_x1 + n_z + (size_t)L + n_small;
        d->bytes = ((floats * sizeof(float) + (size_t)256 * (allocs + 8) + 4095) / 4096) * 4096;
        void* mem = mmap(nullptr, d->bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem == MAP_FAILED) return;
        if (hipHostRegister(mem, d->bytes, hipExtHostRegisterCoarseGrained) != hipSuccess) {
            munmap(mem, d->bytes);
            return;
        }
        d->base = static_cast<char*>(mem);
        d->xin = d->alloc(n_xin);
        d->u[0] = d->alloc(n_u);
        d->u[1] = d->alloc(n_u);
        d->p = d->alloc(n_p);
        d->x1 = d->alloc(n_x1);
        d->z = d->alloc(n_z);
        d->outw = d->alloc((size_t)L);
        d->tin.assign(L_COUNT, nullptr);
        d->tout.assign(L_COUNT, nullptr);
        auto snake_of = [&](const Snake& sn) {
            CodecDev::SnakeDev sd;
            sd.ea = d->alloc((size_t)sn.c);
            sd.ib = d->alloc((size_t)sn.c);
            for (int c = 0; c < sn.c; ++c) {  // the host's values (snake)
                sd.ea[c] = std::exp(sn.alpha[c]);
                sd.ib[c] = 1.0f / (std::exp(sn.beta[c]) + 1e-9f);
            }
            return sd;
        };
        auto tails = [&](int id, int chn, int g) {
            if (g <= 0) return;
            d->tin[id] = d->alloc((size_t)chn * g);
            d->tout[id] = d->alloc((size_t)chn * g);
        };
        ch = m->dec_dim;
        for (int b = 0; b < 4; ++b) {
            const int base = L_DEC_BLOCK0 + 7 * b;
            const DecBlock& B = m->dec[b];
            tails(base, ch, (B.up.k - 1) / B.up.stride);
            d->bsnake[b] = snake_of(B.snake);
            ch = B.up.cout;
            for (int r = 0; r < 3; ++r) {
                tails(base + 1 + 2 * r, ch, (B.res[r].conv1.k - 1) * B.res[r].conv1.dil);
                d->rsnake[b][r][0] = snake_of(B.res[r].act1);
                d->rsnake[b][r][1] = snake_of(B.res[r].act2);
            }
        }
        tails(L_OUT, ch, gout);
        d->osnake = snake_of(m->out_snake);
        size_t map_size = 0;
        const void* wb = gguf_map_base(m->g, &map_size);
        if (!wb || hipHostRegister(const_cast<void*>(wb), map_size, hipHostRegisterReadOnly | hipExtHostRegisterCoarseGrained) != hipSuccess)
            return;
        d->wbase = wb;
        if (hipStreamCreateWithFlags(&d->stream, hipStreamNonBlocking) != hipSuccess) return;
        m->dev = std::move(d);
    } catch (const std::exception&) {
        // (the host's decoder runs)
    }
}

// The decoder's convolutions of a piece on the device: cur [C][L0] (the upsamplers' output) through the input convolution,
// the four blocks and the output convolution; wav gets the piece's samples. The histories of these layers pass through
// the arena (in before the launches, out after the one sync).
void decode_dev(const qwen3tts_codec* m, DecState& st, const float* cur, int L0, float* wav) {
    CodecDev& d = *m->dev;
    std::lock_guard<std::mutex> lk(d.mu);
    const int C = m->latent, gin = (m->dec_in.k - 1) * m->dec_in.dil;
    {  // the input convolution's padded input on the host: its history, then cur; its next history is the last columns
        std::vector<float>& tl = st.tail[L_DEC_IN];
        const size_t ld = (size_t)gin + L0;
        for (int c = 0; c < C; ++c) {
            std::memcpy(d.xin + c * ld, &tl[(size_t)c * gin], sizeof(float) * (size_t)gin);
            std::memcpy(d.xin + c * ld + gin, cur + (size_t)c * L0, sizeof(float) * (size_t)L0);
        }
        for (int c = 0; c < C; ++c) std::memcpy(&tl[(size_t)c * gin], d.xin + c * ld + L0, sizeof(float) * (size_t)gin);
    }
    for (int id = L_DEC_BLOCK0; id < L_COUNT; ++id)
        if (d.tin[id]) std::memcpy(d.tin[id], st.tail[id].data(), sizeof(float) * st.tail[id].size());
    int ui = 0, L = L0, ch = m->dec_dim;
    float* dcur = d.u[0];
    cdev::conv(d, d.xin, gin + L0, m->dec_in, dcur, L0, L0, 0, nullptr);
    for (int b = 0; b < 4; ++b) {
        const int base = L_DEC_BLOCK0 + 7 * b;
        const DecBlock& B = m->dec[b];
        const int g = (B.up.k - 1) / B.up.stride;
        cdev::pad_act(d, dcur, L, d.tin[base], d.p, d.tout[base], ch, L, g, &d.bsnake[b]);
        ui ^= 1;
        float* up = d.u[ui];
        cdev::tconv(d, d.p, g + L, B.up, up, L * B.rate, L);
        L *= B.rate;
        dcur = up;
        ch = B.up.cout;
        for (int r = 0; r < 3; ++r) {
            const ResUnit& R = B.res[r];
            const int id = base + 1 + 2 * r, g1 = (R.conv1.k - 1) * R.conv1.dil;
            cdev::pad_act(d, dcur, L, d.tin[id], d.x1, d.tout[id], ch, L, g1, &d.rsnake[b][r][0]);
            cdev::conv(d, d.x1, g1 + L, R.conv1, d.z, L, L, 2, &d.rsnake[b][r][1]);
            cdev::conv(d, d.z, L, R.conv2, dcur, L, L, 1, nullptr);
        }
    }
    const int gout = (m->out_conv.k - 1) * m->out_conv.dil;
    cdev::pad_act(d, dcur, L, d.tin[L_OUT], d.x1, d.tout[L_OUT], ch, L, gout, &d.osnake);
    cdev::conv(d, d.x1, gout + L, m->out_conv, d.outw, L, L, 0, nullptr);
    if (hipStreamSynchronize(d.stream) != hipSuccess) throw std::runtime_error("qwen3tts_codec device: the stream failed");
    for (int id = L_DEC_BLOCK0; id < L_COUNT; ++id)
        if (d.tout[id]) std::memcpy(st.tail[id].data(), d.tout[id], sizeof(float) * st.tail[id].size());
    for (int i = 0; i < L; ++i) wav[i] = std::min(1.0f, std::max(-1.0f, d.outw[i]));
}

// The decoder on n new frames (codes [n][nq]) from the state st, which it advances; wav gets n * 1920 samples. The same
// operations as the whole sequence's decode, in the order it does them (see DecState).
void decode_stream(const qwen3tts_codec* m, DecState& st, const int* codes, int n, float* wav) {
    if (n <= 0) return;
    if (m->dev && n > DEV_FRAMES) {  // the device path decodes pieces of DEV_FRAMES frames (its arena's size): the same bits
        const int spf = qwen3tts_codec_samples_per_frame(m);
        for (int off = 0; off < n; off += DEV_FRAMES)
            decode_stream(m, st, codes + (size_t)off * m->nq, std::min(DEV_FRAMES, n - off), wav + (size_t)off * spf);
        return;
    }
    const int E = m->emb, C = m->latent, TF = m->tfC;
    const bool prof = std::getenv("QWEN3TTS_CODEC_PROFILE") != nullptr;
    const std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    // the residual quantizer: the first codebook's embeddings through its projection, plus the others' sum through theirs
    std::vector<float> first((size_t)n * E, 0.0f), rest((size_t)n * E, 0.0f);
    for (int t = 0; t < n; ++t) {
        for (int qq = 0; qq < m->nq; ++qq) {
            const int id = codes[(size_t)t * m->nq + qq];
            if (id < 0 || id >= m->cb) throw std::runtime_error("qwen3tts_codec: code out of range");
            const float* row = m->emb_tables[qq].data() + (size_t)id * E;
            float* dst = qq == 0 ? &first[(size_t)t * E] : &rest[(size_t)t * E];
            for (int e = 0; e < E; ++e) dst[e] += row[e];
        }
    }
    std::vector<float> qf((size_t)n * TF), qr((size_t)n * TF), qsum((size_t)n * TF);
    linear(first.data(), n, m->out_first, qf.data());
    linear(rest.data(), n, m->out_rest, qr.data());
    for (size_t i = 0; i < qsum.size(); ++i) qsum[i] = qf[i] + qr[i];
    // [T][TF] -> [TF][T] for the causal convolution
    std::vector<float> xc((size_t)TF * n), pre((size_t)C * n);
    for (int t = 0; t < n; ++t)
        for (int c = 0; c < TF; ++c) xc[(size_t)c * n + t] = qsum[(size_t)t * TF + c];
    stage_conv(st, L_PRE, xc.data(), TF, n, m->pre_conv, pre.data());
    // the transformer: [T][C] -> in -> the blocks -> [T][TF]
    std::vector<float> x((size_t)n * TF);
    {
        std::vector<float> xt((size_t)n * C);
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < C; ++c) xt[(size_t)t * C + c] = pre[(size_t)c * n + t];
        linear(xt.data(), n, m->tf_in, x.data());
    }
    tf_stream(m, st, x, n);
    // the final norm and the output projection: [T][TF] -> [T][C] -> [C][T]
    std::vector<float> cur((size_t)C * n);
    {
        std::vector<float> nrm((size_t)n * TF);
        rms_rows(x.data(), n, TF, m->tf_norm, m->tf_eps, nrm.data());
        std::vector<float> outp((size_t)n * C);
        linear(nrm.data(), n, m->tf_out, outp.data());
        for (int t = 0; t < n; ++t)
            for (int c = 0; c < C; ++c) cur[(size_t)c * n + t] = outp[(size_t)t * C + c];
    }
    // the upsamplers (ratio 2 each): transposed convolution, then the ConvNeXt block
    int L = n;
    for (int s2 = 0; s2 < 2; ++s2) {
        std::vector<float> nxt((size_t)C * L * 2);
        stage_tconv(st, L_UP0 + 2 * s2, cur.data(), C, L, m->up[s2].conv, nxt.data());
        L *= 2;
        stage_convnext(st, L_UP0_DW + 2 * s2, nxt.data(), L, m->up[s2].cnx);
        cur.swap(nxt);
    }
    // the decoder: the input convolution, four blocks (snake, transposed convolution, three residual units), the output
    if (m->dev) {  // on the device (codec_gpu.hip): the same layers, the histories through the arena
        const std::chrono::steady_clock::time_point t_dev = std::chrono::steady_clock::now();
        decode_dev(m, st, cur.data(), L, wav);
        st.frame0 += n;
        if (prof)  // QWEN3TTS_CODEC_PROFILE: a piece's stages (stderr)
            std::fprintf(stderr, "qwen3tts_codec profile (gpu): %d frames: host stages %.1f ms, the device's convolutions %.1f ms\n", n,
                         std::chrono::duration<double, std::milli>(t_dev - t_start).count(),
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dev).count());
        return;
    }
    std::vector<float> dcur((size_t)m->dec_dim * L);
    stage_conv(st, L_DEC_IN, cur.data(), C, L, m->dec_in, dcur.data());
    int ch = m->dec_dim;
    for (int b = 0; b < 4; ++b) {
        const int base = L_DEC_BLOCK0 + 7 * b;
        const DecBlock& B = m->dec[b];
        snake(dcur.data(), ch, L, B.snake);
        const int out = B.up.cout;
        std::vector<float> up((size_t)out * L * B.rate);
        stage_tconv(st, base, dcur.data(), ch, L, B.up, up.data());
        L *= B.rate;
        dcur.swap(up);
        ch = out;
        for (int r = 0; r < 3; ++r) {
            const ResUnit& R = B.res[r];
            std::vector<float> y = dcur, z((size_t)ch * L);
            snake(y.data(), ch, L, R.act1);
            stage_conv(st, base + 1 + 2 * r, y.data(), ch, L, R.conv1, z.data());
            snake(z.data(), ch, L, R.act2);
            stage_conv(st, base + 2 + 2 * r, z.data(), ch, L, R.conv2, y.data());
            for (size_t i = 0; i < dcur.size(); ++i) dcur[i] += y[i];
        }
    }
    snake(dcur.data(), ch, L, m->out_snake);
    std::vector<float> outw((size_t)L);
    stage_conv(st, L_OUT, dcur.data(), ch, L, m->out_conv, outw.data());
    for (int i = 0; i < L; ++i) wav[i] = std::min(1.0f, std::max(-1.0f, outw[i]));
    st.frame0 += n;
}

// the whole piece of frames at once (from the start of the sequence)
void decode_frames(const qwen3tts_codec* m, const int* codes, int n, float* wav) {
    DecState st;
    init_state(m, st);
    decode_stream(m, st, codes, n, wav);
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
        dev_init(m.get());  // the device path, when a device is there (QWEN3TTS_CODEC_DEVICE=cpu: the host's decoder)
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

// the reference frames' hash (the cache's key): FNV-1a over the codes, and the count
static unsigned long long codes_key(const int* p, size_t n, int R) {
    unsigned long long h = 1469598103934665603ull;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n * sizeof(int); ++i) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    return h ^ (static_cast<unsigned long long>(R) * 0x9E3779B97F4A7C15ull);
}

int qwen3tts_codec_decode_after(qwen3tts_codec_t* m, const int* ref_codes, int R, const int* codes, int T, float** wav,
                                size_t* n_out) {
    if (!m || (!ref_codes && R > 0) || R < 0 || !codes || T <= 0 || !wav || !n_out) {
        g_error = "qwen3tts_codec_decode_after: bad arguments";
        return -1;
    }
    try {
        const int S = qwen3tts_codec_samples_per_frame(m);
        const unsigned long long key = codes_key(ref_codes, (size_t)R * m->nq, R);
        DecState st;
        bool hit = false;
        {
            std::lock_guard<std::mutex> lk(m->cache_mu);
            for (const auto& e : m->cache)
                if (e.first == key) {
                    st = e.second;
                    hit = true;
                    break;
                }
        }
        if (!hit) {  // the reference decoded once: its samples are the context's, not the output's
            init_state(m, st);
            if (R > 0) {
                std::vector<float> ctx((size_t)R * S);
                decode_stream(m, st, ref_codes, R, ctx.data());
            }
            std::lock_guard<std::mutex> lk(m->cache_mu);
            m->cache.emplace_back(key, st);
            if (m->cache.size() > 8) m->cache.erase(m->cache.begin());
        }
        float* out = static_cast<float*>(std::malloc(sizeof(float) * (size_t)T * S));
        if (!out) throw std::runtime_error("qwen3tts_codec_decode_after: no memory");
        decode_stream(m, st, codes, T, out);
        *wav = out;
        *n_out = (size_t)T * S;
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
