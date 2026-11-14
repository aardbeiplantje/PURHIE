// Qwen3-TTS's talker and code predictor on the host in float32 (qwen3tts.h): the prefill of the in-context voice clone
// (Qwen3TTSForConditionalGeneration.generate and generate_icl_prompt, qwen-tts 0.1.1), the talker's decoder (Qwen3 blocks:
// RMS norms, per-head q and k norms, GQA, SiLU-gated MLP, causal attention), and the code predictor (the same blocks,
// 16 positions per frame). The rotary is the standard one: the talker's M-RoPE with its three position axes equal (text
// only, no padding) is the same rotation.
#include "qwen3tts.h"

#include "gguf_mmap.h"
#include "mel.h"
#include "qmatvec.h"
#include "tokenizer.h"

#include <hip/hip_runtime.h>

#include "kernel_loader.h"
#include <sys/mman.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string g_error;

// A weight matrix read in place from the GGUF map (UMA: no copy): F32, or BF16 as the checkpoint stores it (the bf16
// values are widened inside the products: half the bytes of F32, and the same numbers).
struct Mat {
    const float* f = nullptr;     // F32 [rows][cols]
    const uint16_t* h = nullptr;  // BF16 [rows][cols]
    int rows = 0, cols = 0;
};

inline float bf16_val(uint16_t v) {
    const uint32_t u = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// the dot products (the reductions vectorize with the simd pragma: AVX-512 with the host object's flags)
inline float dot_f32(const float* a, const float* b, int n) {
    float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

inline float dot_bf16(const float* a, const uint16_t* b, int n) {
    float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
    for (int i = 0; i < n; ++i) acc += a[i] * bf16_val(b[i]);
    return acc;
}

// the dot product of row r of M with x (M.cols values)
inline float mat_dot(const Mat& M, int r, const float* x) {
    return M.f ? dot_f32(x, M.f + (size_t)r * M.cols, M.cols) : dot_bf16(x, M.h + (size_t)r * M.cols, M.cols);
}

// row r of M as floats (the embedding lookups)
inline void mat_row(const Mat& M, int r, float* out) {
    if (M.f) {
        std::memcpy(out, M.f + (size_t)r * M.cols, sizeof(float) * M.cols);
    } else {
        const uint16_t* src = M.h + (size_t)r * M.cols;
        for (int i = 0; i < M.cols; ++i) out[i] = bf16_val(src[i]);
    }
}

// The device paths (UMA: the GPU reads the registered GGUF mapping in place; the activations pass through two pinned buffers,
// a few KB per call against MB of weights). QWEN3TTS_DEVICE selects: "gpu" (the default when a device is present: the
// talker's and the code predictor's blocks run on the device, one token at a time, see dev:: below), "linear" (one
// staged launch per BF16 linear: measured 77 ms against the host's 52 ms per frame, kept for comparison), "cpu" (host
// only). The staging buffers are shared by the process (gpu::mu); one fused talker runs at a time (gpu::run_mu).
namespace gpu {
std::mutex mu;      // the registered mappings and the staging buffers (linear mode)
std::mutex run_mu;  // one fused talker at a time (the device path of qwen3tts_generate)
std::vector<std::pair<const void*, size_t>> mapped;  // the registered GGUF mappings (read-only)
float* stage_in = nullptr;
float* stage_out = nullptr;
size_t stage_cap = 0;  // floats
int state = -1;        // -1: not tried, 0: host only, 1: device
bool per_linear = false;  // QWEN3TTS_DEVICE=linear: the staged BF16 linears (in_mapping is then true for the mappings)

// The device path is on when a device is present, the kernels load and the staging buffers are pinned.
bool ready() {
    std::lock_guard<std::mutex> lk(mu);
    if (state >= 0) return state == 1;
    state = 0;
    const char* dev = getenv("QWEN3TTS_DEVICE");
    const std::string want = dev ? dev : "gpu";
    if (want != "gpu" && want != "linear") return false;
    per_linear = want == "linear";
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) return false;
    if (qmatvec_init("matvec_q.co") != 0) return false;
    stage_cap = (size_t)1 << 20;  // 4 MB each: a prefill's 84 rows of the widest input or output fit many times over
    if (hipHostMalloc(reinterpret_cast<void**>(&stage_in), sizeof(float) * stage_cap, hipHostMallocDefault) != hipSuccess) return false;
    if (hipHostMalloc(reinterpret_cast<void**>(&stage_out), sizeof(float) * stage_cap, hipHostMallocDefault) != hipSuccess) return false;
    state = 1;
    return true;
}

// Register a GGUF mapping for the device (read-only, GPU-cacheable: the weights are read once per token, the cache holds
// the rows of one matvec). Returns 1 when registered.
int map(const void* base, size_t size) {
    if (!ready() || !base) return 0;
    if (hipHostRegister(const_cast<void*>(base), size, hipHostRegisterReadOnly | hipExtHostRegisterCoarseGrained) != hipSuccess)
        return 0;
    std::lock_guard<std::mutex> lk(mu);
    mapped.emplace_back(base, size);
    return 1;
}

void unmap(const void* base) {
    if (!base) return;
    std::lock_guard<std::mutex> lk(mu);
    for (size_t i = 0; i < mapped.size(); ++i)
        if (mapped[i].first == base) {
            (void)hipHostUnregister(const_cast<void*>(base));
            mapped.erase(mapped.begin() + (long)i);
            return;
        }
}

bool in_mapping(const void* p) {
    std::lock_guard<std::mutex> lk(mu);
    if (!per_linear) return false;  // the fused path reads its weights through dev::, not through linear()
    for (const auto& r : mapped)
        if (p >= r.first && static_cast<const char*>(p) < static_cast<const char*>(r.first) + r.second) return true;
    return false;
}
}  // namespace gpu

struct Lin {
    Mat w;                     // [out][in]
    const float* b = nullptr;  // [out] or NULL
    int in = 0, out = 0;
};

struct Block {
    const float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;  // q/k norm: [head_dim]
    Lin q, k, v, o, gate, up, down;
};

const float* tensor(const gguf_model_t* g, const std::string& name, size_t count, bool optional = false) {
    size_t bytes = 0;
    const void* p = gguf_get_tensor_ptr(g, name.c_str(), &bytes);
    if (!p) {
        if (optional) return nullptr;
        throw std::runtime_error("qwen3tts: no tensor " + name);
    }
    if (bytes != count * sizeof(float))
        throw std::runtime_error("qwen3tts: " + name + " has " + std::to_string(bytes) + " bytes, expected " +
                                 std::to_string(count * sizeof(float)));
    return static_cast<const float*>(p);
}

// the GGUF type of a tensor by name (-1: absent); the scan is short (a few hundred tensors, read once per model)
int gguf_type_of(const gguf_model_t* g, const std::string& name) {
    const int64_t n = gguf_n_tensors(g);
    for (int64_t i = 0; i < n; ++i)
        if (name == gguf_tensor_name(g, i)) return gguf_tensor_type(g, i);
    return -1;
}

// a [rows][cols] matrix of the GGUF: BF16 read in place, or F32 (tensor())
Mat mat_of(const gguf_model_t* g, const std::string& name, int rows, int cols) {
    Mat M;
    M.rows = rows;
    M.cols = cols;
    if (gguf_type_of(g, name) == 30) {  // ggml's BF16
        size_t bytes = 0;
        const void* p = gguf_get_tensor_ptr(g, name.c_str(), &bytes);
        if (!p) throw std::runtime_error("qwen3tts: no tensor " + name);
        if (bytes != (size_t)rows * cols * 2) throw std::runtime_error("qwen3tts: " + name + " has the wrong size");
        M.h = static_cast<const uint16_t*>(p);
    } else {
        M.f = tensor(g, name, (size_t)rows * cols);
    }
    return M;
}


// The device path reads the talker's and the predictor's matrices as BF16 in place: a GGUF with F32 matrices (the
// talker-f32 file) runs on the host. A matrix is a tensor with two or more rows (the norms and biases are vectors).
bool bf16_matrices(const gguf_model_t* g) {
    for (int64_t i = 0; i < gguf_n_tensors(g); ++i) {
        const char* name = gguf_tensor_name(g, i);
        if (strncmp(name, "talker.", 7) && strncmp(name, "pred.", 5)) continue;
        if (gguf_tensor_ne(g, i)[1] > 1 && gguf_tensor_type(g, i) != 30) return false;  // 30: ggml's BF16
    }
    return true;
}

// y[t][o] = x[t][:] . w[o][:] + b[o]: parallel over the output rows o, and each row is read once for all T inputs (a
// prefill or a code predictor's batch reuses the row from cache; a decode step has one input and the rows are the work)
void linear(const float* x, int T, const Lin& L, float* y) {
    if (L.w.h && T > 0 && (size_t)T * L.in <= gpu::stage_cap && (size_t)T * L.out <= gpu::stage_cap && gpu::in_mapping(L.w.h)) {
        std::lock_guard<std::mutex> lk(gpu::mu);
        std::memcpy(gpu::stage_in, x, sizeof(float) * (size_t)T * L.in);
        if (qmatvec_launch_bf16f(L.w.h, sizeof(uint16_t) * (size_t)L.in, gpu::stage_in, gpu::stage_out, L.out, L.in, T, 0) ==
                hipSuccess &&
            hipStreamSynchronize(0) == hipSuccess) {
            for (int t = 0; t < T; ++t)
                for (int o = 0; o < L.out; ++o)
                    y[(size_t)t * L.out + o] = gpu::stage_out[(size_t)t * L.out + o] + (L.b ? L.b[o] : 0.0f);
            return;
        }
        // (a launch that fails falls through to the host path below)
    }
#pragma omp parallel for schedule(static)
    for (int o = 0; o < L.out; ++o) {
        const float bias = L.b ? L.b[o] : 0.0f;
        for (int t = 0; t < T; ++t) y[(size_t)t * L.out + o] = bias + mat_dot(L.w, o, x + (size_t)t * L.in);
    }
}

// RMSNorm over the last n values of each of the T rows: y = x / sqrt(mean(x^2) + eps) * w (the weight has n values)
void rms_norm(const float* x, int T, int n, float eps, const float* w, float* y) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t) {
        const float* xr = x + (size_t)t * n;
        float* yr = y + (size_t)t * n;
        double ss = 0.0;
        for (int c = 0; c < n; ++c) ss += (double)xr[c] * xr[c];
        const float inv = 1.0f / std::sqrt((float)(ss / n) + eps);
        for (int c = 0; c < n; ++c) yr[c] = xr[c] * inv * w[c];
    }
}

// The rotary of q [T][heads][D] and k [T][kv_heads][D] at the positions pos[t] (the standard rotate-half form)
void rope(float* x, int T, int heads, int D, const int* pos, double theta) {
    const int half = D / 2;
    std::vector<double> inv(half);
    for (int i = 0; i < half; ++i) inv[i] = std::pow(theta, -2.0 * i / D);
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < heads; ++h) {
            float* v = x + ((size_t)t * heads + h) * D;
            for (int i = 0; i < half; ++i) {
                const double f = (double)pos[t] * inv[i];
                const float c = (float)std::cos(f), s = (float)std::sin(f);
                const float x1 = v[i], x2 = v[i + half];
                v[i] = x1 * c - x2 * s;
                v[i + half] = x2 * c + x1 * s;
            }
        }
}

struct Dims {
    int C, H, Hkv, D, FF;
    float eps;
    double theta;
};

// One pre-norm block over the n rows x [n][C] (in place) at the absolute positions pos0 .. pos0 + n - 1: their keys and
// values go to the cache rows kc / vc (indexed by absolute position: [pos0 + n][Hkv * D]), and each row attends
// causally to the cache up to its own position (GQA), then the gated MLP
void block_step(const Block& B, const Dims& d, float* x, int n, int pos0, float* kc, float* vc) {
    const int C = d.C, HD = d.H * d.D, KD = d.Hkv * d.D;
    std::vector<float> h((size_t)n * C), q((size_t)n * HD), k((size_t)n * KD), v((size_t)n * KD), a((size_t)n * HD),
        o((size_t)n * C), g((size_t)n * d.FF), u((size_t)n * d.FF);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = pos0 + i;
    rms_norm(x, n, C, d.eps, B.attn_norm, h.data());
    linear(h.data(), n, B.q, q.data());
    linear(h.data(), n, B.k, k.data());
    linear(h.data(), n, B.v, v.data());
    rms_norm(q.data(), n * d.H, d.D, d.eps, B.q_norm, q.data());  // per head
    rms_norm(k.data(), n * d.Hkv, d.D, d.eps, B.k_norm, k.data());
    rope(q.data(), n, d.H, d.D, pos.data(), d.theta);
    rope(k.data(), n, d.Hkv, d.D, pos.data(), d.theta);
    std::memcpy(kc + (size_t)pos0 * KD, k.data(), sizeof(float) * (size_t)n * KD);
    std::memcpy(vc + (size_t)pos0 * KD, v.data(), sizeof(float) * (size_t)n * KD);
    // causal attention: row i (position p) sees the cache's positions 0..p; query head hq uses kv head hq / (H / Hkv)
    const int rep = d.H / d.Hkv;
    const float scale = 1.0f / std::sqrt((float)d.D);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const int p = pos0 + i;
        std::vector<float> sc(p + 1);
        for (int hq = 0; hq < d.H; ++hq) {
            const float* qr = q.data() + ((size_t)i * d.H + hq) * d.D;
            const int hk = hq / rep;
            float mx = -INFINITY;
            for (int j = 0; j <= p; ++j) {
                const float* kr = kc + ((size_t)j * d.Hkv + hk) * d.D;
                float dot = 0.0f;
                for (int c = 0; c < d.D; ++c) dot += qr[c] * kr[c];
                sc[j] = dot * scale;
                mx = std::max(mx, sc[j]);
            }
            double sum = 0.0;
            for (int j = 0; j <= p; ++j) {
                sc[j] = std::exp(sc[j] - mx);
                sum += sc[j];
            }
            float* orow = a.data() + ((size_t)i * d.H + hq) * d.D;
            for (int c = 0; c < d.D; ++c) orow[c] = 0.0f;
            for (int j = 0; j <= p; ++j) {
                const float pr = (float)(sc[j] / sum);
                const float* vr = vc + ((size_t)j * d.Hkv + hk) * d.D;
                for (int c = 0; c < d.D; ++c) orow[c] += pr * vr[c];
            }
        }
    }
    linear(a.data(), n, B.o, o.data());
    for (size_t i = 0; i < (size_t)n * C; ++i) x[i] += o[i];
    rms_norm(x, n, C, d.eps, B.ffn_norm, h.data());
    linear(h.data(), n, B.gate, g.data());
    linear(h.data(), n, B.up, u.data());
    for (size_t i = 0; i < (size_t)n * d.FF; ++i) {
        const float gi = g[i];
        g[i] = gi / (1.0f + std::exp(-gi)) * u[i];  // SiLU(gate) * up
    }
    linear(g.data(), n, B.down, o.data());
    for (size_t i = 0; i < (size_t)n * C; ++i) x[i] += o[i];
}

}  // namespace

// The speaker encoder (qwen-tts's Qwen3TTSSpeakerEncoder: an ECAPA-TDNN, T5 of PLAN.md). 1-D convolutions over [channels][time]
// with 'same' padding by reflection (dilation * (k - 1) in total, the left half the smaller), the SE-Res2Net blocks, the
// multi-layer feature aggregation, the attentive statistics pooling (weights from a TDNN over the frame, the frame's
// statistics appended) and the final 1x1 convolution. Host float32; the checkpoint's module names as spk.* tensors.
struct Conv1 {
    const float *w = nullptr, *b = nullptr;  // w [cout][cin][k], b [cout]
    int cin = 0, cout = 0, k = 1, dil = 1;
};

Conv1 conv1_of(const gguf_model_t* g, const std::string& base, int cin, int cout, int k, int dil) {
    Conv1 c;
    c.cin = cin;
    c.cout = cout;
    c.k = k;
    c.dil = dil;
    c.w = tensor(g, base + ".weight", (size_t)cout * cin * k);
    c.b = tensor(g, base + ".bias", (size_t)cout);
    return c;
}

struct SERes2 {
    int C = 0, scale = 0, k = 1, dil = 1;
    Conv1 tdnn1, tdnn2, se1, se2;
    std::vector<Conv1> res;  // the Res2Net's scale - 1 convolutions, C / scale channels each
};

struct Speaker {
    int mel = 0, enc = 0, att = 0, scale = 0, se = 0;
    std::vector<int> ch, ks, dl;  // the encoder's channels, kernels and dilations (the first TDNN, then the SE-Res2Nets)
    Conv1 first, mfa, asp_tdnn, asp_conv, fc;
    std::vector<SERes2> blk;
};

void conv_same(const Conv1& c, const float* x, int T, float* y) {
    const int total = c.dil * (c.k - 1), left = total / 2;
    #pragma omp parallel for schedule(static)
    for (int o = 0; o < c.cout; ++o) {
        for (int t = 0; t < T; ++t) {
            float acc = c.b[o];
            for (int i = 0; i < c.cin; ++i) {
                const float* wr = c.w + ((size_t)o * c.cin + i) * c.k;
                const float* xr = x + (size_t)i * T;
                for (int j = 0; j < c.k; ++j) {
                    int idx = t - left + j * c.dil;
                    if (idx < 0) idx = -idx;                    // reflection: the edge is not repeated
                    if (idx >= T) idx = 2 * (T - 1) - idx;
                    acc += wr[j] * xr[idx];
                }
            }
            y[(size_t)o * T + t] = acc;
        }
    }
}

void relu_in(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// the speaker embedding of a log-mel [mel][T] (the mel's channels first, as the encoder's input)
void speaker_forward(const Speaker& sp, const float* mel, int T, float* emb) {
    std::vector<std::vector<float>> outs;
    std::vector<float> x((size_t)sp.ch[0] * T);
    conv_same(sp.first, mel, T, x.data());
    relu_in(x.data(), x.size());
    outs.push_back(x);
    for (const SERes2& b : sp.blk) {
        const int C = b.C, sub = C / b.scale;
        const std::vector<float> residual = x;
        std::vector<float> t1((size_t)C * T);
        conv_same(b.tdnn1, x.data(), T, t1.data());
        relu_in(t1.data(), t1.size());
        // the Res2Net: the first part passes, the second goes through its convolution, each later one takes its own part
        // plus the previous part's output
        std::vector<float> r2((size_t)C * T), prev((size_t)sub * T), in((size_t)sub * T), o((size_t)sub * T);
        for (int i = 0; i < b.scale; ++i) {
            const float* part = t1.data() + (size_t)i * sub * T;
            if (i == 0) {
                o.assign(part, part + (size_t)sub * T);
            } else {
                if (i == 1) {
                    in.assign(part, part + (size_t)sub * T);
                } else {
                    for (size_t q = 0; q < in.size(); ++q) in[q] = part[q] + prev[q];
                }
                conv_same(b.res[i - 1], in.data(), T, o.data());
                relu_in(o.data(), o.size());
            }
            std::memcpy(&r2[(size_t)i * sub * T], o.data(), sizeof(float) * (size_t)sub * T);
            prev = o;
        }
        std::vector<float> t2((size_t)C * T);
        conv_same(b.tdnn2, r2.data(), T, t2.data());
        relu_in(t2.data(), t2.size());
        // squeeze-excitation: the channel means over time, a bottleneck, a sigmoid gate
        std::vector<float> mean(C), s1(b.se1.cout), s2(C);
        for (int c = 0; c < C; ++c) {
            double acc = 0.0;
            for (int t = 0; t < T; ++t) acc += t2[(size_t)c * T + t];
            mean[c] = (float)(acc / T);
        }
        conv_same(b.se1, mean.data(), 1, s1.data());
        relu_in(s1.data(), s1.size());
        conv_same(b.se2, s1.data(), 1, s2.data());
        for (int c = 0; c < C; ++c) {
            const float g = 1.0f / (1.0f + std::exp(-s2[c]));
            for (int t = 0; t < T; ++t) x[(size_t)c * T + t] = t2[(size_t)c * T + t] * g + residual[(size_t)c * T + t];
        }
        outs.push_back(x);
    }
    // the multi-layer aggregation: the blocks after the first, channel-wise
    size_t Ctot = 0;
    for (size_t i = 1; i < outs.size(); ++i) Ctot += outs[i].size() / T;
    std::vector<float> cat((size_t)Ctot * T), h((size_t)Ctot * T);
    size_t off = 0;
    for (size_t i = 1; i < outs.size(); ++i) {
        std::memcpy(&cat[off], outs[i].data(), sizeof(float) * outs[i].size());
        off += outs[i].size();
    }
    conv_same(sp.mfa, cat.data(), T, h.data());
    relu_in(h.data(), h.size());
    // attentive statistics pooling: the frame weights from [h, mean, std] (each of the mean and std over the frames, uniform
    // weights 1/T), a tanh TDNN and a 1x1 convolution, softmax over time, the weighted mean and std
    const int L = (int)Ctot;
    std::vector<float> a((size_t)3 * L * T), mu(L), sd(L);
    for (int c = 0; c < L; ++c) {
        double m = 0.0, v = 0.0;
        for (int t = 0; t < T; ++t) m += h[(size_t)c * T + t];
        m /= T;
        for (int t = 0; t < T; ++t) {
            const double d = h[(size_t)c * T + t] - m;
            v += d * d;
        }
        mu[c] = (float)m;
        sd[c] = (float)std::sqrt(std::max(v / T, 1e-12));
    }
    for (int c = 0; c < L; ++c)
        for (int t = 0; t < T; ++t) {
            a[(size_t)c * T + t] = h[(size_t)c * T + t];
            a[((size_t)(L + c)) * T + t] = mu[c];
            a[((size_t)(2 * L + c)) * T + t] = sd[c];
        }
    std::vector<float> a1((size_t)sp.att * T), logit((size_t)L * T);
    conv_same(sp.asp_tdnn, a.data(), T, a1.data());
    relu_in(a1.data(), a1.size());
    for (float& v : a1) v = std::tanh(v);
    conv_same(sp.asp_conv, a1.data(), T, logit.data());
    for (int c = 0; c < L; ++c) {
        float* lr = &logit[(size_t)c * T];
        const float mx = *std::max_element(lr, lr + T);
        double z = 0.0;
        for (int t = 0; t < T; ++t) {
            lr[t] = std::exp(lr[t] - mx);
            z += lr[t];
        }
        for (int t = 0; t < T; ++t) lr[t] = (float)(lr[t] / z);
    }
    std::vector<float> pooled((size_t)2 * L);
    for (int c = 0; c < L; ++c) {
        const float* wr = &logit[(size_t)c * T];
        const float* hr = &h[(size_t)c * T];
        double m = 0.0, v = 0.0;
        for (int t = 0; t < T; ++t) m += (double)wr[t] * hr[t];
        for (int t = 0; t < T; ++t) {
            const double d = hr[t] - m;
            v += (double)wr[t] * d * d;
        }
        pooled[c] = (float)m;
        pooled[L + c] = (float)std::sqrt(std::max(v, 1e-12));
    }
    conv_same(sp.fc, pooled.data(), 1, emb);
}

std::unique_ptr<Speaker> load_speaker(gguf_model_t* g, const std::function<int(const char*)>& kv,
                                      const std::function<std::vector<int>(const char*)>& arr) {
    std::unique_ptr<Speaker> sp(new Speaker());
    sp->mel = kv("spk.mel_dim");
    sp->enc = kv("spk.enc_dim");
    sp->att = kv("spk.attention_channels");
    sp->scale = kv("spk.res2net_scale");
    sp->se = kv("spk.se_channels");
    sp->ch = arr("spk.channels");
    sp->ks = arr("spk.kernels");
    sp->dl = arr("spk.dilations");
    const int n = (int)sp->ch.size();
    if (n < 3 || (int)sp->ks.size() != n || (int)sp->dl.size() != n) throw std::runtime_error("qwen3tts: the speaker encoder's arrays");
    sp->first = conv1_of(g, "spk.blocks.0.conv", sp->mel, sp->ch[0], sp->ks[0], sp->dl[0]);
    for (int i = 1; i < n - 1; ++i) {
        SERes2 b;
        const std::string base = "spk.blocks." + std::to_string(i);
        const int in = sp->ch[i - 1], C = sp->ch[i];
        b.C = C;
        b.scale = sp->scale;
        b.k = sp->ks[i];
        b.dil = sp->dl[i];
        if (C % b.scale) throw std::runtime_error("qwen3tts: the speaker encoder's channels");
        b.tdnn1 = conv1_of(g, base + ".tdnn1.conv", in, C, 1, 1);
        for (int r = 0; r < b.scale - 1; ++r)
            b.res.push_back(conv1_of(g, base + ".res2net_block.blocks." + std::to_string(r) + ".conv", C / b.scale, C / b.scale,
                                     b.k, b.dil));
        b.tdnn2 = conv1_of(g, base + ".tdnn2.conv", C, C, 1, 1);
        b.se1 = conv1_of(g, base + ".se_block.conv1", C, sp->se, 1, 1);
        b.se2 = conv1_of(g, base + ".se_block.conv2", sp->se, C, 1, 1);
        sp->blk.push_back(std::move(b));
    }
    int cat = 0;
    for (int i = 1; i < n - 1; ++i) cat += sp->ch[i];
    sp->mfa = conv1_of(g, "spk.mfa.conv", cat, sp->ch[n - 1], sp->ks[n - 1], sp->dl[n - 1]);
    sp->asp_tdnn = conv1_of(g, "spk.asp.tdnn.conv", 3 * sp->ch[n - 1], sp->att, 1, 1);
    sp->asp_conv = conv1_of(g, "spk.asp.conv", sp->att, sp->ch[n - 1], 1, 1);
    sp->fc = conv1_of(g, "spk.fc", 2 * sp->ch[n - 1], sp->enc, 1, 1);
    return sp;
}

struct qwen3tts_model {
    gguf_model_t* g = nullptr;
    tokenizer_t* tok = nullptr;  // the GGUF's vocabulary (NULL: none)
    int n_layer = 0, C = 0, H = 0, Hkv = 0, D = 0, FF = 0, text_vocab = 0, text_C = 0, codec_vocab = 0, ncb = 0;
    float eps = 1e-6f;
    double theta = 1e6;
    int pn_layer = 0, pC = 0, pH = 0, pHkv = 0, pD = 0, pFF = 0, pvocab = 0;
    int tts_bos = 0, tts_eos = 0, tts_pad = 0, codec_pad = 0, codec_bos = 0, codec_eos = 0, codec_think = 0,
        codec_nothink = 0, codec_think_bos = 0, codec_think_eos = 0;
    std::vector<std::string> languages;
    std::unique_ptr<Speaker> spk;       // the speaker encoder (absent in a GGUF without spk.* tensors)
    const void* gpu_base = nullptr;     // the GGUF mapping registered for the device (NULL: host only)
    std::vector<int> language_ids;
    Mat text_embd;                      // [text_vocab][text_C]
    Mat codec_embd;                     // [codec_vocab][C]
    Lin fc1, fc2;                       // the text projection: text_C -> text_C -> C (SiLU between)
    std::vector<Block> blk;
    const float* out_norm = nullptr;
    Lin codec_head;                     // C -> codec_vocab, no bias
    std::vector<Mat> pred_embd;         // ncb - 1 tables [pvocab][C]
    std::vector<Block> pblk;
    const float* pred_out_norm = nullptr;
    std::vector<Lin> pred_head;         // ncb - 1 heads [pvocab][pC]
    ~qwen3tts_model() {
        if (g) gguf_mmap_close(g);
    }
};

namespace {

// ---- The device path of the talker (QWEN3TTS_DEVICE=gpu): one token's block on the device, as talker_gpu.hip's kernels
// and the BF16 matvecs (qmatvec_launch_bf16f) on registered host memory: the caches, the activations and the rope tables
// are in an arena of coarse-grained registered pages (UMA: the host writes and reads them directly, the GPU reads and
// writes the same pages; the host syncs the stream before reading a result).
namespace dev {

hipModule_t mod = nullptr;
hipFunction_t f_rmsnorm = nullptr, f_rope = nullptr, f_kv = nullptr, f_attn = nullptr, f_silu = nullptr, f_add = nullptr;

bool load() {
    if (f_attn) return true;
    if (kernel_module_load(&mod, "talker_gpu.co") != hipSuccess) return false;
    hipFunction_t* fns[] = {&f_rmsnorm, &f_rope, &f_kv, &f_attn, &f_silu, &f_add};
    const char* names[] = {"tk_rmsnorm", "tk_rope", "tk_kv_store", "tk_attn", "tk_silu_mul", "tk_add"};
    for (int i = 0; i < 6; ++i)
        if (hipModuleGetFunction(fns[i], mod, names[i]) != hipSuccess) return false;
    return true;
}

// the arena: registered anonymous pages, carved into 256-byte aligned float arrays
struct Arena {
    char* base = nullptr;
    size_t size = 0, used = 0;
    bool open(size_t bytes) {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return false;
        if (hipHostRegister(p, bytes, hipExtHostRegisterCoarseGrained) != hipSuccess) {
            munmap(p, bytes);
            return false;
        }
        base = static_cast<char*>(p);
        size = bytes;
        used = 0;
        return true;
    }
    float* alloc(size_t n) {
        used = (used + 255) & ~(size_t)255;
        if (used + n * sizeof(float) > size) throw std::runtime_error("qwen3tts device: the arena is too small");
        float* p = reinterpret_cast<float*>(base + used);
        used += n * sizeof(float);
        return p;
    }
    void close() {
        if (!base) return;
        (void)hipHostUnregister(base);
        munmap(base, size);
        base = nullptr;
    }
    ~Arena() { close(); }
};

void launch(hipFunction_t f, unsigned blocks, unsigned threads, unsigned shm, void** args) {
    if (hipModuleLaunchKernel(f, blocks, 1, 1, threads, 1, 1, shm, 0, args, nullptr) != hipSuccess)
        throw std::runtime_error("qwen3tts device: a kernel launch failed");
}

// y = W x for one token (the BF16 matrix in place; fp32 activations)
void mv(const Mat& W, const float* x, float* y) {
    if (!W.h) throw std::runtime_error("qwen3tts device: a matrix is not BF16");
    if (qmatvec_launch_bf16f(W.h, sizeof(uint16_t) * (size_t)W.cols, x, y, W.rows, W.cols, 1, 0) != hipSuccess)
        throw std::runtime_error("qwen3tts device: the matvec launch failed");
}

// rows of n values: y = rms_norm(x) * w, one block per row (y may be x)
void rms(const float* x, float* y, int rows, int n, const float* w, float eps) {
    void* args[] = {(void*)&x, (void*)&w, (void*)&y, (void*)&n, (void*)&eps};
    launch(f_rmsnorm, (unsigned)rows, 256, 0, args);
}

void rope(float* x, int heads, int D, int pos, const float* cs, const float* sn) {
    const int threads = heads * (D / 2);
    void* args[] = {(void*)&x, (void*)&cs, (void*)&sn, (void*)&heads, (void*)&D, (void*)&pos};
    launch(f_rope, (unsigned)((threads + 255) / 256), 256, 0, args);
}

// one pre-norm block for one token at position pos: x [C] in place; the layer's cache rows kc / vc [maxL][Hkv * D]
struct Work {
    float *h, *q, *k, *v, *att, *o, *g, *u;
};

void block(const Block& B, const Dims& d, float* x, float* kc, float* vc, int pos, const float* cs, const float* sn,
           const Work& w) {
    const int C = d.C, KD = d.Hkv * d.D;
    rms(x, w.h, 1, C, B.attn_norm, d.eps);
    mv(B.q.w, w.h, w.q);
    mv(B.k.w, w.h, w.k);
    mv(B.v.w, w.h, w.v);
    rms(w.q, w.q, d.H, d.D, B.q_norm, d.eps);
    rms(w.k, w.k, d.Hkv, d.D, B.k_norm, d.eps);
    rope(w.q, d.H, d.D, pos, cs, sn);
    rope(w.k, d.Hkv, d.D, pos, cs, sn);
    {
        const float *k = w.k, *v = w.v;
        void* args[] = {(void*)&k, (void*)&v, (void*)&kc, (void*)&vc, (void*)&pos, (void*)&KD};
        launch(f_kv, (unsigned)((KD + 255) / 256), 256, 0, args);
    }
    {
        const float* q = w.q;
        const int n_ctx = pos + 1, H = d.H, Hkv = d.Hkv, D = d.D;
        void* args[] = {(void*)&q, (void*)&kc, (void*)&vc, (void*)&w.att, (void*)&n_ctx, (void*)&H, (void*)&Hkv, (void*)&D};
        launch(f_attn, (unsigned)d.H, 256, (unsigned)(n_ctx * sizeof(float)), args);
    }
    mv(B.o.w, w.att, w.o);
    {
        const float* o = w.o;
        void* args[] = {(void*)&x, (void*)&o, (void*)&C};
        launch(f_add, (unsigned)((C + 255) / 256), 256, 0, args);
    }
    rms(x, w.h, 1, C, B.ffn_norm, d.eps);
    mv(B.gate.w, w.h, w.g);
    mv(B.up.w, w.h, w.u);
    {
        const int FF = d.FF;
        const float* u = w.u;
        void* args[] = {(void*)&w.g, (void*)&u, (void*)&FF};
        launch(f_silu, (unsigned)((FF + 255) / 256), 256, 0, args);
    }
    mv(B.down.w, w.g, w.o);
    {
        const float* o = w.o;
        void* args[] = {(void*)&x, (void*)&o, (void*)&C};
        launch(f_add, (unsigned)((C + 255) / 256), 256, 0, args);
    }
}

// the head: rms_norm (the final norm) then the matrix, for one token: normed [C], out [rows]
void head(const float* x, const float* norm_w, const Mat& W, float eps, int C, float* normed, float* out) {
    rms(x, normed, 1, C, norm_w, eps);
    mv(W, normed, out);
}

// cos and sin of pos * theta^(-2i/D) for pos < rows, as the host's rope (double angles, rounded to float)
void rope_tables(int rows, int D, double theta, float* cs, float* sn) {
    const int half = D / 2;
    for (int p = 0; p < rows; ++p)
        for (int i = 0; i < half; ++i) {
            const double f = (double)p * std::pow(theta, -2.0 * i / D);
            cs[(size_t)p * half + i] = (float)std::cos(f);
            sn[(size_t)p * half + i] = (float)std::sin(f);
        }
}

}  // namespace dev

Dims talker_dims(const qwen3tts_model* m) { return Dims{m->C, m->H, m->Hkv, m->D, m->FF, m->eps, m->theta}; }
Dims pred_dims(const qwen3tts_model* m) { return Dims{m->pC, m->pH, m->pHkv, m->pD, m->pFF, m->eps, m->theta}; }

// The text projection of T text rows (text_C each) into C values: fc2(SiLU(fc1(x)))
void text_project(const qwen3tts_model* m, const float* x, int T, float* y) {
    std::vector<float> h((size_t)T * m->text_C);
    linear(x, T, m->fc1, h.data());
    for (float& v : h) v = v / (1.0f + std::exp(-v));
    linear(h.data(), T, m->fc2, y);
}

// The rows of a text table for the ids: [n][text_C] -> projected [n][C]
void text_rows_projected(const qwen3tts_model* m, const std::vector<int>& ids, float* out) {
    const int n = (int)ids.size();
    std::vector<float> raw((size_t)n * m->text_C);
    for (int i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= m->text_vocab) throw std::runtime_error("qwen3tts: text id out of range");
        mat_row(m->text_embd, ids[i], &raw[(size_t)i * m->text_C]);
    }
    text_project(m, raw.data(), n, out);
}

void codec_rows(const qwen3tts_model* m, const std::vector<int>& ids, float* out) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= m->codec_vocab) throw std::runtime_error("qwen3tts: codec id out of range");
        mat_row(m->codec_embd, ids[i], out + i * m->C);
    }
}

}  // namespace

extern "C" {

// splitmix64's next draw (the state advances) and one token from logits[n]: the temperature, then the top_k most likely
// kept (0: all), drawn from the softmax; temperature <= 0: the first maximum
uint64_t next_u64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

int sample_token(const float* logits, int n, float temperature, int top_k, uint64_t* state) {
    if (temperature <= 0.0f) {
        int best = 0;
        for (int v = 1; v < n; ++v)
            if (logits[v] > logits[best]) best = v;
        return best;
    }
    std::vector<int> idx;
    idx.reserve((size_t)n);
    for (int v = 0; v < n; ++v)
        if (logits[v] > -INFINITY) idx.push_back(v);
    if (idx.empty()) return 0;
    if (top_k > 0 && top_k < (int)idx.size()) {
        std::nth_element(idx.begin(), idx.begin() + (top_k - 1), idx.end(),
                         [&](int a, int b) { return logits[a] > logits[b]; });
        idx.resize((size_t)top_k);
    }
    float mx = -INFINITY;
    for (int v : idx) mx = std::max(mx, logits[v]);
    std::vector<double> p(idx.size());
    double sum = 0.0;
    for (size_t i = 0; i < idx.size(); ++i) {
        p[i] = std::exp((double)(logits[idx[i]] - mx) / (double)temperature);
        sum += p[i];
    }
    double r = (double)(next_u64(state) >> 11) * (1.0 / 9007199254740992.0) * sum;
    for (size_t i = 0; i < idx.size(); ++i) {
        r -= p[i];
        if (r < 0.0) return idx[i];
    }
    return idx.back();
}

qwen3tts_model_t* qwen3tts_open(const char* gguf_path) {
    try {
        std::unique_ptr<qwen3tts_model> m(new qwen3tts_model());
        m->g = gguf_mmap_open(gguf_path);
        if (!m->g) {
            g_error = std::string("cannot open ") + (gguf_path ? gguf_path : "(null)");
            return nullptr;
        }
        if (std::string(gguf_arch(m->g)) != "qwen3tts") throw std::runtime_error("not a qwen3tts GGUF");
        // (the qwen3tts.* keys by their suffix: gguf_arch_int reads "<architecture>.<suffix>")
        auto kv = [&](const char* key) { return (int)gguf_arch_int(m->g, key, -1); };
        m->n_layer = kv("talker.n_layer");
        m->C = kv("talker.n_embd");
        m->H = kv("talker.n_head");
        m->Hkv = kv("talker.n_head_kv");
        m->D = kv("talker.head_dim");
        m->FF = kv("talker.n_ff");
        m->theta = gguf_arch_float(m->g, "talker.rope_theta", 1e6f);
        m->eps = gguf_arch_float(m->g, "talker.rms_eps", 1e-6f);
        m->text_vocab = kv("talker.text_vocab");
        m->text_C = kv("talker.text_n_embd");
        m->codec_vocab = kv("talker.codec_vocab");
        m->ncb = kv("n_codebooks");
        m->pn_layer = kv("pred.n_layer");
        m->pC = kv("pred.n_embd");
        m->pH = kv("pred.n_head");
        m->pHkv = kv("pred.n_head_kv");
        m->pD = kv("pred.head_dim");
        m->pFF = kv("pred.n_ff");
        m->pvocab = kv("pred.vocab");
        m->tts_bos = kv("tts_bos_token");
        m->tts_eos = kv("tts_eos_token");
        m->tts_pad = kv("tts_pad_token");
        m->codec_pad = kv("codec_pad_id");
        m->codec_bos = kv("codec_bos_id");
        m->codec_eos = kv("codec_eos_id");
        m->codec_think = kv("codec_think_id");
        m->codec_nothink = kv("codec_nothink_id");
        m->codec_think_bos = kv("codec_think_bos_id");
        m->codec_think_eos = kv("codec_think_eos_id");
        if (m->n_layer <= 0 || m->C <= 0 || m->H <= 0 || m->Hkv <= 0 || m->D <= 0 || m->FF <= 0 || m->text_vocab <= 0 ||
            m->text_C <= 0 || m->codec_vocab <= 0 || m->ncb <= 1 || m->pn_layer <= 0 || m->pC <= 0 || m->pH <= 0 ||
            m->pHkv <= 0 || m->pD <= 0 || m->pFF <= 0 || m->pvocab <= 0 || m->H % m->Hkv != 0 || m->pH % m->pHkv != 0)
            throw std::runtime_error("qwen3tts: missing or bad hyperparameters");
        if (m->pC != m->C) throw std::runtime_error("qwen3tts: the code predictor's width differs from the talker's");
        // the languages (the names and their codec ids)
        {
            const char** names = nullptr;
            size_t n = 0;
            if (gguf_kv_get_string_array(m->g, "qwen3tts.languages", &names, &n) == 0)
                for (size_t i = 0; i < n; ++i) m->languages.push_back(names[i]);
            int32_t type = 0;
            const void* data = nullptr;
            size_t count = 0;
            if (gguf_kv_get_array(m->g, "qwen3tts.language_ids", &type, &data, &count) == 0 && count == m->languages.size())
                for (size_t i = 0; i < count; ++i) m->language_ids.push_back((int)static_cast<const uint32_t*>(data)[i]);
        }

        m->text_embd = mat_of(m->g, "talker.text_embd.weight", m->text_vocab, m->text_C);
        m->codec_embd = mat_of(m->g, "talker.codec_embd.weight", m->codec_vocab, m->C);
        m->fc1 = {mat_of(m->g, "talker.text_proj.fc1.weight", m->text_C, m->text_C),
                  tensor(m->g, "talker.text_proj.fc1.bias", m->text_C), m->text_C, m->text_C};
        m->fc2 = {mat_of(m->g, "talker.text_proj.fc2.weight", m->C, m->text_C),
                  tensor(m->g, "talker.text_proj.fc2.bias", m->C), m->text_C, m->C};
        m->out_norm = tensor(m->g, "talker.out_norm.weight", m->C);
        m->codec_head = {mat_of(m->g, "talker.codec_head.weight", m->codec_vocab, m->C), nullptr, m->C, m->codec_vocab};
        auto lin = [&](const std::string& base, int in, int out, bool bias) {
            Lin L;
            L.in = in;
            L.out = out;
            L.w = mat_of(m->g, base + ".weight", out, in);
            L.b = bias ? tensor(m->g, base + ".bias", out) : nullptr;
            return L;
        };
        auto block_of = [&](const std::string& p, int C, int H, int Hkv, int D, int FF) {
            Block B;
            B.attn_norm = tensor(m->g, p + "attn_norm.weight", C);
            B.ffn_norm = tensor(m->g, p + "ffn_norm.weight", C);
            B.q_norm = tensor(m->g, p + "attn_q_norm.weight", D);
            B.k_norm = tensor(m->g, p + "attn_k_norm.weight", D);
            B.q = lin(p + "attn_q", C, H * D, false);
            B.k = lin(p + "attn_k", C, Hkv * D, false);
            B.v = lin(p + "attn_v", C, Hkv * D, false);
            B.o = lin(p + "attn_o", H * D, C, false);
            B.gate = lin(p + "ffn_gate", C, FF, false);
            B.up = lin(p + "ffn_up", C, FF, false);
            B.down = lin(p + "ffn_down", FF, C, false);
            return B;
        };
        for (int l = 0; l < m->n_layer; ++l) m->blk.push_back(block_of("talker.blk." + std::to_string(l) + ".", m->C, m->H, m->Hkv, m->D, m->FF));
        for (int i = 0; i < m->ncb - 1; ++i) {
            m->pred_embd.push_back(mat_of(m->g, "pred.embd." + std::to_string(i) + ".weight", m->pvocab, m->C));
            m->pred_head.push_back(lin("pred.head." + std::to_string(i), m->pC, m->pvocab, false));
        }
        for (int l = 0; l < m->pn_layer; ++l)
            m->pblk.push_back(block_of("pred.blk." + std::to_string(l) + ".", m->pC, m->pH, m->pHkv, m->pD, m->pFF));
        m->pred_out_norm = tensor(m->g, "pred.out_norm.weight", m->pC);
        m->tok = tokenizer_load_from_gguf(m->g);  // (NULL when the GGUF has no vocabulary)
        // the device path: the GGUF's mapping registered (one mapping: an unsplit file; BF16 matrices, see bf16_matrices)
        if (gguf_n_maps(m->g) == 1 && bf16_matrices(m->g)) {
            size_t map_size = 0;
            const void* base = gguf_map_base(m->g, &map_size);
            if (gpu::map(base, map_size)) m->gpu_base = base;
        }
        // the speaker encoder, when the GGUF has it (spk.* tensors; a talker-only GGUF has none)
        size_t spk_bytes = 0;
        if (gguf_get_tensor_ptr(m->g, "spk.blocks.0.conv.weight", &spk_bytes)) {
            auto arr = [&](const char* suffix) {
                const std::string key = std::string("qwen3tts.") + suffix;
                int32_t type = 0;
                const void* data = nullptr;
                size_t count = 0;
                if (gguf_kv_get_array(m->g, key.c_str(), &type, &data, &count) != 0) throw std::runtime_error("qwen3tts: no " + key);
                std::vector<int> v(count);
                for (size_t i = 0; i < count; ++i) v[i] = (int)static_cast<const uint32_t*>(data)[i];
                return v;
            };
            m->spk = load_speaker(m->g, kv, arr);
        }
        return m.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void qwen3tts_close(qwen3tts_model_t* model) {
    if (!model) return;
    gpu::unmap(model->gpu_base);
    tokenizer_free(model->tok);
    delete model;
}

const char* qwen3tts_last_error(void) { return g_error.c_str(); }

int qwen3tts_n_embd(const qwen3tts_model_t* m) { return m ? m->C : 0; }
int qwen3tts_n_codebooks(const qwen3tts_model_t* m) { return m ? m->ncb : 0; }
int qwen3tts_codec_vocab(const qwen3tts_model_t* m) { return m ? m->codec_vocab : 0; }
int qwen3tts_pred_vocab(const qwen3tts_model_t* m) { return m ? m->pvocab : 0; }

int qwen3tts_speaker_dim(const qwen3tts_model_t* m) { return m && m->spk ? m->spk->enc : 0; }

int qwen3tts_speaker_embedding(qwen3tts_model_t* m, const float* samples, size_t n, float* spk) {
    if (!m || !samples || !spk) { g_error = "qwen3tts_speaker_embedding: null argument"; return -1; }
    if (!m->spk) { g_error = "qwen3tts_speaker_embedding: the GGUF has no speaker encoder (spk.* tensors)"; return -1; }
    float* mel = nullptr;
    int T = 0;
    if (speaker_log_mel(samples, n, &mel, &T) != 0) {
        g_error = "qwen3tts_speaker_embedding: the clip is too short (at least 257 samples at 24 kHz)";
        return -1;
    }
    try {
        speaker_forward(*m->spk, mel, T, spk);
    } catch (const std::exception& e) {
        speaker_free_mel(mel);
        g_error = e.what();
        return -1;
    }
    speaker_free_mel(mel);
    return 0;
}

int qwen3tts_language_id(const qwen3tts_model_t* m, const char* name) {
    if (!m || !name) return -1;
    std::string lower;
    for (const char* p = name; *p; ++p) lower += (char)std::tolower((unsigned char)*p);
    for (size_t i = 0; i < m->languages.size(); ++i)
        if (m->languages[i] == lower) return m->language_ids[i];
    return -1;
}

int qwen3tts_build_prefill(qwen3tts_model_t* m, const int* text_ids, int n_text, const int* ref_ids, int n_ref,
                           const int* ref_codes, int R, const float* spk, int language_id, float* out, int max_positions,
                           int* positions, float* trailing, int max_trailing, int* n_trailing, float* pad) {
    if (!m || !text_ids || !ref_ids || !ref_codes || !spk || !out || !positions || !trailing || !n_trailing || !pad) {
        g_error = "qwen3tts_build_prefill: null argument";
        return -1;
    }
    if (n_text < 9 || n_ref < 8 || R < 1) { g_error = "qwen3tts_build_prefill: the text, the reference or its frames are too short"; return -1; }
    try {
        const int C = m->C;
        // the assistant template's parts: the role (the first three ids), the text (after them, before the last five)
        std::vector<int> role(text_ids, text_ids + 3);
        std::vector<int> text(text_ids + 3, text_ids + n_text - 5);
        std::vector<int> ref_text(ref_ids + 3, ref_ids + n_ref - 2);
        // the TTS special embeddings: bos, eos, pad (projected)
        std::vector<int> specials = {m->tts_bos, m->tts_eos, m->tts_pad};
        std::vector<float> sp((size_t)3 * C);
        text_rows_projected(m, specials, sp.data());
        const float* tts_bos = sp.data();
        const float* tts_eos = sp.data() + C;
        const float* tts_pad = sp.data() + 2 * C;
        // the codec prefix: the language (or no-think) tags, the speaker, the bos
        std::vector<int> cprefix = {m->codec_think, m->codec_think_bos, language_id, m->codec_think_eos};
        std::vector<float> codec_in((size_t)7 * C);  // 4 tags + speaker + pad, bos
        codec_rows(m, cprefix, codec_in.data());
        std::memcpy(&codec_in[(size_t)4 * C], spk, sizeof(float) * C);
        std::vector<int> cpad = {m->codec_pad, m->codec_bos};
        codec_rows(m, cpad, &codec_in[(size_t)5 * C]);
        // the talker's input for the role and the codec prefix: role (3), then pad * 5 and bos, plus codec_in[:6]
        std::vector<float> role_e((size_t)3 * C);
        text_rows_projected(m, role, role_e.data());
        std::vector<float> talker_pre((size_t)9 * C);
        std::memcpy(talker_pre.data(), role_e.data(), sizeof(float) * 3 * C);
        for (int j = 0; j < 6; ++j) {
            const float* add = j < 5 ? tts_pad : tts_bos;
            for (int c = 0; c < C; ++c) talker_pre[(size_t)(3 + j) * C + c] = add[c] + codec_in[(size_t)j * C + c];
        }
        // the in-context reference: the reference text and the text with the eos, the reference's codec frames with the bos
        std::vector<int> both = ref_text;
        both.insert(both.end(), text.begin(), text.end());
        std::vector<float> tproj((size_t)(both.size() + 1) * C);
        text_rows_projected(m, both, tproj.data());
        std::memcpy(&tproj[both.size() * C], tts_eos, sizeof(float) * C);
        const int text_len = (int)both.size() + 1;
        // codec: the bos, then each frame's sum over its 16 codebooks
        const int codec_len = R + 1;
        std::vector<float> codec((size_t)codec_len * C, 0.0f);
        {
            std::vector<int> bos = {m->codec_bos};
            codec_rows(m, bos, codec.data());
            for (int r = 0; r < R; ++r) {
                float* dst = &codec[(size_t)(r + 1) * C];
                for (int gi = 0; gi < m->ncb; ++gi) {
                    const int id = ref_codes[(size_t)r * m->ncb + gi];
                    if (id < 0) throw std::runtime_error("qwen3tts: negative code");
                    const Mat& table = gi == 0 ? m->codec_embd : m->pred_embd[gi - 1];
                    if (id >= (gi == 0 ? m->codec_vocab : m->pvocab)) throw std::runtime_error("qwen3tts: code out of range");
                    std::vector<float> row(C);
                    mat_row(table, id, row.data());
                    for (int c = 0; c < C; ++c) dst[c] += row[c];
                }
            }
        }
        // the ICL block: text and codec frames side by side (the shorter padded with the pad), the text left over trailing
        std::vector<float> icl;
        std::vector<float> trail;
        if (text_len > codec_len) {
            icl.resize((size_t)codec_len * C);
            for (size_t i = 0; i < icl.size(); ++i) icl[i] = tproj[i] + codec[i];
            trail.assign(tproj.begin() + (size_t)codec_len * C, tproj.begin() + (size_t)text_len * C);
        } else {
            icl.resize((size_t)codec_len * C);
            for (int i = 0; i < codec_len; ++i)
                for (int c = 0; c < C; ++c) {
                    const float tv = i < text_len ? tproj[(size_t)i * C + c] : tts_pad[c];
                    icl[(size_t)i * C + c] = tv + codec[(size_t)i * C + c];
                }
            trail.assign(tts_pad, tts_pad + C);
        }
        const int P = 9 + codec_len;
        if (P > max_positions) { g_error = "qwen3tts_build_prefill: out is too small"; return -1; }
        std::memcpy(out, talker_pre.data(), sizeof(float) * 9 * C);
        std::memcpy(out + (size_t)9 * C, icl.data(), sizeof(float) * icl.size());
        *positions = P;
        const int nt = (int)trail.size() / C;
        if (nt > max_trailing) { g_error = "qwen3tts_build_prefill: trailing is too small"; return -1; }
        std::memcpy(trailing, trail.data(), sizeof(float) * trail.size());
        *n_trailing = nt;
        std::memcpy(pad, tts_pad, sizeof(float) * C);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int qwen3tts_frame_input(qwen3tts_model_t* m, const int* codes, const float* trailing, float* out) {
    if (!m || !codes || !out) { g_error = "qwen3tts_frame_input: null argument"; return -1; }
    const int C = m->C;
    std::fill(out, out + C, 0.0f);
    for (int gi = 0; gi < m->ncb; ++gi) {
        const int id = codes[gi];
        const Mat& table = gi == 0 ? m->codec_embd : m->pred_embd[gi - 1];
        if (id < 0 || id >= (gi == 0 ? m->codec_vocab : m->pvocab)) { g_error = "qwen3tts_frame_input: code out of range"; return -1; }
        std::vector<float> row(C);
        mat_row(table, id, row.data());
        for (int c = 0; c < C; ++c) out[c] += row[c];
    }
    if (trailing)
        for (int c = 0; c < C; ++c) out[c] += trailing[c];
    return 0;
}

int qwen3tts_talker_forward(qwen3tts_model_t* m, const float* X, int L, float* hidden, float* logits) {
    if (!m || !X || L <= 0 || !hidden || !logits) { g_error = "qwen3tts_talker_forward: bad arguments"; return -1; }
    try {
        const int C = m->C;
        std::vector<float> x(X, X + (size_t)L * C);
        const Dims d = talker_dims(m);
        std::vector<float> kc((size_t)L * d.Hkv * d.D), vc((size_t)L * d.Hkv * d.D);
        for (const Block& B : m->blk) block_step(B, d, x.data(), L, 0, kc.data(), vc.data());
        rms_norm(x.data(), L, C, m->eps, m->out_norm, hidden);
        linear(hidden, L, m->codec_head, logits);
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int qwen3tts_code_predictor(qwen3tts_model_t* m, const float* H, const int* codes, int T, float* logits) {
    if (!m || !H || !codes || T <= 0 || !logits) { g_error = "qwen3tts_code_predictor: bad arguments"; return -1; }
    try {
        const int C = m->C, ncb = m->ncb;
        const int rows = ncb;  // H, the talker's embedding of c0, then the predictor's of c1 .. c_{ncb-2} (c_{ncb-1} is never an input)
        const Dims d = pred_dims(m);
        std::vector<float> x((size_t)rows * C);
        for (int t = 0; t < T; ++t) {
            // the inputs of frame t: [H_t, E0(c0), E1(c1) .. E_{ncb-2}(c_{ncb-2})]; the talker's table for c0, the code
            // predictor's table j for codebook j + 1
            std::memcpy(x.data(), H + (size_t)t * C, sizeof(float) * C);
            const int* c = codes + (size_t)t * ncb;
            if (c[0] < 0 || c[0] >= m->codec_vocab) { g_error = "qwen3tts_code_predictor: code out of range"; return -1; }
            mat_row(m->codec_embd, c[0], x.data() + C);
            for (int j = 0; j + 2 < rows; ++j) {
                const int id = c[j + 1];
                if (id < 0 || id >= m->pvocab) { g_error = "qwen3tts_code_predictor: code out of range"; return -1; }
                mat_row(m->pred_embd[j], id, x.data() + (size_t)(j + 2) * C);
            }
            std::vector<float> kc((size_t)rows * d.Hkv * d.D), vc((size_t)rows * d.Hkv * d.D);
            for (const Block& B : m->pblk) block_step(B, d, x.data(), rows, 0, kc.data(), vc.data());
            std::vector<float> hn((size_t)rows * C);
            rms_norm(x.data(), rows, C, m->eps, m->pred_out_norm, hn.data());
            // codebook k + 1 from the position k + 1 through head k
            for (int k = 0; k < ncb - 1; ++k) {
                const float* hr = hn.data() + (size_t)(k + 1) * C;
                float* lr = logits + ((size_t)t * (ncb - 1) + k) * m->pvocab;
                const Lin& hd = m->pred_head[k];
                linear(hr, 1, hd, lr);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int qwen3tts_sample(const float* logits, int n, float temperature, int top_k, uint64_t* state) {
    if (!logits || n <= 0 || !state) return -1;
    return sample_token(logits, n, temperature, top_k, state);
}

qwen3tts_sampling_t qwen3tts_sampling_default(void) {
    qwen3tts_sampling_t s;
    s.temperature = 0.9f;
    s.top_k = 50;
    s.repetition_penalty = 1.05f;
    s.sub_temperature = 0.9f;
    s.sub_top_k = 50;
    s.seed = 0;
    return s;
}

int qwen3tts_tokenize_chat(qwen3tts_model_t* m, const char* text, int* ids, int max_ids) {
    if (!m || !text || !ids || max_ids <= 0) { g_error = "qwen3tts_tokenize_chat: bad arguments"; return -1; }
    if (!m->tok) { g_error = "qwen3tts_tokenize_chat: the GGUF has no tokenizer"; return -1; }
    const std::string s = std::string("<|im_start|>assistant\n") + text + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> buf(s.size() + 2);  // (tokenizer_encode's bound: the text's length plus 2)
    const int n = tokenizer_encode(m->tok, s.c_str(), buf.data(), (int)buf.size());
    if (n < 0) { g_error = "qwen3tts_tokenize_chat: encoding failed"; return -1; }
    if (n > max_ids) { g_error = "qwen3tts_tokenize_chat: ids is too small"; return -1; }
    std::memcpy(ids, buf.data(), sizeof(int) * (size_t)n);
    return n;
}

int qwen3tts_generate(qwen3tts_model_t* m, const int* text_ids, int n_text, const int* ref_ids, int n_ref,
                      const int* ref_codes, int R, const float* spk, int language_id, const qwen3tts_sampling_t* s,
                      int max_frames, int* out_codes) {
    if (!m || !text_ids || !ref_ids || !ref_codes || !spk || !s || !out_codes || max_frames <= 0 ||
        s->repetition_penalty <= 0.0f) {
        g_error = "qwen3tts_generate: bad arguments";
        return -1;
    }
    try {
        const int C = m->C, V = m->codec_vocab, ncb = m->ncb;
        const Dims d = talker_dims(m);
        const Dims pd = pred_dims(m);
        const int KD = d.Hkv * d.D, pKD = pd.Hkv * pd.D, HD = d.H * d.D, pHD = pd.H * pd.D;
        // the prefill's rows: the role, the codec prefix, and the reference's frames with the text (its length: the frames
        // plus one, with the nine rows before)
        const int maxP = 10 + R;
        const int maxT = n_text + n_ref + 2;  // (the text left after the frames: at most the texts' ids)
        std::vector<float> prefill((size_t)maxP * C), trailing((size_t)maxT * C), pad(C);
        int P = 0, nt = 0;
        if (qwen3tts_build_prefill(m, text_ids, n_text, ref_ids, n_ref, ref_codes, R, spk, language_id, prefill.data(), maxP,
                                   &P, trailing.data(), maxT, &nt, pad.data()) != 0)
            return -1;
        const int maxL = P + max_frames + 1;
        const bool dev_on = gpu::ready() && m->gpu_base && dev::load();
        if (dev_on && pd.C != C) throw std::runtime_error("qwen3tts device: the predictor's width is not the talker's");
        // one fused talker on the device at a time: the runs share stream 0 and each syncs it (the host's runs are not held)
        std::unique_lock<std::mutex> run_lock(gpu::run_mu, std::defer_lock);
        if (dev_on) run_lock.lock();
        // the caches and the activations: the arena (registered pages) on the device path, the host's vectors otherwise
        dev::Arena arena;
        std::vector<std::vector<float>> kc_h, vc_h, pk_h, pv_h;
        std::vector<float> xrow_h(C), hid_h(C), logits_h(V), lv_h(m->pvocab), pin_h((size_t)C), pn_h(C);
        std::vector<float*> kc(m->n_layer), vc(m->n_layer), pkc(m->pn_layer), pvc(m->pn_layer);
        float *xrow = xrow_h.data(), *hid = hid_h.data(), *logits = logits_h.data(), *plogits = lv_h.data(), *pin = pin_h.data(),
              *pn = pn_h.data(), *rcs = nullptr, *rsn = nullptr, *prcs = nullptr, *prsn = nullptr;
        dev::Work work{};
        if (dev_on) {
            const int mxC = std::max(C, pd.C), mxHD = std::max(HD, pHD), mxKD = std::max(KD, pKD), mxFF = std::max(d.FF, pd.FF);
            const size_t floats = (size_t)2 * m->n_layer * maxL * KD + (size_t)2 * m->pn_layer * (ncb + 1) * pKD +
                                  (size_t)maxL * (d.D / 2) * 2 + (size_t)(ncb + 1) * (pd.D / 2) * 2 +
                                  (size_t)(3 * C + V + m->pvocab + 2 * mxC + mxC + C) + (size_t)(mxC + 2 * mxKD + 2 * mxHD + mxC + 2 * mxFF);
            const size_t slack_allocs = (size_t)2 * m->n_layer + 2 * m->pn_layer + 4 + 14 + 8;
            if (!arena.open(floats * sizeof(float) + 256 * slack_allocs + 4096)) throw std::runtime_error("qwen3tts device: arena");
            for (int l = 0; l < m->n_layer; ++l) {
                kc[l] = arena.alloc((size_t)maxL * KD);
                vc[l] = arena.alloc((size_t)maxL * KD);
            }
            for (int l = 0; l < m->pn_layer; ++l) {
                pkc[l] = arena.alloc((size_t)(ncb + 1) * pKD);
                pvc[l] = arena.alloc((size_t)(ncb + 1) * pKD);
            }
            rcs = arena.alloc((size_t)maxL * (d.D / 2));
            rsn = arena.alloc((size_t)maxL * (d.D / 2));
            prcs = arena.alloc((size_t)(ncb + 1) * (pd.D / 2));
            prsn = arena.alloc((size_t)(ncb + 1) * (pd.D / 2));
            xrow = arena.alloc(C);
            hid = arena.alloc(C);
            logits = arena.alloc(V);
            plogits = arena.alloc(m->pvocab);
            pin = arena.alloc(mxC);
            pn = arena.alloc(pd.C);
            work.h = arena.alloc(mxC);
            work.q = arena.alloc(mxHD);
            work.k = arena.alloc(mxKD);
            work.v = arena.alloc(mxKD);
            work.att = arena.alloc(mxHD);
            work.o = arena.alloc(mxC);
            work.g = arena.alloc(mxFF);
            work.u = arena.alloc(mxFF);
            dev::rope_tables(maxL, d.D, d.theta, rcs, rsn);
            dev::rope_tables(ncb + 1, pd.D, pd.theta, prcs, prsn);
        } else {
            kc_h.assign(m->n_layer, std::vector<float>((size_t)maxL * KD));
            vc_h.assign(m->n_layer, std::vector<float>((size_t)maxL * KD));
            for (int l = 0; l < m->n_layer; ++l) {
                kc[l] = kc_h[l].data();
                vc[l] = vc_h[l].data();
            }
        }
        // the prefill (on the host, into the caches)
        std::vector<float> x(prefill.begin(), prefill.begin() + (size_t)P * C);
        for (int l = 0; l < m->n_layer; ++l) block_step(m->blk[l], d, x.data(), P, 0, kc[l], vc[l]);
        auto head_host = [&](const float* last) {
            rms_norm(last, 1, C, m->eps, m->out_norm, hid);
            linear(hid, 1, m->codec_head, logits);
        };
        head_host(x.data() + (size_t)(P - 1) * C);
        int pos = P;
        std::vector<int> history;
        const int eos = m->codec_eos;
        uint64_t rng = s->seed;
        int n_out = 0;
        std::vector<int> frame(ncb);
        std::vector<float> lv(m->pvocab);  // the host predictor's logits (the device path writes plogits)
        auto sync = [&]() {
            if (hipStreamSynchronize(0) != hipSuccess) throw std::runtime_error("qwen3tts device: the stream failed");
        };
        auto pred_stack = [&](int p) {  // one token of the predictor at position p, through its layers (pin in place)
            for (int l = 0; l < m->pn_layer; ++l) dev::block(m->pblk[l], pd, pin, pkc[l], pvc[l], p, prcs, prsn, work);
        };
        for (int t = 0; t < max_frames; ++t) {
            // the talker's logits, processed as qwen-tts does (repetition penalty, the suppressed range, the minimum)
            std::vector<float> lg(logits, logits + V);
            // (once per distinct token, as the transformers' processor: its scatter over the unique ids)
            std::vector<int> seen(history);
            std::sort(seen.begin(), seen.end());
            seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
            for (int h : seen) {
                float& l = lg[h];
                l = l < 0.0f ? l * s->repetition_penalty : l / s->repetition_penalty;
            }
            for (int v = V - 1024; v < V; ++v)
                if (v != eos) lg[v] = -INFINITY;
            if (t < 2) lg[eos] = -INFINITY;
            const int c0 = sample_token(lg.data(), V, s->temperature, s->top_k, &rng);
            if (c0 == eos) break;
            history.push_back(c0);
            // the code predictor: the rows H, E(c0), then each new codebook's embedding; the output row k gives codebook k
            // through head k - 1 (causal: one new row per codebook)
            std::fill(frame.begin(), frame.end(), 0);
            frame[0] = c0;
            if (!dev_on) {
                const int pKDh = m->pHkv * m->pD;
                std::vector<std::vector<float>> pk(m->pn_layer, std::vector<float>((size_t)(ncb + 1) * pKDh)),
                    pv(m->pn_layer, std::vector<float>((size_t)(ncb + 1) * pKDh));
                std::vector<float> px((size_t)2 * C), pnh((size_t)C);
                std::memcpy(px.data(), hid, sizeof(float) * C);
                mat_row(m->codec_embd, c0, px.data() + C);
                for (int l = 0; l < m->pn_layer; ++l) block_step(m->pblk[l], pd, px.data(), 2, 0, pk[l].data(), pv[l].data());
                std::vector<float> row(C);
                for (int k = 1; k < ncb; ++k) {
                    // the output row k (the last one processed) predicts codebook k through head k - 1
                    const float* last = k == 1 ? px.data() + C : row.data();
                    rms_norm(last, 1, C, m->eps, m->pred_out_norm, pnh.data());
                    const Lin& hd = m->pred_head[k - 1];
                    linear(pnh.data(), 1, hd, lv.data());
                    const int best = sample_token(lv.data(), m->pvocab, s->sub_temperature, s->sub_top_k, &rng);
                    frame[k] = best;
                    if (k + 1 < ncb) {
                        // the next row: codebook k's embedding in the table k - 1, at position k + 1
                        mat_row(m->pred_embd[k - 1], best, row.data());
                        for (int l = 0; l < m->pn_layer; ++l) block_step(m->pblk[l], pd, row.data(), 1, k + 1, pk[l].data(), pv[l].data());
                    }
                }
            } else {
                // the same on the device: pin holds the predictor's current row (the output of its last token)
                std::memcpy(pin, hid, sizeof(float) * C);
                pred_stack(0);
                sync();  // the host writes pin next: the stack above must have finished with it
                mat_row(m->codec_embd, c0, pin);
                pred_stack(1);
                for (int k = 1; k < ncb; ++k) {
                    dev::head(pin, m->pred_out_norm, m->pred_head[k - 1].w, m->eps, pd.C, pn, plogits);
                    sync();
                    const int best = sample_token(plogits, m->pvocab, s->sub_temperature, s->sub_top_k, &rng);
                    frame[k] = best;
                    if (k + 1 < ncb) {
                        mat_row(m->pred_embd[k - 1], best, pin);
                        pred_stack(k + 1);
                    }
                }
            }
            std::memcpy(out_codes + (size_t)n_out * ncb, frame.data(), sizeof(int) * ncb);
            ++n_out;
            if (t + 1 == max_frames) break;
            // the next input: the frame's sum plus the trailing text (the pad after the trailing rows)
            const float* tr = t < nt ? &trailing[(size_t)t * C] : pad.data();
            if (qwen3tts_frame_input(m, frame.data(), tr, xrow) != 0) return -1;
            if (dev_on) {
                for (int l = 0; l < m->n_layer; ++l) dev::block(m->blk[l], d, xrow, kc[l], vc[l], pos, rcs, rsn, work);
                ++pos;
                dev::head(xrow, m->out_norm, m->codec_head.w, m->eps, C, hid, logits);
                sync();
            } else {
                for (int l = 0; l < m->n_layer; ++l) block_step(m->blk[l], d, xrow, 1, pos, kc[l], vc[l]);
                ++pos;
                head_host(xrow);
            }
        }
        arena.close();
        return n_out;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}

int qwen3tts_generate_greedy(qwen3tts_model_t* m, const int* text_ids, int n_text, const int* ref_ids, int n_ref,
                             const int* ref_codes, int R, const float* spk, int language_id, float repetition_penalty,
                             int max_frames, int* out_codes) {
    qwen3tts_sampling_t g = qwen3tts_sampling_default();
    g.temperature = 0.0f;
    g.sub_temperature = 0.0f;
    g.top_k = 0;
    g.sub_top_k = 0;
    g.repetition_penalty = repetition_penalty;
    return qwen3tts_generate(m, text_ids, n_text, ref_ids, n_ref, ref_codes, R, spk, language_id, &g, max_frames, out_codes);
}

}  // extern "C"
