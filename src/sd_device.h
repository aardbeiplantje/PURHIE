// The image engine's device path (sd.cpp; PLAN.md I6): the activations are float32 in registered UMA memory (Floats), the
// F16 weights are read in place from the GGUF mapping (register_weights), and each op is one of sd_gpu.hip's kernels. The
// host code of sd.cpp is the fallback (SD_DEVICE=cpu, or no device): the same functions, on the same memory.
#ifndef SD_DEVICE_H
#define SD_DEVICE_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace sddev {

// The device path is on when SD_DEVICE is unset or "gpu", a device is present and the kernels load.
bool enabled();

// The GGUF mapping [base, base + bytes) registered read-only for the device (the weights are read in place). Returns false
// when the registration fails. unregister_weights is the mapping's release (a no-op for an unregistered base).
bool register_weights(const void* base, size_t bytes);
void unregister_weights(const void* base);

// One sd_model at a time on the device: the callers hold this for a whole UNet or VAE call (the streams sync per op).
std::recursive_mutex& run_mutex();

// Float memory (zeroed unless zero is false): pinned and device-visible when the device path is on, else calloc. Pooled.
void* alloc(size_t bytes, bool zero = true);  // zero = false: the caller writes every byte (no memset: a CPU pass)
void release(void* p, size_t bytes);

// Float32 activations: the memory of alloc, zeroed. Copies are deep (the image's tensors are small next to the weights).
class Floats {
   public:
    Floats() = default;
    explicit Floats(size_t n) : n_(n), p_(n ? static_cast<float*>(alloc(n * sizeof(float))) : nullptr) {}
    ~Floats() { reset(); }
    Floats(const Floats& o) : Floats(o.n_) {
        if (n_) std::memcpy(p_, o.p_, n_ * sizeof(float));
    }
    Floats(Floats&& o) noexcept : n_(o.n_), p_(o.p_) {
        o.n_ = 0;
        o.p_ = nullptr;
    }
    Floats& operator=(const Floats& o) {
        if (this != &o) {
            Floats tmp(o);
            swap(tmp);
        }
        return *this;
    }
    Floats& operator=(Floats&& o) noexcept {
        if (this != &o) {
            reset();
            n_ = o.n_;
            p_ = o.p_;
            o.n_ = 0;
            o.p_ = nullptr;
        }
        return *this;
    }
    float* data() { return p_; }
    const float* data() const { return p_; }
    size_t size() const { return n_; }
    float& operator[](size_t i) { return p_[i]; }
    const float& operator[](size_t i) const { return p_[i]; }
    void swap(Floats& o) noexcept {
        size_t n = n_;
        float* p = p_;
        n_ = o.n_;
        p_ = o.p_;
        o.n_ = n;
        o.p_ = p;
    }

   private:
    void reset() {
        release(p_, n_ * sizeof(float));
        p_ = nullptr;
        n_ = 0;
    }
    size_t n_ = 0;
    float* p_ = nullptr;
};

// The ops. Activations: float32 in alloc'd memory; weights: F16 (uint16_t) in the registered mapping; biases F16 or null.
// Conv: x [cin][H][W] -> y [cout][Ho][Wo], H' = (H + 2p - k) / st + 1. Linear: token rows x[t][in] -> y[t][out]; the
// channel-plane input x[c][t] (linear_cm); the channel-plane output y[o][t] (linear_to_cm, accumulating into y when set).
void conv2d(const float* x, int cin, int H, int W, const uint16_t* w, int cout, const uint16_t* b, int k, int p, int st, float* y,
            int Ho, int Wo);
void linear_tok(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y);
void linear_cm(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y);
// The same with float32 weights and bias (b may be null): y[t][o] = b[o] + sum_i x[t][i] W[o][i]
void linear_f32(const float* x, int T, const float* W, int in, int out, const float* b, float* y);
// conv2d with F32 weights [cout][cin][k][k] (read in place) and F32 bias; the same im2col + GEMM as conv2d
void conv2d_f32(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y, int Ho, int Wo);
// Qwen-Image's FP8 E4M3 linear (sd_gpu.hip qi_linear_f8): W [out][in] raw bytes read in place, times wscale; the bias is
// bias_kind 0 none, 1 E4M3, 2 BF16 bits, 3 F32. y[t][o] = wscale sum_i x[t][i] W[o][i] + b[o]
void linear_f8(const float* x, int T, const uint8_t* W, float wscale, int in, int out, const void* b, int bias_kind, float* y);
// Qwen-Image's norms and elementwise ops (sd_gpu.hip): the RMS norm with E4M3 weights; x[t][c] = x[t][c] (1 + s[c]) + b[c] in
// place; GELU (tanh approximation) in place
void rms_norm_f8(const float* x, int T, int C, const uint8_t* w, float eps, float* y);
void affine(float* x, int T, int C, const float* s, const float* b);
void gelu_tanh(float* x, size_t n);
// Qwen-Image's VAE decoder ops (sd_gpu.hip), token-major x[h][w][c] (H x W pixels, C channels): the channel RMS norm (gamma
// BF16); the convolution patches for the GEMM (mode 1: 2D 3 x 3, K = 9C; mode 3: causal 3 x 3 x 3 on one frame, K = 27C) of the
// pixel rows p0..p0+n-1 into P [n][K]; a BF16 bias added to rows of C; the nearest 2x upsampling
void rms_tok(const float* x, int HW, int C, const uint16_t* g, float* y);
void patches(const float* x, int H, int W, int C, int mode, int p0, int n, float* P);
void bias_tok(float* y, long rows, int C, const uint16_t* b);
void upsample2_tok(const float* x, int H, int W, int C, float* y);
// The same with bfloat16 weights (the checkpoint's BF16, raw bits): y[t][o] = b[o] + sum_i x[t][i] W[o][i]
void linear_bf16(const float* x, int T, const uint16_t* W, int in, int out, const float* b, float* y);
void linear_to_cm(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y, bool accumulate);
void group_norm(const float* x, int C, int HW, int groups, const uint16_t* gw, const uint16_t* gb, float eps, float* y);
void silu(float* x, size_t n);
void add(float* out, const float* x, size_t n);
void upsample2(const float* x, int C, int H, int W, float* y);
void layer_norm(const float* x, int T, int C, const uint16_t* w, const uint16_t* b, float eps, float* y);
void rms_norm(const float* x, int T, int C, const uint16_t* w, float eps, float* y);
// Z-Image's block ops (sd_gpu.hip): x[t][c] *= s[c]; out[t][c] += g[c] * y[t][c]; g = silu(a) * b; the RoPE rotation of
// q [T][heads][128] by fr [T][64][2] = (cos, sin) per pair
void modulate(float* x, int T, int C, const float* s);
// LayerNorm of each row without weights: y = (x - mean) rsqrt(var + eps)
void layer_norm_plain(const float* x, int T, int C, float eps, float* y);
// GroupNorm with F32 affine weights on the channel-major [C][HW] x: y = groupnorm(x) * gamma + beta (per channel)
// transposes between channel-major [C][HW] and token-major [HW][C]
// Qwen3's text encoder ops (sd_gpu.hip): RoPE by rotate-half (x [T][heads][128], cos/sin [T][64]); causal attention with
// grouped KV heads (T <= 64): out [T][H][128] from q [T][H][128], k and v [T][KVH][128]
void rope_half(float* x, int T, int heads, const float* cos, const float* sin);
void causal_attn(const float* q, const float* k, const float* v, int T, int H, int KVH, float* out);
void to_tokens(const float* x, int C, int HW, float* y);
void to_chw(const float* x, int C, int HW, float* y);
void group_norm_f32(const float* x, int C, int HW, int groups, const float* gamma, const float* beta, float eps, float* y);
void gate_add(float* out, int T, int C, const float* g, const float* y);
void swiglu(const float* a, const float* b, float* g, size_t n);
void rope(float* q, int T, int heads, const float* fr);  // y = x rsqrt(mean(x^2) + eps) w
void geglu(const float* pj, long rows, int ff, float* g);
// The attention over token rows: q [Tq][C], k and v [Tk][C] (heads of width C / heads), out [Tq][C]; scores scaled by 1/sqrt(D).
void attention(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out);

}  // namespace sddev

#endif  // SD_DEVICE_H
