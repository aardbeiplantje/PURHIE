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

// Zeroed float memory: pinned and device-visible when the device path is on, else calloc. Pooled (a freed block is reused).
void* alloc(size_t bytes);
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
void linear_to_cm(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y, bool accumulate);
void group_norm(const float* x, int C, int HW, int groups, const uint16_t* gw, const uint16_t* gb, float eps, float* y);
void silu(float* x, size_t n);
void add(float* out, const float* x, size_t n);
void upsample2(const float* x, int C, int H, int W, float* y);
void layer_norm(const float* x, int T, int C, const uint16_t* w, const uint16_t* b, float eps, float* y);
void geglu(const float* pj, long rows, int ff, float* g);
// The attention over token rows: q [Tq][C], k and v [Tk][C] (heads of width C / heads), out [Tq][C]; scores scaled by 1/sqrt(D).
void attention(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out);

}  // namespace sddev

#endif  // SD_DEVICE_H
