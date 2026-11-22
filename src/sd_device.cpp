// The image engine's device path (sd_device.h): the launches of sd_gpu.hip's kernels on the null stream, each op synchronous
// (the host reads the activations after it). The GEMM covers the convolutions (weights x im2col), the linear layers and the
// attention; the attention's score rows are chunked so the scores of a chunk stay within 16 MB.
#include "sd_device.h"

#include "kernel_loader.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sddev {
namespace {

std::recursive_mutex g_run;
std::mutex g_pool_mu;
struct Block {
    void* p;
    size_t cap;
};
std::vector<Block> g_free;  // the released blocks, reused by alloc (best fit)

int g_state = -1;  // -1: not tried, 0: host only, 1: device
hipModule_t g_mod = nullptr;
hipFunction_t f_gemm_ah = nullptr, f_gemm_af = nullptr, f_im2col = nullptr, f_gn_sum = nullptr, f_gn_sq = nullptr,
              f_gn_apply = nullptr, f_silu = nullptr, f_add = nullptr, f_upsample2 = nullptr, f_softmax = nullptr,
              f_layer_norm = nullptr, f_rms_norm = nullptr, f_zm_modulate = nullptr, f_zm_gate_add = nullptr, f_zm_swiglu = nullptr, f_zm_rope = nullptr, f_zm_ln_plain = nullptr, f_gn_f32 = nullptr, f_to_tok = nullptr, f_to_chw = nullptr, f_zq_rope = nullptr, f_zq_attn = nullptr, f_geglu = nullptr, f_gemm_bf = nullptr, f_gemm_wmma = nullptr, f_gemm_wmma128 = nullptr, f_softmax_bf16w = nullptr, f_flash = nullptr, f_gemm_wmma256 = nullptr, f_gemm_wmma256b = nullptr, f_gemm_wmma128bb = nullptr, f_transpose_pad_bf16 = nullptr,
              f_to_bf16 = nullptr, f_transpose_pad = nullptr, f_softmax_bf16 = nullptr, f_qi_linear_f8 = nullptr, f_qi_rms_f8 = nullptr, f_qi_affine = nullptr, f_qi_gelu = nullptr, f_qv_rms = nullptr, f_qv_patch = nullptr, f_qv_bias = nullptr, f_qv_upsample = nullptr, f_qi_hadamard = nullptr, f_qi_linear_i8 = nullptr, f_qi21_zc = nullptr, f_qi21_gather = nullptr, f_qi21_mul = nullptr, f_qi21_gated = nullptr, f_qi21_attn = nullptr, f_qi21_lbf = nullptr;

constexpr int kThreads = 256;
constexpr int kGroupSplit = 16;  // the group norm's partial sums per group
constexpr size_t kColFloats = (size_t)1 << 23;  // the im2col chunk: at most 32 MB
constexpr size_t kScoreFloats = (size_t)1 << 22;  // the attention's score chunk: at most 16 MB

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("sd device: ") + what + ": " + hipGetErrorString(e));
}

bool init() {
    if (g_state >= 0) return g_state == 1;
    g_state = 0;
    const char* env = std::getenv("SD_DEVICE");
    const std::string want = env ? env : "gpu";
    if (want != "gpu") return false;
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) return false;
    if (kernel_module_load(&g_mod, "sd_gpu.co") != hipSuccess) return false;
    struct Entry {
        hipFunction_t* f;
        const char* name;
    };
    const Entry entries[] = {
        {&f_gemm_ah, "sd_gemm_ah"},     {&f_gemm_af, "sd_gemm_af"},     {&f_im2col, "sd_im2col"}, {&f_gemm_bf, "sd_gemm_bf"}, {&f_gemm_wmma, "sd_gemm_wmma"}, {&f_gemm_wmma128, "sd_gemm_wmma128"}, {&f_gemm_wmma128bb, "sd_gemm_wmma128bb"}, {&f_gemm_wmma256, "sd_gemm_wmma256"}, {&f_gemm_wmma256b, "sd_gemm_wmma256b"}, {&f_flash, "sd_flash_attn"}, {&f_transpose_pad_bf16, "sd_transpose_pad_bf16"}, {&f_to_bf16, "sd_to_bf16"}, {&f_qi_linear_f8, "qi_linear_f8"}, {&f_qi_rms_f8, "qi_rms_norm_f8"}, {&f_qi_affine, "qi_affine"}, {&f_qi_gelu, "qi_gelu_tanh"}, {&f_qv_rms, "qv_rms_tok"}, {&f_qv_patch, "qv_patch"}, {&f_qv_bias, "qv_bias_tok"}, {&f_qv_upsample, "qv_upsample2_tok"}, {&f_qi_hadamard, "qi_hadamard256"}, {&f_qi_linear_i8, "qi_linear_i8"}, {&f_qi21_zc, "qi21_zc_rms"}, {&f_qi21_gather, "qi21_gather_mod"}, {&f_qi21_mul, "qi21_mul_1p"}, {&f_qi21_gated, "qi21_gated_add"}, {&f_qi21_attn, "qi21_attn"}, {&f_qi21_lbf, "qi21_linear_bf16"},
        {&f_transpose_pad, "sd_transpose_pad"}, {&f_softmax_bf16, "sd_softmax_bf16"}, {&f_softmax_bf16w, "sd_softmax_bf16_w"},
        {&f_gn_sum, "sd_gn_sum"},       {&f_gn_sq, "sd_gn_sq"},         {&f_gn_apply, "sd_gn_apply"},
        {&f_silu, "sd_silu"},           {&f_add, "sd_add"},             {&f_upsample2, "sd_upsample2"},
        {&f_softmax, "sd_softmax"},     {&f_layer_norm, "sd_layer_norm"}, {&f_rms_norm, "sd_rms_norm"}, {&f_zm_modulate, "zimage_modulate"}, {&f_zm_gate_add, "zimage_gate_add"}, {&f_zm_swiglu, "zimage_swiglu"}, {&f_zm_rope, "zimage_rope"}, {&f_zm_ln_plain, "zimage_layer_norm_plain"}, {&f_gn_f32, "zimage_group_norm_f32"}, {&f_to_tok, "zimage_to_tokens"}, {&f_to_chw, "zimage_to_chw"}, {&f_zq_rope, "zq_rope_half"}, {&f_zq_attn, "zq_causal_attn"}, {&f_geglu, "sd_geglu"},
    };
    for (const Entry& e : entries)
        if (hipModuleGetFunction(e.f, g_mod, e.name) != hipSuccess) return false;
    g_state = 1;
    return true;
}

void launch(hipFunction_t f, dim3 grid, void** args) {
    check(hipModuleLaunchKernel(f, grid.x, grid.y, grid.z, kThreads, 1, 1, 0, nullptr, args, nullptr), "launch");
}

void sync() { check(hipStreamSynchronize(nullptr), "stream"); }

// C[m * cs_m + n * cs_n] = alpha * sum_k A(m, k) B(k, n) + bias[m] + beta * C, the strides of gemm_body (sd_gpu.hip)
template <typename TA>
void gemm(hipFunction_t f, const TA* A, long lda, const float* B, long sbk, long sbn, float* C, long cs_m, long cs_n,
          const uint16_t* bias, float alpha, float beta, int M, int N, int K, const float* biasf = nullptr) {
    if (M <= 0 || N <= 0) return;
    const TA* a = A;
    const float* b = B;
    float* c = C;
    const uint16_t* bi = bias;
    const float* bf = biasf;
    void* args[] = {(void*)&a, (void*)&lda, (void*)&b, (void*)&sbk, (void*)&sbn, (void*)&c, (void*)&cs_m, (void*)&cs_n,
                    (void*)&bi, (void*)&bf, (void*)&alpha, (void*)&beta, (void*)&M, (void*)&N, (void*)&K};
    launch(f, dim3((unsigned)((N + 63) / 64), (unsigned)((M + 63) / 64), 1), args);
}

size_t round_up_page(size_t bytes) { return (bytes + 4095) & ~(size_t)4095; }

unsigned blocks(size_t n) { return (unsigned)((n + kThreads - 1) / kThreads); }

}  // namespace

bool enabled() {
    std::lock_guard<std::recursive_mutex> lk(g_run);
    return init();
}

bool register_weights(const void* base, size_t bytes) {
    if (!enabled() || !base) return false;
    return hipHostRegister(const_cast<void*>(base), bytes, hipHostRegisterReadOnly | hipExtHostRegisterCoarseGrained) == hipSuccess;
}

void unregister_weights(const void* base) {
    if (base && g_state == 1) (void)hipHostUnregister(const_cast<void*>(base));
}

std::recursive_mutex& run_mutex() { return g_run; }

void* alloc(size_t bytes, bool zero) {
    if (bytes == 0) return nullptr;
    const size_t need = round_up_page(bytes);
    if (!enabled()) {
        void* p = zero ? std::calloc(1, need) : std::malloc(need);
        if (!p) throw std::runtime_error("sd: out of memory");
        return p;
    }
    {
        std::lock_guard<std::mutex> lk(g_pool_mu);
        size_t best = (size_t)-1;
        for (size_t i = 0; i < g_free.size(); ++i)
            if (g_free[i].cap >= need && (best == (size_t)-1 || g_free[i].cap < g_free[best].cap)) best = i;
        if (best != (size_t)-1) {
            void* p = g_free[best].p;
            g_free.erase(g_free.begin() + (long)best);
            if (zero) std::memset(p, 0, need);
            return p;
        }
    }
    void* p = nullptr;
    if (hipHostMalloc(&p, need, hipHostMallocDefault) != hipSuccess) throw std::runtime_error("sd device: out of pinned memory");
    if (zero) std::memset(p, 0, need);
    return p;
}

void release(void* p, size_t bytes) {
    if (!p) return;
    if (!enabled()) {
        std::free(p);
        return;
    }
    std::lock_guard<std::mutex> lk(g_pool_mu);
    g_free.push_back({p, round_up_page(bytes)});
}

void conv2d(const float* x, int cin, int H, int W, const uint16_t* w, int cout, const uint16_t* b, int k, int p, int st, float* y,
            int Ho, int Wo) {
    const long HWo = (long)Ho * Wo;
    const int K = cin * k * k;
    if (HWo <= 0 || K <= 0) return;
    const long nc_max = std::max<long>(1, std::min<long>(HWo, (long)(kColFloats / (size_t)K)));
    float* col = static_cast<float*>(alloc(sizeof(float) * (size_t)K * nc_max));
    for (long n0l = 0; n0l < HWo; n0l += nc_max) {
        const int n0 = (int)n0l;
        const int nc = (int)std::min<long>(nc_max, HWo - n0l);
        void* a1[] = {(void*)&x, (void*)&cin, (void*)&H, (void*)&W, (void*)&k, (void*)&p, (void*)&st, (void*)&Wo, (void*)&n0, (void*)&nc, (void*)&col};
        const dim3 g((unsigned)((nc + kThreads - 1) / kThreads), (unsigned)K, 1);
        launch(f_im2col, g, a1);
        gemm<uint16_t>(f_gemm_ah, w, K, col, nc, 1, y + n0l, HWo, 1, b, 1.0f, 0.0f, cout, nc, K);
    }
    sync();
    release(col, sizeof(float) * (size_t)K * nc_max);
}

void conv2d_f32(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y,
                int Ho, int Wo) {
    const long HWo = (long)Ho * Wo;
    const int K = cin * k * k;
    if (HWo <= 0 || K <= 0) return;
    const long nc_max = std::max<long>(1, std::min<long>(HWo, (long)(kColFloats / (size_t)K)));
    float* col = static_cast<float*>(alloc(sizeof(float) * (size_t)K * nc_max));
    for (long n0l = 0; n0l < HWo; n0l += nc_max) {
        const int n0 = (int)n0l;
        const int nc = (int)std::min<long>(nc_max, HWo - n0l);
        void* a1[] = {(void*)&x, (void*)&cin, (void*)&H, (void*)&W, (void*)&k, (void*)&p, (void*)&st, (void*)&Wo, (void*)&n0, (void*)&nc, (void*)&col};
        const dim3 g((unsigned)((nc + kThreads - 1) / kThreads), (unsigned)K, 1);
        launch(f_im2col, g, a1);
        gemm<float>(f_gemm_af, w, K, col, nc, 1, y + n0l, HWo, 1, nullptr, 1.0f, 0.0f, cout, nc, K, b);
    }
    sync();
    release(col, sizeof(float) * (size_t)K * nc_max);
}

void linear_tok(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y) {
    gemm<uint16_t>(f_gemm_ah, W, in, x, 1, in, y, 1, out, b, 1.0f, 0.0f, out, T, in);
    sync();
}

void linear_f32(const float* x, int T, const float* W, int in, int out, const float* b, float* y) {
    gemm<float>(f_gemm_af, W, in, x, 1, in, y, 1, out, nullptr, 1.0f, 0.0f, out, T, in, b);
    sync();
}

// Qwen-Image's FP8 linear (sd_gpu.hip qi_linear_f8): the weights (E4M3 bytes, read in place) times wscale, plus the bias
// (bias_kind: 0 none, 1 E4M3, 2 BF16 bits, 3 F32)
void linear_f8(const float* x, int T, const uint8_t* W, float wscale, int in, int out, const void* b, int bias_kind, float* y) {
    if (out <= 0 || T <= 0) return;
    if (in % 16 != 0) throw std::runtime_error("linear_f8: the input width must be a multiple of 16");
    void* a[] = {(void*)&x, (void*)&T, (void*)&W, (void*)&wscale, (void*)&in, (void*)&out, (void*)&b, (void*)&bias_kind, (void*)&y};
    check(hipModuleLaunchKernel(f_qi_linear_f8, (unsigned)((out + 63) / 64), (unsigned)((T + 63) / 64), 1, 256, 1, 1, 0, nullptr, a,
                                nullptr),
          "qi_linear_f8");
    sync();
}

// Qwen-Image's RMS norm with E4M3 weights (no scale): y = x rsqrt(mean(x^2) + eps) w, one row of C per token
void rms_norm_f8(const float* x, int T, int C, const uint8_t* w, float eps, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&w, (void*)&eps, (void*)&y};
    launch(f_qi_rms_f8, dim3((unsigned)T, 1, 1), a);
    sync();
}

// x[t][c] = x[t][c] (1 + s[c]) + b[c], in place
void affine(float* x, int T, int C, const float* s, const float* b) {
    long n = (long)T * C;
    void* a[] = {(void*)&x, (void*)&s, (void*)&b, (void*)&C, (void*)&n};
    launch(f_qi_affine, dim3((unsigned)((n + kThreads - 1) / kThreads)), a);
    sync();
}

// GELU (tanh approximation), in place
void gelu_tanh(float* x, size_t n) {
    long nn = (long)n;
    void* a[] = {(void*)&x, (void*)&nn};
    launch(f_qi_gelu, dim3((unsigned)((nn + kThreads - 1) / kThreads)), a);
    sync();
}

// Qwen-Image's VAE decoder ops (token-major x[h][w][c]): the channel RMS norm (gamma BF16); the convolution patches (mode 1: 2D 3 x 3,
// mode 3: causal 3 x 3 x 3 on one frame) for the GEMM; a BF16 bias added to rows of C; the nearest 2x upsampling
void rms_tok(const float* x, int HW, int C, const uint16_t* g, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&g, (void*)&y};
    launch(f_qv_rms, dim3((unsigned)HW, 1, 1), a);
    sync();
}

void patches(const float* x, int H, int W, int C, int mode, int p0, int n, float* P) {
    long total = (long)n * (mode == 3 ? 27 : 9) * C;
    void* a[] = {(void*)&x, (void*)&H, (void*)&W, (void*)&C, (void*)&mode, (void*)&p0, (void*)&n, (void*)&P};
    launch(f_qv_patch, dim3((unsigned)((total + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void bias_tok(float* y, long rows, int C, const uint16_t* b) {
    long n = rows * C;
    void* a[] = {(void*)&y, (void*)&b, (void*)&C, (void*)&n};
    launch(f_qv_bias, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void upsample2_tok(const float* x, int H, int W, int C, float* y) {
    long n = 4L * H * W * C;
    void* a[] = {(void*)&x, (void*)&H, (void*)&W, (void*)&C, (void*)&y};
    launch(f_qv_upsample, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

// Qwen-Image 2.1's ConvRot linear (see sd_gpu.hip): x [T][in] rotated by H (the regular Hadamard of 256, normalised, 256 x 256)
// in blocks of the input, then y[t][o] = scale[o] sum_i xr[t][i] W[o][i] with the int8 weights read in place. xr is caller memory
// in sddev::alloc (T * in floats).
void linear_convrot(const float* x, int T, const int8_t* W, const float* wscale, const float* H, int in, int out, float* xr, float* y) {
    if (out <= 0 || T <= 0) return;
    if (in % 256 != 0) throw std::runtime_error("linear_convrot: the input width must be a multiple of 256");
    void* ha[] = {(void*)&x, (void*)&T, (void*)&in, (void*)&H, (void*)&xr};
    check(hipModuleLaunchKernel(f_qi_hadamard, (unsigned)(in / 256), (unsigned)T, 1, 256, 1, 1, 0, nullptr, ha, nullptr), "qi_hadamard256");
    void* la[] = {(void*)&xr, (void*)&T, (void*)&W, (void*)&wscale, (void*)&in, (void*)&out, (void*)&y};
    check(hipModuleLaunchKernel(f_qi_linear_i8, (unsigned)((out + 63) / 64), (unsigned)((T + 63) / 64), 1, 256, 1, 1, 0, nullptr, la,
                                nullptr),
          "qi_linear_i8");
    sync();
}

// Qwen-Image 2.1 ops (sd_gpu.hip): the zero-centred RMS norm (BF16 weights w, scale w + 1); the per-token modulation gather (mod is
// [2][C]: row 0 the sampled timestep's, row 1 t = 0's; sel[j] > 0.5 picks row 1); x *= 1 + s; h += tanh(g) y; and the masked attention
void zc_rms(const float* x, int T, int C, const uint16_t* w, float eps, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&w, (void*)&eps, (void*)&y};
    launch(f_qi21_zc, dim3((unsigned)T, 1, 1), a);
    sync();
}

void gather_mod(const float* mod, const float* sel, int J, int C, float* out) {
    long n = (long)J * C;
    void* a[] = {(void*)&mod, (void*)&sel, (void*)&C, (void*)&n, (void*)&out};
    launch(f_qi21_gather, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void mul_1p(float* x, const float* s, size_t n) {
    long nn = (long)n;
    void* a[] = {(void*)&x, (void*)&s, (void*)&nn};
    launch(f_qi21_mul, dim3((unsigned)((nn + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void gated_add(float* h, const float* g, const float* y, size_t n) {
    long nn = (long)n;
    void* a[] = {(void*)&h, (void*)&g, (void*)&y, (void*)&nn};
    launch(f_qi21_gated, dim3((unsigned)((nn + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

// Qwen-Image 2.1's BF16 linear with FP32 activations (no rounding of x): y = x W^T, W [out][in] BF16 bits in place, in % 16 == 0
void linear_bf16_f32(const float* x, int T, const uint16_t* W, int in, int out, float* y) {
    if (out <= 0 || T <= 0) return;
    if (in % 16 != 0) throw std::runtime_error("linear_bf16_f32: the input width must be a multiple of 16");
    void* a[] = {(void*)&x, (void*)&T, (void*)&W, (void*)&in, (void*)&out, (void*)&y};
    check(hipModuleLaunchKernel(f_qi21_lbf, (unsigned)((out + 63) / 64), (unsigned)((T + 63) / 64), 1, 256, 1, 1, 0, nullptr, a, nullptr),
          "qi21_linear_bf16");
    sync();
}

void attn21_text(const float* q, const float* k, const float* v, int J, int L, int heads, float* out) {
    long n = (long)L * heads;
    void* a[] = {(void*)&q, (void*)&k, (void*)&v, (void*)&J, (void*)&L, (void*)&heads, (void*)&out};
    launch(f_qi21_attn, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void linear_bf16(const float* x, int T, const uint16_t* W, int in, int out, const float* b, float* y) {
    if (out <= 0 || T <= 0) return;
    // the tensor-core tile (sd_gemm_wmma, 128 x 128 when the output has the rows for it; the F32 activations rounded on load:
    // measured faster than converting them once, see PLAN.md)
    if (in % 64 == 0) {
        const bool big = out >= 128;
        const bool huge = out >= 256 && T >= 1024;  // the 256 x 128 tile for the large token counts
        const float alpha = 1.0f, beta = 0.0f;
        const int M = out, N = T, K = in;
        const long bs0 = 0;
        if (huge && f_gemm_wmma256b) {
            // the activations converted to BF16 once (the tile re-reads them once per 256 output rows: the F32 bytes were the
            // bottleneck, see sd_gemm_wmma256b)
            const long nb = (long)T * in;
            uint16_t* xb = static_cast<uint16_t*>(alloc(sizeof(uint16_t) * (size_t)nb, false));
            const float* xs = x;
            void* ac[] = {(void*)&xs, (void*)&xb, (void*)&nb};
            launch(f_to_bf16, dim3((unsigned)((nb + kThreads - 1) / kThreads)), ac);
            const uint16_t* A = W;
            const uint16_t* B = xb;
            float* C = y;
            const long lda = in, sbn = in, cs_m = 1, cs_n = out;
            const float* bf = b;
            void* args[] = {(void*)&A, (void*)&lda, (void*)&B, (void*)&sbn, (void*)&C, (void*)&cs_m, (void*)&cs_n,
                            (void*)&bf, (void*)&alpha, (void*)&beta, (void*)&M, (void*)&N, (void*)&K, (void*)&bs0, (void*)&bs0,
                            (void*)&bs0};
            check(hipModuleLaunchKernel(f_gemm_wmma256b, (unsigned)((N + 127) / 128), (unsigned)((M + 255) / 256), 1, 256, 1, 1, 0,
                                        nullptr, args, nullptr),
                  "wmma gemm 256b");
            sync();
            release(xb, sizeof(uint16_t) * (size_t)nb);
            return;
        }
        const uint16_t* A = W;
        const float* B = x;
        float* C = y;
        const long lda = in, sbn = in, cs_m = 1, cs_n = out;
        const float* bf = b;
        void* args[] = {(void*)&A, (void*)&lda, (void*)&B, (void*)&sbn, (void*)&C, (void*)&cs_m, (void*)&cs_n,
                        (void*)&bf, (void*)&alpha, (void*)&beta, (void*)&M, (void*)&N, (void*)&K, (void*)&bs0, (void*)&bs0,
                        (void*)&bs0};
        if (huge)
            check(hipModuleLaunchKernel(f_gemm_wmma256, (unsigned)((N + 127) / 128), (unsigned)((M + 255) / 256), 1, 256, 1, 1, 0,
                                        nullptr, args, nullptr),
                  "wmma gemm 256");
        else
            check(hipModuleLaunchKernel(big ? f_gemm_wmma128 : f_gemm_wmma, (unsigned)((N + (big ? 127 : 63)) / (big ? 128 : 64)),
                                        (unsigned)((M + (big ? 127 : 63)) / (big ? 128 : 64)), 1, big ? 256 : 128, 1, 1, 0, nullptr,
                                        args, nullptr),
                  "wmma gemm");
        sync();
        return;
    }
    gemm<uint16_t>(f_gemm_bf, W, in, x, 1, in, y, 1, out, nullptr, 1.0f, 0.0f, out, T, in, b);
    sync();
}

void linear_cm(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y) {
    gemm<uint16_t>(f_gemm_ah, W, in, x, T, 1, y, 1, out, b, 1.0f, 0.0f, out, T, in);
    sync();
}

void linear_to_cm(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y, bool accumulate) {
    gemm<uint16_t>(f_gemm_ah, W, in, x, 1, in, y, T, 1, b, 1.0f, accumulate ? 1.0f : 0.0f, out, T, in);
    sync();
}

void group_norm(const float* x, int C, int HW, int groups, const uint16_t* gw, const uint16_t* gb, float eps, float* y) {
    const int per = C / groups;
    const long n_group = (long)per * HW;
    const int S = kGroupSplit;
    float* psum = static_cast<float*>(alloc(sizeof(float) * (size_t)groups * S));
    float* psq = static_cast<float*>(alloc(sizeof(float) * (size_t)groups * S));
    {
        const float* xp = x;
        void* a[] = {(void*)&xp, (void*)&n_group, (void*)&S, (void*)&psum};
        launch(f_gn_sum, dim3((unsigned)S, (unsigned)groups, 1), a);
    }
    {
        const float* xp = x;
        const float* ps = psum;
        void* a[] = {(void*)&xp, (void*)&n_group, (void*)&S, (void*)&ps, (void*)&psq};
        launch(f_gn_sq, dim3((unsigned)S, (unsigned)groups, 1), a);
    }
    {
        const float* xp = x;
        const float* ps = psum;
        const float* pq = psq;
        const int Cc = C, HWc = HW, perc = per;
        const float e = eps;
        void* a[] = {(void*)&xp, (void*)&Cc, (void*)&HWc, (void*)&perc, (void*)&n_group, (void*)&S, (void*)&ps, (void*)&pq,
                     (void*)&gw, (void*)&gb, (void*)&e, (void*)&y};
        launch(f_gn_apply, dim3(blocks((size_t)C * HW), 1, 1), a);
    }
    sync();
    release(psum, sizeof(float) * (size_t)groups * S);
    release(psq, sizeof(float) * (size_t)groups * S);
}

void silu(float* x, size_t n) {
    if (!n) return;
    long nn = (long)n;
    void* a[] = {(void*)&x, (void*)&nn};
    launch(f_silu, dim3(blocks(n), 1, 1), a);
    sync();
}

void add(float* out, const float* x, size_t n) {
    if (!n) return;
    long nn = (long)n;
    void* a[] = {(void*)&out, (void*)&x, (void*)&nn};
    launch(f_add, dim3(blocks(n), 1, 1), a);
    sync();
}

void upsample2(const float* x, int C, int H, int W, float* y) {
    const size_t total = (size_t)C * 2 * H * 2 * W;
    void* a[] = {(void*)&x, (void*)&C, (void*)&H, (void*)&W, (void*)&y};
    launch(f_upsample2, dim3(blocks(total), 1, 1), a);
    sync();
}

void rope_half(float* x, int T, int heads, const float* cos, const float* sin) {
    long n = (long)T * heads * 64;
    void* a[] = {(void*)&x, (void*)&T, (void*)&heads, (void*)&cos, (void*)&sin, (void*)&n};
    launch(f_zq_rope, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void causal_attn(const float* q, const float* k, const float* v, int T, int H, int KVH, float* out) {
    void* a[] = {(void*)&q, (void*)&k, (void*)&v, (void*)&T, (void*)&H, (void*)&KVH, (void*)&out};
    check(hipModuleLaunchKernel(f_zq_attn, (unsigned)T, (unsigned)H, 1, 128, 1, 1, 0, nullptr, a, nullptr), "causal attention");
    sync();
}

void to_tokens(const float* x, int C, int HW, float* y) {
    const long n = (long)C * HW;
    void* a[] = {(void*)&x, (void*)&C, (void*)&HW, (void*)&y};
    launch(f_to_tok, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void to_chw(const float* x, int C, int HW, float* y) {
    const long n = (long)C * HW;
    void* a[] = {(void*)&x, (void*)&C, (void*)&HW, (void*)&y};
    launch(f_to_chw, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void group_norm_f32(const float* x, int C, int HW, int groups, const float* gamma, const float* beta, float eps, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&HW, (void*)&groups, (void*)&gamma, (void*)&beta, (void*)&eps, (void*)&y};
    launch(f_gn_f32, dim3((unsigned)groups, 1, 1), a);
    sync();
}

void layer_norm_plain(const float* x, int T, int C, float eps, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&eps, (void*)&y};
    launch(f_zm_ln_plain, dim3((unsigned)T, 1, 1), a);
    sync();
}

void modulate(float* x, int T, int C, const float* s) {
    long n = (long)T * C;
    void* a[] = {(void*)&x, (void*)&s, (void*)&C, (void*)&n};
    launch(f_zm_modulate, dim3((unsigned)((n + kThreads - 1) / kThreads)), a);
    sync();
}

void gate_add(float* out, int T, int C, const float* g, const float* y) {
    long n = (long)T * C;
    void* a[] = {(void*)&out, (void*)&g, (void*)&y, (void*)&C, (void*)&n};
    launch(f_zm_gate_add, dim3((unsigned)((n + kThreads - 1) / kThreads)), a);
    sync();
}

void swiglu(const float* a0, const float* b0, float* g, size_t n0) {
    long n = (long)n0;
    void* a[] = {(void*)&a0, (void*)&b0, (void*)&g, (void*)&n};
    launch(f_zm_swiglu, dim3((unsigned)((n + kThreads - 1) / kThreads)), a);
    sync();
}

void rope(float* q, int T, int heads, const float* fr) {
    long n = (long)T * heads * 64;
    void* a[] = {(void*)&q, (void*)&fr, (void*)&T, (void*)&heads, (void*)&n};
    launch(f_zm_rope, dim3((unsigned)((n + kThreads - 1) / kThreads)), a);
    sync();
}

void rms_norm(const float* x, int T, int C, const uint16_t* w, float eps, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&w, (void*)&eps, (void*)&y};
    launch(f_rms_norm, dim3((unsigned)T, 1, 1), a);
    sync();
}

void layer_norm(const float* x, int T, int C, const uint16_t* w, const uint16_t* b, float eps, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&w, (void*)&b, (void*)&eps, (void*)&y};
    launch(f_layer_norm, dim3((unsigned)T, 1, 1), a);
    sync();
}

void geglu(const float* pj, long rows, int ff, float* g) {
    const size_t n = (size_t)rows * ff;
    void* a[] = {(void*)&pj, (void*)&rows, (void*)&ff, (void*)&g};
    launch(f_geglu, dim3(blocks(n), 1, 1), a);
    sync();
}

void attention_f32(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out) {
    const int D = C / heads;
    const float scale = 1.0f / std::sqrt((float)D);
    if (Tq <= 0 || Tk <= 0) return;
    const int R = std::max(1, std::min(Tq, (int)(kScoreFloats / (size_t)Tk)));
    float* S = static_cast<float*>(alloc(sizeof(float) * (size_t)R * Tk));
    for (int hd = 0; hd < heads; ++hd)
        for (int r0 = 0; r0 < Tq; r0 += R) {
            const int Rc = std::min(R, Tq - r0);
            // scores = scale * Q K^T for the chunk's rows, softmaxed, then the weighted values into the head's columns
            gemm<float>(f_gemm_af, q + (size_t)r0 * C + hd * D, C, k + hd * D, 1, C, S, Tk, 1, nullptr, scale, 0.0f, Rc, Tk, D);
            int N = Tk;
            void* sm[] = {(void*)&S, (void*)&N};
            launch(f_softmax, dim3((unsigned)Rc, 1, 1), sm);
            gemm<float>(f_gemm_af, S, Tk, v + hd * D, C, 1, out + (size_t)r0 * C + hd * D, C, 1, nullptr, 1.0f, 0.0f, Rc, D, Tk);
        }
    sync();
    release(S, sizeof(float) * (size_t)R * Tk);
}

// A batch of BF16 WMMA GEMMs (sd_gemm_wmma): C[z] = alpha * A[z] B[z] for z < batches; the operands are offset by the batch
// strides bsA, bsB, bsC (elements); A has row stride lda (k contiguous), B has n stride sbn (k contiguous)
void wmma_batched(const uint16_t* A, long lda, long bsA, const float* B, long sbn, long bsB, float* C, long cs_m, long cs_n,
                  long bsC, float alpha, int M, int N, int K, int batches) {
    const uint16_t* a = A;
    const float* b = B;
    float* c = C;
    const float* bi = nullptr;
    const float beta = 0.0f;
    void* args[] = {(void*)&a, (void*)&lda, (void*)&b, (void*)&sbn, (void*)&c, (void*)&cs_m, (void*)&cs_n, (void*)&bi,
                    (void*)&alpha, (void*)&beta, (void*)&M, (void*)&N, (void*)&K, (void*)&bsA, (void*)&bsB, (void*)&bsC};
    const bool big = M >= 128;
    check(hipModuleLaunchKernel(big ? f_gemm_wmma128 : f_gemm_wmma, (unsigned)((N + (big ? 127 : 63)) / (big ? 128 : 64)),
                                (unsigned)((M + (big ? 127 : 63)) / (big ? 128 : 64)), (unsigned)batches, big ? 256 : 128, 1, 1, 0,
                                nullptr, args, nullptr),
          "wmma batched");
}

// The attention on the tensor cores (head width 128): the scores and the weighted values are BF16 WMMA batches over the heads
// (sd_gemm_wmma with the head as the batch), the softmax in F32 and written as BF16 with zero pads (the value's rows are padded
// to a multiple of 64 with the transposed values). Matches the reference's bf16 attention in precision.
void attention_tc(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out) {
    const int D = C / heads;
    const float scale = 1.0f / std::sqrt((float)D);
    const int Tkp = (Tk + 63) / 64 * 64;
    uint16_t* qb = static_cast<uint16_t*>(alloc(sizeof(uint16_t) * (size_t)Tq * C));
    float* vt = static_cast<float*>(alloc(sizeof(float) * (size_t)C * Tkp));
    float* sc = static_cast<float*>(alloc(sizeof(float) * (size_t)heads * Tq * Tk));
    uint16_t* pb = static_cast<uint16_t*>(alloc(sizeof(uint16_t) * (size_t)heads * Tq * Tkp));
    {
        long n = (long)Tq * C;
        void* a1[] = {(void*)&q, (void*)&qb, (void*)&n};
        launch(f_to_bf16, dim3((unsigned)((n + kThreads - 1) / kThreads)), a1);
    }
    {
        int tk = Tk, tkp = Tkp, c = C;
        const float* vp = v;
        float* vtp = vt;
        void* a2[] = {(void*)&vp, (void*)&tk, (void*)&tkp, (void*)&c, (void*)&vtp};
        launch(f_transpose_pad, dim3((unsigned)(((long)C * Tkp + kThreads - 1) / kThreads)), a2);
    }
    // scores: S[h][m][n] = scale * sum_d Q[m][h d] K[n][h d]
    wmma_batched(qb, C, D, k, C, D, sc, Tk, 1, (long)Tq * Tk, scale, Tq, Tk, D, heads);
    {
        int tq = Tq, tk = Tk, tkp = Tkp;
        const float* sp = sc;
        uint16_t* pp = pb;
        void* a3[] = {(void*)&sp, (void*)&pp, (void*)&tq, (void*)&tk, (void*)&tkp};
        const long rows = (long)heads * Tq;
        void* a3w[] = {(void*)&sp, (void*)&pp, (void*)&tq, (void*)&tk, (void*)&tkp, (void*)&rows};
        check(hipModuleLaunchKernel(f_softmax_bf16w, (unsigned)((rows + 7) / 8), 1, 1, 256, 1, 1, 0, nullptr, a3w, nullptr),
              "softmax_bf16_w");
        (void)a3;
    }
    // out[m][h d] = sum_n P[h][m][n] V[n][h d]
    wmma_batched(pb, Tkp, (long)Tq * Tkp, vt, Tkp, (long)D * Tkp, out, C, 1, D, 1.0f, Tq, D, Tkp, heads);
    check(hipStreamSynchronize(nullptr), "stream");
    release(qb, sizeof(uint16_t) * (size_t)Tq * C);
    release(vt, sizeof(float) * (size_t)C * Tkp);
    release(sc, sizeof(float) * (size_t)heads * Tq * Tk);
    release(pb, sizeof(uint16_t) * (size_t)heads * Tq * Tkp);
}

// The flash attention (sd_flash_attn): Q, K BF16 row-major, the values transposed and padded to a multiple of 64 keys (BF16)
void attention_flash(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out) {
    const int Tkp = (Tk + 63) / 64 * 64;
    // (written in full by the conversions and the transpose: no zeroing pass)
    uint16_t* qb = static_cast<uint16_t*>(alloc(sizeof(uint16_t) * (size_t)Tq * C, false));
    uint16_t* kb = static_cast<uint16_t*>(alloc(sizeof(uint16_t) * (size_t)Tk * C, false));
    uint16_t* vt = static_cast<uint16_t*>(alloc(sizeof(uint16_t) * (size_t)C * Tkp, false));
    {
        long n = (long)Tq * C;
        void* a1[] = {(void*)&q, (void*)&qb, (void*)&n};
        launch(f_to_bf16, dim3((unsigned)((n + kThreads - 1) / kThreads)), a1);
        n = (long)Tk * C;
        const float* kp = k;
        void* a2[] = {(void*)&kp, (void*)&kb, (void*)&n};
        launch(f_to_bf16, dim3((unsigned)((n + kThreads - 1) / kThreads)), a2);
    }
    {
        int tk = Tk, tkp = Tkp, c = C;
        const float* vp = v;
        uint16_t* vtp = vt;
        void* a3[] = {(void*)&vp, (void*)&tk, (void*)&tkp, (void*)&c, (void*)&vtp};
        launch(f_transpose_pad_bf16, dim3((unsigned)(((C + 31) / 32) * ((Tkp + 31) / 32))), a3);  // one block per 32 x 32 tile
    }
    {
        const float scale = 1.0f / std::sqrt((float)(C / heads));
        int tq = Tq, tk = Tk, tkp = Tkp, c = C;
        void* a4[] = {(void*)&qb, (void*)&kb, (void*)&vt, (void*)&out, (void*)&tq, (void*)&tk, (void*)&tkp, (void*)&c, (void*)&scale};
        check(hipModuleLaunchKernel(f_flash, (unsigned)((Tq + 127) / 128), (unsigned)heads, 1, 256, 1, 1, 0, nullptr, a4, nullptr),
              "flash attention");
    }
    check(hipStreamSynchronize(nullptr), "stream");
    release(qb, sizeof(uint16_t) * (size_t)Tq * C);
    release(kb, sizeof(uint16_t) * (size_t)Tk * C);
    release(vt, sizeof(uint16_t) * (size_t)C * Tkp);
}

void attention(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out) {
    if (Tq <= 0 || Tk <= 0) return;
    if (C / heads == 128) attention_flash(q, Tq, k, v, Tk, C, heads, out);
    else attention_f32(q, Tq, k, v, Tk, C, heads, out);
}

}  // namespace sddev
