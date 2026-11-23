// The image engine's device path (sd_device.h): the launches of sd_gpu.hip's kernels on the null stream, each op synchronous
// (the host reads the activations after it). The GEMM covers the convolutions (weights x im2col), the linear layers and the
// attention; the attention's score rows are chunked so the scores of a chunk stay within 16 MB.
#include "sd_device.h"

#include "kernel_loader.h"
#include "blt_calib.h"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
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
hipFunction_t f_audio_gemm = nullptr, f_audio_ln = nullptr, f_audio_attn = nullptr, f_audio_attn_g = nullptr, f_audio_gelu_erf = nullptr;
hipFunction_t f_gemm_ah = nullptr, f_gemm_af = nullptr, f_im2col = nullptr, f_gn_sum = nullptr, f_gn_sq = nullptr,
              f_gn_apply = nullptr, f_silu = nullptr, f_add = nullptr, f_upsample2 = nullptr, f_softmax = nullptr,
              f_layer_norm = nullptr, f_rms_norm = nullptr, f_zm_modulate = nullptr, f_zm_gate_add = nullptr, f_zm_swiglu = nullptr, f_zm_rope = nullptr, f_zm_ln_plain = nullptr, f_gn_f32 = nullptr, f_to_tok = nullptr, f_to_chw = nullptr, f_zq_rope = nullptr, f_zq_attn = nullptr, f_geglu = nullptr, f_gemm_bf = nullptr, f_gemm_wmma = nullptr, f_gemm_wmma128 = nullptr, f_softmax_bf16w = nullptr, f_flash = nullptr, f_gemm_wmma256 = nullptr, f_gemm_wmma256b = nullptr, f_gemm_wmma128bb = nullptr, f_transpose_pad_bf16 = nullptr,
              f_to_bf16 = nullptr, f_transpose_pad = nullptr, f_softmax_bf16 = nullptr, f_qi_linear_f8 = nullptr, f_qi_rms_f8 = nullptr, f_qi_affine = nullptr, f_qi_gelu = nullptr, f_qv_rms = nullptr, f_qv_patch = nullptr, f_qv_bias = nullptr, f_qv_upsample = nullptr, f_qi_hadamard = nullptr, f_qi_linear_i8 = nullptr, f_qi21_zc = nullptr, f_qi21_gather = nullptr, f_qi21_mul = nullptr, f_qi21_gated = nullptr, f_qi21_attn = nullptr, f_qi21_lbf = nullptr, f_qi21_dupup = nullptr, f_gemm_af128 = nullptr, f_transpose_bf16 = nullptr, f_bias_cm = nullptr, f_zm_rms_mod_bf16 = nullptr, f_zm_swiglu_bf16 = nullptr, f_bf16_to_f32 = nullptr, f_round_bf16 = nullptr, f_gn_sum_part = nullptr, f_gn_var_part = nullptr, f_gn_apply_part = nullptr, f_im2col_bf16_tile = nullptr, f_hnr16 = nullptr, f_count_diff = nullptr, f_tp16 = nullptr, f_rga16 = nullptr, f_sw16 = nullptr;

constexpr int kThreads = 256;
constexpr int kGroupSplit = 16;  // the group norm's partial sums per group
constexpr size_t kColFloats = (size_t)1 << 25;  // the im2col chunk: at most 128 MB (a launch covers enough tiles to fill the GPU)
constexpr size_t kScoreFloats = (size_t)1 << 22;  // the attention's score chunk: at most 16 MB

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("sd device: ") + what + ": " + hipGetErrorString(e));
}

// SD_TIME=1: every launch is timed (events around it, synchronous: the results do not change) and the totals per kernel are
// printed at exit; SD_TIME_MS=n also prints each launch that takes longer than n ms (default 100)
struct TimeStat {
    long count = 0;
    double total = 0, max = 0;
};
bool g_time_on = false;
double g_time_slow_ms = 100;
hipEvent_t g_ev0 = nullptr, g_ev1 = nullptr;
std::mutex g_time_mu;
std::map<std::string, TimeStat> g_time_stats;
std::vector<std::pair<hipFunction_t, const char*>> g_names;

const char* kernel_name(hipFunction_t f) {
    for (const auto& p : g_names)
        if (p.first == f) return p.second;
    return "?";
}

void print_timing() {
    std::lock_guard<std::mutex> lk(g_time_mu);
    std::fprintf(stderr, "sd timing (ms): kernel, launches, total, max\n");
    for (const auto& kv : g_time_stats)
        std::fprintf(stderr, "  %-24s %8ld %12.2f %10.2f\n", kv.first.c_str(), kv.second.count, kv.second.total, kv.second.max);
}

// hipModuleLaunchKernel, timed when SD_TIME is set
hipError_t hip_launch(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned sh,
                      hipStream_t stream, void** args, void** extra) {
    if (!g_time_on) return hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, stream, args, extra);
    check(hipEventRecord(g_ev0, stream), "event");
    const hipError_t e = hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, sh, stream, args, extra);
    if (e != hipSuccess) return e;
    check(hipEventRecord(g_ev1, stream), "event");
    check(hipEventSynchronize(g_ev1), "event");
    float ms = 0;
    check(hipEventElapsedTime(&ms, g_ev0, g_ev1), "event");
    const char* name = kernel_name(f);
    {
        std::lock_guard<std::mutex> lk(g_time_mu);
        TimeStat& s = g_time_stats[name];
        s.count += 1;
        s.total += ms;
        s.max = std::max(s.max, (double)ms);
    }
    if (ms > g_time_slow_ms)
        std::fprintf(stderr, "sd timing: %s grid %u x %u x %u block %u took %.1f ms\n", name, gx, gy, gz, bx, ms);
    return hipSuccess;
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
        {&f_gemm_ah, "sd_gemm_ah"},     {&f_gemm_af, "sd_gemm_af"}, {&f_gemm_af128, "sd_gemm_af128"}, {&f_transpose_bf16, "sd_transpose_bf16"}, {&f_bias_cm, "sd_bias_cm"}, {&f_zm_rms_mod_bf16, "zimage_rms_mod_bf16"}, {&f_zm_swiglu_bf16, "zimage_swiglu_bf16"}, {&f_bf16_to_f32, "sd_bf16_to_f32"}, {&f_round_bf16, "sd_round_bf16"}, {&f_gn_sum_part, "zimage_gn_sum"}, {&f_gn_var_part, "zimage_gn_var"}, {&f_gn_apply_part, "zimage_gn_apply"}, {&f_im2col_bf16_tile, "sd_im2col_bf16_tile"}, {&f_hnr16, "zimage_head_norm_rope_bf16"}, {&f_count_diff, "sd_count_diff"}, {&f_tp16, "zimage_transpose_pad_bf16_in"}, {&f_rga16, "zimage_rms_gate_add_bf16"}, {&f_sw16, "zimage_swiglu_bf16_in"},     {&f_im2col, "sd_im2col"}, {&f_gemm_bf, "sd_gemm_bf"}, {&f_gemm_wmma, "sd_gemm_wmma"}, {&f_gemm_wmma128, "sd_gemm_wmma128"}, {&f_gemm_wmma128bb, "sd_gemm_wmma128bb"}, {&f_gemm_wmma256, "sd_gemm_wmma256"}, {&f_gemm_wmma256b, "sd_gemm_wmma256b"}, {&f_flash, "sd_flash_attn"}, {&f_transpose_pad_bf16, "sd_transpose_pad_bf16"}, {&f_to_bf16, "sd_to_bf16"}, {&f_qi_linear_f8, "qi_linear_f8"}, {&f_qi_rms_f8, "qi_rms_norm_f8"}, {&f_qi_affine, "qi_affine"}, {&f_qi_gelu, "qi_gelu_tanh"}, {&f_qv_rms, "qv_rms_tok"}, {&f_qv_patch, "qv_patch"}, {&f_qv_bias, "qv_bias_tok"}, {&f_qv_upsample, "qv_upsample2_tok"}, {&f_qi_hadamard, "qi_hadamard256"}, {&f_qi_linear_i8, "qi_linear_i8"}, {&f_qi21_zc, "qi21_zc_rms"}, {&f_qi21_gather, "qi21_gather_mod"}, {&f_qi21_mul, "qi21_mul_1p"}, {&f_qi21_gated, "qi21_gated_add"}, {&f_qi21_attn, "qi21_attn"}, {&f_qi21_lbf, "qi21_linear_bf16"}, {&f_qi21_dupup, "qi21_dupup"},
        {&f_audio_gemm, "audio_gemm_f32"}, {&f_audio_ln, "audio_ln_affine_f32"}, {&f_audio_attn, "audio_causal_attn_f32"},
        {&f_audio_attn_g, "audio_attn_f32"}, {&f_audio_gelu_erf, "audio_gelu_erf_f32"},
        {&f_transpose_pad, "sd_transpose_pad"}, {&f_softmax_bf16, "sd_softmax_bf16"}, {&f_softmax_bf16w, "sd_softmax_bf16_w"},
        {&f_gn_sum, "sd_gn_sum"},       {&f_gn_sq, "sd_gn_sq"},         {&f_gn_apply, "sd_gn_apply"},
        {&f_silu, "sd_silu"},           {&f_add, "sd_add"},             {&f_upsample2, "sd_upsample2"},
        {&f_softmax, "sd_softmax"},     {&f_layer_norm, "sd_layer_norm"}, {&f_rms_norm, "sd_rms_norm"}, {&f_zm_modulate, "zimage_modulate"}, {&f_zm_gate_add, "zimage_gate_add"}, {&f_zm_swiglu, "zimage_swiglu"}, {&f_zm_rope, "zimage_rope"}, {&f_zm_ln_plain, "zimage_layer_norm_plain"}, {&f_gn_f32, "zimage_group_norm_f32"}, {&f_to_tok, "zimage_to_tokens"}, {&f_to_chw, "zimage_to_chw"}, {&f_zq_rope, "zq_rope_half"}, {&f_zq_attn, "zq_causal_attn"}, {&f_geglu, "sd_geglu"},
    };
    for (const Entry& e : entries)
        if (hipModuleGetFunction(e.f, g_mod, e.name) != hipSuccess) return false;
    for (const Entry& e : entries) g_names.push_back({*e.f, e.name});
    if (const char* t = std::getenv("SD_TIME"); t && std::atoi(t) != 0) {
        g_time_on = true;
        if (const char* m = std::getenv("SD_TIME_MS")) g_time_slow_ms = std::atof(m);
        check(hipEventCreate(&g_ev0), "event");
        check(hipEventCreate(&g_ev1), "event");
        std::atexit(print_timing);
    }
    g_state = 1;
    return true;
}

void launch(hipFunction_t f, dim3 grid, void** args) {
    check(hip_launch(f, grid.x, grid.y, grid.z, kThreads, 1, 1, 0, nullptr, args, nullptr), "launch");
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

namespace {
std::mutex g_dpool_mu;
std::vector<Block> g_dfree;  // the released device blocks, reused by dalloc (best fit)
}  // namespace

void* dalloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    const size_t need = round_up_page(bytes);
    if (!enabled()) return std::malloc(need);
    {
        std::lock_guard<std::mutex> lk(g_dpool_mu);
        size_t best = (size_t)-1;
        for (size_t i = 0; i < g_dfree.size(); ++i)
            if (g_dfree[i].cap >= need && (best == (size_t)-1 || g_dfree[i].cap < g_dfree[best].cap)) best = i;
        if (best != (size_t)-1) {
            void* p = g_dfree[best].p;
            g_dfree.erase(g_dfree.begin() + (long)best);
            return p;
        }
    }
    void* p = nullptr;
    if (hipMalloc(&p, need) != hipSuccess) throw std::runtime_error("sd device: out of device memory");
    return p;
}

void drelease(void* p, size_t bytes) {
    if (!p) return;
    if (!enabled()) {
        std::free(p);
        return;
    }
    std::lock_guard<std::mutex> lk(g_dpool_mu);
    g_dfree.push_back({p, round_up_page(bytes)});
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

// the convolutions' FP32 GEMM with 128 x 128 tiles (sd_gemm_af128), the same arguments as gemm<float>
void gemm_conv_f32(const float* A, long lda, const float* B, long sbk, long sbn, float* C, long cs_m, long cs_n, float alpha,
                   float beta, int M, int N, int K, const float* biasf) {
    if (M <= 0 || N <= 0) return;
    const float* a = A;
    const float* b = B;
    float* c = C;
    const uint16_t* bi = nullptr;
    const float* bf = biasf;
    void* args[] = {(void*)&a, (void*)&lda, (void*)&b, (void*)&sbk, (void*)&sbn, (void*)&c, (void*)&cs_m, (void*)&cs_n,
                    (void*)&bi, (void*)&bf, (void*)&alpha, (void*)&beta, (void*)&M, (void*)&N, (void*)&K};
    launch(f_gemm_af128, dim3((unsigned)((N + 127) / 128), (unsigned)((M + 127) / 128), 1), args);
}

// The convolution on the tensor cores (SD_VAE_BF16=1): the weights and the im2col activations rounded to BF16 (the products
// and the sums stay FP32 in the WMMA), the output FP32 and channel-major. The GEMM's rows are the output pixels (the tile
// stores 8 consecutive rows at once, so the output's stride along the rows is 1); the bias is added in a pass after it.
void conv2d_bf16(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y,
                 int Ho, int Wo) {
    const long HWo = (long)Ho * Wo;
    const int K = cin * k * k;
    const long nc_max = std::max<long>(1, std::min<long>(HWo, (long)(kColFloats / (size_t)K)));
    uint16_t* wb = static_cast<uint16_t*>(alloc(sizeof(uint16_t) * (size_t)cout * K, false));
    {
        const long nw = (long)cout * K;
        const float* wf = w;
        void* a[] = {(void*)&wf, (void*)&wb, (void*)&nw};
        launch(f_to_bf16, dim3(blocks((size_t)nw)), a);
    }
    float* colf = static_cast<float*>(alloc(sizeof(float) * (size_t)K * nc_max, false));
    uint16_t* col = static_cast<uint16_t*>(alloc(sizeof(uint16_t) * (size_t)K * nc_max, false));
    const long bs0 = 0;
    for (long n0l = 0; n0l < HWo; n0l += nc_max) {
        const int n0 = (int)n0l;
        const int nc = (int)std::min<long>(nc_max, HWo - n0l);
        {
            void* a1[] = {(void*)&x, (void*)&cin, (void*)&H, (void*)&W, (void*)&k, (void*)&p, (void*)&st, (void*)&Wo, (void*)&n0,
                          (void*)&nc, (void*)&colf};
            launch(f_im2col, dim3((unsigned)((nc + kThreads - 1) / kThreads), (unsigned)K, 1), a1);
            const int Kc = K, ncc = nc;
            void* a2[] = {(void*)&colf, (void*)&Kc, (void*)&ncc, (void*)&col};
            check(hip_launch(f_transpose_bf16, (unsigned)((nc + 31) / 32), (unsigned)((K + 31) / 32), 1, 32, 8, 1, 0, nullptr,
                             a2, nullptr),
                  "transpose bf16");
        }
        // C[pixel][channel] = sum_k col[pixel][k] wb[channel][k]: M = the chunk's pixels, N = the output channels
        const uint16_t* A = col;
        const uint16_t* B = wb;
        float* C = y + n0l;
        const long lda = K, sbn = K, cs_m = 1, cs_n = HWo;
        const float alpha = 1.0f, beta = 0.0f;
        const int M = nc, N = cout, KK = K;
        const float* bf = nullptr;
        void* args[] = {(void*)&A, (void*)&lda, (void*)&B, (void*)&sbn, (void*)&C, (void*)&cs_m, (void*)&cs_n,
                        (void*)&bf, (void*)&alpha, (void*)&beta, (void*)&M, (void*)&N, (void*)&KK, (void*)&bs0, (void*)&bs0,
                        (void*)&bs0};
        check(hip_launch(f_gemm_wmma256b, (unsigned)((N + 127) / 128), (unsigned)((M + 255) / 256), 1, 256, 1, 1, 0, nullptr,
                         args, nullptr),
              "wmma conv");
        {
            const long ne = (long)cout * nc;
            const int ncc = nc, cc = cout;
            const float* bb = b;
            float* yy = y + n0l;
            void* a2[] = {(void*)&yy, (void*)&HWo, (void*)&ncc, (void*)&cc, (void*)&bb};
            launch(f_bias_cm, dim3(blocks((size_t)ne)), a2);
        }
    }
    sync();
    release(colf, sizeof(float) * (size_t)K * nc_max);
    release(col, sizeof(uint16_t) * (size_t)K * nc_max);
    release(wb, sizeof(uint16_t) * (size_t)cout * K);
}

namespace blt {
bool gemm_f32(int M, int N, int K, const float* A, long lda, const float* B, long sbk, long sbn, float* C, long cs_m, float alpha);
bool gemm_bf16(int M, int N, int K, const uint16_t* A, long lda, const uint16_t* B, long sbk, long sbn, float* C, long cs_m, float alpha);
}

/* The convolution in BF16 by hipBLASLt (the default of the VAE's convolutions; SD_VAE_CONV=fp32 keeps the FP32 kernel): the
 * im2col written as BF16 token-major in one pass, the weights as BF16, one product per convolution (no
 * chunks: one plan per shape) with the FP32 output channel-major (C[c][p] at c HW + p: the product's rows are the pixels, the
 * output's row stride is 1), the bias as a pass. Returns false when the device path does not take it. */
bool conv2d_blt_bf16(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y,
                     int Ho, int Wo) {
    const long HWo = (long)Ho * Wo;
    const int K = cin * k * k;
    if (HWo <= 0 || K <= 0 || !f_im2col_bf16_tile || !f_bias_cm || !f_to_bf16) return false;
    /* the tile kernel: stride 1, k <= 3, whole 64-pixel row segments, 8-channel blocks */
    if (st != 1 || k > 3 || k < 1 || Wo % 64 != 0 || cin % 8 != 0 || Ho != H) return false;
    uint16_t* wb = static_cast<uint16_t*>(dalloc(sizeof(uint16_t) * (size_t)cout * K));
    uint16_t* col = static_cast<uint16_t*>(dalloc(sizeof(uint16_t) * (size_t)HWo * K));
    {
        const long nw = (long)cout * K;
        const float* wf = w;
        void* a[] = {(void*)&wf, (void*)&wb, (void*)&nw};
        launch(f_to_bf16, dim3(blocks((size_t)nw)), a);
    }
    {
        const float* xs = x;
        const int Kc = K;
        void* a[] = {(void*)&xs, (void*)&cin, (void*)&H, (void*)&W, (void*)&k, (void*)&p, (void*)&Wo, (void*)&Kc, (void*)&col};
        launch(f_im2col_bf16_tile, dim3((unsigned)(Ho * (Wo / 64)), (unsigned)(cin / 8), 1), a);
    }
    const bool ok = blt::gemm_bf16(cout, (int)HWo, K, wb, K, col, 1, K, y, HWo, 1.0f);
    if (ok) {
        const long ne = (long)cout * HWo;
        const int ncc = (int)HWo, cc = cout;
        const long hw = HWo;
        const float* bb = b;
        void* a[] = {(void*)&y, (void*)&hw, (void*)&ncc, (void*)&cc, (void*)&bb};
        launch(f_bias_cm, dim3(blocks((size_t)ne)), a);
    }
    sync();
    drelease(col, sizeof(uint16_t) * (size_t)HWo * K);
    drelease(wb, sizeof(uint16_t) * (size_t)cout * K);
    return ok;
}

/* SD_CONV_KERNEL=blt: the FP32 convolutions by hipBLASLt (measured slower than the 128 x 128 kernel on the decoder: 8.65 s
 * against 8.11 s a step); the default is the kernel */
bool blt_conv_on() {
    static const bool on = [] {
        const char* e = std::getenv("SD_CONV_KERNEL");
        return e && std::strcmp(e, "blt") == 0;
    }();
    return on;
}

void conv2d_f32(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y,
                int Ho, int Wo) {
    const long HWo = (long)Ho * Wo;
    const int K = cin * k * k;
    if (HWo <= 0 || K <= 0) return;
    static const bool conv_fp32 = [] {
        const char* e = std::getenv("SD_VAE_CONV");
        return e && std::strcmp(e, "fp32") == 0;
    }();
    if (!conv_fp32 && (K % 8) == 0 && conv2d_blt_bf16(x, cin, H, W, w, cout, b, k, p, st, y, Ho, Wo)) return;
    static const bool bf16_conv = [] {
        const char* e = std::getenv("SD_VAE_BF16");
        return e && std::atoi(e) != 0;
    }();
    if (bf16_conv && cout >= 128 && K % 64 == 0 && f_im2col && f_transpose_bf16 && f_bias_cm && f_gemm_wmma256b) {
        conv2d_bf16(x, cin, H, W, w, cout, b, k, p, st, y, Ho, Wo);
        return;
    }
    const long nc_max = std::max<long>(1, std::min<long>(HWo, (long)(kColFloats / (size_t)K)));
    float* col = static_cast<float*>(alloc(sizeof(float) * (size_t)K * nc_max));
    for (long n0l = 0; n0l < HWo; n0l += nc_max) {
        const int n0 = (int)n0l;
        const int nc = (int)std::min<long>(nc_max, HWo - n0l);
        void* a1[] = {(void*)&x, (void*)&cin, (void*)&H, (void*)&W, (void*)&k, (void*)&p, (void*)&st, (void*)&Wo, (void*)&n0, (void*)&nc, (void*)&col};
        const dim3 g((unsigned)((nc + kThreads - 1) / kThreads), (unsigned)K, 1);
        launch(f_im2col, g, a1);
        /* SD_CONV_KERNEL=blt: hipBLASLt in FP32 (the weights times the columns, then the bias as a pass) */
        if (blt_conv_on() && blt::gemm_f32(cout, nc, K, w, K, col, nc, 1, y + n0l, HWo, 1.0f)) {
            const long ne = (long)cout * nc;
            const int ncc = nc, cc = cout;
            const float* bb = b;
            float* yy = y + n0l;
            void* a2[] = {(void*)&yy, (void*)&HWo, (void*)&ncc, (void*)&cc, (void*)&bb};
            launch(f_bias_cm, dim3(blocks((size_t)ne)), a2);
        } else {
            gemm_conv_f32(w, K, col, nc, 1, y + n0l, HWo, 1, 1.0f, 0.0f, cout, nc, K, b);
        }
    }
    sync();
    release(col, sizeof(float) * (size_t)K * nc_max);
}

void linear_tok(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y) {
    gemm<uint16_t>(f_gemm_ah, W, in, x, 1, in, y, 1, out, b, 1.0f, 0.0f, out, T, in);
    sync();
}

bool linear_huge_xb(const uint16_t* xb, int T, const uint16_t* W, int in, int out, const float* b, void* y, bool out16);


void linear_bf16_xb(const uint16_t* xb, int T, const uint16_t* W, int in, int out, const float* b, float* y) {
    if (out <= 0 || T <= 0) return;
    if (in % 64 == 0 && out >= 256 && T >= 1024 && f_gemm_wmma256b) {
        linear_huge_xb(xb, T, W, in, out, b, y, false);
        return;
    }
    // the other shapes take FP32 activations: the BF16 values widened (exact)
    const size_t nb = (size_t)T * in;
    float* xf = static_cast<float*>(dalloc(sizeof(float) * nb));
    long n = (long)nb;
    const uint16_t* xbp = xb;
    void* a[] = {(void*)&xbp, (void*)&xf, (void*)&n};
    launch(f_bf16_to_f32, dim3(blocks(nb)), a);
    linear_bf16(xf, T, W, in, out, b, y);
    drelease(xf, sizeof(float) * nb);
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
    check(hip_launch(f_qi_linear_f8, (unsigned)((out + 63) / 64), (unsigned)((T + 63) / 64), 1, 256, 1, 1, 0, nullptr, a,
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
    check(hip_launch(f_qi_hadamard, (unsigned)(in / 256), (unsigned)T, 1, 256, 1, 1, 0, nullptr, ha, nullptr), "qi_hadamard256");
    void* la[] = {(void*)&xr, (void*)&T, (void*)&W, (void*)&wscale, (void*)&in, (void*)&out, (void*)&y};
    check(hip_launch(f_qi_linear_i8, (unsigned)((out + 63) / 64), (unsigned)((T + 63) / 64), 1, 256, 1, 1, 0, nullptr, la,
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
    check(hip_launch(f_qi21_lbf, (unsigned)((out + 63) / 64), (unsigned)((T + 63) / 64), 1, 256, 1, 1, 0, nullptr, a, nullptr),
          "qi21_linear_bf16");
    sync();
}

// Qwen-Image 2.1's residual up block shortcut (qi21_dupup): token-major [H][W][in] -> [2H][2W][out] (factor_t 2 or 1: the time slot kept)
void dupup(const float* x, int H, int W, int in, int out, int factor_t, float* y) {
    long n = 4L * H * W * out;
    void* a[] = {(void*)&x, (void*)&H, (void*)&W, (void*)&in, (void*)&out, (void*)&factor_t, (void*)&y};
    launch(f_qi21_dupup, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void attn21_text(const float* q, const float* k, const float* v, int J, int L, int heads, float* out) {
    long n = (long)L * heads;
    void* a[] = {(void*)&q, (void*)&k, (void*)&v, (void*)&J, (void*)&L, (void*)&heads, (void*)&out};
    launch(f_qi21_attn, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

/* Device-only scratch: hipMalloc memory for a buffer the host never reads or writes (the BF16 copy of a large linear's input).
 * The GEMMs read it faster than the pinned allocation (hipBLASLt: 27 against 21 TFLOPS on the DiT's QKV, sd_gemm_bench).
 * One buffer, grown when a call needs more; the device path's calls are serialised. */
void* device_scratch(size_t bytes) {
    static void* buf = nullptr;
    static size_t cap = 0;
    if (bytes > cap) {
        if (buf) hipFree(buf);
        buf = nullptr;
        cap = 0;
        if (hipMalloc(&buf, bytes) != hipSuccess) throw std::runtime_error("sd device: out of device memory");
        cap = bytes;
    }
    return buf;
}

/* hipBLASLt for the large linears (the DiT's: out >= 256 and T >= 1024): y[t][o] = sum_i xb[t][i] W[o][i] + b[o], the BF16
 * operands as the WMMA kernels take them, FP32 output and accumulation, the bias as the GEMM's epilogue. Column-major, as
 * the library takes it: C (out x T) = W^T (out x in: the weights as stored) times xb^T (in x T). One plan per shape (and
 * whether there is a bias); its algorithm is chosen once, by timing the heuristic's candidates on that shape: the first
 * candidate ran at a quarter of the best one's speed on the DiT's shapes (PLAN.md). SD_LINEAR_KERNEL=256b keeps the
 * kernel. A plan that fails leaves the kernel in charge of that shape. */
namespace blt {

struct Plan {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    size_t ws = 0;
    bool tuned = false, failed = false;
};

hipblasLtHandle_t g_handle = nullptr;
bool g_init_failed = false;
std::map<std::array<int, 5>, Plan> g_plans;
void* g_ws = nullptr;
size_t g_ws_cap = 0;

bool ensure_workspace(size_t bytes) {
    if (bytes <= g_ws_cap) return true;
    if (g_ws) hipFree(g_ws);
    g_ws = nullptr;
    g_ws_cap = 0;
    if (hipMalloc(&g_ws, bytes) != hipSuccess) return false;  // device-only: the host never touches it
    g_ws_cap = bytes;
    return true;
}

/* the number of differing 32-bit words of two device buffers (bytes a multiple of 4): 0 when the two algorithms gave the same bits */
unsigned diff_words(const void* a, const void* b, size_t bytes) {
    unsigned* cnt = static_cast<unsigned*>(dalloc(sizeof(unsigned)));
    unsigned zero = 0;
    hipMemcpy(cnt, &zero, sizeof(unsigned), hipMemcpyHostToDevice);
    const long n = (long)(bytes / 4);
    const void* ap = a;
    const void* bp = b;
    void* args[] = {(void*)&ap, (void*)&bp, (void*)&n, (void*)&cnt};
    launch(f_count_diff, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), args);
    unsigned host = 0;
    hipMemcpy(&host, cnt, sizeof(unsigned), hipMemcpyDeviceToHost);
    drelease(cnt, sizeof(unsigned));
    return host;
}

bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("SD_LINEAR_KERNEL");
        return !(e && std::strcmp(e, "256b") == 0);
    }();
    return on && !g_init_failed;
}

// one product with the algorithm given (its workspace: p.ws bytes of g_ws)
bool run_once(const Plan& p, const hipblasLtMatmulAlgo_t& algo, const uint16_t* W, const uint16_t* xb, void* y,
              const void* bias_ptr, bool has_bias) {
    if (has_bias) hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr));
    const float alpha = 1.0f, beta = 0.0f;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = hipblasLtMatmul(g_handle, p.desc, &alpha, W, p.la, xb, p.lb, &beta, y, p.lc, y, p.lc, &algo, g_ws, p.ws, nullptr) ==
                    HIPBLAS_STATUS_SUCCESS;
    if (g_time_on) {  /* SD_TIME: the products' wall time (synchronous), in the same table as the kernels' */
        hipStreamSynchronize(nullptr);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::lock_guard<std::mutex> lk(g_time_mu);
        TimeStat& st = g_time_stats["hipblaslt_gemm"];
        st.count += 1;
        st.total += ms;
        st.max = std::max(st.max, ms);
    }
    return ok;
}

// the plan for (out, T, in, bias): created and tuned on its first call (with that call's operands)
Plan* plan_for(int out, int T, int in, bool has_bias, bool out16, const uint16_t* W, const uint16_t* xb, void* y, const float* b) {
    if (!g_handle && hipblasLtCreate(&g_handle) != HIPBLAS_STATUS_SUCCESS) {
        g_init_failed = true;
        g_handle = nullptr;
        return nullptr;
    }
    const std::array<int, 5> key = {out, T, in, has_bias ? 1 : 0, out16 ? 1 : 0};
    auto it = g_plans.find(key);
    if (it == g_plans.end()) {
        Plan p;
        bool ok = hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS;
        const hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        ok = ok && hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)) == HIPBLAS_STATUS_SUCCESS;
        ok = ok && hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)) == HIPBLAS_STATUS_SUCCESS;
        if (has_bias) {
            const hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_BIAS;
            const hipDataType bt = HIP_R_32F;
            ok = ok && hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)) == HIPBLAS_STATUS_SUCCESS;
            ok = ok && hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)) == HIPBLAS_STATUS_SUCCESS;
        }
        ok = ok && hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16BF, in, out, in) == HIPBLAS_STATUS_SUCCESS;   // W^T: in x out
        ok = ok && hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16BF, in, T, in) == HIPBLAS_STATUS_SUCCESS;     // xb^T: in x T
        ok = ok && hipblasLtMatrixLayoutCreate(&p.lc, out16 ? HIP_R_16BF : HIP_R_32F, out, T, out) == HIPBLAS_STATUS_SUCCESS;  // y: out x T
        ok = ok && hipblasLtMatmulPreferenceCreate(&p.pref) == HIPBLAS_STATUS_SUCCESS;
        const uint64_t maxws = (uint64_t)256 << 20;
        ok = ok && hipblasLtMatmulPreferenceSetAttribute(p.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxws, sizeof(maxws)) ==
                       HIPBLAS_STATUS_SUCCESS;
        p.failed = !ok;
        it = g_plans.emplace(key, p).first;
    }
    Plan& p = it->second;
    if (p.failed || p.tuned) return p.failed ? nullptr : &p;
    /* the heuristic's candidates, each timed on this call's operands (the results overwrite y, which the call overwrites anyway) */
    hipblasLtMatmulHeuristicResult_t heur[32];
    int n = 0;
    if (hipblasLtMatmulAlgoGetHeuristic(g_handle, p.desc, p.la, p.lb, p.lc, p.lc, p.pref, 32, heur, &n) != HIPBLAS_STATUS_SUCCESS || n < 1) {
        p.failed = true;
        return nullptr;
    }
    size_t maxws = 0;
    for (int i = 0; i < n; ++i) maxws = std::max(maxws, (size_t)heur[i].workspaceSize);
    if (!ensure_workspace(std::max<size_t>(maxws, 1))) {
        p.failed = true;
        return nullptr;
    }
    const void* bias_ptr = b;
    /* the candidates' outputs: the first candidate's is the reference, and a candidate whose bits differ is never chosen (the
     * algorithms differ in their numerics; the choice must not depend on the timing) */
    const size_t nbytes = (size_t)out * T * (out16 ? 2 : 4);
    void* ref = dalloc(nbytes);
    bool have_ref = false;
    double best = 1e30;
    int best_i = -1;
    for (int i = 0; i < n; ++i) {
        p.ws = heur[i].workspaceSize;
        /* one warm-up run (the library loads the algorithm's kernels then), then one timed run; a candidate whose warm-up is
         * already well behind the best is not timed again (the tuning is a one-time cost of the first call) */
        const auto w0 = std::chrono::steady_clock::now();
        if (!run_once(p, heur[i].algo, W, xb, y, bias_ptr, has_bias)) continue;
        hipDeviceSynchronize();
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        if (!have_ref) {
            hipMemcpy(ref, y, nbytes, hipMemcpyDeviceToDevice);
            have_ref = true;
        } else if (diff_words(y, ref, nbytes) != 0) {
            if (std::getenv("SD_BLT_LOG"))
                std::fprintf(stderr, "blt: out %d T %d in %d bias %d out16 %d candidate %d: other numerics, not chosen\n", out, T, in,
                             has_bias ? 1 : 0, out16 ? 1 : 0, i);
            continue;
        }
        if (best < 1e30 && t > 2.5 * best) {
            if (std::getenv("SD_BLT_LOG"))
                std::fprintf(stderr, "blt: out %d T %d in %d bias %d candidate %d: %.2f ms (warm-up only)\n", out, T, in,
                             has_bias ? 1 : 0, i, t * 1e3);
            continue;
        }
        {
            const auto t0 = std::chrono::steady_clock::now();
            run_once(p, heur[i].algo, W, xb, y, bias_ptr, has_bias);
            hipDeviceSynchronize();
            t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
        if (std::getenv("SD_BLT_LOG"))
            std::fprintf(stderr, "blt: out %d T %d in %d bias %d out16 %d candidate %d: %.2f ms (workspace %zu)\n", out, T, in,
                         has_bias ? 1 : 0, out16 ? 1 : 0, i, t * 1e3, (size_t)heur[i].workspaceSize);
        if (t < best) {
            best = t;
            best_i = i;
        }
    }
    drelease(ref, nbytes);
    if (best_i < 0) {
        p.failed = true;
        return nullptr;
    }
    p.algo = heur[best_i].algo;
    p.ws = heur[best_i].workspaceSize;
    p.tuned = true;
    return &p;
}

}  // namespace blt

namespace blt {
/* the FP32 product C[m][n] = alpha sum_k A[m][k] B(k, n) with the strides of gemm<float> (A rows contiguous in k: lda; B(k, n)
 * at k sbk + n sbn; C at m cs_m + n, cs_n == 1), by hipBLASLt in FP32 (its algorithm tuned once per shape, as the BF16 plans).
 * The attention's two products (sd_device attention_f32) take it. */
struct PlanF32 {
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    size_t ws = 0;
    bool tuned = false, failed = false;
};
std::map<std::array<long, 8>, PlanF32> g_f32_plans;

bool gemm_impl(bool bf16, int M, int N, int K, const void* A, long lda, const void* B, long sbk, long sbn, float* C, long cs_m, float alpha) {
    if (M <= 0 || N <= 0 || K <= 0) return true;
    if (!g_handle && hipblasLtCreate(&g_handle) != HIPBLAS_STATUS_SUCCESS) {
        g_init_failed = true;
        g_handle = nullptr;
        return false;
    }
    /* B as the library reads it (column-major): B(k, n) at k + n ldb (op N: the stored matrix is n x k, ld = sbk; the
     * transposed one: stored k x n with ld = sbn, when B's k is the contiguous one) */
    const hipDataType dt = bf16 ? HIP_R_16BF : HIP_R_32F;
    const bool opN = (sbn == 1);
    const long ldb = opN ? sbk : sbn;
    const std::array<long, 8> key = {M, N, K, lda, ldb, opN ? 0L : 1L, cs_m, bf16 ? 1L : 0L};
    auto it = g_f32_plans.find(key);
    if (it == g_f32_plans.end()) {
        PlanF32 p;
        bool ok = hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS;
        const hipblasOperation_t opA = opN ? HIPBLAS_OP_N : HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
        ok = ok && hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)) == HIPBLAS_STATUS_SUCCESS;
        ok = ok && hipblasLtMatmulDescSetAttribute(p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)) == HIPBLAS_STATUS_SUCCESS;
        /* the stored B: n x k (opN, ld = sbk) or k x n (ld = sbn); A: k x m (ld = lda); C: n x m (ld = cs_m) */
        ok = ok && hipblasLtMatrixLayoutCreate(&p.la, dt, opN ? N : K, opN ? K : N, ldb) == HIPBLAS_STATUS_SUCCESS;
        ok = ok && hipblasLtMatrixLayoutCreate(&p.lb, dt, K, M, lda) == HIPBLAS_STATUS_SUCCESS;
        ok = ok && hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_32F, N, M, cs_m) == HIPBLAS_STATUS_SUCCESS;
        ok = ok && hipblasLtMatmulPreferenceCreate(&p.pref) == HIPBLAS_STATUS_SUCCESS;
        const uint64_t maxws = (uint64_t)512 << 20;
        ok = ok && hipblasLtMatmulPreferenceSetAttribute(p.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxws, sizeof(maxws)) ==
                       HIPBLAS_STATUS_SUCCESS;
        p.failed = !ok;
        it = g_f32_plans.emplace(key, p).first;
    }
    PlanF32& p = it->second;
    if (p.failed) return false;
    const float beta = 0.0f;
    auto run = [&](const hipblasLtMatmulAlgo_t& algo, size_t wsz) {
        return hipblasLtMatmul(g_handle, p.desc, &alpha, B, p.la, A, p.lb, &beta, C, p.lc, C, p.lc, &algo, g_ws, wsz, nullptr) ==
               HIPBLAS_STATUS_SUCCESS;
    };
    if (!p.tuned) {
        hipblasLtMatmulHeuristicResult_t heur[16];
        int n = 0;
        if (hipblasLtMatmulAlgoGetHeuristic(g_handle, p.desc, p.la, p.lb, p.lc, p.lc, p.pref, 16, heur, &n) != HIPBLAS_STATUS_SUCCESS ||
            n < 1) {
            p.failed = true;
            return false;
        }
        size_t maxws = 0;
        for (int i = 0; i < n; ++i) maxws = std::max(maxws, (size_t)heur[i].workspaceSize);
        if (!ensure_workspace(std::max<size_t>(maxws, 1))) {
            p.failed = true;
            return false;
        }
        /* the calibrated algorithm of this shape (blt_calib.h), when the heuristic offers it */
        const bool calib_mode = std::getenv("SD_BLT_CALIBRATE") != nullptr;
        if (!calib_mode && !std::getenv("SD_BLT_NOCALIB")) {
            for (const BltCalib& ce : kBltCalib) {
                if (ce.M != M || ce.N != N || ce.K != K || ce.lda != lda || ce.ldb != ldb || ce.opN != (opN ? 1L : 0L) ||
                    ce.cs_m != cs_m || ce.bf16 != (bf16 ? 1 : 0))
                    continue;
                for (int i = 0; i < n && !p.tuned; ++i)
                    if (std::memcmp(heur[i].algo.data, ce.algo, sizeof ce.algo) == 0) {
                        p.algo = heur[i].algo;
                        p.ws = heur[i].workspaceSize;
                        p.tuned = true;
                    }
                if (std::getenv("SD_BLT_LOG"))
                    std::fprintf(stderr, "calib table: M %d N %d K %d: %s\n", M, N, K, p.tuned ? "taken" : "not offered by the heuristic");
                break;
            }
        }
        if (!p.tuned) {
            /* the candidates' outputs: the first candidate's is the reference; a candidate whose bits differ is never chosen (the
             * choice must not depend on the timing). The span: the rows M of cs_m words. (Calibration: every candidate is timed.) */
            const size_t span = (size_t)M * (size_t)cs_m * sizeof(float);
            void* ref = dalloc(span);
            bool have_ref = false;
            double best = 1e30;
            int best_i = -1;
            for (int i = 0; i < n; ++i) {
                if (!run(heur[i].algo, heur[i].workspaceSize)) continue;
                hipDeviceSynchronize();
                if (!have_ref) {
                    hipMemcpy(ref, C, span, hipMemcpyDeviceToDevice);
                    have_ref = true;
                } else if (!calib_mode && diff_words(C, ref, span) != 0) {
                    continue;
                }
                const auto t0 = std::chrono::steady_clock::now();
                run(heur[i].algo, heur[i].workspaceSize);
                hipDeviceSynchronize();
                const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                if (calib_mode) {
                    std::fprintf(stderr, "calib: M %ld N %ld K %ld lda %ld ldb %ld opN %d cs_m %ld bf16 %d candidate %d: %.3f ms algo ",
                                 (long)M, (long)N, (long)K, lda, ldb, opN ? 1 : 0, cs_m, bf16 ? 1 : 0, i, t * 1e3);
                    for (int b = 0; b < 16; ++b) std::fprintf(stderr, "%02x", heur[i].algo.data[b]);
                    std::fprintf(stderr, "\n");
                } else if (std::getenv("SD_BLT_LOG")) {
                    std::fprintf(stderr, "blt product: M %d N %d K %d lda %ld ldb %ld cs_m %ld bf16 %d candidate %d: %.3f ms\n", M, N, K,
                                 lda, ldb, cs_m, bf16 ? 1 : 0, i, t * 1e3);
                }
                if (t < best) {
                    best = t;
                    best_i = i;
                }
            }
            drelease(ref, span);
            if (best_i < 0) {
                p.failed = true;
                return false;
            }
            p.algo = heur[best_i].algo;
            p.ws = heur[best_i].workspaceSize;
            p.tuned = true;
        }
    }
    if (!ensure_workspace(std::max<size_t>(p.ws, 1))) return false;
    const auto tp0 = std::chrono::steady_clock::now();
    if (!run(p.algo, p.ws)) {
        p.failed = true;
        return false;
    }
    if (g_time_on) {  /* SD_TIME: the convolutions' and attention's products, synchronous, in the table */
        hipStreamSynchronize(nullptr);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
        std::lock_guard<std::mutex> lk(g_time_mu);
        TimeStat& st = g_time_stats["blt_product"];
        st.count += 1;
        st.total += ms;
        st.max = std::max(st.max, ms);
    }
    return true;
}

bool gemm_f32(int M, int N, int K, const float* A, long lda, const float* B, long sbk, long sbn, float* C, long cs_m, float alpha) {
    return gemm_impl(false, M, N, K, A, lda, B, sbk, sbn, C, cs_m, alpha);
}

bool gemm_bf16(int M, int N, int K, const uint16_t* A, long lda, const uint16_t* B, long sbk, long sbn, float* C, long cs_m, float alpha) {
    return gemm_impl(true, M, N, K, A, lda, B, sbk, sbn, C, cs_m, alpha);
}
}  // namespace blt

/* the large linear with its input in BF16 (xb: T x in, row-major): hipBLASLt, or the 256 x 128 WMMA kernel when hipBLASLt is off
 * or has no plan for the shape. The callers check the shape (out >= 256, T >= 1024, in % 64 == 0). */
bool linear_huge_xb(const uint16_t* xb, int T, const uint16_t* W, int in, int out, const float* b, void* y, bool out16) {
    const float alpha = 1.0f, beta = 0.0f;
    const int M = out, N = T, K = in;
    const long bs0 = 0;
    const uint16_t* A = W;
    const uint16_t* B = xb;
    float* C = static_cast<float*>(y);
    const long lda = in, sbn = in, cs_m = 1, cs_n = out;
    const float* bf = b;
    void* args[] = {(void*)&A, (void*)&lda, (void*)&B, (void*)&sbn, (void*)&C, (void*)&cs_m, (void*)&cs_n,
                    (void*)&bf, (void*)&alpha, (void*)&beta, (void*)&M, (void*)&N, (void*)&K, (void*)&bs0, (void*)&bs0,
                    (void*)&bs0};
    bool done = false;
    if (blt::enabled()) {
        // hipBLASLt, with the bias as the epilogue (no bias: the plan without it)
        blt::Plan* p = blt::plan_for(out, T, in, b != nullptr, out16, W, xb, y, b);
        if (p) {
            done = blt::run_once(*p, p->algo, W, xb, y, b, b != nullptr);
            if (!done) p->failed = true;
        }
    }
    if (out16 && !done) {
        sync();
        return false;  /* the BF16 output is hipBLASLt's only: the caller takes the FP32 path */
    }
    if (!done)
        check(hip_launch(f_gemm_wmma256b, (unsigned)((N + 127) / 128), (unsigned)((M + 255) / 256), 1, 256, 1, 1, 0, nullptr, args,
                         nullptr),
              "wmma gemm 256b");
    sync();
    /* SD_ROUND_OUT=1 (experiment): the output rounded to BF16, as a BF16 GEMM's output is */
    static const bool round_out = [] {
        const char* e = std::getenv("SD_ROUND_OUT");
        return e && std::atoi(e) != 0;
    }();
    if (round_out && !out16) {  /* (the BF16 output is already rounded) */
        const long ny = (long)T * out;
        float* yf = static_cast<float*>(y);
        void* ar[] = {(void*)&yf, (void*)&ny};
        launch(f_round_bf16, dim3(blocks((size_t)ny)), ar);
        sync();
    }
    return true;
}

bool linear_bf16_out16(const uint16_t* xb, int T, const uint16_t* W, int in, int out, const float* b, uint16_t* y) {
    if (out <= 0 || T <= 0 || in % 64 != 0 || out < 256 || T < 1024 || !f_gemm_wmma256b || !blt::enabled()) return false;
    return linear_huge_xb(xb, T, W, in, out, b, y, true);
}

/* the linear with BF16 output, whatever the shape: hipBLASLt's BF16 output when it takes the shape, else the FP32 product
 * rounded to BF16 */
bool linear_bf16_xb16(const uint16_t* xb, int T, const uint16_t* W, int in, int out, const float* b, uint16_t* y16) {
    if (linear_bf16_out16(xb, T, W, in, out, b, y16)) return true;
    float* tmp = static_cast<float*>(dalloc(sizeof(float) * (size_t)T * out));
    linear_bf16_xb(xb, T, W, in, out, b, tmp);
    const long n = (long)T * out;
    const float* tp = tmp;
    void* a[] = {(void*)&tp, (void*)&y16, (void*)&n};
    launch(f_to_bf16, dim3(blocks((size_t)n)), a);
    sync();
    drelease(tmp, sizeof(float) * (size_t)T * out);
    return true;
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
            uint16_t* xb = static_cast<uint16_t*>(device_scratch(sizeof(uint16_t) * (size_t)nb));
            const float* xs = x;
            void* ac[] = {(void*)&xs, (void*)&xb, (void*)&nb};
            launch(f_to_bf16, dim3((unsigned)((nb + kThreads - 1) / kThreads)), ac);
            linear_huge_xb(xb, T, W, in, out, b, y, false);
            return;  // (device_scratch: kept for the next call)
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
            check(hip_launch(f_gemm_wmma256, (unsigned)((N + 127) / 128), (unsigned)((M + 255) / 256), 1, 256, 1, 1, 0,
                                        nullptr, args, nullptr),
                  "wmma gemm 256");
        else
            check(hip_launch(big ? f_gemm_wmma128 : f_gemm_wmma, (unsigned)((N + (big ? 127 : 63)) / (big ? 128 : 64)),
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
    check(hip_launch(f_zq_attn, (unsigned)T, (unsigned)H, 1, 128, 1, 1, (unsigned)(sizeof(float) * (size_t)T), nullptr, a, nullptr), "causal attention");
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
    /* the three passes over 64 chunks per group (zimage_gn_sum / _var / _apply); the one-block version when the groups are odd */
    const int S = 64;
    const int per = C / groups;
    if (groups <= 0 || C % groups != 0 || !f_gn_sum_part) {
        void* a[] = {(void*)&x, (void*)&C, (void*)&HW, (void*)&groups, (void*)&gamma, (void*)&beta, (void*)&eps, (void*)&y};
        launch(f_gn_f32, dim3((unsigned)groups, 1, 1), a);
        sync();
        return;
    }
    float* part_sum = static_cast<float*>(alloc(sizeof(float) * (size_t)groups * S, false));
    float* part_q = static_cast<float*>(alloc(sizeof(float) * (size_t)groups * S, false));
    const int Sv = S;
    void* a1[] = {(void*)&x, (void*)&HW, (void*)&per, (void*)&Sv, (void*)&part_sum};
    launch(f_gn_sum_part, dim3((unsigned)S, (unsigned)groups, 1), a1);
    void* a2[] = {(void*)&x, (void*)&HW, (void*)&per, (void*)&Sv, (void*)&part_sum, (void*)&part_q};
    launch(f_gn_var_part, dim3((unsigned)S, (unsigned)groups, 1), a2);
    void* a3[] = {(void*)&x, (void*)&C, (void*)&HW, (void*)&groups, (void*)&Sv, (void*)&part_sum, (void*)&part_q, (void*)&gamma,
                  (void*)&beta, (void*)&eps, (void*)&y};
    launch(f_gn_apply_part, dim3((unsigned)S, (unsigned)groups, 1), a3);
    sync();
    release(part_sum, sizeof(float) * (size_t)groups * S);
    release(part_q, sizeof(float) * (size_t)groups * S);
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

void rms_mod_bf16(const float* x, int T, int C, const uint16_t* w, float eps, const float* s, uint16_t* y) {
    static const int ex = std::getenv("ZI_NO_EXACT_RESID") == nullptr;  /* diffusers' BF16 rounding order (ZI_NO_EXACT_RESID=1: off) */
    void* a[] = {(void*)&x, (void*)&C, (void*)&w, (void*)&eps, (void*)&s, (void*)&y, (void*)&ex};
    launch(f_zm_rms_mod_bf16, dim3((unsigned)T, 1, 1), a);
    sync();
}

void swiglu_bf16(const float* a0, const float* b0, uint16_t* g, size_t n0) {
    long n = (long)n0;
    void* a[] = {(void*)&a0, (void*)&b0, (void*)&g, (void*)&n};
    launch(f_zm_swiglu_bf16, dim3((unsigned)((n + kThreads - 1) / kThreads)), a);
    sync();
}

void head_norm_rope_bf16(const uint16_t* x, int T, int H, const uint16_t* w, float eps, const float* fr, uint16_t* y) {
    void* a[] = {(void*)&x, (void*)&H, (void*)&w, (void*)&eps, (void*)&fr, (void*)&y};
    check(hip_launch(f_hnr16, (unsigned)((long)T * H), 1, 1, 128, 1, 1, 0, nullptr, a, nullptr), "head norm rope");
    sync();
}

void rms_gate_add_bf16(float* x, int T, int C, const uint16_t* o, const uint16_t* w, float eps, const float* g) {
    static const int ex = std::getenv("ZI_NO_EXACT_RESID") == nullptr;  /* diffusers' BF16 rounding order */
    void* a[] = {(void*)&x, (void*)&C, (void*)&o, (void*)&w, (void*)&eps, (void*)&g, (void*)&ex};
    launch(f_rga16, dim3((unsigned)T, 1, 1), a);
    sync();
}

void download_bf16_as_f32(const uint16_t* d, float* h, size_t n) {
    std::vector<uint16_t> t(n);
    check(hipMemcpy(t.data(), d, sizeof(uint16_t) * n, hipMemcpyDeviceToHost), "download");
    for (size_t i = 0; i < n; ++i) {
        const uint32_t u = (uint32_t)t[i] << 16;
        std::memcpy(&h[i], &u, 4);
    }
}

void audio_gemm_f32(const float* x, int T, int in, const float* W, const float* b, int out, float* y) {
    if (T <= 0 || out <= 0) return;
    void* a[] = {(void*)&x, (void*)&T, (void*)&in, (void*)&W, (void*)&b, (void*)&out, (void*)&y};
    launch(f_audio_gemm, dim3(blocks((size_t)T * out), 1, 1), a);
    sync();
}

void audio_layer_norm_f32(const float* x, int T, int C, const float* w, const float* b, float eps, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&w, (void*)&b, (void*)&eps, (void*)&y};
    launch(f_audio_ln, dim3((unsigned)T, 1, 1), a);
    sync();
}

// qkv [T][3 H 64] -> out [T][H 64], causal, scale 1/8
// q, k, v [T][C] with row stride ld, C = H * 64; causal or not; kmask (T floats on the device, 0 for a padding key) or null
void audio_attention_f32(const float* q, const float* k, const float* v, int ld, int T, int H, bool causal, const float* kmask, float* out) {
    int tt = T, hh = H, cc = causal ? 1 : 0, lld = ld;
    const float* km = kmask;
    void* a[] = {(void*)&q, (void*)&k, (void*)&v, (void*)&lld, (void*)&tt, (void*)&hh, (void*)&cc, (void*)&km, (void*)&out};
    check(hip_launch(f_audio_attn_g, (unsigned)T, (unsigned)H, 1, 64, 1, 1, (unsigned)(sizeof(float) * (64 + (size_t)T)), nullptr, a, nullptr),
          "audio attention");
    sync();
}

// exact GELU (erf), in place
void audio_gelu_erf(float* x, size_t n) {
    long nn = (long)n;
    void* a[] = {(void*)&x, (void*)&nn};
    launch(f_audio_gelu_erf, dim3(blocks(n), 1, 1), a);
    sync();
}

void audio_causal_attn_f32(const float* qkv, int T, int H, float* out) {
    const float scale = 0.125f;
    int tt = T, hh = H;
    void* a[] = {(void*)&qkv, (void*)&tt, (void*)&hh, (void*)&out, (void*)&scale};
    check(hip_launch(f_audio_attn, (unsigned)T, (unsigned)H, 1, 64, 1, 1, (unsigned)(sizeof(float) * (64 + (size_t)T)), nullptr, a, nullptr),
          "audio causal attention");
    sync();
}

void round_bf16_inplace(float* x, size_t n0) {
    long n = (long)n0;
    void* a[] = {(void*)&x, (void*)&n};
    launch(f_round_bf16, dim3(blocks((size_t)n)), a);
    sync();
}

void to_bf16(const float* x, uint16_t* y, size_t n0) {
    long n = (long)n0;
    void* a[] = {(void*)&x, (void*)&y, (void*)&n};
    launch(f_to_bf16, dim3(blocks((size_t)n)), a);
    sync();
}

void swiglu_bf16_in(const uint16_t* a0, const uint16_t* b0, uint16_t* g, size_t n0) {
    long n = (long)n0;
    void* a[] = {(void*)&a0, (void*)&b0, (void*)&g, (void*)&n};
    launch(f_sw16, dim3((unsigned)((n + kThreads - 1) / kThreads)), a);
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

/* SD_ATTN_KERNEL=scalar keeps the scalar FP32 products of attention_f32 */
bool blt_attn_on() {
    static const bool on = [] {
        const char* e = std::getenv("SD_ATTN_KERNEL");
        return !(e && std::strcmp(e, "scalar") == 0);
    }();
    return on;
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
            const bool blt_ok = blt_attn_on() && blt::gemm_f32(Rc, Tk, D, q + (size_t)r0 * C + hd * D, C, k + hd * D, 1, C, S, Tk, scale);
            if (!blt_ok) gemm<float>(f_gemm_af, q + (size_t)r0 * C + hd * D, C, k + hd * D, 1, C, S, Tk, 1, nullptr, scale, 0.0f, Rc, Tk, D);
            int N = Tk;
            void* sm[] = {(void*)&S, (void*)&N};
            launch(f_softmax, dim3((unsigned)Rc, 1, 1), sm);
            const bool blt_pv = blt_attn_on() && blt::gemm_f32(Rc, D, Tk, S, Tk, v + hd * D, C, 1, out + (size_t)r0 * C + hd * D, C, 1.0f);
            if (!blt_pv) gemm<float>(f_gemm_af, S, Tk, v + hd * D, C, 1, out + (size_t)r0 * C + hd * D, C, 1, nullptr, 1.0f, 0.0f, Rc, D, Tk);
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
    check(hip_launch(big ? f_gemm_wmma128 : f_gemm_wmma, (unsigned)((N + (big ? 127 : 63)) / (big ? 128 : 64)),
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
        check(hip_launch(f_softmax_bf16w, (unsigned)((rows + 7) / 8), 1, 1, 256, 1, 1, 0, nullptr, a3w, nullptr),
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
        check(hip_launch(f_flash, (unsigned)((Tq + 127) / 128), (unsigned)heads, 1, 256, 1, 1, 0, nullptr, a4, nullptr),
              "flash attention");
    }
    check(hipStreamSynchronize(nullptr), "stream");
    release(qb, sizeof(uint16_t) * (size_t)Tq * C);
    release(kb, sizeof(uint16_t) * (size_t)Tk * C);
    release(vt, sizeof(uint16_t) * (size_t)C * Tkp);
}

void attention_bf16(const uint16_t* q, int Tq, const uint16_t* k, const uint16_t* v, int Tk, int C, int heads, float* out) {
    if (C / heads != 128) throw std::runtime_error("sd device: attention_bf16 takes the head width 128");
    if (Tq <= 0 || Tk <= 0) return;
    const int Tkp = (Tk + 63) / 64 * 64;
    uint16_t* vt = static_cast<uint16_t*>(dalloc(sizeof(uint16_t) * (size_t)C * Tkp));
    {
        int tk = Tk, tkp = Tkp, c = C;
        const uint16_t* vp = v;
        uint16_t* vtp = vt;
        void* a3[] = {(void*)&vp, (void*)&tk, (void*)&tkp, (void*)&c, (void*)&vtp};
        launch(f_tp16, dim3((unsigned)(((C + 31) / 32) * ((Tkp + 31) / 32))), a3);
    }
    {
        const float scale = 1.0f / std::sqrt((float)(C / heads));
        int tq = Tq, tk = Tk, tkp = Tkp, c = C;
        const uint16_t* qp = q;
        const uint16_t* kp = k;
        void* a4[] = {(void*)&qp, (void*)&kp, (void*)&vt, (void*)&out, (void*)&tq, (void*)&tk, (void*)&tkp, (void*)&c, (void*)&scale};
        check(hip_launch(f_flash, (unsigned)((Tq + 127) / 128), (unsigned)heads, 1, 256, 1, 1, 0, nullptr, a4, nullptr),
              "flash attention");
    }
    sync();
    drelease(vt, sizeof(uint16_t) * (size_t)C * Tkp);
}

void attention(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out) {
    if (Tq <= 0 || Tk <= 0) return;
    if (C / heads == 128) attention_flash(q, Tq, k, v, Tk, C, heads, out);
    else attention_f32(q, Tq, k, v, Tk, C, heads, out);
}

}  // namespace sddev
