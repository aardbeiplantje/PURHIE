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
#include <cstdint>
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
hipFunction_t f_audio_gemm = nullptr, f_audio_ln = nullptr, f_audio_attn = nullptr, f_audio_attn_g = nullptr, f_audio_gelu_erf = nullptr, f_audio_rms = nullptr, f_audio_gelu_mul = nullptr, f_audio_add_channel = nullptr, f_audio_copy = nullptr, f_audio_conv1d = nullptr, f_audio_convt1d = nullptr, f_audio_leaky = nullptr, f_audio_scale = nullptr, f_audio_tanh = nullptr, f_audio_upnear = nullptr;
hipFunction_t f_gemm_ah = nullptr, f_gemm_af = nullptr, f_im2col = nullptr, f_gn_sum = nullptr, f_gn_sq = nullptr,
              f_gn_apply = nullptr, f_silu = nullptr, f_add = nullptr, f_upsample2 = nullptr, f_softmax = nullptr,
              f_layer_norm = nullptr, f_rms_norm = nullptr, f_zm_modulate = nullptr, f_zm_gate_add = nullptr, f_zm_swiglu = nullptr, f_zm_rope = nullptr, f_zm_ln_plain = nullptr, f_gn_f32 = nullptr, f_to_tok = nullptr, f_to_chw = nullptr, f_zq_rope = nullptr, f_zq_attn = nullptr, f_geglu = nullptr, f_gemm_bf = nullptr, f_gemm_wmma = nullptr, f_gemm_wmma128 = nullptr, f_softmax_bf16w = nullptr, f_flash = nullptr, f_gemm_wmma256 = nullptr, f_gemm_wmma256b = nullptr, f_gemm_wmma128bb = nullptr, f_transpose_pad_bf16 = nullptr,
              f_to_bf16 = nullptr, f_transpose_pad = nullptr, f_softmax_bf16 = nullptr, f_qi_linear_f8 = nullptr, f_qi_rms_f8 = nullptr, f_qi_affine = nullptr, f_qi_gelu = nullptr, f_qv_rms = nullptr, f_qv_patch = nullptr, f_qv_bias = nullptr, f_qv_upsample = nullptr, f_qi_hadamard = nullptr, f_qi_linear_i8 = nullptr, f_qi21_zc = nullptr, f_qi21_gather = nullptr, f_qi21_mul = nullptr, f_qi21_gated = nullptr, f_qi21_attn = nullptr, f_qi21_lbf = nullptr, f_qi21_dupup = nullptr, f_gemm_af128 = nullptr, f_transpose_bf16 = nullptr, f_bias_cm = nullptr, f_zm_rms_mod_bf16 = nullptr, f_zm_swiglu_bf16 = nullptr, f_bf16_to_f32 = nullptr, f_round_bf16 = nullptr, f_gn_sum_part = nullptr, f_gn_var_part = nullptr, f_gn_apply_part = nullptr, f_im2col_bf16_tile = nullptr, f_hnr16 = nullptr, f_count_diff = nullptr, f_tp16 = nullptr, f_rga16 = nullptr, f_sw16 = nullptr;

hipFunction_t f_ss_affine = nullptr, f_ss_gate_add = nullptr, f_ss_head_norm_rope = nullptr; hipFunction_t f_sp_hash_insert = nullptr, f_sp_neighbor_map = nullptr, f_sp_subm_conv_f16w = nullptr; hipFunction_t f_sp_c2s_mask = nullptr, f_sp_exclusive_scan_serial = nullptr, f_sp_c2s_fill = nullptr, f_sp_c2s_gather = nullptr; hipFunction_t f_simp_qem = nullptr, f_simp_cost = nullptr, f_simp_propagate = nullptr, f_simp_collapse = nullptr, f_simp_compress_v = nullptr, f_simp_compress_f = nullptr; hipFunction_t f_chart_face_geom = nullptr, f_chart_cones = nullptr, f_chart_perims = nullptr, f_chart_costs = nullptr, f_chart_propagate = nullptr, f_chart_collapse = nullptr, f_chart_refine = nullptr;
hipFunction_t f_attn_flash128 = nullptr, f_attn_flash64 = nullptr;
hipFunction_t f_dino_rope = nullptr; hipFunction_t f_mesh_vertices = nullptr, f_mesh_edge_queries = nullptr, f_sp_lookup = nullptr, f_mesh_edge_valid = nullptr, f_mesh_quads = nullptr, f_mesh_triangles = nullptr; hipFunction_t f_ss_im2col3 = nullptr, f_ss_gemm_f16w = nullptr, f_ss_pixel_shuffle3 = nullptr;
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
        {&f_audio_attn_g, "audio_attn_f32"}, {&f_audio_gelu_erf, "audio_gelu_erf_f32"}, {&f_audio_rms, "audio_rms_f32"}, {&f_audio_gelu_mul, "audio_gelu_tanh_mul_f32"}, {&f_audio_add_channel, "audio_add_channel_f32"}, {&f_audio_copy, "audio_copy_f32"},
        {&f_audio_conv1d, "audio_conv1d_f32"}, {&f_audio_convt1d, "audio_conv_transpose1d_f32"}, {&f_audio_leaky, "audio_leaky_relu_f32"},
        {&f_audio_scale, "audio_scale_f32"}, {&f_audio_tanh, "audio_tanh_f32"}, {&f_audio_upnear, "audio_upsample_nearest_f32"},
        {&f_transpose_pad, "sd_transpose_pad"}, {&f_softmax_bf16, "sd_softmax_bf16"}, {&f_softmax_bf16w, "sd_softmax_bf16_w"},
        {&f_gn_sum, "sd_gn_sum"},       {&f_gn_sq, "sd_gn_sq"},         {&f_gn_apply, "sd_gn_apply"},
        {&f_silu, "sd_silu"},           {&f_add, "sd_add"},             {&f_upsample2, "sd_upsample2"},
        {&f_softmax, "sd_softmax"},     {&f_layer_norm, "sd_layer_norm"}, {&f_rms_norm, "sd_rms_norm"}, {&f_zm_modulate, "zimage_modulate"}, {&f_zm_gate_add, "zimage_gate_add"}, {&f_zm_swiglu, "zimage_swiglu"}, {&f_zm_rope, "zimage_rope"}, {&f_zm_ln_plain, "zimage_layer_norm_plain"}, {&f_gn_f32, "zimage_group_norm_f32"}, {&f_to_tok, "zimage_to_tokens"}, {&f_to_chw, "zimage_to_chw"}, {&f_zq_rope, "zq_rope_half"}, {&f_zq_attn, "zq_causal_attn"}, {&f_geglu, "sd_geglu"},
        {&f_ss_affine, "ss_affine"}, {&f_ss_gate_add, "ss_gate_add"}, {&f_ss_head_norm_rope, "ss_head_norm_rope"},
        {&f_ss_im2col3, "ss_im2col3"}, {&f_ss_gemm_f16w, "ss_gemm_f16w"}, {&f_ss_pixel_shuffle3, "ss_pixel_shuffle3"},
        {&f_sp_hash_insert, "sp_hash_insert"}, {&f_sp_neighbor_map, "sp_neighbor_map"}, {&f_sp_subm_conv_f16w, "sp_subm_conv_f16w"},
        {&f_sp_c2s_mask, "sp_c2s_mask"}, {&f_sp_exclusive_scan_serial, "sp_exclusive_scan_serial"}, {&f_sp_c2s_fill, "sp_c2s_fill"}, {&f_sp_c2s_gather, "sp_c2s_gather"},
        {&f_simp_qem, "simp_qem"}, {&f_simp_cost, "simp_cost"}, {&f_simp_propagate, "simp_propagate"}, {&f_simp_collapse, "simp_collapse"}, {&f_simp_compress_v, "simp_compress_v"}, {&f_simp_compress_f, "simp_compress_f"}, {&f_chart_face_geom, "chart_face_geom"}, {&f_chart_cones, "chart_cones"}, {&f_chart_perims, "chart_perims"}, {&f_chart_costs, "chart_costs"}, {&f_chart_propagate, "chart_propagate"}, {&f_chart_collapse, "chart_collapse"}, {&f_chart_refine, "chart_refine"}, {&f_attn_flash128, "attn_flash_d128"}, {&f_attn_flash64, "attn_flash_d64"}, {&f_dino_rope, "dino_rope"}, {&f_mesh_vertices, "mesh_vertices"}, {&f_mesh_edge_queries, "mesh_edge_queries"}, {&f_sp_lookup, "sp_lookup"}, {&f_mesh_edge_valid, "mesh_edge_valid"}, {&f_mesh_quads, "mesh_quads"}, {&f_mesh_triangles, "mesh_triangles"},
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

void conv2d_f32_fp32(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y,
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
    conv2d_f32_fp32(x, cin, H, W, w, cout, b, k, p, st, y, Ho, Wo);
}

// The convolution in FP32 throughout (the im2col and the FP32 GEMM; no BF16 route): for the layers whose accuracy is checked
// against the FP32 oracle (AudioLDM 2's VAE: the BF16 route is 1e-3 off at 128 x 128)
void conv2d_f32_exact(const float* x, int cin, int H, int W, const float* w, int cout, const float* b, int k, int p, int st, float* y,
                      int Ho, int Wo) {
    conv2d_f32_fp32(x, cin, H, W, w, cout, b, k, p, st, y, Ho, Wo);
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
// q [Tq][C], k, v [Tk][C] with row stride ld, C = H * D (D <= 128); causal (Tq = Tk) or not; kmask (Tk floats on the device, 0 for a
// padding key) or null; scores * scale + bias [H][Tq][Tk] (device-accessible) or none; out [Tq][C]
void audio_attention_f32(const float* q, const float* k, const float* v, int ld, int Tq, int Tk, int H, int D, bool causal, const float* kmask,
                         float scale, const float* bias, float* out) {
    int tq = Tq, tk = Tk, hh = H, dd = D, cc = causal ? 1 : 0, lld = ld;
    const float* km = kmask;
    const float* bi = bias;
    void* a[] = {(void*)&q, (void*)&k, (void*)&v, (void*)&lld, (void*)&tq, (void*)&tk, (void*)&hh, (void*)&dd, (void*)&cc, (void*)&km, (void*)&scale, (void*)&bi, (void*)&out};
    check(hip_launch(f_audio_attn_g, (unsigned)Tq, (unsigned)H, 1, 128, 1, 1, (unsigned)(sizeof(float) * (128 + (size_t)Tk)), nullptr, a, nullptr),
          "audio attention");
    sync();
}

// x = RMS norm (T5's: no mean, no bias) over the last dimension, x [T][C] -> y
void audio_rms_norm_f32(const float* x, int T, int C, const float* w, float eps, float* y) {
    void* a[] = {(void*)&x, (void*)&C, (void*)&w, (void*)&eps, (void*)&y};
    launch(f_audio_rms, dim3((unsigned)T, 1, 1), a);
    sync();
}

// a = gelu_new(a) * b, in place (T5's gated GELU)
void audio_gelu_tanh_mul(float* a, const float* b, size_t n) {
    long nn = (long)n;
    void* args[] = {(void*)&a, (void*)&b, (void*)&nn};
    launch(f_audio_gelu_mul, dim3(blocks(n), 1, 1), args);
    sync();
}

// h[c][p] += t[c] (HW positions per channel), in place
void audio_add_channel(float* h, const float* t, int C, int HW) {
    const size_t n = (size_t)C * HW;
    void* a[] = {(void*)&h, (void*)&t, (void*)&C, (void*)&HW};
    launch(f_audio_add_channel, dim3(blocks(n), 1, 1), a);
    sync();
}

// dst[0..n) = src[0..n)
void audio_copy(float* dst, const float* src, size_t n) {
    long nn = (long)n;
    void* a[] = {(void*)&dst, (void*)&src, (void*)&nn};
    launch(f_audio_copy, dim3(blocks(n), 1, 1), a);
    sync();
}

// 1-D convolution, stride 1: x [cin][T] -> y [cout][Tout], Tout = T + 2 pad - dil (k - 1); w [cout][cin][k]
void audio_conv1d(const float* x, int cin, int T, const float* w, const float* b, int cout, int k, int pad, int dil, float* y, int Tout) {
    const size_t n = (size_t)cout * Tout;
    void* a[] = {(void*)&x, (void*)&w, (void*)&b, (void*)&cin, (void*)&cout, (void*)&T, (void*)&k, (void*)&pad, (void*)&dil, (void*)&Tout, (void*)&y};
    launch(f_audio_conv1d, dim3(blocks(n), 1, 1), a);
    sync();
}

// 1-D transposed convolution: x [cin][T] -> y [cout][Tout], Tout = (T - 1) stride - 2 pad + k; w [cin][cout][k]
void audio_conv_transpose1d(const float* x, int cin, int T, const float* w, const float* b, int cout, int k, int stride, int pad, float* y,
                            int Tout) {
    const size_t n = (size_t)cout * Tout;
    void* a[] = {(void*)&x, (void*)&w, (void*)&b, (void*)&cin, (void*)&cout, (void*)&T, (void*)&k, (void*)&stride, (void*)&pad, (void*)&Tout, (void*)&y};
    launch(f_audio_convt1d, dim3(blocks(n), 1, 1), a);
    sync();
}

// x = leaky_relu(x, slope) in place
void audio_leaky_relu(float* x, float slope, size_t n) {
    long nn = (long)n;
    void* a[] = {(void*)&x, (void*)&slope, (void*)&nn};
    launch(f_audio_leaky, dim3(blocks(n), 1, 1), a);
    sync();
}

// x = x * s in place
void audio_scale(float* x, float s, size_t n) {
    long nn = (long)n;
    void* a[] = {(void*)&x, (void*)&s, (void*)&nn};
    launch(f_audio_scale, dim3(blocks(n), 1, 1), a);
    sync();
}

// x = tanh(x) in place
void audio_tanh(float* x, size_t n) {
    long nn = (long)n;
    void* a[] = {(void*)&x, (void*)&nn};
    launch(f_audio_tanh, dim3(blocks(n), 1, 1), a);
    sync();
}

// nearest resampling of x [C][H][W] to [C][Ho][Wo] (2x for the UNet and VAE upsamplers; any size for the UNet's skip sizes)
void audio_upsample_nearest(const float* x, int C, int H, int W, int Ho, int Wo, float* y) {
    const size_t n = (size_t)C * Ho * Wo;
    void* a[] = {(void*)&x, (void*)&C, (void*)&H, (void*)&W, (void*)&Ho, (void*)&Wo, (void*)&y};
    launch(f_audio_upnear, dim3(blocks(n), 1, 1), a);
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
    attention_f32_strided(q, C, Tq, k, C, v, C, Tk, C, heads, out);
}

void attention_f32_strided(const float* q, long ldq, int Tq, const float* k, long ldk, const float* v, long ldv, int Tk, int C,
                           int heads, float* out) {
    const int D = C / heads;
    const float scale = 1.0f / std::sqrt((float)D);
    if (Tq <= 0 || Tk <= 0) return;
    // the fused kernel (sd_gpu.hip attn_flash_*: no score matrix in memory), for the head sizes it is built for; SD_ATTN_FLASH=0 keeps
    // the GEMM path below
    static const bool flash = [] {
        const char* e = std::getenv("SD_ATTN_FLASH");
        return !(e && std::strcmp(e, "0") == 0);
    }();
    // the kernel reads float4 along the rows: the pointers and the row strides must be 16-byte aligned, else the GEMM path runs
    auto a16 = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    if (flash && (D == 64 || D == 128) && a16(q) && a16(k) && a16(v) && a16(out) && ldq % 4 == 0 && ldk % 4 == 0 && ldv % 4 == 0 &&
        C % 4 == 0) {
        void* a[] = {(void*)&q, (void*)&ldq, (void*)&Tq, (void*)&k, (void*)&ldk, (void*)&v, (void*)&ldv, (void*)&Tk, (void*)&C, (void*)&heads,
                     (void*)&out, (void*)&scale};
        launch(D == 64 ? f_attn_flash64 : f_attn_flash128, dim3((unsigned)((Tq + 63) / 64), (unsigned)heads, 1), a);
        sync();
        return;
    }
    const int R = std::max(1, std::min(Tq, (int)(kScoreFloats / (size_t)Tk)));
    float* S = static_cast<float*>(alloc(sizeof(float) * (size_t)R * Tk));
    for (int hd = 0; hd < heads; ++hd)
        for (int r0 = 0; r0 < Tq; r0 += R) {
            const int Rc = std::min(R, Tq - r0);
            // scores = scale * Q K^T for the chunk's rows, softmaxed, then the weighted values into the head's columns
            const float* qh = q + (size_t)r0 * ldq + hd * D;
            const float* kh = k + hd * D;
            const float* vh = v + hd * D;
            float* oh = out + (size_t)r0 * C + hd * D;
            const bool blt_ok = blt_attn_on() && blt::gemm_f32(Rc, Tk, D, qh, ldq, kh, 1, ldk, S, Tk, scale);
            if (!blt_ok) gemm<float>(f_gemm_af, qh, ldq, kh, 1, ldk, S, Tk, 1, nullptr, scale, 0.0f, Rc, Tk, D);
            int N = Tk;
            void* sm[] = {(void*)&S, (void*)&N};
            launch(f_softmax, dim3((unsigned)Rc, 1, 1), sm);
            const bool blt_pv = blt_attn_on() && blt::gemm_f32(Rc, D, Tk, S, Tk, vh, ldv, 1, oh, C, 1.0f);
            if (!blt_pv) gemm<float>(f_gemm_af, S, Tk, vh, ldv, 1, oh, C, 1, nullptr, 1.0f, 0.0f, Rc, D, Tk);
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

// TRELLIS.2 sparse-structure flow (sd_gpu.hip, the ss_* kernels): see sd_device.h
void ss_affine(float* x, int T, int C, const float* s, const float* sh) {
    long n = (long)T * C;
    int cc = C;
    void* a[] = {(void*)&x, (void*)&n, (void*)&cc, (void*)&s, (void*)&sh};
    launch(f_ss_affine, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void ss_gate_add(float* h, int T, int C, const float* g, const float* y) {
    long n = (long)T * C;
    int cc = C;
    void* a[] = {(void*)&h, (void*)&n, (void*)&cc, (void*)&g, (void*)&y};
    launch(f_ss_gate_add, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void ss_head_norm_rope(float* x, int T, int ld, int H, int D, const float* gamma, const float* phases) {
    int ldc = ld, hh = H, dd = D;
    void* a[] = {(void*)&x, (void*)&ldc, (void*)&hh, (void*)&dd, (void*)&gamma, (void*)&phases};
    check(hip_launch(f_ss_head_norm_rope, (unsigned)T, (unsigned)H, 1, 128, 1, 1, 0, nullptr, a, nullptr), "ss head norm rope");
    sync();
}

// TRELLIS.2 sparse-structure decoder (sd_gpu.hip, the ss_im2col3 / ss_gemm_f16w / ss_pixel_shuffle3 kernels): see sd_device.h
void ss_im2col3(const float* x, int C, int n, float* P) {
    long total = (long)n * n * n * C * 27;
    int cc = C, nn = n;
    void* a[] = {(void*)&x, (void*)&cc, (void*)&nn, (void*)&P};
    launch(f_ss_im2col3, dim3((unsigned)((total + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void ss_gemm_f16w(const float* P, long K, const uint16_t* W, const float* b, int Cout, long Vtot, float* y, int v0, int nv, int o0, int no) {
    long kk = K;
    int cout = Cout;
    long vt = Vtot;
    int vv0 = v0, nvv = nv, oo0 = o0, noo = no;
    void* a[] = {(void*)&P, (void*)&kk, (void*)&W, (void*)&b, (void*)&cout, (void*)&vt, (void*)&y, (void*)&vv0, (void*)&nvv, (void*)&oo0, (void*)&noo};
    check(hip_launch(f_ss_gemm_f16w, (unsigned)((nv + 15) / 16), (unsigned)((no + 15) / 16), 1, 16, 16, 1, 0, nullptr, a, nullptr), "ss gemm fp16 weights");
    sync();
}

void ss_pixel_shuffle3(const float* x, int Cout, int n, float* y) {
    long total = (long)(2 * n) * (2 * n) * (2 * n) * Cout;
    int cc = Cout, nn = n;
    void* a[] = {(void*)&x, (void*)&cc, (void*)&nn, (void*)&y};
    launch(f_ss_pixel_shuffle3, dim3((unsigned)((total + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

// sparse voxels (sd_gpu.hip, src/sparse.c): see sd_device.h
void sp_hash_build(const int* coords, int n, unsigned long long* keys, int* vals, unsigned int tsize) {
    // the table starts empty (every key 0xFF..): a host fill of the device-visible table, then the inserts
    std::memset(keys, 0xFF, sizeof(unsigned long long) * tsize);
    unsigned int mask = tsize - 1;
    int nn = n;
    void* a[] = {(void*)&coords, (void*)&nn, (void*)&keys, (void*)&vals, (void*)&mask};
    launch(f_sp_hash_insert, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void sp_neighbor_map(const int* coords, int n, const unsigned long long* keys, const int* vals, unsigned int tsize, int* nb) {
    unsigned int mask = tsize - 1;
    int nn = n;
    long total = (long)n * 27;
    void* a[] = {(void*)&coords, (void*)&nn, (void*)&keys, (void*)&vals, (void*)&mask, (void*)&nb};
    launch(f_sp_neighbor_map, dim3((unsigned)((total + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void sp_subm_conv_f16w(const float* x, int n, int cin, const int* nb, const uint16_t* W, const float* b, int cout, float* y,
                       int o0, int no, int layout) {
    int nn = n, ci = cin, co = cout, oo0 = o0, noo = no, lay = layout;
    void* a[] = {(void*)&x, (void*)&nn, (void*)&ci, (void*)&nb, (void*)&W, (void*)&b, (void*)&co, (void*)&y, (void*)&oo0, (void*)&noo, (void*)&lay};
    check(hip_launch(f_sp_subm_conv_f16w, (unsigned)((n + 15) / 16), (unsigned)((no + 15) / 16), 1, 16, 16, 1, 0, nullptr, a, nullptr),
          "sparse submanifold conv");
    sync();
}

// channel-to-spatial upsampling (sd_gpu.hip, src/sparse.c): see sd_device.h
void sp_c2s_mask(const float* logit, int n, unsigned char* bits, int* cnt) {
    int nn = n;
    void* a[] = {(void*)&logit, (void*)&nn, (void*)&bits, (void*)&cnt};
    launch(f_sp_c2s_mask, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void sp_exclusive_scan(const int* cnt, int n, int* off) {
    int nn = n;
    void* a[] = {(void*)&cnt, (void*)&nn, (void*)&off};
    launch(f_sp_exclusive_scan_serial, dim3(1, 1, 1), a);
    sync();
}

void sp_c2s_fill(const int* coords, const unsigned char* bits, const int* off, int n, int* new_coords, int* idx, int* subidx) {
    int nn = n;
    void* a[] = {(void*)&coords, (void*)&bits, (void*)&off, (void*)&nn, (void*)&new_coords, (void*)&idx, (void*)&subidx};
    launch(f_sp_c2s_fill, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void sp_c2s_gather(const float* src, const int* idx, const int* subidx, long M, int cc, int rep, float* out) {
    long mm = M;
    int c = cc, r = rep;
    void* a[] = {(void*)&src, (void*)&idx, (void*)&subidx, (void*)&mm, (void*)&c, (void*)&r, (void*)&out};
    long total = M * (long)cc * rep;
    launch(f_sp_c2s_gather, dim3((unsigned)((total + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

// the mesh step (sd_gpu.hip, src/trellis_mesh.cpp): see sd_device.h
void mesh_vertices(const int* coords, const float* feats, int n, float vox, float aabb0, float margin, float* v) {
    int nn = n;
    float vx = vox, a0 = aabb0, mg = margin;
    void* a[] = {(void*)&coords, (void*)&feats, (void*)&nn, (void*)&vx, (void*)&a0, (void*)&mg, (void*)&v};
    launch(f_mesh_vertices, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void mesh_edge_queries(const int* coords, int n, int* q) {
    int nn = n;
    void* a[] = {(void*)&coords, (void*)&nn, (void*)&q};
    launch(f_mesh_edge_queries, dim3((unsigned)((n * 12 + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void sp_lookup(const int* q, int nq, const unsigned long long* keys, const int* vals, unsigned int tsize, int* out) {
    unsigned int mask = tsize - 1;
    int nn = nq;
    void* a[] = {(void*)&q, (void*)&nn, (void*)&keys, (void*)&vals, (void*)&mask, (void*)&out};
    launch(f_sp_lookup, dim3((unsigned)((nq + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void mesh_edge_valid(const float* feats, int n, const int* look, int* valid) {
    int nn = n;
    void* a[] = {(void*)&feats, (void*)&nn, (void*)&look, (void*)&valid};
    launch(f_mesh_edge_valid, dim3((unsigned)((n * 3 + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void mesh_quads(const int* valid, const int* pos, const int* look, int ne, int* quads) {
    int nn = ne;
    void* a[] = {(void*)&valid, (void*)&pos, (void*)&look, (void*)&nn, (void*)&quads};
    launch(f_mesh_quads, dim3((unsigned)((ne + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void mesh_triangles(const int* quads, int L, const float* feats, int* faces) {
    int ll = L;
    void* a[] = {(void*)&quads, (void*)&ll, (void*)&feats, (void*)&faces};
    launch(f_mesh_triangles, dim3((unsigned)((L + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

// DINOv3's rotary embedding on the patch tokens (sd_gpu.hip dino_rope): x [T][C] with the heads of width D; cs, sn [P][D] for the P = T - prefix
// patch tokens
void dino_rope(float* x, int T, int prefix, int C, int H, int D, const float* cs, const float* sn) {
    const long n = (long)(T - prefix) * H * (D / 2);
    int tt = T, pf = prefix, cc = C, hh = H, dd = D;
    const float* c = cs;
    const float* sg = sn;
    void* a[] = {(void*)&x, (void*)&tt, (void*)&pf, (void*)&cc, (void*)&hh, (void*)&dd, (void*)&c, (void*)&sg};
    launch(f_dino_rope, dim3((unsigned)((n + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

// mesh simplification (sd_gpu.hip simp_*): one GPU step each over the faces, edges or vertices (trellis_simplify.cpp drives them)
void simp_qem(const float* v, const int* f, const int* v2f, const int* v2f_off, int V, float* qem) {
    int vv = V;
    void* a[] = {(void*)&v, (void*)&f, (void*)&v2f, (void*)&v2f_off, (void*)&vv, (void*)&qem};
    launch(f_simp_qem, dim3((unsigned)((V + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void simp_cost(const float* v, const int* f, const int* v2f, const int* v2f_off, const unsigned long long* edges, const unsigned char* vb,
               const float* qem, int E, float lam_len, float lam_sk, float* cost) {
    int ee = E;
    void* a[] = {(void*)&v, (void*)&f, (void*)&v2f, (void*)&v2f_off, (void*)&edges, (void*)&vb, (void*)&qem, (void*)&ee, (void*)&lam_len,
                 (void*)&lam_sk, (void*)&cost};
    launch(f_simp_cost, dim3((unsigned)((E + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void simp_propagate(const unsigned long long* edges, const int* v2f, const int* v2f_off, const float* cost, int E, unsigned long long* prop) {
    int ee = E;
    void* a[] = {(void*)&edges, (void*)&v2f, (void*)&v2f_off, (void*)&cost, (void*)&ee, (void*)&prop};
    launch(f_simp_propagate, dim3((unsigned)((E + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void simp_collapse(float* v, int* f, const unsigned long long* edges, const int* v2f, const int* v2f_off, const float* cost,
                   const unsigned long long* prop, const unsigned char* vb, int E, float thresh, int* vkept, int* fkept) {
    int ee = E;
    void* a[] = {(void*)&v, (void*)&f, (void*)&edges, (void*)&v2f, (void*)&v2f_off, (void*)&cost, (void*)&prop, (void*)&vb, (void*)&ee,
                 (void*)&thresh, (void*)&vkept, (void*)&fkept};
    launch(f_simp_collapse, dim3((unsigned)((E + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void simp_compress_v(const int* vmap, const float* oldv, int V, float* newv) {
    int vv = V;
    void* a[] = {(void*)&vmap, (void*)&oldv, (void*)&vv, (void*)&newv};
    launch(f_simp_compress_v, dim3((unsigned)((V + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void simp_compress_f(const int* fmap, const int* vmap, const int* oldf, int F, int* newf) {
    int ff = F;
    void* a[] = {(void*)&fmap, (void*)&vmap, (void*)&oldf, (void*)&ff, (void*)&newf};
    launch(f_simp_compress_f, dim3((unsigned)((F + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

// mesh charts (sd_gpu.hip chart_*): one GPU step each over the faces, edges or charts (trellis_charts.cpp drives them)
void chart_face_geom(const float* v, const int* f, int F, float* fn, float* fa) {
    int ff = F;
    void* a[] = {(void*)&v, (void*)&f, (void*)&ff, (void*)&fn, (void*)&fa};
    launch(f_chart_face_geom, dim3((unsigned)((F + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void chart_cones(const int* coff, const int* cface, const float* fn, const float* fa, int C, float* cn, float* carea) {
    int cc = C;
    void* a[] = {(void*)&coff, (void*)&cface, (void*)&fn, (void*)&fa, (void*)&cc, (void*)&cn, (void*)&carea};
    launch(f_chart_cones, dim3((unsigned)((C + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void chart_perims(const int* c2e_off, const int* c2e, const float* elen, int C, float* perim) {
    int cc = C;
    void* a[] = {(void*)&c2e_off, (void*)&c2e, (void*)&elen, (void*)&cc, (void*)&perim};
    launch(f_chart_perims, dim3((unsigned)((C + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void chart_costs(const unsigned long long* ek, const float* elen, const float* cn, const float* carea, const float* perim, float aw, float pw,
                 int E, float* cost) {
    int ee = E;
    void* a[] = {(void*)&ek, (void*)&elen, (void*)&cn, (void*)&carea, (void*)&perim, (void*)&aw, (void*)&pw, (void*)&ee, (void*)&cost};
    launch(f_chart_costs, dim3((unsigned)((E + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void chart_propagate(const int* c2e_off, const int* c2e, const float* cost, int C, unsigned long long* prop) {
    int cc = C;
    void* a[] = {(void*)&c2e_off, (void*)&c2e, (void*)&cost, (void*)&cc, (void*)&prop};
    launch(f_chart_propagate, dim3((unsigned)((C + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void chart_collapse(const unsigned long long* ek, const float* cost, const unsigned long long* prop, float thresh, int E, int* cmap, float* cn,
                    int* flag) {
    int ee = E;
    void* a[] = {(void*)&ek, (void*)&cost, (void*)&prop, (void*)&thresh, (void*)&ee, (void*)&cmap, (void*)&cn, (void*)&flag};
    launch(f_chart_collapse, dim3((unsigned)((E + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

void chart_refine(const float* cn, const float* fn, const float* v, const unsigned long long* ek, const int* f2e, const int* e2f,
                  const int* e2f_off, int F, float lam, const int* ids, int* out) {
    int ff = F;
    void* a[] = {(void*)&cn, (void*)&fn, (void*)&v, (void*)&ek, (void*)&f2e, (void*)&e2f, (void*)&e2f_off, (void*)&ff, (void*)&lam, (void*)&ids,
                 (void*)&out};
    launch(f_chart_refine, dim3((unsigned)((F + kThreads - 1) / kThreads), 1, 1), a);
    sync();
}

}  // namespace sddev
