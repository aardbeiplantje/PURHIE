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
              f_layer_norm = nullptr, f_geglu = nullptr;

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
        {&f_gemm_ah, "sd_gemm_ah"},     {&f_gemm_af, "sd_gemm_af"},     {&f_im2col, "sd_im2col"},
        {&f_gn_sum, "sd_gn_sum"},       {&f_gn_sq, "sd_gn_sq"},         {&f_gn_apply, "sd_gn_apply"},
        {&f_silu, "sd_silu"},           {&f_add, "sd_add"},             {&f_upsample2, "sd_upsample2"},
        {&f_softmax, "sd_softmax"},     {&f_layer_norm, "sd_layer_norm"}, {&f_geglu, "sd_geglu"},
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
          const uint16_t* bias, float alpha, float beta, int M, int N, int K) {
    if (M <= 0 || N <= 0) return;
    const TA* a = A;
    const float* b = B;
    float* c = C;
    const uint16_t* bi = bias;
    void* args[] = {(void*)&a, (void*)&lda, (void*)&b, (void*)&sbk, (void*)&sbn, (void*)&c, (void*)&cs_m, (void*)&cs_n,
                    (void*)&bi, (void*)&alpha, (void*)&beta, (void*)&M, (void*)&N, (void*)&K};
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

void* alloc(size_t bytes) {
    if (bytes == 0) return nullptr;
    const size_t need = round_up_page(bytes);
    if (!enabled()) {
        void* p = std::calloc(1, need);
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
            std::memset(p, 0, need);
            return p;
        }
    }
    void* p = nullptr;
    if (hipHostMalloc(&p, need, hipHostMallocDefault) != hipSuccess) throw std::runtime_error("sd device: out of pinned memory");
    std::memset(p, 0, need);
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

void linear_tok(const float* x, int T, const uint16_t* W, int in, int out, const uint16_t* b, float* y) {
    gemm<uint16_t>(f_gemm_ah, W, in, x, 1, in, y, 1, out, b, 1.0f, 0.0f, out, T, in);
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

void attention(const float* q, int Tq, const float* k, const float* v, int Tk, int C, int heads, float* out) {
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

}  // namespace sddev
