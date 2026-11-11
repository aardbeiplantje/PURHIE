// glm5_gpu.cpp -- see glm5_gpu.h.

#include "glm5_gpu.h"

#include "inference_log.h"
#include "kernel_loader.h"

static hipModule_t g_mod = nullptr;
static hipFunction_t f_rms_flat, f_hc_pre, f_hc_post, f_hc_mean, f_kda_conv, f_kda_l2, f_kda_rule, f_swiglu,
    f_layernorm, f_pool, f_score, f_select, f_mla, f_swap;

int glm5_gpu_init(const char* co_path) {
    if (g_mod) return 0;
    if (kernel_module_load(&g_mod, co_path) != hipSuccess) {
        INFER_LOG("[glm5_gpu] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    const struct { hipFunction_t* f; const char* name; } fs[] = {
        {&f_rms_flat, "glm_rms_flat"}, {&f_hc_pre, "glm_hc_pre"}, {&f_hc_post, "glm_hc_post"},
        {&f_hc_mean, "glm_hc_mean"}, {&f_kda_conv, "kda_conv"}, {&f_kda_l2, "kda_l2"}, {&f_kda_rule, "kda_rule"},
        {&f_swiglu, "glm_swiglu_clamp"}, {&f_layernorm, "glm_layernorm"}, {&f_pool, "glm_pool"},
        {&f_score, "glm_index_score"}, {&f_select, "glm_index_select"}, {&f_mla, "glm_mla_attn"},
        {&f_swap, "glm_heads_swap"}};
    for (const auto& e : fs)
        if (hipModuleGetFunction(e.f, g_mod, e.name) != hipSuccess) {
            INFER_LOG("[glm5_gpu] kernel %s missing in %s\n", e.name, co_path);
            glm5_gpu_free();
            return -1;
        }
    return 0;
}

void glm5_gpu_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
}

static unsigned blocks(long long n) { return (unsigned)((n + 255) / 256); }

static hipError_t launch(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, void** args,
                         hipStream_t stream) {
    if (!g_mod) return hipErrorInvalidDeviceFunction;
    if (gx == 0 || gy == 0 || gz == 0) return hipSuccess;  // nothing to do
    return hipModuleLaunchKernel(f, gx, gy, gz, bx, 1, 1, 0, stream, args, nullptr);
}

hipError_t glm_rms_flat(const float* x, float* y, int n, int T, float eps, hipStream_t stream) {
    void* a[] = {(void*)&x, &y, &n, &eps};
    return launch(f_rms_flat, (unsigned)T, 1, 1, 256, a, stream);
}

hipError_t glm_hc_pre(const float* mix, const float* scale, const float* base, const float* streams, float* x,
                      float* post, float* comb, int n, int T, float eps, int iters, hipStream_t stream) {
    void* a[] = {(void*)&mix, (void*)&scale, (void*)&base, (void*)&streams, &x, &post, &comb, &n, &eps, &iters};
    return launch(f_hc_pre, (unsigned)T, 1, 1, 256, a, stream);
}

hipError_t glm_hc_post(const float* x, const float* streams, const float* post, const float* comb, float* out, int n,
                       int T, hipStream_t stream) {
    void* a[] = {(void*)&x, (void*)&streams, (void*)&post, (void*)&comb, &out, &n};
    return launch(f_hc_post, blocks(4LL * n), (unsigned)T, 1, 256, a, stream);
}

hipError_t glm_hc_mean(const float* streams, float* x, int n, int T, hipStream_t stream) {
    void* a[] = {(void*)&streams, &x, &n};
    return launch(f_hc_mean, blocks(n), (unsigned)T, 1, 256, a, stream);
}

hipError_t kda_conv(const float* proj, float* out, long long ld, float* state, const float* w0, const float* w1,
                    const float* w2, int n, int d_conv, int T, hipStream_t stream) {
    if (d_conv < 1 || d_conv > 9) return hipErrorInvalidValue;  // KDA_MAXC + 1
    void* a[] = {(void*)&proj, &out, &ld, &state, (void*)&w0, (void*)&w1, (void*)&w2, &n, &d_conv, &T};
    return launch(f_kda_conv, blocks(n), 3, 1, 256, a, stream);
}

hipError_t kda_l2(float* qk, long long ld, int hd, int H, int T, float eps, hipStream_t stream) {
    void* a[] = {&qk, &ld, &hd, &H, &eps};
    return launch(f_kda_l2, (unsigned)H, (unsigned)T, 2, 256, a, stream);
}

hipError_t kda_rule(float* S, const float* qkv, long long ld, const float* gf, const float* dt, const float* A,
                    const float* beta, const float* g2, const float* norm_w, float* out, int H, int hd, float lb,
                    float eps, int T, hipStream_t stream) {
    if (hd < 2 || hd > 256 || (hd & (hd - 1))) return hipErrorInvalidValue;  // a block per head, a thread per column
    void* a[] = {&S, (void*)&qkv, &ld, (void*)&gf, (void*)&dt, (void*)&A, (void*)&beta, (void*)&g2, (void*)&norm_w,
                 &out, &H, &lb, &eps, &T};
    return launch(f_kda_rule, (unsigned)H, 1, 1, (unsigned)hd, a, stream);
}

hipError_t glm_swiglu_clamp(const float* g, const float* u, float* out, int rows, int h, float limit,
                            hipStream_t stream) {
    void* a[] = {(void*)&g, (void*)&u, &out, &rows, &h, &limit};
    return launch(f_swiglu, blocks((long long)rows * h), 1, 1, 256, a, stream);
}

hipError_t glm_layernorm(const float* x, const float* g, const float* b, float* y, int n, int T, float eps,
                         hipStream_t stream) {
    void* a[] = {(void*)&x, (void*)&g, (void*)&b, &y, &n, &eps};
    return launch(f_layernorm, (unsigned)T, 1, 1, 256, a, stream);
}

hipError_t glm_pool(const float* ik, const float* ig, const float* ape, float* pooled, int j0, int npools, int kp,
                    int d, hipStream_t stream) {
    if (kp < 1 || kp > 16) return hipErrorInvalidValue;
    void* a[] = {(void*)&ik, (void*)&ig, (void*)&ape, &pooled, &j0, &kp, &d};
    return launch(f_pool, (unsigned)(npools > 0 ? npools : 0), 1, 1, 256, a, stream);
}

hipError_t glm_index_score(const float* iq, const float* w, const float* pooled, float* score, int ld_score,
                           int pos0, int T, int kp, int nh, int d, float ws, hipStream_t stream) {
    const int max_pools = (pos0 + T) / kp;
    void* a[] = {(void*)&iq, (void*)&w, (void*)&pooled, &score, &ld_score, &pos0, &kp, &nh, &d, &ws};
    return launch(f_score, blocks(max_pools), (unsigned)T, 1, 256, a, stream);
}

hipError_t glm_index_select(const float* score, int ld_score, int* sel, int ld_sel, int* nsel, int pos0, int T, int kp,
                            int n_top, int tail, hipStream_t stream) {
    void* a[] = {(void*)&score, &ld_score, &sel, &ld_sel, &nsel, &pos0, &kp, &n_top, &tail};
    return launch(f_select, (unsigned)T, 1, 1, 256, a, stream);
}

hipError_t glm_mla_attn(const float* q_abs, const float* latent, const int* sel, int ld_sel, const int* nsel,
                        float* out, int H, int T, int kl, float scale, hipStream_t stream) {
    if (ld_sel > 4096) return hipErrorInvalidValue;  // GLM_MAX_SEL
    void* a[] = {(void*)&q_abs, (void*)&latent, (void*)&sel, &ld_sel, (void*)&nsel, &out, &T, &kl, &scale};
    return launch(f_mla, (unsigned)H, (unsigned)T, 1, 256, a, stream);
}

hipError_t glm_heads_swap(const float* in, float* out, int H, int T, int D, int to_ht, hipStream_t stream) {
    void* a[] = {(void*)&in, &out, &H, &T, &D, &to_ht};
    return launch(f_swap, blocks(D), (unsigned)H, (unsigned)T, 256, a, stream);
}
