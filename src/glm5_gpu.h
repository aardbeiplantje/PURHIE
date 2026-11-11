// glm5_gpu.h -- launchers of the GLM-5.3-Flash kernels (glm5_gpu.hip: what each computes). All pointers are
// device-visible (the pinned scratch, state and the model mapping); every call is queued on `stream`.

#ifndef GLM5_GPU_H
#define GLM5_GPU_H

#include <hip/hip_runtime.h>

int glm5_gpu_init(const char* co_path);
void glm5_gpu_free(void);

hipError_t glm_hc_pre(const float* mix, const float* scale, const float* base, const float* streams, float* x,
                      float* post, float* comb, int n, int T, float eps, int iters, float rms_eps, hipStream_t stream);
hipError_t glm_hc_post(const float* x, const float* streams, const float* post, const float* comb, float* out, int n,
                       int T, hipStream_t stream);
hipError_t glm_hc_mean(const float* streams, float* x, int n, int T, hipStream_t stream);
hipError_t kda_conv(const float* proj, float* out, long long ld, float* state, const float* w0, const float* w1,
                    const float* w2, int n, int d_conv, int T, hipStream_t stream);
hipError_t kda_l2(float* qk, long long ld, int hd, int H, int T, float eps, hipStream_t stream);
hipError_t kda_rule(float* S, const float* qkv, long long ld, const float* gf, const float* dt, const float* A,
                    const float* beta, const float* g2, const float* norm_w, float* out, int H, int hd, float lb,
                    float eps, int T, hipStream_t stream);  // hd 128
hipError_t glm_swiglu_clamp(const float* g, const float* u, float* out, int rows, int h, float limit,
                            hipStream_t stream);
hipError_t glm_layernorm(const float* x, const float* g, const float* b, float* y, int n, int T, float eps,
                         hipStream_t stream);
hipError_t glm_pool(const float* ik, const float* ig, const float* ape, float* pooled, int j0, int npools, int kp,
                    int d, hipStream_t stream);
hipError_t glm_index_score(const float* iq, const float* w, const float* pooled, float* score, int ld_score,
                           int pos0, int T, int kp, int nh, int d, float ws, hipStream_t stream);
hipError_t glm_index_select(const float* score, int ld_score, int* sel, int ld_sel, int* nsel, int pos0, int T, int kp,
                            int n_top, int tail, hipStream_t stream);
// part_ml [T][H][9][2] and part_acc [T][H][9][kl] floats (GLM_MAX_SPLITS 9): the split partial results
hipError_t glm_mla_attn(const float* q_abs, const float* latent, const int* sel, int ld_sel, const int* nsel,
                        float* part_ml, float* part_acc, float* out, int H, int T, int kl, float scale,
                        hipStream_t stream);
static inline size_t glm_mla_part_floats(int H, int T, int kl) { return (size_t)T * H * 9 * (2 + (size_t)kl); }

#endif
