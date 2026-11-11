// glm5_gpu.hip -- the GLM-5.3-Flash (glm5-next) kernels of the forward pass (glm5_gpu.h). Their reference is
// src/glm5_ref.cpp (the same formulas on the CPU, itself checked against upstream llama.cpp's glm5-next):
//
// glm_rms_flat     y = x / sqrt(mean(x^2) + eps) over a row (the mHC mixer's input, no gamma)
// glm_hc_pre       from the mixes (24 a token): pre / post / comb (softmax + Sinkhorn), the block input
// glm_hc_post      the streams after a block: out[d] = x * post[d] + sum_s streams[s] * comb[d, s]
// glm_hc_mean      the mean of the streams (the final mixer before output_norm)
// kda_conv         KDA's causal conv1d + SiLU of q / k / v, tokens in order, the window state updated
// kda_l2           q / k L2-normed per head: x / sqrt(sum x^2 + eps)
// kda_rule         the delta rule with the per-channel decay, then RMSNorm * gamma * sigmoid(gate)
// glm_swiglu_clamp SiLU(min(g, L)) * clamp(u, -L, L) (L 0: plain SwiGLU)
// glm_layernorm    the indexer key's LayerNorm (gamma, beta)
// glm_pool         the pooled key of each completed k-pool: softmax(gate + ape) over its positions times keys
// glm_index_score  each token's score of each pool it can see
// glm_index_select the top pools (ties to the lower index) + the incomplete tail: the positions attended to
// glm_mla_attn     softmax(q_abs . latent * scale) over the selected positions, the latent-weighted sum
// glm_heads_swap   [T][H][D] <-> [H][T][D]

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

#define GLM_HC 4
#define GLM_MIX ((2 + GLM_HC) * GLM_HC)

__device__ __forceinline__ float glm_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float glm_silu(float x) { return x / (1.0f + expf(-x)); }

// The sum of v over the block (256 threads)
__device__ float glm_block_sum(float v, float* sh) {
    const int tid = threadIdx.x;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// grid T, block 256
extern "C" __global__ void __launch_bounds__(256) glm_rms_flat(const float* x, float* y, int n, float eps) {
    __shared__ float sh[256];
    x += (size_t)blockIdx.x * n;
    y += (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) s += x[i] * x[i];
    const float r = 1.0f / sqrtf(glm_block_sum(s, sh) / n + eps);
    for (int i = threadIdx.x; i < n; i += 256) y[i] = x[i] * r;
}

// grid T, block 256: mix [T][24], streams [T][4][n] -> x [T][n], post [T][4], comb [T][16] (comb[d + 4 s])
extern "C" __global__ void __launch_bounds__(256)
glm_hc_pre(const float* mix, const float* scale, const float* base, const float* streams, float* x, float* post,
           float* comb, int n, float eps, int iters) {
    __shared__ float pre[GLM_HC], cb[GLM_HC * GLM_HC];
    const int t = blockIdx.x;
    mix += (size_t)t * GLM_MIX;
    if (threadIdx.x == 0) {
        for (int i = 0; i < GLM_HC; ++i) {
            pre[i] = glm_sigmoid(mix[i] * scale[0] + base[i]) + eps;
            post[(size_t)t * GLM_HC + i] = 2.0f * glm_sigmoid(mix[GLM_HC + i] * scale[1] + base[GLM_HC + i]);
        }
        for (int s = 0; s < GLM_HC; ++s) {
            float mx = -INFINITY, sum = 0.0f;
            for (int d = 0; d < GLM_HC; ++d) {
                const int idx = d + GLM_HC * s;
                cb[idx] = mix[2 * GLM_HC + idx] * scale[2] + base[2 * GLM_HC + idx];
                mx = fmaxf(mx, cb[idx]);
            }
            for (int d = 0; d < GLM_HC; ++d) sum += cb[d + GLM_HC * s] = expf(cb[d + GLM_HC * s] - mx);
            const float inv = 1.0f / sum;
            for (int d = 0; d < GLM_HC; ++d) cb[d + GLM_HC * s] = cb[d + GLM_HC * s] * inv + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0)  // rows: over dst for each src
                for (int s = 0; s < GLM_HC; ++s) {
                    float sum = eps;
                    for (int d = 0; d < GLM_HC; ++d) sum += cb[d + GLM_HC * s];
                    const float inv = 1.0f / sum;
                    for (int d = 0; d < GLM_HC; ++d) cb[d + GLM_HC * s] *= inv;
                }
            for (int d = 0; d < GLM_HC; ++d) {  // columns: over src for each dst
                float sum = eps;
                for (int s = 0; s < GLM_HC; ++s) sum += cb[d + GLM_HC * s];
                const float inv = 1.0f / sum;
                for (int s = 0; s < GLM_HC; ++s) cb[d + GLM_HC * s] *= inv;
            }
        }
        for (int i = 0; i < GLM_HC * GLM_HC; ++i) comb[(size_t)t * GLM_HC * GLM_HC + i] = cb[i];
    }
    __syncthreads();
    streams += (size_t)t * GLM_HC * n;
    for (int e = threadIdx.x; e < n; e += 256) {
        float v = 0.0f;
        for (int i = 0; i < GLM_HC; ++i) v += pre[i] * streams[(size_t)i * n + e];
        x[(size_t)t * n + e] = v;
    }
}

// grid (ceil(4 n / 256), T), block 256
extern "C" __global__ void __launch_bounds__(256)
glm_hc_post(const float* x, const float* streams, const float* post, const float* comb, float* out, int n) {
    const int t = blockIdx.y, i = blockIdx.x * 256 + threadIdx.x;
    if (i >= GLM_HC * n) return;
    const int d = i / n, e = i % n;
    const float* st = streams + (size_t)t * GLM_HC * n;
    const float* cb = comb + (size_t)t * GLM_HC * GLM_HC;
    float v = x[(size_t)t * n + e] * post[(size_t)t * GLM_HC + d];
    for (int s = 0; s < GLM_HC; ++s) v += st[(size_t)s * n + e] * cb[d + GLM_HC * s];
    out[(size_t)t * GLM_HC * n + i] = v;
}

// grid (ceil(n / 256), T), block 256
extern "C" __global__ void __launch_bounds__(256) glm_hc_mean(const float* streams, float* x, int n) {
    const int t = blockIdx.y, e = blockIdx.x * 256 + threadIdx.x;
    if (e >= n) return;
    const float* st = streams + (size_t)t * GLM_HC * n;
    float v = 0.0f;
    for (int s = 0; s < GLM_HC; ++s) v += st[(size_t)s * n + e];
    x[(size_t)t * n + e] = v * (1.0f / GLM_HC);
}

// grid (ceil(n / 256), 3), block 256: proj / out [3][ld] (c = q, k, v; rows of n per token), state [3][n][d_conv - 1]
#define KDA_MAXC 8
extern "C" __global__ void __launch_bounds__(256)
kda_conv(const float* proj, float* out, long long ld, float* state, const float* w0, const float* w1, const float* w2,
         int n, int d_conv, int T) {
    const int ch = blockIdx.x * 256 + threadIdx.x, c = blockIdx.y, ns = d_conv - 1;
    if (ch >= n) return;
    const float* w = (c == 0 ? w0 : c == 1 ? w1 : w2) + (size_t)ch * d_conv;
    float* st = state + ((size_t)c * n + ch) * ns;
    float h[KDA_MAXC];
    for (int k = 0; k < ns; ++k) h[k] = st[k];
    for (int t = 0; t < T; ++t) {
        const float p = proj[(size_t)c * ld + (size_t)t * n + ch];
        float v = w[ns] * p;
        for (int k = 0; k < ns; ++k) v += w[k] * h[k];
        for (int k = 0; k + 1 < ns; ++k) h[k] = h[k + 1];
        if (ns > 0) h[ns - 1] = p;
        out[(size_t)c * ld + (size_t)t * n + ch] = glm_silu(v);
    }
    for (int k = 0; k < ns; ++k) st[k] = h[k];
}

// grid (H, T, 2), block 256: rows [2][ld] (q, k), heads of hd values
extern "C" __global__ void __launch_bounds__(256) kda_l2(float* qk, long long ld, int hd, int H, float eps) {
    __shared__ float sh[256];
    float* v = qk + (size_t)blockIdx.z * ld + ((size_t)blockIdx.y * H + blockIdx.x) * hd;
    float s = 0.0f;
    for (int i = threadIdx.x; i < hd; i += 256) s += v[i] * v[i];
    const float r = 1.0f / sqrtf(glm_block_sum(s, sh) + eps);
    for (int i = threadIdx.x; i < hd; i += 256) v[i] *= r;
}

// grid H, block hd (<= 256, a power of two): head h of T tokens in order. qkv [3][ld] after conv / L2; gf: the
// f_b output [T][H * hd]; dt [H * hd], A [H] (-exp(A_log)); beta: raw [T][H]; g2: the g_b output [T][H * hd];
// S [H][hd][hd] (S[i][j], i the key channel); out [T][H * hd]. lb: the gate's lower bound (-inf: softplus * A).
extern "C" __global__ void __launch_bounds__(256)
kda_rule(float* S, const float* qkv, long long ld, const float* gf, const float* dt, const float* A, const float* beta,
         const float* g2, const float* norm_w, float* out, int H, float lb, float eps, int T) {
    __shared__ float kk[256], qq[256], dd[256], sh[256];
    const int h = blockIdx.x, j = threadIdx.x, hd = blockDim.x, DI = H * hd;
    float* M = S + (size_t)h * hd * hd;
    const float scale = 1.0f / sqrtf((float)hd), a = A[h];
    for (int t = 0; t < T; ++t) {
        const size_t o = (size_t)t * DI + (size_t)h * hd;
        qq[j] = qkv[o + j];
        kk[j] = qkv[ld + o + j];
        const float g1 = gf[o + j] + dt[h * hd + j];
        dd[j] = expf(lb > -INFINITY ? lb * glm_sigmoid(-(g1 * a)) : (g1 > 20.0f ? g1 : log1pf(expf(g1))) * a);
        const float kq = glm_block_sum(kk[j] * qq[j], sh);  // (also syncs the loads above)
        const float b = glm_sigmoid(beta[(size_t)t * H + h]);
        float sa = 0.0f, sb = 0.0f;
        for (int i = 0; i < hd; ++i) {
            const float m = M[(size_t)i * hd + j] * dd[i];
            sa += m * kk[i];
            sb += m * qq[i];
        }
        const float delta = (qkv[2 * ld + o + j] - sa) * b;
        for (int i = 0; i < hd; ++i) M[(size_t)i * hd + j] = M[(size_t)i * hd + j] * dd[i] + kk[i] * delta;
        const float y = (sb + delta * kq) * scale;
        const float r = 1.0f / sqrtf(glm_block_sum(y * y, sh) / hd + eps);
        out[o + j] = y * r * norm_w[j] * glm_sigmoid(g2[o + j]);
        __syncthreads();
    }
}

// grid ceil(rows * h / 256), block 256: rows of h values (gate / up rows at stride h)
extern "C" __global__ void __launch_bounds__(256)
glm_swiglu_clamp(const float* g, const float* u, float* out, int rows, int h, float limit) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long long)rows * h) return;
    float gv = g[i], uv = u[i];
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    out[i] = glm_silu(gv) * uv;
}

// grid T, block 256
extern "C" __global__ void __launch_bounds__(256)
glm_layernorm(const float* x, const float* g, const float* b, float* y, int n, float eps) {
    __shared__ float sh[256];
    x += (size_t)blockIdx.x * n;
    y += (size_t)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) s += x[i];
    const float mean = glm_block_sum(s, sh) / n;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) v += (x[i] - mean) * (x[i] - mean);
    const float r = 1.0f / sqrtf(glm_block_sum(v, sh) / n + eps);
    for (int i = threadIdx.x; i < n; i += 256) y[i] = (x[i] - mean) * r * g[i] + b[i];
}

// grid npools, block 256: pools j0 .. j0 + npools - 1 of kp positions each; ik / ig [pos][d] (the cache rows), ape
// [kp][d] float, pooled [pool][d]
extern "C" __global__ void __launch_bounds__(256)
glm_pool(const float* ik, const float* ig, const float* ape, float* pooled, int j0, int kp, int d_n) {
    const int j = j0 + blockIdx.x;
    for (int d = threadIdx.x; d < d_n; d += 256) {
        float lg[16], mx = -INFINITY, sum = 0.0f, v = 0.0f;
        for (int m = 0; m < kp; ++m) {
            lg[m] = ig[((size_t)j * kp + m) * d_n + d] + ape[(size_t)m * d_n + d];
            mx = fmaxf(mx, lg[m]);
        }
        for (int m = 0; m < kp; ++m) sum += lg[m] = expf(lg[m] - mx);
        for (int m = 0; m < kp; ++m) v += lg[m] / sum * ik[((size_t)j * kp + m) * d_n + d];
        pooled[(size_t)j * d_n + d] = v;
    }
}

// grid (ceil(max_pools / 256), T), block 256: token t at pos0 + t sees pools j < (pos0 + t + 1) / kp. iq [T][nh][d],
// w [T][nh] (the indexer.proj output), pooled [pool][d]; score [T][ld_score]
extern "C" __global__ void __launch_bounds__(256)
glm_index_score(const float* iq, const float* w, const float* pooled, float* score, int ld_score, int pos0, int kp,
                int nh, int d_n, float ws) {
    const int t = blockIdx.y, j = blockIdx.x * 256 + threadIdx.x;
    if (j >= (pos0 + t + 1) / kp) return;
    const float* q = iq + (size_t)t * nh * d_n;
    const float* k = pooled + (size_t)j * d_n;
    float sc = 0.0f;
    for (int h = 0; h < nh; ++h) {
        float d = 0.0f;
        for (int c = 0; c < d_n; ++c) d += q[(size_t)h * d_n + c] * k[c];
        sc += fmaxf(d, 0.0f) * w[(size_t)t * nh + h] * ws;
    }
    score[(size_t)t * ld_score + j] = sc;
}

// A float as an unsigned key that orders like the float (larger float, larger key)
__device__ __forceinline__ uint32_t glm_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// grid T, block 256: the positions token t (at pos0 + t) attends to, ascending: the cells of its top
// min(nv, n_top) pools by score (equal scores: the lower pool index first) and, with tail, the (pos + 1) % kp
// positions of its incomplete pool. sel [T][ld_sel], nsel [T].
extern "C" __global__ void __launch_bounds__(256)
glm_index_select(const float* score, int ld_score, int* sel, int ld_sel, int* nsel, int pos0, int kp, int n_top,
                 int tail) {
    __shared__ int cnt[256];
    __shared__ uint32_t thr_sh;
    __shared__ int base_sh;
    const int t = blockIdx.x, tid = threadIdx.x, pos = pos0 + t;
    const int nv = (pos + 1) / kp, nt = min(nv, n_top);
    const float* sc = score + (size_t)t * ld_score;
    int* out = sel + (size_t)t * ld_sel;
    // the nt-th largest key: the largest thr with count(key >= thr) >= nt, bit by bit from the top
    uint32_t thr = 0;
    if (nt > 0 && nt < nv) {
        for (int bit = 31; bit >= 0; --bit) {
            const uint32_t cand = thr | (1u << bit);
            int c = 0;
            for (int j = tid; j < nv; j += 256) c += glm_key(sc[j]) >= cand;
            cnt[tid] = c;
            __syncthreads();
            for (int s = 128; s > 0; s >>= 1) {
                if (tid < s) cnt[tid] += cnt[tid + s];
                __syncthreads();
            }
            if (cnt[0] >= nt) thr = cand;
            __syncthreads();
        }
    }
    // how many equal to thr are kept: nt - count(key > thr)
    int need_eq = 0;
    if (nt > 0 && nt < nv) {
        int c = 0;
        for (int j = tid; j < nv; j += 256) c += glm_key(sc[j]) > thr;
        cnt[tid] = c;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) cnt[tid] += cnt[tid + s];
            __syncthreads();
        }
        need_eq = nt - cnt[0];
        __syncthreads();
    }
    if (tid == 0) thr_sh = thr, base_sh = 0;
    __syncthreads();
    // in index order, chunks of 256: keep key > thr, and key == thr while fewer than need_eq were kept
    int eq_seen = 0;
    for (int j0 = 0; j0 < nv; j0 += 256) {
        const int j = j0 + tid;
        const uint32_t k = j < nv ? glm_key(sc[j]) : 0;
        const bool all = nt >= nv;
        const bool eq = !all && j < nv && k == thr_sh;
        cnt[tid] = eq;  // the equal ones before this thread (exclusive scan)
        __syncthreads();
        int before = 0;
        for (int i = 0; i < tid; ++i) before += cnt[i];
        int total_eq = 0;
        for (int i = 0; i < 256; ++i) total_eq += cnt[i];
        __syncthreads();
        const bool keep = j < nv && (all || k > thr_sh || (eq && eq_seen + before < need_eq));
        cnt[tid] = keep;
        __syncthreads();
        int at = 0;
        for (int i = 0; i < tid; ++i) at += cnt[i];
        int kept = 0;
        for (int i = 0; i < 256; ++i) kept += cnt[i];
        if (keep)
            for (int m = 0; m < kp; ++m) out[(base_sh + at) * kp + m] = j * kp + m;
        __syncthreads();
        if (tid == 0) base_sh += kept;
        eq_seen += total_eq;
        __syncthreads();
    }
    if (tid == 0) {
        int n = base_sh * kp;
        if (tail)
            for (int k = (pos + 1) % kp - 1; k >= 0; --k) out[n++] = pos - k;
        nsel[t] = n;
    }
}

// grid (H, T), block 256: q_abs [H][T][kl], latent [pos][kl], sel / nsel as glm_index_select's; out [H][T][kl]
#define GLM_MAX_SEL 4096
extern "C" __global__ void __launch_bounds__(256)
glm_mla_attn(const float* q_abs, const float* latent, const int* sel, int ld_sel, const int* nsel, float* out, int T,
             int kl, float scale) {
    __shared__ float p[GLM_MAX_SEL];
    __shared__ float sh[256];
    const int h = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const float* q = q_abs + ((size_t)h * T + t) * kl;
    const int* s = sel + (size_t)t * ld_sel;
    const int n = min(nsel[t], GLM_MAX_SEL);
    float mx = -INFINITY;
    for (int i = tid; i < n; i += 256) {
        const float* c = latent + (size_t)s[i] * kl;
        float d = 0.0f;
        for (int k = 0; k < kl; ++k) d += q[k] * c[k];
        p[i] = d * scale;
        mx = fmaxf(mx, p[i]);
    }
    sh[tid] = mx;
    __syncthreads();
    for (int r = 128; r > 0; r >>= 1) {
        if (tid < r) sh[tid] = fmaxf(sh[tid], sh[tid + r]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i < n; i += 256) sum += p[i] = expf(p[i] - mx);
    sum = glm_block_sum(sum, sh);
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    float* o = out + ((size_t)h * T + t) * kl;
    for (int k = tid; k < kl; k += 256) {
        float v = 0.0f;
        for (int i = 0; i < n; ++i) v += p[i] * latent[(size_t)s[i] * kl + k];
        o[k] = v * inv;
    }
}

// grid (ceil(D / 256), H, T), block 256: to_ht 1: in [T][H][D] -> out [H][T][D]; 0: the reverse
extern "C" __global__ void __launch_bounds__(256)
glm_heads_swap(const float* in, float* out, int H, int T, int D, int to_ht) {
    const int d = blockIdx.x * 256 + threadIdx.x, h = blockIdx.y, t = blockIdx.z;
    if (d >= D) return;
    const size_t th = ((size_t)t * H + h) * D + d, ht = ((size_t)h * T + t) * D + d;
    if (to_ht) out[ht] = in[th];
    else out[th] = in[ht];
}
