// glm5_gpu.hip -- the GLM-5.3-Flash (glm5-next) kernels of the forward pass (glm5_gpu.h). Their reference is
// src/glm5_ref.cpp (the same formulas on the CPU, itself checked against upstream llama.cpp's glm5-next):
//
// glm_hc_pre       from the mixes (24 a token, of the raw streams; the RMS here): pre / post / comb (softmax +
//                  Sinkhorn), the block input
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

// grid (ceil(n / 256), T), block 256 (every block computes the token's coefficients alike, cache-resident work, and
// writes its 256 values of x; block 0 also post and comb): mix [T][24] = fn . streams (the raw streams: fn . (x r) = r (fn . x), r = 1 / sqrt(mean(x^2) +
// rms_eps) over all 4 n values, computed here), streams [T][4][n] -> x [T][n], post [T][4], comb [T][16]
// (comb[d + 4 s])
extern "C" __global__ void __launch_bounds__(256)
glm_hc_pre(const float* mix, const float* scale, const float* base, const float* streams, float* x, float* post,
           float* comb, int n, float eps, int iters, float rms_eps) {
    __shared__ float pre[GLM_HC], cb[GLM_HC * GLM_HC], sh[256];
    const int t = blockIdx.y;
    mix += (size_t)t * GLM_MIX;
    const float* st = streams + (size_t)t * GLM_HC * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < GLM_HC * n; i += 256) ss += st[i] * st[i];
    const float r = 1.0f / sqrtf(glm_block_sum(ss, sh) / (GLM_HC * n) + rms_eps);
    const bool first = blockIdx.x == 0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < GLM_HC; ++i) {
            pre[i] = glm_sigmoid(mix[i] * r * scale[0] + base[i]) + eps;
            if (first) post[(size_t)t * GLM_HC + i] = 2.0f * glm_sigmoid(mix[GLM_HC + i] * r * scale[1] + base[GLM_HC + i]);
        }
        for (int s = 0; s < GLM_HC; ++s) {
            float mx = -INFINITY, sum = 0.0f;
            for (int d = 0; d < GLM_HC; ++d) {
                const int idx = d + GLM_HC * s;
                cb[idx] = mix[2 * GLM_HC + idx] * r * scale[2] + base[2 * GLM_HC + idx];
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
        if (first)
            for (int i = 0; i < GLM_HC * GLM_HC; ++i) comb[(size_t)t * GLM_HC * GLM_HC + i] = cb[i];
    }
    __syncthreads();
    for (int e = blockIdx.x * 256 + threadIdx.x; e < n; e += 256 * gridDim.x) {
        float v = 0.0f;
        for (int i = 0; i < GLM_HC; ++i) v += pre[i] * st[(size_t)i * n + e];
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

// grid H, block KDA_HD (128 threads, the head size): head h of T tokens in order. qkv [3][ld] after conv / L2; gf: the
// f_b output [T][H * hd]; dt [H * hd], A [H] (-exp(A_log)); beta: raw [T][H]; g2: the g_b output [T][H * hd];
// S [H][hd][hd] (S[i][j], i the key channel); out [T][H * hd]. lb: the gate's lower bound (-inf: softplus * A).
// Thread j keeps column j of the head's state in registers across the tokens: read once, written once.
#define KDA_HD 128
extern "C" __global__ void __launch_bounds__(KDA_HD)
kda_rule(float* S, const float* qkv, long long ld, const float* gf, const float* dt, const float* A, const float* beta,
         const float* g2, const float* norm_w, float* out, int H, float lb, float eps, int T) {
    __shared__ float kk[KDA_HD], qq[KDA_HD], dd[KDA_HD], sh[KDA_HD];
    const int h = blockIdx.x, j = threadIdx.x, DI = H * KDA_HD;
    float* M = S + (size_t)h * KDA_HD * KDA_HD;
    float col[KDA_HD];
#pragma unroll
    for (int i = 0; i < KDA_HD; ++i) col[i] = M[(size_t)i * KDA_HD + j];
    const float scale = 1.0f / sqrtf((float)KDA_HD), a = A[h];
    for (int t = 0; t < T; ++t) {
        const size_t o = (size_t)t * DI + (size_t)h * KDA_HD;
        qq[j] = qkv[o + j];
        kk[j] = qkv[ld + o + j];
        const float g1 = gf[o + j] + dt[h * KDA_HD + j];
        dd[j] = expf(lb > -INFINITY ? lb * glm_sigmoid(-(g1 * a)) : (g1 > 20.0f ? g1 : log1pf(expf(g1))) * a);
        const float kq = glm_block_sum(kk[j] * qq[j], sh);  // (also syncs the loads above)
        const float b = glm_sigmoid(beta[(size_t)t * H + h]);
        float sa = 0.0f, sb = 0.0f;
#pragma unroll
        for (int i = 0; i < KDA_HD; ++i) {
            col[i] *= dd[i];
            sa += col[i] * kk[i];
            sb += col[i] * qq[i];
        }
        const float delta = (qkv[2 * ld + o + j] - sa) * b;
#pragma unroll
        for (int i = 0; i < KDA_HD; ++i) col[i] += kk[i] * delta;
        const float y = (sb + delta * kq) * scale;
        const float r = 1.0f / sqrtf(glm_block_sum(y * y, sh) / KDA_HD + eps);
        out[o + j] = y * r * norm_w[j] * glm_sigmoid(g2[o + j]);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < KDA_HD; ++i) M[(size_t)i * KDA_HD + j] = col[i];
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

// grid (ceil(max_pools / 64), T), block 256: token t at pos0 + t sees pools j < (pos0 + t + 1) / kp. iq [T][nh][d],
// w [T][nh] (the indexer.proj output), pooled [pool][d]; score [T][ld_score]. The token's queries in shared memory;
// a wave (32 lanes) per pool, each lane d / 32 of its key's values, one wave reduction per head. d % 32 == 0,
// nh * d <= 8192.
#define GLM_IX_MAXQ 8192
extern "C" __global__ void __launch_bounds__(256)
glm_index_score(const float* iq, const float* w, const float* pooled, float* score, int ld_score, int pos0, int kp,
                int nh, int d_n, float ws) {
    __shared__ float qs[GLM_IX_MAXQ], wsh[64];
    const int t = blockIdx.y, nv = (pos0 + t + 1) / kp, lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int j0 = blockIdx.x * 64;
    if (j0 >= nv) return;
    for (int i = threadIdx.x; i < nh * d_n; i += 256) qs[i] = iq[(size_t)t * nh * d_n + i];
    for (int h = threadIdx.x; h < nh; h += 256) wsh[h] = w[(size_t)t * nh + h] * ws;
    __syncthreads();
    const int per = d_n / 32;
    for (int j = j0 + wave; j < min(j0 + 64, nv); j += 8) {
        float k[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) k[i] = i < per ? pooled[(size_t)j * d_n + lane + 32 * i] : 0.0f;
        float sc = 0.0f;
        for (int h = 0; h < nh; ++h) {
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (i < per) d += qs[h * d_n + lane + 32 * i] * k[i];
            for (int o = 16; o > 0; o >>= 1) d += __shfl_xor(d, o, 32);
            sc += fmaxf(d, 0.0f) * wsh[h];
        }
        if (lane == 0) score[(size_t)t * ld_score + j] = sc;
    }
}

// A float as an unsigned key that orders like the float (larger float, larger key)
__device__ __forceinline__ uint32_t glm_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive prefix sum of v over the block (256 threads) in buf; afterwards buf[i] holds the inclusive sum shifted:
// buf[tid] = sum of v over threads < tid, and the block's total is buf[255] + the last thread's own v (eq_last)
__device__ int glm_block_excl_scan(int v, int* buf) {
    const int tid = threadIdx.x;
    buf[tid] = v;
    __syncthreads();
    for (int o = 1; o < 256; o <<= 1) {  // Hillis-Steele inclusive scan
        const int a = tid >= o ? buf[tid - o] : 0;
        __syncthreads();
        buf[tid] += a;
        __syncthreads();
    }
    const int incl = buf[tid];
    __syncthreads();
    buf[tid] = incl - v;  // exclusive
    if (tid == 255) buf[256] = v;
    __syncthreads();
    return incl - v;
}
__device__ __forceinline__ int eq_last(const int* buf) { return buf[256]; }

// grid T, block 256: the positions token t (at pos0 + t) attends to, ascending: the cells of its top
// min(nv, n_top) pools by score (equal scores: the lower pool index first) and, with tail, the (pos + 1) % kp
// positions of its incomplete pool. sel [T][ld_sel], nsel [T].
extern "C" __global__ void __launch_bounds__(256)
glm_index_select(const float* score, int ld_score, int* sel, int ld_sel, int* nsel, int pos0, int kp, int n_top,
                 int tail) {
    __shared__ int cnt[257];
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
    const bool all = nt >= nv;
    for (int j0 = 0; j0 < nv; j0 += 256) {
        const int j = j0 + tid;
        const uint32_t k = j < nv ? glm_key(sc[j]) : 0;
        const int eq = !all && j < nv && k == thr_sh;
        const int eq_before = glm_block_excl_scan(eq, cnt) ;  // the equal ones before this thread in the chunk
        const int eq_total = cnt[255] + eq_last(cnt);
        __syncthreads();
        const int keep = j < nv && (all || k > thr_sh || (eq && eq_seen + eq_before < need_eq));
        const int at = glm_block_excl_scan(keep, cnt);
        const int kept = cnt[255] + eq_last(cnt);
        if (keep)
            for (int m = 0; m < kp; ++m) out[(base_sh + at) * kp + m] = j * kp + m;
        __syncthreads();
        if (tid == 0) base_sh += kept;
        eq_seen += eq_total;
        __syncthreads();
    }
    if (tid == 0) {
        int n = base_sh * kp;
        if (tail)
            for (int k = (pos + 1) % kp - 1; k >= 0; --k) out[n++] = pos - k;
        nsel[t] = n;
    }
}

// Sparse MLA attention in two kernels. glm_mla_part: grid (H / hg, T, ceil(n_max / 256)), block 256: heads
// hg * blockIdx.x .. + hg - 1 (hg <= 4) of token t over its selected positions [256 z, 256 z + 256) (sel / nsel
// as glm_index_select's): q_abs [T][H][kl], latent [pos][kl]. The heads' queries in shared memory; each wave takes
// every 8th position, reads its latent row once (coalesced, kl / 32 values a lane) for all hg heads and keeps an
// online softmax (max, sum, the weighted latent sum in registers); the 8 waves combine in a fixed order. Out:
// part_ml [T][H][GLM_MAX_SPLITS][2] (max, sum) and part_acc [T][H][GLM_MAX_SPLITS][kl] (unnormalized). The split
// depends only on the positions, never on T: a batch sums in the order a single token does.
// glm_mla_combine: grid (H, T), block 256: out [T][H][kl] = the splits combined. kl % 32 == 0, kl <= 512.
#define GLM_HG 4
#define GLM_MAX_KL 512
#define GLM_SPLIT 256
#define GLM_MAX_SPLITS 9
extern "C" __global__ void __launch_bounds__(256)
glm_mla_part(const float* q_abs, const float* latent, const int* sel, int ld_sel, const int* nsel, float* part_ml,
             float* part_acc, int H, int kl, float scale, int hg) {
    __shared__ float qs[GLM_HG][GLM_MAX_KL];
    __shared__ float wm[8][GLM_HG], wl[8][GLM_HG];
    __shared__ float accb[4][GLM_HG][GLM_MAX_KL];
    const int h0 = blockIdx.x * hg, t = blockIdx.y, z = blockIdx.z, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int* s = sel + (size_t)t * ld_sel;
    const int n = nsel[t], i0 = z * GLM_SPLIT, i1 = min(n, i0 + GLM_SPLIT), per = kl / 32;
    for (int i = tid; i < hg * kl; i += 256) qs[i / kl][i % kl] = q_abs[((size_t)t * H + h0) * kl + i];
    __syncthreads();
    float m[GLM_HG], l[GLM_HG], acc[GLM_HG][GLM_MAX_KL / 32];
#pragma unroll
    for (int g = 0; g < GLM_HG; ++g) {
        m[g] = -INFINITY, l[g] = 0.0f;
#pragma unroll
        for (int k = 0; k < GLM_MAX_KL / 32; ++k) acc[g][k] = 0.0f;
    }
    for (int i = i0 + wave; i < i1; i += 8) {
        const float* c = latent + (size_t)s[i] * kl;
        float lv[GLM_MAX_KL / 32];
#pragma unroll
        for (int k = 0; k < GLM_MAX_KL / 32; ++k) lv[k] = k < per ? c[lane + 32 * k] : 0.0f;
#pragma unroll
        for (int g = 0; g < GLM_HG; ++g) {
            if (g >= hg) break;
            float d = 0.0f;
#pragma unroll
            for (int k = 0; k < GLM_MAX_KL / 32; ++k)
                if (k < per) d += qs[g][lane + 32 * k] * lv[k];
            for (int o = 16; o > 0; o >>= 1) d += __shfl_xor(d, o, 32);
            const float sc = d * scale, mn = fmaxf(m[g], sc), cr = expf(m[g] - mn), e = expf(sc - mn);
            l[g] = l[g] * cr + e;
#pragma unroll
            for (int k = 0; k < GLM_MAX_KL / 32; ++k) acc[g][k] = acc[g][k] * cr + e * lv[k];
            m[g] = mn;
        }
    }
    if (lane == 0)
#pragma unroll
        for (int g = 0; g < GLM_HG; ++g)
            if (g < hg) wm[wave][g] = m[g], wl[wave][g] = l[g];
    __syncthreads();
    float M[GLM_HG] = {}, Lsum[GLM_HG] = {};  // the block's max and sum per head (every thread computes them alike)
#pragma unroll
    for (int g = 0; g < GLM_HG; ++g) {
        if (g >= hg) break;
        M[g] = -INFINITY;
        for (int w = 0; w < 8; ++w) M[g] = fmaxf(M[g], wm[w][g]);
        Lsum[g] = 0.0f;
        for (int w = 0; w < 8; ++w) Lsum[g] += M[g] == -INFINITY ? 0.0f : wl[w][g] * expf(wm[w][g] - M[g]);
    }
    float o[GLM_HG][2] = {};  // thread tid: values tid and tid + 256 of each head's sum
    for (int r = 0; r < 2; ++r) {  // waves 4 r .. 4 r + 3 hand their scaled sums over, in wave order
        if (wave >= 4 * r && wave < 4 * r + 4)
#pragma unroll
            for (int g = 0; g < GLM_HG; ++g) {
                if (g >= hg) break;
                const float f = M[g] == -INFINITY ? 0.0f : expf(m[g] - M[g]);
#pragma unroll
                for (int k = 0; k < GLM_MAX_KL / 32; ++k)
                    if (k < per) accb[wave - 4 * r][g][lane + 32 * k] = acc[g][k] * f;
            }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < GLM_HG; ++g)
#pragma unroll
            for (int q = 0; q < 2; ++q) {
                if (g >= hg) break;
                const int k = tid + 256 * q;
                if (k < kl)
                    for (int w = 0; w < 4; ++w) o[g][q] += accb[w][g][k];
            }
        __syncthreads();
    }
    const size_t base = ((size_t)t * H + h0) * GLM_MAX_SPLITS + z;
#pragma unroll
    for (int g = 0; g < GLM_HG; ++g) {
        if (g >= hg) break;
        const size_t b = base + (size_t)g * GLM_MAX_SPLITS;
        if (tid == 0) part_ml[b * 2] = M[g], part_ml[b * 2 + 1] = Lsum[g];
#pragma unroll
        for (int q = 0; q < 2; ++q)
            if (tid + 256 * q < kl) part_acc[b * kl + tid + 256 * q] = o[g][q];
    }
}

extern "C" __global__ void __launch_bounds__(256)
glm_mla_combine(const float* part_ml, const float* part_acc, const int* nsel, float* out, int H, int kl) {
    const int h = blockIdx.x, t = blockIdx.y, nz = (nsel[t] + GLM_SPLIT - 1) / GLM_SPLIT;
    const size_t base = ((size_t)t * H + h) * GLM_MAX_SPLITS;
    float M = -INFINITY;
    for (int z = 0; z < nz; ++z) M = fmaxf(M, part_ml[(base + z) * 2]);
    float L = 0.0f;
    for (int z = 0; z < nz; ++z) L += M == -INFINITY ? 0.0f : part_ml[(base + z) * 2 + 1] * expf(part_ml[(base + z) * 2] - M);
    const float inv = L > 0.0f ? 1.0f / L : 0.0f;
    for (int k = threadIdx.x; k < kl; k += 256) {
        float v = 0.0f;
        for (int z = 0; z < nz; ++z)
            v += M == -INFINITY ? 0.0f : part_acc[(base + z) * kl + k] * expf(part_ml[(base + z) * 2] - M);
        out[((size_t)t * H + h) * kl + k] = v * inv;
    }
}
