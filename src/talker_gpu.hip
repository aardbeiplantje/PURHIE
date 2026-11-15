// The Qwen3-TTS talker's fused block on the device (qwen3tts.cpp: the device path, QWEN3TTS_DEVICE=gpu): one pre-norm block
// for one token in six launches, fp32 activations, the weights BF16 in place. The launches are the cost of a frame (about
// 17 us each on this machine: the talker's block was 19 launches, the predictor's stack 95), so each one fuses the
// neighbours it can: the rms norm into the matvecs that read its output (the QKV, the gate and the up, the heads), the
// q/k norm, the rotary and the cache store into one kernel, the residual adds into the matvecs that feed them. The
// arithmetic per output is the host's (rms norm in float, the rotary from the table of cos and sin of the position's
// angles, GQA over the cache, SiLU(gate) * up); the dot products sum in another order than the matvec_q kernels.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define TK_THREADS 256
#define TK_ROWS (TK_THREADS / 32)  // rows per block: one 32-lane group per row

__device__ __forceinline__ float tk_bf16(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    return __uint_as_float(u);
}

// the block's rms norm of x [n] into xs [n] (xs = x * rsqrt(mean(x^2) + eps) * w): each wave sums its squares by shuffles,
// the waves' sums are added in one order by every thread (two barriers in all). red has TK_THREADS / 32 floats at least.
__device__ void tk_rms_lds(const float *x, const float *w, int n, float eps, float *xs, float *red) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += TK_THREADS) s += x[i] * x[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    float tot = 0.0f;
    for (int wv = 0; wv < TK_THREADS / 32; ++wv) tot += red[wv];
    const float inv = 1.0f / sqrtf(tot / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += TK_THREADS) xs[i] = x[i] * inv * w[i];
    __syncthreads();
}

// the dot product of one BF16 row [n] (n a multiple of 8, 16-byte aligned) with xs [n], by one 32-lane group. Four 16-byte
// loads per lane are issued before their products are added: the row is bandwidth-bound, and one load in flight per lane
// left the matvecs at 70 to 140 GB/s.
__device__ __forceinline__ float tk_dot8(const uint4 v, const float *xp) {
    return tk_bf16((uint16_t)(v.x & 0xFFFFu)) * xp[0] + tk_bf16((uint16_t)(v.x >> 16)) * xp[1] +
           tk_bf16((uint16_t)(v.y & 0xFFFFu)) * xp[2] + tk_bf16((uint16_t)(v.y >> 16)) * xp[3] +
           tk_bf16((uint16_t)(v.z & 0xFFFFu)) * xp[4] + tk_bf16((uint16_t)(v.z >> 16)) * xp[5] +
           tk_bf16((uint16_t)(v.w & 0xFFFFu)) * xp[6] + tk_bf16((uint16_t)(v.w >> 16)) * xp[7];
}

__device__ __forceinline__ float tk_row_dot(const uint16_t *wr, const float *xs, int n, int lane) {
    // (n is a multiple of 1024: the four loads of a step are unconditional; a load inside a branch serialized them)
    const uint4 *w4 = reinterpret_cast<const uint4 *>(wr);
    const int n8 = n / 8;
    float acc = 0.0f;
    for (int j = lane; j < n8; j += 128) {
        const uint4 v0 = w4[j], v1 = w4[j + 32], v2 = w4[j + 64], v3 = w4[j + 96];
        acc += tk_dot8(v0, xs + 8 * j) + tk_dot8(v1, xs + 8 * (j + 32)) + tk_dot8(v2, xs + 8 * (j + 64)) +
               tk_dot8(v3, xs + 8 * (j + 96));
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    return acc;
}

// q = Wq rms(x), k = Wk rms(x), v = Wv rms(x): the rows of the three matrices as one launch (row r < nq: q[r]; then k, then v).
// grid: ceil((nq + nk + nv) / TK_ROWS); dynamic shared memory: C floats
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_qkv(const float *x, const float *nw, const uint16_t *Wq, const uint16_t *Wk, const uint16_t *Wv, int C, int nq, int nk,
       int nv, float eps, float *q, float *k, float *v) {
    extern __shared__ float xs[];
    __shared__ float red[TK_THREADS];
    tk_rms_lds(x, nw, C, eps, xs, red);
    const int row = blockIdx.x * TK_ROWS + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= nq + nk + nv) return;
    const uint16_t *W;
    float *out;
    if (row < nq) {
        W = Wq + (size_t)row * C;
        out = q + row;
    } else if (row < nq + nk) {
        W = Wk + (size_t)(row - nq) * C;
        out = k + (row - nq);
    } else {
        W = Wv + (size_t)(row - nq - nk) * C;
        out = v + (row - nq - nk);
    }
    const float acc = tk_row_dot(W, xs, C, lane);
    if (lane == 0) *out = acc;
}

// The q and k heads: each head's rms norm (q_norm / k_norm, [D]), the rotary at *posp (cs, sn: [pos][D/2]); the k heads and
// the v heads go to the cache row pos of kc / vc ([maxL][Hkv * D]). Blocks 0 .. H - 1: the query heads (in place);
// blocks H .. H + Hkv - 1: the key heads (in place) and their cache rows. dynamic shared memory: D floats.
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_qk_rope_kv(float *q, float *k, const float *v, const float *qn, const float *kn, const float *cs, const float *sn,
              const int *posp, float *kc, float *vc, int H, int Hkv, int D, int KD, float eps) {
    extern __shared__ float xs[];
    __shared__ float red[TK_THREADS];
    const int pos = *posp;  // the position is read on the device: one graph serves every position
    const int b = blockIdx.x;
    const bool is_k = b >= H;
    const int head = is_k ? b - H : b;
    float *vec = (is_k ? k : q) + (size_t)head * D;
    const float *nw = is_k ? kn : qn;
    tk_rms_lds(vec, nw, D, eps, xs, red);  // (vec is read here, written below)
    const int half = D / 2;
    if (threadIdx.x < half) {
        const int i = threadIdx.x;
        const float c = cs[(size_t)pos * half + i], s = sn[(size_t)pos * half + i];
        const float x1 = xs[i], x2 = xs[i + half];
        vec[i] = x1 * c - x2 * s;
        vec[i + half] = x2 * c + x1 * s;
    }
    __syncthreads();  // the rotated vector is complete before the cache row reads it
    if (is_k) {
        // the cache row: the rotated key and the value of this head
        for (int i = threadIdx.x; i < D; i += TK_THREADS) {
            kc[(size_t)pos * KD + head * D + i] = vec[i];
            vc[(size_t)pos * KD + head * D + i] = v[head * D + i];
        }
    }
}

// one query token over the cache rows 0 .. *posp: one block per query head (GQA: head hq reads kv head hq / (H / Hkv));
// the scores in the dynamic shared memory (the largest context's floats)
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_attn(const float *q, const float *kc, const float *vc, float *out, const int *posp, int H, int Hkv, int D) {
    extern __shared__ float sc[];
    __shared__ float red[TK_THREADS];
    const int n_ctx = *posp + 1;  // (the dynamic shared memory was sized for the largest context)
    const int hq = blockIdx.x, hk = hq / (H / Hkv);
    const float *qr = q + (size_t)hq * D;
    const float scale = 1.0f / sqrtf((float)D);
    for (int j = threadIdx.x; j < n_ctx; j += TK_THREADS) {
        const float *kr = kc + ((size_t)j * Hkv + hk) * D;
        float d = 0.0f;
        for (int c = 0; c < D; ++c) d += qr[c] * kr[c];
        sc[j] = d * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n_ctx; j += TK_THREADS) mx = fmaxf(mx, sc[j]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int off = TK_THREADS / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_ctx; j += TK_THREADS) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int off = TK_THREADS / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += TK_THREADS) {
        float acc = 0.0f;
        for (int j = 0; j < n_ctx; ++j) acc += sc[j] * vc[((size_t)j * Hkv + hk) * D + d];
        out[(size_t)hq * D + d] = acc * inv;
    }
}

// x[r] += Wo[r] . att for the C rows of Wo [C][K] (the attention's output projection and its residual add).
// dynamic shared memory: K floats
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_oproj_add(const float *att, const uint16_t *Wo, float *x, int K, int C) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += TK_THREADS) xs[i] = att[i];
    __syncthreads();
    const int row = blockIdx.x * TK_ROWS + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= C) return;
    const float acc = tk_row_dot(Wo + (size_t)row * K, xs, K, lane);
    if (lane == 0) x[row] += acc;
}

// g[j] = SiLU(Wg[j] . n) * (Wu[j] . n) for the FF rows of the gate and the up, n = rms(x) * nw (the FFN's input). Dynamic
// shared memory: C floats
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_gateup(const float *x, const float *nw, const uint16_t *Wg, const uint16_t *Wu, int C, int FF, float eps, float *g) {
    extern __shared__ float xs[];
    __shared__ float red[TK_THREADS];
    tk_rms_lds(x, nw, C, eps, xs, red);
    const int row = blockIdx.x * TK_ROWS + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= FF) return;
    const float gate = tk_row_dot(Wg + (size_t)row * C, xs, C, lane);
    const float up = tk_row_dot(Wu + (size_t)row * C, xs, C, lane);
    if (lane == 0) g[row] = gate / (1.0f + expf(-gate)) * up;
}

// x[r] += Wd[r] . g for the C rows of Wd [C][FF] (the FFN's down projection and its residual add). Dynamic shared memory:
// FF floats
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_down_add(const float *g, const uint16_t *Wd, float *x, int FF, int C) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < FF; i += TK_THREADS) xs[i] = g[i];
    __syncthreads();
    const int row = blockIdx.x * TK_ROWS + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= C) return;
    const float acc = tk_row_dot(Wd + (size_t)row * FF, xs, FF, lane);
    if (lane == 0) x[row] += acc;
}

// out[r] = Wh[r] . rms(x) * nw for the rows of a head (the codec head, the predictor's heads); normed (optional) gets rms(x) * nw.
// Dynamic shared memory: C floats
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_head(const float *x, const float *nw, const uint16_t *Wh, int C, int rows, float eps, float *normed, float *out) {
    extern __shared__ float xs[];
    __shared__ float red[TK_THREADS];
    tk_rms_lds(x, nw, C, eps, xs, red);
    if (normed && blockIdx.x == 0)  // the normed input (the predictor's first row is the talker's normed output)
        for (int i = threadIdx.x; i < C; i += TK_THREADS) normed[i] = xs[i];
    const int row = blockIdx.x * TK_ROWS + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float acc = tk_row_dot(Wh + (size_t)row * C, xs, C, lane);
    if (lane == 0) out[row] = acc;
}

// ---- The prefill: the rows of a prompt (Tn tokens at positions pos0 ..) through a block, the batched form of the decode
// step's kernels (qmatvec's batched BF16 product sums each token as its single-token product does; the norms, the rotary,
// the cache and the attention below are the decode step's, per row).

// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, one block per row (the same reduction as tk_rms_lds)
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_rms_rows(const float *x, const float *w, int C, float eps, float *y) {
    __shared__ float red[TK_THREADS / 32];
    const float *xr = x + (size_t)blockIdx.x * C;
    float s = 0.0f;
    for (int i = threadIdx.x; i < C; i += TK_THREADS) s += xr[i] * xr[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o);
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = s;
    __syncthreads();
    float tot = 0.0f;
    for (int wv = 0; wv < TK_THREADS / 32; ++wv) tot += red[wv];
    const float inv = 1.0f / sqrtf(tot / (float)C + eps);
    float *yr = y + (size_t)blockIdx.x * C;
    for (int i = threadIdx.x; i < C; i += TK_THREADS) yr[i] = xr[i] * inv * w[i];
}

// the q and k heads of row t (position pos0 + t): the norms, the rotary; the k heads and the v rows go to the cache at that
// position. grid (Tn, H + Hkv); dynamic shared memory: D floats
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_qk_rows(float *q, float *k, const float *v, const float *qn, const float *kn, const float *cs, const float *sn, int pos0,
           float *kc, float *vc, int H, int Hkv, int D, int HD, int KD, float eps) {
    extern __shared__ float xs[];
    __shared__ float red[TK_THREADS];
    const int t = blockIdx.x, b = blockIdx.y;
    const bool is_k = b >= H;
    const int head = is_k ? b - H : b;
    const int pos = pos0 + t;
    float *vec = (is_k ? k + (size_t)t * KD : q + (size_t)t * HD) + (size_t)head * D;
    const float *nw = is_k ? kn : qn;
    tk_rms_lds(vec, nw, D, eps, xs, red);
    const int half = D / 2;
    if (threadIdx.x < half) {
        const int i = threadIdx.x;
        const float c = cs[(size_t)pos * half + i], s = sn[(size_t)pos * half + i];
        const float x1 = xs[i], x2 = xs[i + half];
        vec[i] = x1 * c - x2 * s;
        vec[i + half] = x2 * c + x1 * s;
    }
    __syncthreads();
    if (is_k) {
        for (int i = threadIdx.x; i < D; i += TK_THREADS) {
            kc[(size_t)pos * KD + head * D + i] = vec[i];
            vc[(size_t)pos * KD + head * D + i] = v[(size_t)t * KD + head * D + i];
        }
    }
}

// the attention of row t over the cache's positions 0 .. pos0 + t: grid (H, Tn); dynamic shared memory: (pos0 + Tn) floats
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_attn_rows(const float *q, const float *kc, const float *vc, float *out, int pos0, int H, int Hkv, int D, int HD) {
    extern __shared__ float sc[];
    __shared__ float red[TK_THREADS];
    const int hq = blockIdx.x, t = blockIdx.y, hk = hq / (H / Hkv);
    const int n_ctx = pos0 + t + 1;
    const float *qr = q + (size_t)t * HD + (size_t)hq * D;
    const float scale = 1.0f / sqrtf((float)D);
    for (int j = threadIdx.x; j < n_ctx; j += TK_THREADS) {
        const float *kr = kc + ((size_t)j * Hkv + hk) * D;
        float d = 0.0f;
        for (int c = 0; c < D; ++c) d += qr[c] * kr[c];
        sc[j] = d * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n_ctx; j += TK_THREADS) mx = fmaxf(mx, sc[j]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int off = TK_THREADS / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + off]);
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n_ctx; j += TK_THREADS) {
        const float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int off = TK_THREADS / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = 1.0f / red[0];
    __syncthreads();
    float *orow = out + (size_t)t * HD + (size_t)hq * D;
    for (int d = threadIdx.x; d < D; d += TK_THREADS) {
        float acc = 0.0f;
        for (int j = 0; j < n_ctx; ++j) acc += sc[j] * vc[((size_t)j * Hkv + hk) * D + d];
        orow[d] = acc * inv;
    }
}

// x[i] += y[i] over n values
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_add_rows(float *x, const float *y, int n) {
    const int i = blockIdx.x * TK_THREADS + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// g[i] = SiLU(g[i]) * u[i] over n values (the decode step's gate)
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_silu_mul2(float *g, const float *u, int n) {
    const int i = blockIdx.x * TK_THREADS + threadIdx.x;
    if (i < n) {
        const float gi = g[i];
        g[i] = gi / (1.0f + expf(-gi)) * u[i];
    }
}

// ---- The prefill's matmul: Y [T][N] = X [T][K] . W^T (W: BF16 [N][K], X: fp32). A block computes a tile of 16 output rows
// by 32 tokens: the weights and the activations of each chunk of 64 inputs go to the shared memory once, then every element
// sums its chunk in order (fp32). grid (ceil(N / 16), ceil(T / 32)); K a multiple of 64.
#define TK_GM 16
#define TK_GT 32
#define TK_GK 64
extern "C" __global__ void __launch_bounds__(256)
tk_gemm(const uint16_t *W, const float *X, float *Y, int K, int N, int T) {
    __shared__ float Ws[TK_GM][TK_GK + 1];
    __shared__ float Xs[TK_GT][TK_GK + 1];
    const int n0 = blockIdx.x * TK_GM, t0 = blockIdx.y * TK_GT;
    const int r = threadIdx.x / 16, tq = threadIdx.x % 16;  // the output row r; the tokens tq and tq + 16
    float a0 = 0.0f, a1 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TK_GK) {
        for (int i = threadIdx.x; i < TK_GM * TK_GK; i += 256) {
            const int rr = i / TK_GK, kk = i % TK_GK;
            Ws[rr][kk] = (n0 + rr < N) ? tk_bf16(W[(size_t)(n0 + rr) * K + k0 + kk]) : 0.0f;
        }
        for (int i = threadIdx.x; i < TK_GT * TK_GK; i += 256) {
            const int tt = i / TK_GK, kk = i % TK_GK;
            Xs[tt][kk] = (t0 + tt < T) ? X[(size_t)(t0 + tt) * K + k0 + kk] : 0.0f;
        }
        __syncthreads();
        for (int kk = 0; kk < TK_GK; ++kk) {
            const float w = Ws[r][kk];
            a0 += w * Xs[tq][kk];
            a1 += w * Xs[tq + 16][kk];
        }
        __syncthreads();
    }
    if (n0 + r < N) {
        if (t0 + tq < T) Y[(size_t)(t0 + tq) * N + n0 + r] = a0;
        if (t0 + tq + 16 < T) Y[(size_t)(t0 + tq + 16) * N + n0 + r] = a1;
    }
}
