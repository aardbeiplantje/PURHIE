// The Qwen3-TTS talker's fused block on the device (qwen3tts.cpp: the device path, QWEN3TTS_DEVICE=gpu): the small kernels
// of one pre-norm block for one token, fp32 activations, the weights (BF16) through matvec_bf16f. The arithmetic follows the
// host's block_step (rms norm in float, the rotary from a table of cos and sin of the position's angles, GQA over the cache,
// SiLU(gate) * up). Every kernel is one launch; the host sequences them on one stream.
#include <hip/hip_runtime.h>

#define TK_THREADS 256

// one block per row: y[row][i] = x[row][i] * rsqrt(mean(x[row]^2) + eps) * w[i]; rows = tokens or heads (n = head_dim)
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    __shared__ float red[TK_THREADS];
    const int row = blockIdx.x;
    const float *xr = x + (size_t)row * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += TK_THREADS) s += xr[i] * xr[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int off = TK_THREADS / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float inv = 1.0f / sqrtf(red[0] / (float)n + eps);
    float *yr = y + (size_t)row * n;
    for (int i = threadIdx.x; i < n; i += TK_THREADS) yr[i] = xr[i] * inv * w[i];
}

// the rotary (rotate-half) of x [heads][D] at one position: cs and sn are [pos][D/2] tables of cos and sin
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_rope(float *x, const float *cs, const float *sn, int heads, int D, int pos) {
    const int half = D / 2;
    const int idx = blockIdx.x * TK_THREADS + threadIdx.x;
    if (idx >= heads * half) return;
    const int h = idx / half, i = idx % half;
    const float c = cs[(size_t)pos * half + i], s = sn[(size_t)pos * half + i];
    float *v = x + (size_t)h * D;
    const float x1 = v[i], x2 = v[i + half];
    v[i] = x1 * c - x2 * s;
    v[i + half] = x2 * c + x1 * s;
}

// the cache row of one position: kc[pos][:] = k, vc[pos][:] = v (KD values each)
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_kv_store(const float *k, const float *v, float *kc, float *vc, int pos, int KD) {
    const int i = blockIdx.x * TK_THREADS + threadIdx.x;
    if (i >= KD) return;
    kc[(size_t)pos * KD + i] = k[i];
    vc[(size_t)pos * KD + i] = v[i];
}

// attention of one query token over the cache rows 0 .. n_ctx-1: one block per query head (GQA: head hq reads kv head
// hq / (H / Hkv)); the scores in the dynamic shared memory (n_ctx floats)
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_attn(const float *q, const float *kc, const float *vc, float *out, int n_ctx, int H, int Hkv, int D) {
    extern __shared__ float sc[];
    __shared__ float red[TK_THREADS];
    const int hq = blockIdx.x, hk = hq / (H / Hkv);
    const float *qr = q + (size_t)hq * D;
    const float scale = 1.0f / sqrtf((float)D);
    // the scores
    for (int j = threadIdx.x; j < n_ctx; j += TK_THREADS) {
        const float *kr = kc + ((size_t)j * Hkv + hk) * D;
        float d = 0.0f;
        for (int c = 0; c < D; ++c) d += qr[c] * kr[c];
        sc[j] = d * scale;
    }
    __syncthreads();
    // the maximum
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
    // the exponentials and their sum
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
    // the weighted values: one output dimension per thread
    for (int d = threadIdx.x; d < D; d += TK_THREADS) {
        float acc = 0.0f;
        for (int j = 0; j < n_ctx; ++j) acc += sc[j] * vc[((size_t)j * Hkv + hk) * D + d];
        out[(size_t)hq * D + d] = acc * inv;
    }
}

// g[i] = SiLU(g[i]) * u[i]
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_silu_mul(float *g, const float *u, int n) {
    const int i = blockIdx.x * TK_THREADS + threadIdx.x;
    if (i >= n) return;
    const float gi = g[i];
    g[i] = gi / (1.0f + expf(-gi)) * u[i];
}

// x[i] += y[i]
extern "C" __global__ void __launch_bounds__(TK_THREADS)
tk_add(float *x, const float *y, int n) {
    const int i = blockIdx.x * TK_THREADS + threadIdx.x;
    if (i < n) x[i] += y[i];
}
