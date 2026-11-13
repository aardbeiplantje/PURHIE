// MiniMax-H3-Image's VAE decoder kernels (minimax_vae.cpp; PLAN.md M2): the ViT's elementwise and normalization ops on
// float32 token rows. The projections and the attention are the image engine's (sd_device: linear_f32, attention).
#include <hip/hip_runtime.h>

#define MV_THREADS 256

__device__ __forceinline__ float mv_block_sum(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int off = MV_THREADS / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float s = red[0];
    __syncthreads();
    return s;
}

// y[r][i] = x[r][i] * rsqrt(mean_i(x[r]^2) + eps) * (w ? w[i] : 1): RMS norm of each row of n (one block per row)
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mv_rmsnorm(const float* x, int n, const float* w, float eps, float* y) {
    __shared__ float red[MV_THREADS];
    const float* xr = x + (long)blockIdx.x * n;
    float* yr = y + (long)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += MV_THREADS) s += xr[i] * xr[i];
    const float inv = rsqrtf(mv_block_sum(s, red) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += MV_THREADS) yr[i] = xr[i] * inv * (w ? w[i] : 1.0f);
}

// LayerNorm with affine weight and bias, row by row: y = (x - mean) rsqrt(var + eps) * w + b
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mv_layernorm(const float* x, int n, const float* w, const float* b, float eps, float* y) {
    __shared__ float red[MV_THREADS];
    const float* xr = x + (long)blockIdx.x * n;
    float* yr = y + (long)blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += MV_THREADS) s += xr[i];
    const float mean = mv_block_sum(s, red) / (float)n;
    float q = 0.0f;
    for (int i = threadIdx.x; i < n; i += MV_THREADS) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(mv_block_sum(q, red) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += MV_THREADS) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// The rotary (rotate-half) of the first rd channels of each head: x [T][C] with heads of D; cs and sn are [T][rd] tables of
// cos and sin per token (the second half of rd repeats the first half's frequencies). One thread per (token, head, pair).
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mv_rope(float* x, int T, int C, int heads, int D, int rd, const float* cs, const float* sn) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;
    const int h2 = rd / 2;
    const long total = (long)T * heads * h2;
    if (i >= total) return;
    const int j = (int)(i % h2);
    const long th = i / h2;
    const int hd = (int)(th % heads), t = (int)(th / heads);
    float* xr = x + (long)t * C + (long)hd * D;
    const float* c = cs + (long)t * rd;
    const float* s = sn + (long)t * rd;
    const float a = xr[j], b = xr[j + h2];
    xr[j] = a * c[j] - b * s[j];
    xr[j + h2] = b * c[j + h2] + a * s[j + h2];
}

// x[t][c] += y[t][c] * scale[c]
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mv_scale_add(float* x, const float* y, const float* scale, int T, int C) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;
    if (i >= (long)T * C) return;
    const int c = (int)(i % C);
    x[i] += y[i] * scale[c];
}

// SwiGLU of the projection rows pj [T][2F]: g[t][j] = pj[t][j] * silu(pj[t][F + j])
extern "C" __global__ void __launch_bounds__(MV_THREADS) mv_swiglu(const float* pj, int T, int F, float* g) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;
    if (i >= (long)T * F) return;
    const long t = i / F;
    const int j = (int)(i % F);
    const float gate = pj[t * 2 * F + F + j];
    g[i] = pj[t * 2 * F + j] * (gate / (1.0f + expf(-gate)));
}

// The output pixels of one tile: img[c][h*16 + i][w*16 + j] = tok[h*W + w][c*256 + i*16 + j] (p = 16 patches, C = 3)
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mv_unpatch(const float* tok, int H, int W, int p, float* img) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;
    const int OW = W * p, OH = H * p;
    const long total = 3L * OH * OW;
    if (i >= total) return;
    const int xw = (int)(i % OW);
    const int yh = (int)((i / OW) % OH);
    const int c = (int)(i / ((long)OW * OH));
    const int h = yh / p, ii = yh % p, w = xw / p, jj = xw % p;
    img[i] = tok[((long)h * W + w) * (3L * p * p) + (long)c * p * p + ii * p + jj];
}
