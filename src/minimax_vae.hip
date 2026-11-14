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

// ---- the DiT's AdaLN (minimax_dit.cpp). The modulation table m holds, per modality row r (0 image, 1 text), six chunks of
// H values: m[r * 6H + j * H + c] for j = shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp (0..5).

// y[t][c] = x[t][c] * (1 + m[r_t][js][c]) + m[r_t][jsh][c], r_t = rows[t] (0 or 1)
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mm_mod_rows(const float* x, const float* rows, const float* m, int S, int H, int jsh, int js, float* y) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;
    if (i >= (long)S * H) return;
    const int t = (int)(i / H), c = (int)(i % H);
    const long r = (long)rows[t];
    const float* mr = m + r * 6L * H;
    y[i] = x[i] * (1.0f + mr[js * H + c]) + mr[jsh * H + c];
}

// x[t][c] += o[t][c] * m[r_t][jg][c]
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mm_gate_rows(float* x, const float* o, const float* rows, const float* m, int S, int H, int jg) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;
    if (i >= (long)S * H) return;
    const int t = (int)(i / H), c = (int)(i % H);
    const long r = (long)rows[t];
    x[i] += o[i] * m[r * 6L * H + jg * H + c];
}

// y[t][c] = x[t][c] * (1 + sc[c]) + sh[c] for every row (the final layer, one timestep)
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mm_mod_cols(const float* x, const float* sc, const float* sh, int S, int H, float* y) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;
    if (i >= (long)S * H) return;
    const int c = (int)(i % H);
    y[i] = x[i] * (1.0f + sc[c]) + sh[c];
}

// ---- the checkpoint's own layouts (the diffusers-format release, checked 2026-11-13): the FFN's projection rows hold the GATE
// first and the VALUE second, so g = pj[F + j] * silu(pj[j]); the fused QKV rows are grouped per head: head h holds q, k, v of D
// rows each at (h * 3 + part) * D.

// SwiGLU of the checkpoint's projection rows pj [T][2F] (gate first, value second): g[t][j] = pj[t][F + j] * silu(pj[t][j])
extern "C" __global__ void __launch_bounds__(MV_THREADS) mm_swiglu_native(const float* pj, int T, int F, float* g) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;
    if (i >= (long)T * F) return;
    const long t = i / F;
    const int j = (int)(i % F);
    const float gate = pj[t * 2 * F + j];
    g[i] = pj[t * 2 * F + F + j] * (gate / (1.0f + expf(-gate)));
}

// Split the head-grouped QKV rows in [T][heads * 3 * D] into q, k, v [T][heads * D]
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mm_split_qkv(const float* in, int T, int heads, int D, float* q, float* k, float* v) {
    const long i = (long)blockIdx.x * MV_THREADS + threadIdx.x;  // over T * heads * D
    if (i >= (long)T * heads * D) return;
    const long t = i / ((long)heads * D);
    const int r = (int)(i % ((long)heads * D));
    const int h = r / D, d = r % D;
    const float* row = in + t * (long)heads * 3 * D + (long)h * 3 * D;
    q[i] = row[d];
    k[i] = row[D + d];
    v[i] = row[2 * D + d];
}

// Q8_0 weights [N][K] decoded to BF16 (the engine's decoder's F32 products, rounded to nearest even): one thread per value
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mm_dequant_q8_0_bf16(const uint8_t* W, long long row_bytes, int N, int K, uint16_t* out) {
    const long long i = (long long)blockIdx.x * MV_THREADS + threadIdx.x;  // over N * K
    if (i >= (long long)N * K) return;
    const long long r = i / K;
    const int c = (int)(i % K);
    const uint8_t* b = W + r * row_bytes + (long long)(c / 32) * 34;
    const uint16_t dbits = (uint16_t)(b[0] | (b[1] << 8));
    _Float16 dh;
    __builtin_memcpy(&dh, &dbits, 2);
    const float v = (float)((int8_t)b[2 + (c % 32)]) * (float)dh;
    uint32_t u;
    __builtin_memcpy(&u, &v, 4);
    if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
    out[i] = (uint16_t)(u >> 16);
}

// Q4_0_ROCMFP4 (fast = 0: 18-byte blocks, two scales; fast = 1: Q4_0_ROCMFP4_FAST, 17-byte blocks, one scale) [N][K]
// decoded to BF16, as src/quant_types.cpp's decoders: the codebook value times the unsigned E4M3 half scale, exact in F32
extern "C" __global__ void __launch_bounds__(MV_THREADS)
mm_dequant_fp4_bf16(const uint8_t* W, long long row_bytes, int N, int K, int fast, uint16_t* out) {
    const long long i = (long long)blockIdx.x * MV_THREADS + threadIdx.x;  // over N * K
    if (i >= (long long)N * K) return;
    const long long r = i / K;
    const int c = (int)(i % K);
    const int j = c % 32;
    const uint8_t* b = W + r * row_bytes + (long long)(c / 32) * (fast ? 17 : 18);
    const int8_t cb[16] = {0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10};
    const int nib = j < 16 ? (b[j] & 0x0f) : (b[j - 16] >> 4);
    const uint8_t e = fast ? b[16] : (j < 16 ? b[16] : b[17]);
    float d = 0.0f;
    if (e <= 0x7e) {
        const int ex = e >> 3, m = e & 7;
        d = ex == 0 ? (float)m * 0x1p-10f : (float)(8 + m) * ldexpf(1.0f, ex - 11);
    }
    const float v = (float)cb[nib] * d;
    uint32_t u;
    __builtin_memcpy(&u, &v, 4);
    if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);
    out[i] = (uint16_t)(u >> 16);
}
