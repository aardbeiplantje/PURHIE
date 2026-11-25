// The image engine's device kernels (sd_device.cpp; PLAN.md I6): float32 activations in registered UMA memory, the F16 weights
// read in place from the GGUF mapping. One GEMM serves the convolutions (weights x im2col), the linear layers in either
// layout (token rows or channel planes, by the strides) and the attention (scores and the weighted values, by the strides).
#include <hip/hip_runtime.h>

#include <stdint.h>

#define SD_THREADS 256
#define G_BM 64
#define G_BN 64
#define G_BK 16

__device__ __forceinline__ float bf2f(uint16_t u) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = (uint32_t)u << 16;
    return v.f;
}

__device__ __forceinline__ float h2f(uint16_t u) {
    union {
        uint16_t u;
        _Float16 h;
    } v;
    v.u = u;
    return (float)v.h;
}

// C[m * cs_m + n * cs_n] = alpha * sum_k A(m, k) B(k, n) + bias[m] + beta * C, with A(m, k) = A[m * lda + k] (F16 or float) and
// B(k, n) = B[k * sbk + n * sbn] (float). A 64 x 64 output tile per block, 16 x 16 threads of 4 x 4 outputs each.
template <typename TA, int MODE>  // MODE: 0 float A, 1 binary16 A, 2 bfloat16 A
__device__ __forceinline__ void gemm_body(const TA* __restrict__ A, long lda, const float* __restrict__ B, long sbk, long sbn,
                                          float* __restrict__ C, long cs_m, long cs_n, const uint16_t* __restrict__ bias,
                                          const float* __restrict__ biasf, float alpha, float beta, int M, int N, int K,
                                          float (*As)[G_BM + 1],
                                          float (*Bs)[G_BN + 1]) {
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int bm = blockIdx.y * G_BM, bn = blockIdx.x * G_BN;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < K; k0 += G_BK) {
        for (int e = tid; e < G_BM * G_BK; e += SD_THREADS) {
            const int m = e / G_BK, k = e % G_BK;
            const int gm = bm + m, gk = k0 + k;
            float v = 0.0f;
            if (gm < M && gk < K) {
                const TA a = A[(long)gm * lda + gk];
                if (MODE == 1)
                    v = h2f((uint16_t)a);
                else if (MODE == 2)
                    v = bf2f((uint16_t)a);
                else
                    v = (float)a;
            }
            As[k][m] = v;
        }
        for (int e = tid; e < G_BK * G_BN; e += SD_THREADS) {
            const int k = e / G_BN, n = e % G_BN;
            const int gk = k0 + k, gn = bn + n;
            float v = 0.0f;
            if (gk < K && gn < N) v = B[(long)gk * sbk + (long)gn * sbn];
            Bs[k][n] = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < G_BK; ++kk) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = As[kk][ty * 4 + i];
                b[i] = Bs[kk][tx * 4 + i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = bm + ty * 4 + i, n = bn + tx * 4 + j;
            if (m < M && n < N) {
                float v = alpha * acc[i][j];
                if (bias) v += h2f(bias[m]);
                if (biasf) v += biasf[m];
                const long off = (long)m * cs_m + (long)n * cs_n;
                if (beta != 0.0f) v += beta * C[off];
                C[off] = v;
            }
        }
}

extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_gemm_ah(const uint16_t* A, long lda, const float* B, long sbk, long sbn, float* C, long cs_m, long cs_n, const uint16_t* bias,
           const float* biasf, float alpha, float beta, int M, int N, int K) {
    __shared__ float As[G_BK][G_BM + 1];
    __shared__ float Bs[G_BK][G_BN + 1];
    gemm_body<uint16_t, 1>(A, lda, B, sbk, sbn, C, cs_m, cs_n, bias, biasf, alpha, beta, M, N, K, As, Bs);
}

extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_gemm_af(const float* A, long lda, const float* B, long sbk, long sbn, float* C, long cs_m, long cs_n, const uint16_t* bias,
           const float* biasf, float alpha, float beta, int M, int N, int K) {
    __shared__ float As[G_BK][G_BM + 1];
    __shared__ float Bs[G_BK][G_BN + 1];
    gemm_body<float, 0>(A, lda, B, sbk, sbn, C, cs_m, cs_n, bias, biasf, alpha, beta, M, N, K, As, Bs);
}

// The FP32 GEMM for the convolutions: 128 x 128 tiles, 8 x 8 outputs per thread (the 64 x 64 body above does 4 x 4), so each
// loaded value feeds twice the products. Same epilogue and FP32 arithmetic; only the summation order differs.
extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_gemm_af128(const float* __restrict__ A, long lda, const float* __restrict__ B, long sbk, long sbn, float* __restrict__ C,
              long cs_m, long cs_n, const uint16_t* __restrict__ bias, const float* __restrict__ biasf, float alpha, float beta,
              int M, int N, int K) {
    __shared__ float As[16][132];
    __shared__ float Bs[16][132];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int bm = blockIdx.y * 128, bn = blockIdx.x * 128;
    float acc[8][8] = {};
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int e = tid; e < 128 * 16; e += SD_THREADS) {  // A: m = e / 16 (rows of A are contiguous in k)
            const int m = e >> 4, k = e & 15;
            const int gm = bm + m, gk = k0 + k;
            As[k][m] = (gm < M && gk < K) ? A[(long)gm * lda + gk] : 0.0f;
        }
        for (int e = tid; e < 16 * 128; e += SD_THREADS) {  // B: k = e / 128, n = e % 128
            const int k = e >> 7, n = e & 127;
            const int gk = k0 + k, gn = bn + n;
            Bs[k][n] = (gk < K && gn < N) ? B[(long)gk * sbk + (long)gn * sbn] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[8], b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                a[i] = As[kk][ty * 8 + i];
                b[i] = Bs[kk][tx * 8 + i];
            }
#pragma unroll
            for (int i = 0; i < 8; ++i)
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int m = bm + ty * 8 + i, n = bn + tx * 8 + j;
            if (m < M && n < N) {
                float v = alpha * acc[i][j];
                if (bias) v += h2f(bias[m]);
                if (biasf) v += biasf[m];
                const long off = (long)m * cs_m + (long)n * cs_n;
                if (beta != 0.0f) v += beta * C[off];
                C[off] = v;
            }
        }
}

// col[r][j] (r < cin k k, j < nc): the input under kernel tap r at output position n0 + j (zero outside the input)
extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_im2col(const float* x, int cin, int H, int W, int k, int p, int st, int Wo, int n0, int nc, float* col) {
    const int r = blockIdx.y;
    const int j = blockIdx.x * SD_THREADS + threadIdx.x;
    if (j >= nc) return;
    const int kk = k * k;
    const int c = r / kk, rem = r % kk, ky = rem / k, kx = rem % k;
    const int n = n0 + j, yy = n / Wo, xx = n % Wo;
    const int sy = yy * st + ky - p, sx = xx * st + kx - p;
    float v = 0.0f;
    if (c < cin && sy >= 0 && sy < H && sx >= 0 && sx < W) v = x[((long)c * H + sy) * W + sx];
    col[(long)r * nc + j] = v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int off = SD_THREADS / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    const float s = red[0];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int off = SD_THREADS / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + off]);
        __syncthreads();
    }
    const float s = red[0];
    __syncthreads();
    return s;
}

// GroupNorm, in three passes over the groups (a group is a contiguous run of per x HW floats): the partial sums (S per
// group, block (s, g)), the partial sums of squared deviations from the mean, and the normalized output.
extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_gn_sum(const float* x, long n_group, int S, float* psum) {
    __shared__ float red[SD_THREADS];
    const int s = blockIdx.x, g = blockIdx.y;
    const float* base = x + (long)g * n_group;
    float acc = 0.0f;
    for (long i = (long)s * SD_THREADS + threadIdx.x; i < n_group; i += (long)S * SD_THREADS) acc += base[i];
    const float tot = block_sum(acc, red);
    if (threadIdx.x == 0) psum[g * S + s] = tot;
}

extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_gn_sq(const float* x, long n_group, int S, const float* psum, float* psq) {
    __shared__ float red[SD_THREADS];
    const int s = blockIdx.x, g = blockIdx.y;
    const float* base = x + (long)g * n_group;
    float sum = 0.0f;
    for (int i = 0; i < S; ++i) sum += psum[g * S + i];
    const float mean = sum / (float)n_group;
    float acc = 0.0f;
    for (long i = (long)s * SD_THREADS + threadIdx.x; i < n_group; i += (long)S * SD_THREADS) {
        const float d = base[i] - mean;
        acc += d * d;
    }
    const float tot = block_sum(acc, red);
    if (threadIdx.x == 0) psq[g * S + s] = tot;
}

extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_gn_apply(const float* x, int C, int HW, int per, long n_group, int S, const float* psum, const float* psq, const uint16_t* gw,
            const uint16_t* gb, float eps, float* y) {
    const long idx = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (idx >= (long)C * HW) return;
    const int ch = (int)(idx / HW), g = ch / per;
    float sum = 0.0f, sq = 0.0f;
    for (int i = 0; i < S; ++i) {
        sum += psum[g * S + i];
        sq += psq[g * S + i];
    }
    const float mean = sum / (float)n_group, var = sq / (float)n_group;
    const float sc = rsqrtf(var + eps) * h2f(gw[ch]);
    y[idx] = (x[idx] - mean) * sc + h2f(gb[ch]);
}

// y = x / (1 + exp(-x)), in place
extern "C" __global__ void __launch_bounds__(SD_THREADS) sd_silu(float* x, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}

// out += x
extern "C" __global__ void __launch_bounds__(SD_THREADS) sd_add(float* out, const float* x, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) out[i] += x[i];
}

// nearest 2x upsampling of [C][H][W] into [C][2H][2W]
extern "C" __global__ void __launch_bounds__(SD_THREADS) sd_upsample2(const float* x, int C, int H, int W, float* y) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    const long total = (long)C * 2 * H * 2 * W;
    if (i >= total) return;
    const int w2 = (int)(i % (2 * W)), h2 = (int)((i / (2 * W)) % (2 * H)), c = (int)(i / (2L * W * 2 * H));
    y[i] = x[((long)c * H + h2 / 2) * W + w2 / 2];
}

// softmax of each row of S [rows][N], in place
extern "C" __global__ void __launch_bounds__(SD_THREADS) sd_softmax(float* S, int N) {
    __shared__ float red[SD_THREADS];
    float* r = S + (long)blockIdx.x * N;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < N; j += SD_THREADS) mx = fmaxf(mx, r[j]);
    mx = block_max(mx, red);
    float s = 0.0f;
    for (int j = threadIdx.x; j < N; j += SD_THREADS) {
        const float e = expf(r[j] - mx);
        r[j] = e;
        s += e;
    }
    const float inv = 1.0f / block_sum(s, red);
    for (int j = threadIdx.x; j < N; j += SD_THREADS) r[j] *= inv;
}

// LayerNorm of each token row x[t][C] (biased variance), y = (x - mean) rsqrt(var + eps) * w + b
extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_layer_norm(const float* x, int C, const uint16_t* w, const uint16_t* b, float eps, float* y) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    float* yr = y + (long)blockIdx.x * C;
    float s = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) s += xr[i];
    const float mean = block_sum(s, red) / (float)C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) yr[i] = (xr[i] - mean) * inv * h2f(w[i]) + h2f(b[i]);
}

// RMSNorm of each row x[t][C] (no mean subtraction): y = x * rsqrt(mean(x^2) + eps) * w, w BF16 (Z-Image's norms)
extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_rms_norm(const float* x, int C, const uint16_t* w, float eps, float* y) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    float* yr = y + (long)blockIdx.x * C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) q += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) yr[i] = xr[i] * inv * bf2f(w[i]);  /* BF16 weights (the Z-Image norms) */
}

// LayerNorm without weights (the Z-Image final layer): y = (x - mean) rsqrt(var + eps) of each row x[t][C]
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_layer_norm_plain(const float* x, int C, float eps, float* y) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    float* yr = y + (long)blockIdx.x * C;
    float s = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) s += xr[i];
    const float mean = block_sum(s, red) / (float)C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) q += (xr[i] - mean) * (xr[i] - mean);
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) yr[i] = (xr[i] - mean) * inv;
}

// GroupNorm with F32 affine weights on [C][HW] (channel-major): one block per group; y = (x - mean) rsqrt(var + eps) * g + b
extern "C" __global__ void __launch_bounds__(SD_THREADS)
zimage_group_norm_f32(const float* x, int C, int HW, int groups, const float* g, const float* bt, float eps, float* y) {
    __shared__ float red[SD_THREADS];
    const int per = C / groups;
    const int grp = blockIdx.x;
    const long n = (long)per * HW;
    const float* xg = x + (long)grp * per * HW;
    float s = 0.0f;
    for (long i = threadIdx.x; i < n; i += SD_THREADS) s += xg[i];
    const float mean = block_sum(s, red) / (float)n;
    float q = 0.0f;
    for (long i = threadIdx.x; i < n; i += SD_THREADS) { const float d = xg[i] - mean; q += d * d; }
    const float inv = rsqrtf(block_sum(q, red) / (float)n + eps);
    for (long i = threadIdx.x; i < n; i += SD_THREADS) {
        const int c = grp * per + (int)(i / HW);
        y[(long)grp * per * HW + i] = (xg[i] - mean) * inv * g[c] + bt[c];
    }
}

// GroupNorm in three passes over S chunks per group (grid (S, groups)): the one-block-per-group kernel above left 32 blocks
// for the GPU's 40 CUs. The partial sums are reduced in a fixed order (every block sums the group's S partials the same way),
// so the result does not depend on the scheduling: the same two-pass statistics (mean, then the squared deviations).
extern "C" __global__ void __launch_bounds__(SD_THREADS)
zimage_gn_sum(const float* x, int HW, int per, int S, float* part) {
    __shared__ float red[SD_THREADS];
    const int sc = blockIdx.x, grp = blockIdx.y;
    const long n = (long)per * HW;
    const long len = (n + S - 1) / S;
    const long i0 = (long)sc * len, i1 = i0 + len < n ? i0 + len : n;
    const float* xg = x + (long)grp * n;
    float s = 0.0f;
    for (long i = i0 + threadIdx.x; i < i1; i += SD_THREADS) s += xg[i];
    const float tot = block_sum(s, red);
    if (threadIdx.x == 0) part[(long)grp * S + sc] = tot;
}

extern "C" __global__ void __launch_bounds__(SD_THREADS)
zimage_gn_var(const float* x, int HW, int per, int S, const float* part_sum, float* part_q) {
    __shared__ float red[SD_THREADS];
    const int sc = blockIdx.x, grp = blockIdx.y;
    const long n = (long)per * HW;
    const long len = (n + S - 1) / S;
    const long i0 = (long)sc * len, i1 = i0 + len < n ? i0 + len : n;
    float tot = 0.0f;
    for (int k = 0; k < S; ++k) tot += part_sum[(long)grp * S + k];
    const float mean = tot / (float)n;
    const float* xg = x + (long)grp * n;
    float q = 0.0f;
    for (long i = i0 + threadIdx.x; i < i1; i += SD_THREADS) {
        const float d = xg[i] - mean;
        q += d * d;
    }
    const float qt = block_sum(q, red);
    if (threadIdx.x == 0) part_q[(long)grp * S + sc] = qt;
}

extern "C" __global__ void __launch_bounds__(SD_THREADS)
zimage_gn_apply(const float* x, int C, int HW, int groups, int S, const float* part_sum, const float* part_q, const float* g,
                const float* bt, float eps, float* y) {
    const int sc = blockIdx.x, grp = blockIdx.y;
    const int per = C / groups;
    const long n = (long)per * HW;
    const long len = (n + S - 1) / S;
    const long i0 = (long)sc * len, i1 = i0 + len < n ? i0 + len : n;
    float tot = 0.0f, totq = 0.0f;
    for (int k = 0; k < S; ++k) {
        tot += part_sum[(long)grp * S + k];
        totq += part_q[(long)grp * S + k];
    }
    const float mean = tot / (float)n;
    const float inv = rsqrtf(totq / (float)n + eps);
    const float* xg = x + (long)grp * n;
    float* yg = y + (long)grp * n;
    for (long i = i0 + threadIdx.x; i < i1; i += SD_THREADS) {
        const int c = grp * per + (int)(i / HW);
        yg[i] = (xg[i] - mean) * inv * g[c] + bt[c];
    }
}

// [C][HW] -> [HW][C] and back (the decoder's attention works on tokens)
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_to_tokens(const float* x, int C, int HW, float* y) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= (long)C * HW) return;
    const int c = (int)(i / HW), p = (int)(i % HW);
    y[(long)p * C + c] = x[i];
}
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_to_chw(const float* x, int C, int HW, float* y) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= (long)C * HW) return;
    const int p = (int)(i / C), c = (int)(i % C);
    y[(long)c * HW + p] = x[i];
}

// ---- the text encoder (Qwen3, zimage_text.cpp) ----
// RoPE, rotate-half convention: x [T][heads][128]; pairs (p, p + 64) of each head rotated by cos/sin [T][64]
extern "C" __global__ void __launch_bounds__(SD_THREADS)
zq_rope_half(float* x, int T, int heads, const float* cs, const float* sn, long n_pairs) {
    const long j = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (j >= n_pairs) return;
    const int P = 64, D = 128;
    const long t = j / (heads * P);
    const long rem = j % (heads * P);
    const int h = (int)(rem / P), p = (int)(rem % P);
    float* v = x + t * heads * D + h * D;
    const float c = cs[t * P + p], s0 = sn[t * P + p];
    const float a = v[p], b = v[p + P];
    v[p] = a * c - b * s0;
    v[p + P] = b * c + a * s0;
}
// causal attention with grouped KV heads: out[t][h][d] = softmax_j<=t(q[t][h] . k[j][h / (H / KVH)] / sqrt(128)) v[j][kvh][d].
// One block per (query t, head h); the scores in dynamic shared memory (T floats: the launch passes 4 T bytes).
extern "C" __global__ void __launch_bounds__(128)
zq_causal_attn(const float* q, const float* k, const float* v, int T, int H, int KVH, float* out) {
    extern __shared__ float sc[];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int D = 128, kvh = h / (H / KVH);
    const float scale = rsqrtf((float)D);
    const float* qv = q + ((long)t * H + h) * D;
    for (int j = d; j <= t; j += 128) {
        const float* kv = k + ((long)j * KVH + kvh) * D;
        float acc = 0.0f;
        for (int e = 0; e < D; ++e) acc += qv[e] * kv[e];
        sc[j] = acc * scale;
    }
    __syncthreads();
    if (d == 0) {
        float mx = -1e30f;
        for (int j = 0; j <= t; ++j) mx = fmaxf(mx, sc[j]);
        float sum = 0.0f;
        for (int j = 0; j <= t; ++j) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
        for (int j = 0; j <= t; ++j) sc[j] /= sum;
    }
    __syncthreads();
    float acc = 0.0f;
    for (int j = 0; j <= t; ++j) acc += sc[j] * v[((long)j * KVH + kvh) * D + d];
    out[((long)t * H + h) * D + d] = acc;
}

// ---- Z-Image's block ops (zimage_dit.cpp): the AdaLN scale, the gated residual, SwiGLU and the RoPE rotation ----
// x[i] *= s[i % C]: the scale of each channel over all tokens (x [T][C] row-major)
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_modulate(float* x, const float* s, int C, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) x[i] *= s[i % C];
}
// out[i] += g[i % C] * y[i]: the gated residual
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_gate_add(float* out, const float* g, const float* y, int C, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) out[i] += g[i % C] * y[i];
}
// SwiGLU: g[i] = silu(a[i]) * b[i]
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_swiglu(const float* a, const float* b, float* g, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) g[i] = a[i] / (1.0f + expf(-a[i])) * b[i];
}
// RoPE on q [T][heads][D] (D = 128: 64 complex pairs; the pairs (2p, 2p+1) of each head), fr [T][64][2] = (cos, sin)
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_rope(float* q, const float* fr, int T, int heads, long n_pairs) {
    const long j = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (j >= n_pairs) return;
    const int D = 128, P = 64;
    const long t = j / (heads * P);
    const long rem = j % (heads * P);
    const int h = (int)(rem / P), p = (int)(rem % P);
    float* v = q + t * heads * D + h * D + 2 * p;
    const float c = fr[(t * P + p) * 2], sn = fr[(t * P + p) * 2 + 1];
    const float re = v[0], im = v[1];
    v[0] = re * c - im * sn;
    v[1] = re * sn + im * c;
}

// GEGLU: g[t][j] = hid * gelu(gate) for pj[t] = (hid [ff], gate [ff])
extern "C" __global__ void __launch_bounds__(SD_THREADS) sd_geglu(const float* pj, long rows, int ff, float* g) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= rows * ff) return;
    const long t = i / ff;
    const int j = (int)(i % ff);
    const float hid = pj[t * 2 * ff + j], gate = pj[t * 2 * ff + ff + j];
    g[i] = hid * (0.5f * gate * (1.0f + erff(gate / sqrtf(2.0f))));
}

// bfloat16 A (the DiT's weights, read in place from the mapping): the same tile, the operand's bits shifted up
extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_gemm_bf(const uint16_t* A, long lda, const float* B, long sbk, long sbn, float* C, long cs_m, long cs_n, const uint16_t* bias,
           const float* biasf, float alpha, float beta, int M, int N, int K) {
    __shared__ float As[G_BK][G_BM + 1];
    __shared__ float Bs[G_BK][G_BN + 1];
    gemm_body<uint16_t, 2>(A, lda, B, sbk, sbn, C, cs_m, cs_n, bias, biasf, alpha, beta, M, N, K, As, Bs);
}

// ---- the tensor-core GEMM for BF16 weights (the DiT's linears): C[m * cs_m + n * cs_n] = alpha * sum_k A(m, k) B(k, n) +
// biasf[m] + beta * C, A(m, k) = A[m * lda + k] (BF16, k contiguous), B(k, n) = B[n * sbn + k] (F32, k contiguous: sbk 1; rounded
// to BF16 on load: the reference runs its activations in BF16 too). A 64 x 64 tile per block of 4 waves (wave32), a k-tile of 64
// (four 16-wide WMMA steps), each wave a 2 x 2 grid of 16 x 16 x 16 WMMA tiles. Fragment layout (RDNA3, wave32): a lane holds one
// row of A (lane & 15) and the same row again in the other half-wave; likewise one column of B; the accumulator's lane holds
// rows 2 i + (lane >> 4), column lane & 15. Requires K a multiple of 64, lda a multiple of 8, sbk 1 (the host checks).
typedef __bf16 bf16x16_t __attribute__((ext_vector_type(16)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));

__device__ __forceinline__ uint16_t f2bf(float f) {  // round to nearest, ties to even (the inputs are finite)
    union {
        float f;
        uint32_t u;
    } v;
    v.f = f;
    const uint32_t r = v.u + 0x7FFFu + ((v.u >> 16) & 1u);
    return (uint16_t)(r >> 16);
}

extern "C" __global__ void __launch_bounds__(128)
sd_gemm_wmma(const uint16_t* __restrict__ A0, long lda, const float* __restrict__ B0, long sbn, float* __restrict__ C0, long cs_m,
             long cs_n, const float* __restrict__ biasf, float alpha, float beta, int M, int N, int K, long bsA, long bsB,
             long bsC) {
    // a batch (blockIdx.z): the operands and the output offset by the batch strides (the attention's heads)
    const uint16_t* A = A0 + blockIdx.z * bsA;
    const float* B = B0 + blockIdx.z * bsB;
    float* C = C0 + blockIdx.z * bsC;
    __shared__ __attribute__((aligned(16))) uint16_t As[64][72];  // 144-byte rows: 16-byte aligned
    __shared__ __attribute__((aligned(16))) uint16_t Bs[64][72];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    const int wm = w >> 1, wn = w & 1;
    const int bm = blockIdx.y * 64, bn = blockIdx.x * 64;
    f32x8_t acc[2][2];
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int i = 0; i < 8; ++i) acc[a][b][i] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 64) {
        {  // A: thread t loads 32 values of row t / 2 at k0 + 32 * (t & 1)
            const int r = t >> 1, h = (t & 1) * 32, gm = bm + r;
            uint4 v[4];
            for (int j = 0; j < 4; ++j) {
                v[j] = {0u, 0u, 0u, 0u};
                if (gm < M) v[j] = *reinterpret_cast<const uint4*>(A + (long)gm * lda + k0 + h + 8 * j);
            }
            for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(&As[r][h + 8 * j]) = v[j];
        }
        {  // B: thread t converts 32 values of column t / 2 at k0 + 32 * (t & 1): eight float4 loads
            const int n = t >> 1, h = (t & 1) * 32, gn = bn + n;
            uint16_t q[32];
            for (int j = 0; j < 8; ++j) {
                float4 x = {0.0f, 0.0f, 0.0f, 0.0f};
                if (gn < N) x = *reinterpret_cast<const float4*>(B + (long)gn * sbn + k0 + h + 4 * j);
                q[4 * j] = f2bf(x.x);
                q[4 * j + 1] = f2bf(x.y);
                q[4 * j + 2] = f2bf(x.z);
                q[4 * j + 3] = f2bf(x.w);
            }
            for (int j = 0; j < 4; ++j) {
                uint4 v;
                __builtin_memcpy(&v, &q[8 * j], sizeof v);
                *reinterpret_cast<uint4*>(&Bs[n][h + 8 * j]) = v;
            }
        }
        __syncthreads();
        for (int ks = 0; ks < 4; ++ks) {
            bf16x16_t af[2], bfr[2];
            for (int mt = 0; mt < 2; ++mt) {
                const int row = wm * 32 + mt * 16 + (l & 15);
                __builtin_memcpy(&af[mt], &As[row][ks * 16], sizeof(bf16x16_t));
            }
            for (int nt = 0; nt < 2; ++nt) {
                const int col = wn * 32 + nt * 16 + (l & 15);
                __builtin_memcpy(&bfr[nt], &Bs[col][ks * 16], sizeof(bf16x16_t));
            }
            for (int mt = 0; mt < 2; ++mt)
                for (int nt = 0; nt < 2; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[mt], bfr[nt], acc[mt][nt]);
        }
        __syncthreads();
    }
    for (int mt = 0; mt < 2; ++mt)
        for (int nt = 0; nt < 2; ++nt)
            for (int i = 0; i < 8; ++i) {
                const int gm = bm + wm * 32 + mt * 16 + 2 * i + (l >> 4);
                const int gn = bn + wn * 32 + nt * 16 + (l & 15);
                if (gm < M && gn < N) {
                    float v = alpha * acc[mt][nt][i];
                    if (biasf) v += biasf[gm];
                    const long off = (long)gm * cs_m + (long)gn * cs_n;
                    if (beta != 0.0f) v += beta * C[off];
                    C[off] = v;
                }
            }
}

// fp32 -> bf16 (round to nearest even), n values
extern "C" __global__ void __launch_bounds__(256) sd_to_bf16(const float* x, uint16_t* y, long n) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i < n) y[i] = f2bf(x[i]);
}

// The DiT's producers of GEMM inputs, in BF16 (the same operations in the same order as the FP32 kernels they replace, one
// rounding at the end: the results are bit-identical to rounding the FP32 outputs). y[t][c] = BF16(rms(x[t])[c] w[c] s[c]),
// the RMS norm (sd_rms_norm) and the AdaLN scale (zimage_modulate) in one pass
extern "C" __global__ void __launch_bounds__(SD_THREADS)
zimage_rms_mod_bf16(const float* x, int C, const uint16_t* w, float eps, const float* s, uint16_t* y, int ex) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    uint16_t* yr = y + (long)blockIdx.x * C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) q += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) {
        if (ex) {  // diffusers' RMSNorm (BF16 x * rsqrt, then BF16 times the weight) and the BF16 scale
            const float n1 = bf2f(f2bf(xr[i] * inv));
            yr[i] = f2bf(bf2f(f2bf(n1 * bf2f(w[i]))) * s[i]);
        } else {
            yr[i] = f2bf(xr[i] * inv * bf2f(w[i]) * s[i]);
        }
    }
}

// g[i] = BF16(silu(a[i]) * b[i]): zimage_swiglu's values, rounded to BF16 (the feed-forward's down projection takes them)
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_swiglu_bf16(const float* a, const float* b, uint16_t* g, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) g[i] = f2bf(a[i] / (1.0f + expf(-a[i])) * b[i]);
}

// The im2col in BF16 through shared memory (stride 1, k <= 3, the output width a multiple of 64, the channels a multiple of
// 8: the VAE's convolutions). A block is 64 consecutive output pixels of one row (blockIdx.x) and 8 input channels
// (blockIdx.y): the input patch (8 channels x 3 rows x 66 columns) is read once and the block writes its 72 taps (8 x 9) of
// each pixel as one contiguous run of colT (token-major: colT[n][r], n = y Wo + x, r = c k k + ky k + kx).
#define IM2COL_CB 8
extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_im2col_bf16_tile(const float* x, int cin, int H, int W, int k, int p, int Wo, int K, uint16_t* colT) {
    __shared__ float patch[IM2COL_CB][3][66];
    const int tiles_x = Wo / 64;
    const int yy = (int)blockIdx.x / tiles_x;
    const int xseg = ((int)blockIdx.x % tiles_x) * 64;
    const int cb = blockIdx.y;
    const int tid = threadIdx.x;
    const int kk = k * k;
    for (int idx = tid; idx < IM2COL_CB * 3 * 66; idx += SD_THREADS) {
        const int cl = idx / (3 * 66), rem = idx % (3 * 66), ry = rem / 66, cx = rem % 66;
        const int c = cb * IM2COL_CB + cl, gy = yy - p + ry, gx = xseg - p + cx;
        float v = 0.0f;
        if (c < cin && gy >= 0 && gy < H && gx >= 0 && gx < W) v = x[((long)c * H + gy) * W + gx];
        patch[cl][ry][cx] = v;
    }
    __syncthreads();
    const int taps = IM2COL_CB * kk;
    for (int idx = tid; idx < 64 * taps; idx += SD_THREADS) {
        const int j = idx / taps, rr = idx % taps;
        const int cl = rr / kk, rem = rr % kk, ky = rem / k, kx = rem % k;
        const long n = (long)yy * Wo + xseg + j;
        colT[n * K + cb * taps + rr] = f2bf(patch[cl][ky][j + kx]);
    }
}

// The DiT's BF16 chain (the linears' outputs in BF16). head_norm_rope_bf16: per (token, head) row of 128 values (x in BF16,
// y BF16, may be x): y = rope(rms(x) w), the same operations as sd_rms_norm (rows of 128) and zimage_rope, one rounding
extern "C" __global__ void __launch_bounds__(128)
zimage_head_norm_rope_bf16(const uint16_t* x, int H, const uint16_t* w, float eps, const float* fr, uint16_t* y) {
    __shared__ float red[128];
    __shared__ float nrm[128];
    const long row = blockIdx.x;
    const int t = (int)(row / H);
    const int i = threadIdx.x;
    const float xv = bf2f(x[row * 128 + i]);
    red[i] = xv * xv;
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) {
        if (i < off) red[i] += red[i + off];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / 128.0f + eps);
    nrm[i] = xv * inv * bf2f(w[i]);
    __syncthreads();
    if (i < 64) {
        const float c = fr[(t * 64 + i) * 2], sn = fr[(t * 64 + i) * 2 + 1];
        const float re = nrm[2 * i], im = nrm[2 * i + 1];
        y[row * 128 + 2 * i] = f2bf(re * c - im * sn);
        y[row * 128 + 2 * i + 1] = f2bf(re * sn + im * c);
    }
}

// Vt[c * Tkp + j] = BF16 V[j * C + c] (zero for Tk <= j < Tkp): sd_transpose_pad_bf16 for V in BF16 (the same tile form)
extern "C" __global__ void __launch_bounds__(256) zimage_transpose_pad_bf16_in(const uint16_t* V, int Tk, int Tkp, int C, uint16_t* Vt) {
    __shared__ float tile[32][33];
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    const int tiles_c = (C + 31) / 32;
    const int bc = blockIdx.x % tiles_c, bj = blockIdx.x / tiles_c;
    const int c0 = bc * 32, j0 = bj * 32;
    for (int k = 0; k < 4; ++k) {
        const int j = j0 + ty + 8 * k, c = c0 + tx;
        tile[ty + 8 * k][tx] = (j < Tk && c < C) ? bf2f(V[(long)j * C + c]) : 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < 4; ++k) {
        const int c = c0 + ty + 8 * k, j = j0 + tx;
        if (c < C && j < Tkp) Vt[(long)c * Tkp + j] = f2bf(tile[tx][ty + 8 * k]);
    }
}

// x[t][c] += g[c] * (rms(o[t]) w[c]): the RMS norm of a BF16 row (sd_rms_norm's operations) and the gated residual
// (zimage_gate_add), in place on the FP32 residual x
extern "C" __global__ void __launch_bounds__(SD_THREADS)
zimage_rms_gate_add_bf16(float* x, int C, const uint16_t* o, const uint16_t* w, float eps, const float* g, int ex) {
    __shared__ float red[SD_THREADS];
    const uint16_t* orow = o + (long)blockIdx.x * C;
    float* xr = x + (long)blockIdx.x * C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) {
        const float ov = bf2f(orow[i]);
        q += ov * ov;
    }
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) {
        if (ex) {  // diffusers: the normed value in BF16, the gate times it in BF16, the residual sum in BF16
            const float nv = bf2f(f2bf(bf2f(f2bf(bf2f(orow[i]) * inv)) * bf2f(w[i])));
            xr[i] = bf2f(f2bf(xr[i] + bf2f(f2bf(g[i] * nv))));
        } else {
            xr[i] += g[i] * (bf2f(orow[i]) * inv * bf2f(w[i]));
        }
    }
}

// g = BF16(silu(a) b) with a, b in BF16 (the SwiGLU of the BF16 products)
extern "C" __global__ void __launch_bounds__(SD_THREADS) zimage_swiglu_bf16_in(const uint16_t* a, const uint16_t* b, uint16_t* g, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) {
        const float av = bf2f(a[i]);
        g[i] = f2bf(av / (1.0f + expf(-av)) * bf2f(b[i]));
    }
}

// The number of 32-bit words that differ (the tuner's check that two algorithms give the same bits): count += 1 per word
extern "C" __global__ void __launch_bounds__(SD_THREADS) sd_count_diff(const uint32_t* a, const uint32_t* b, long n, unsigned* count) {
    for (long i = (long)blockIdx.x * SD_THREADS + threadIdx.x; i < n; i += (long)gridDim.x * SD_THREADS)
        if (a[i] != b[i]) atomicAdd(count, 1u);
}

// y[i] = BF16(y[i]) as FP32 (in place): a linear's output rounded to BF16 (the precision experiment: SD_ROUND_OUT)
extern "C" __global__ void __launch_bounds__(SD_THREADS) sd_round_bf16(float* y, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) y[i] = bf2f(f2bf(y[i]));
}

// y[i] = BF16 widened to FP32 (exact): the small-token linears take FP32 activations
extern "C" __global__ void __launch_bounds__(SD_THREADS) sd_bf16_to_f32(const uint16_t* x, float* y, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) y[i] = bf2f(x[i]);
}

// colT[j][r] = BF16(col[r][j]) for col [K][nc] (sd_im2col's columns): a 32 x 32 tile through LDS, so the reads run along
// the columns and the writes along the rows (block 32 x 8)
extern "C" __global__ void __launch_bounds__(256) sd_transpose_bf16(const float* col, int K, int nc, uint16_t* colT) {
    __shared__ float tile[32][33];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int j0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int i = 0; i < 32; i += 8) {
        const int r = r0 + ty + i, j = j0 + tx;
        if (r < K && j < nc) tile[ty + i][tx] = col[(long)r * nc + j];
    }
    __syncthreads();
    for (int i = 0; i < 32; i += 8) {
        const int j = j0 + ty + i, r = r0 + tx;
        if (j < nc && r < K) colT[(long)j * K + r] = f2bf(tile[tx][ty + i]);
    }
}

// y[c * HW + p] += b[c] for p < nc, c < C: the convolution's bias on the channel-major output (the chunk at its column)
extern "C" __global__ void __launch_bounds__(SD_THREADS)
sd_bias_cm(float* y, long HW, int nc, int C, const float* b) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= (long)C * nc) return;
    const int c = (int)(e / nc), p = (int)(e % nc);
    y[(long)c * HW + p] += b[c];
}

// Vt[c * Tkp + j] = V[j * C + c] for j < Tk, zero for Tk <= j < Tkp (the attention's values, transposed and padded)
extern "C" __global__ void __launch_bounds__(256) sd_transpose_pad(const float* V, int Tk, int Tkp, int C, float* Vt) {
    const long i = (long)blockIdx.x * 256 + threadIdx.x;
    if (i >= (long)C * Tkp) return;
    const int c = (int)(i / Tkp), j = (int)(i % Tkp);
    Vt[i] = j < Tk ? V[(long)j * C + c] : 0.0f;
}

// Each row of the scores S [heads][Tq][Tk] (f32): softmax, written as bf16 P [heads][Tq][Tkp] with zeros in the pad columns.
// One block per row (grid = heads * Tq).
extern "C" __global__ void __launch_bounds__(256)
sd_softmax_bf16(const float* S, uint16_t* P, int Tq, int Tk, int Tkp) {
    __shared__ float red[256];
    const long row = blockIdx.x;
    const float* s = S + row * Tk;
    uint16_t* p = P + row * Tkp;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < Tk; j += 256) mx = fmaxf(mx, s[j]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < Tk; j += 256) sum += expf(s[j] - mx);
    const float inv = 1.0f / block_sum(sum, red);
    for (int j = threadIdx.x; j < Tkp; j += 256) p[j] = j < Tk ? f2bf(expf(s[j] - mx) * inv) : (uint16_t)0;
}

// The 128 x 128 tile of sd_gemm_wmma: eight waves in a 4 x 2 arrangement, each a 32 x 64 region (2 x 4 WMMA tiles). Same operands,
// strides, batches and tails as sd_gemm_wmma; blocks of 256 threads. A k-tile of 64, LDS 2 x 18 KB.
extern "C" __global__ void __launch_bounds__(256)
sd_gemm_wmma128(const uint16_t* __restrict__ A0, long lda, const float* __restrict__ B0, long sbn, float* __restrict__ C0, long cs_m,
                long cs_n, const float* __restrict__ biasf, float alpha, float beta, int M, int N, int K, long bsA, long bsB,
                long bsC) {
    const uint16_t* A = A0 + blockIdx.z * bsA;
    const float* B = B0 + blockIdx.z * bsB;
    float* C = C0 + blockIdx.z * bsC;
    __shared__ __attribute__((aligned(16))) uint16_t As[128][72];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[128][72];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    const int wm = w >> 1, wn = w & 1;  // wm 0..3 (32 rows each), wn 0..1 (64 columns each)
    const int bm = blockIdx.y * 128, bn = blockIdx.x * 128;
    f32x8_t acc[2][4];
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < 8; ++i) acc[a][b][i] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 64) {
        {  // A: thread t loads 32 values of row t / 2 at k0 + 32 * (t & 1)
            const int r = t >> 1, h = (t & 1) * 32, gm = bm + r;
            uint4 v[4];
            for (int j = 0; j < 4; ++j) {
                v[j] = {0u, 0u, 0u, 0u};
                if (gm < M) v[j] = *reinterpret_cast<const uint4*>(A + (long)gm * lda + k0 + h + 8 * j);
            }
            for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(&As[r][h + 8 * j]) = v[j];
        }
        {  // B: thread t converts 32 values of column t / 2 at k0 + 32 * (t & 1)
            const int n = t >> 1, h = (t & 1) * 32, gn = bn + n;
            uint16_t q[32];
            for (int j = 0; j < 8; ++j) {
                float4 x = {0.0f, 0.0f, 0.0f, 0.0f};
                if (gn < N) x = *reinterpret_cast<const float4*>(B + (long)gn * sbn + k0 + h + 4 * j);
                q[4 * j] = f2bf(x.x);
                q[4 * j + 1] = f2bf(x.y);
                q[4 * j + 2] = f2bf(x.z);
                q[4 * j + 3] = f2bf(x.w);
            }
            for (int j = 0; j < 4; ++j) {
                uint4 v;
                __builtin_memcpy(&v, &q[8 * j], sizeof v);
                *reinterpret_cast<uint4*>(&Bs[n][h + 8 * j]) = v;
            }
        }
        __syncthreads();
        for (int ks = 0; ks < 4; ++ks) {
            bf16x16_t af[2], bfr[4];
            for (int mt = 0; mt < 2; ++mt) {
                const int row = wm * 32 + mt * 16 + (l & 15);
                __builtin_memcpy(&af[mt], &As[row][ks * 16], sizeof(bf16x16_t));
            }
            for (int nt = 0; nt < 4; ++nt) {
                const int col = wn * 64 + nt * 16 + (l & 15);
                __builtin_memcpy(&bfr[nt], &Bs[col][ks * 16], sizeof(bf16x16_t));
            }
            for (int mt = 0; mt < 2; ++mt)
                for (int nt = 0; nt < 4; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[mt], bfr[nt], acc[mt][nt]);
        }
        __syncthreads();
    }
    for (int mt = 0; mt < 2; ++mt)
        for (int nt = 0; nt < 4; ++nt)
            for (int i = 0; i < 8; ++i) {
                const int gm = bm + wm * 32 + mt * 16 + 2 * i + (l >> 4);
                const int gn = bn + wn * 64 + nt * 16 + (l & 15);
                if (gm < M && gn < N) {
                    float v = alpha * acc[mt][nt][i];
                    if (biasf) v += biasf[gm];
                    const long off = (long)gm * cs_m + (long)gn * cs_n;
                    if (beta != 0.0f) v += beta * C[off];
                    C[off] = v;
                }
            }
}

// The 256 x 128 tile of sd_gemm_wmma (for the large token counts: the activations are re-read once per 256 output rows): eight waves
// in a 4 x 2 arrangement, each a 64 x 64 region (4 x 4 WMMA tiles). Same operands,
// strides, batches and tails as sd_gemm_wmma; blocks of 256 threads. A k-tile of 64, LDS 2 x 18 KB.
extern "C" __global__ void __launch_bounds__(256)
sd_gemm_wmma256(const uint16_t* __restrict__ A0, long lda, const float* __restrict__ B0, long sbn, float* __restrict__ C0, long cs_m,
                long cs_n, const float* __restrict__ biasf, float alpha, float beta, int M, int N, int K, long bsA, long bsB,
                long bsC) {
    const uint16_t* A = A0 + blockIdx.z * bsA;
    const float* B = B0 + blockIdx.z * bsB;
    float* C = C0 + blockIdx.z * bsC;
    __shared__ __attribute__((aligned(16))) uint16_t As[256][72];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[128][72];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    const int wm = w >> 1, wn = w & 1;  // wm 0..3 (64 rows each), wn 0..1 (64 columns each)
    const int bm = blockIdx.y * 256, bn = blockIdx.x * 128;
    f32x8_t acc[4][4];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < 8; ++i) acc[a][b][i] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 64) {
        {  // A: thread t loads the 64 values of row t at k0 (256 rows, 256 threads)
            const int r = t, gm = bm + r;
            uint4 v[8];
            for (int j = 0; j < 8; ++j) {
                v[j] = {0u, 0u, 0u, 0u};
                if (gm < M) v[j] = *reinterpret_cast<const uint4*>(A + (long)gm * lda + k0 + 8 * j);
            }
            for (int j = 0; j < 8; ++j) *reinterpret_cast<uint4*>(&As[r][8 * j]) = v[j];
        }
        {  // B: thread t converts 32 values of column t / 2 at k0 + 32 * (t & 1)
            const int n = t >> 1, h = (t & 1) * 32, gn = bn + n;
            uint16_t q[32];
            for (int j = 0; j < 8; ++j) {
                float4 x = {0.0f, 0.0f, 0.0f, 0.0f};
                if (gn < N) x = *reinterpret_cast<const float4*>(B + (long)gn * sbn + k0 + h + 4 * j);
                q[4 * j] = f2bf(x.x);
                q[4 * j + 1] = f2bf(x.y);
                q[4 * j + 2] = f2bf(x.z);
                q[4 * j + 3] = f2bf(x.w);
            }
            for (int j = 0; j < 4; ++j) {
                uint4 v;
                __builtin_memcpy(&v, &q[8 * j], sizeof v);
                *reinterpret_cast<uint4*>(&Bs[n][h + 8 * j]) = v;
            }
        }
        __syncthreads();
        for (int ks = 0; ks < 4; ++ks) {
            bf16x16_t af[4], bfr[4];
            for (int mt = 0; mt < 4; ++mt) {
                const int row = wm * 64 + mt * 16 + (l & 15);
                __builtin_memcpy(&af[mt], &As[row][ks * 16], sizeof(bf16x16_t));
            }
            for (int nt = 0; nt < 4; ++nt) {
                const int col = wn * 64 + nt * 16 + (l & 15);
                __builtin_memcpy(&bfr[nt], &Bs[col][ks * 16], sizeof(bf16x16_t));
            }
            for (int mt = 0; mt < 4; ++mt)
                for (int nt = 0; nt < 4; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[mt], bfr[nt], acc[mt][nt]);
        }
        __syncthreads();
    }
    for (int mt = 0; mt < 4; ++mt)
        for (int nt = 0; nt < 4; ++nt)
            for (int i = 0; i < 8; ++i) {
                const int gm = bm + wm * 64 + mt * 16 + 2 * i + (l >> 4);
                const int gn = bn + wn * 64 + nt * 16 + (l & 15);
                if (gm < M && gn < N) {
                    float v = alpha * acc[mt][nt][i];
                    if (biasf) v += biasf[gm];
                    const long off = (long)gm * cs_m + (long)gn * cs_n;
                    if (beta != 0.0f) v += beta * C[off];
                    C[off] = v;
                }
            }
}


// The 128 x 128 tile of sd_gemm_wmma with BF16 B too (sd_gemm_wmma128bb): the activations converted once, read as 16-byte vectors.
// The 128 x 128 tile of sd_gemm_wmma: eight waves in a 4 x 2 arrangement, each a 32 x 64 region (2 x 4 WMMA tiles). Same operands,
// strides, batches and tails as sd_gemm_wmma; blocks of 256 threads. A k-tile of 64, LDS 2 x 18 KB.
extern "C" __global__ void __launch_bounds__(256)
sd_gemm_wmma128bb(const uint16_t* __restrict__ A0, long lda, const uint16_t* __restrict__ B0, long sbn, float* __restrict__ C0, long cs_m,
                long cs_n, const float* __restrict__ biasf, float alpha, float beta, int M, int N, int K, long bsA, long bsB,
                long bsC) {
    const uint16_t* A = A0 + blockIdx.z * bsA;
    const uint16_t* B = B0 + blockIdx.z * bsB;
    float* C = C0 + blockIdx.z * bsC;
    __shared__ __attribute__((aligned(16))) uint16_t As[128][72];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[128][72];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    const int wm = w >> 1, wn = w & 1;  // wm 0..3 (32 rows each), wn 0..1 (64 columns each)
    const int bm = blockIdx.y * 128, bn = blockIdx.x * 128;
    f32x8_t acc[2][4];
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < 8; ++i) acc[a][b][i] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 64) {
        {  // A: thread t loads 32 values of row t / 2 at k0 + 32 * (t & 1)
            const int r = t >> 1, h = (t & 1) * 32, gm = bm + r;
            uint4 v[4];
            for (int j = 0; j < 4; ++j) {
                v[j] = {0u, 0u, 0u, 0u};
                if (gm < M) v[j] = *reinterpret_cast<const uint4*>(A + (long)gm * lda + k0 + h + 8 * j);
            }
            for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(&As[r][h + 8 * j]) = v[j];
        }
        {  // B: thread t copies 32 BF16 values of column t / 2 at k0 + 32 * (t & 1)
            const int n = t >> 1, h = (t & 1) * 32, gn = bn + n;
            for (int j = 0; j < 4; ++j) {
                uint4 v = {0u, 0u, 0u, 0u};
                if (gn < N) v = *reinterpret_cast<const uint4*>(B + (long)gn * sbn + k0 + h + 8 * j);
                *reinterpret_cast<uint4*>(&Bs[n][h + 8 * j]) = v;
            }
        }
        __syncthreads();
        for (int ks = 0; ks < 4; ++ks) {
            bf16x16_t af[2], bfr[4];
            for (int mt = 0; mt < 2; ++mt) {
                const int row = wm * 32 + mt * 16 + (l & 15);
                __builtin_memcpy(&af[mt], &As[row][ks * 16], sizeof(bf16x16_t));
            }
            for (int nt = 0; nt < 4; ++nt) {
                const int col = wn * 64 + nt * 16 + (l & 15);
                __builtin_memcpy(&bfr[nt], &Bs[col][ks * 16], sizeof(bf16x16_t));
            }
            for (int mt = 0; mt < 2; ++mt)
                for (int nt = 0; nt < 4; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[mt], bfr[nt], acc[mt][nt]);
        }
        __syncthreads();
    }
    for (int mt = 0; mt < 2; ++mt)
        for (int nt = 0; nt < 4; ++nt)
            for (int i = 0; i < 8; ++i) {
                const int gm = bm + wm * 32 + mt * 16 + 2 * i + (l >> 4);
                const int gn = bn + wn * 64 + nt * 16 + (l & 15);
                if (gm < M && gn < N) {
                    float v = alpha * acc[mt][nt][i];
                    if (biasf) v += biasf[gm];
                    const long off = (long)gm * cs_m + (long)gn * cs_n;
                    if (beta != 0.0f) v += beta * C[off];
                    C[off] = v;
                }
            }
}

// Vt[c * Tkp + j] = V[j * C + c] in BF16 (zero for Tk <= j < Tkp): the attention's values for the batched GEMM. A 32 x 32 tile
// through LDS (one block of 256 threads per tile, the grid is the tiles in row-major order over (channel, key) blocks), so the
// reads of V and the writes of Vt are both contiguous. (The direct form read V with a stride of C per thread: 101 ms per call.)
extern "C" __global__ void __launch_bounds__(256) sd_transpose_pad_bf16(const float* V, int Tk, int Tkp, int C, uint16_t* Vt) {
    __shared__ float tile[32][33];
    const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    const int tiles_c = (C + 31) / 32;
    const int bc = blockIdx.x % tiles_c, bj = blockIdx.x / tiles_c;  // channel block, key block
    const int c0 = bc * 32, j0 = bj * 32;
    for (int k = 0; k < 4; ++k) {  // V rows (keys) j0 + ty + 8k, columns (channels) c0 + tx
        const int j = j0 + ty + 8 * k, c = c0 + tx;
        tile[ty + 8 * k][tx] = (j < Tk && c < C) ? V[(long)j * C + c] : 0.0f;
    }
    __syncthreads();
    for (int k = 0; k < 4; ++k) {  // Vt rows (channels) c0 + ty + 8k, columns (keys) j0 + tx
        const int c = c0 + ty + 8 * k, j = j0 + tx;
        if (c < C && j < Tkp) Vt[(long)c * Tkp + j] = f2bf(tile[tx][ty + 8 * k]);
    }
}

// The same softmax with one wave (32 lanes) per row, eight rows per 256-thread block: the reductions are wave shuffles, no block
// barriers (the rows are ~1000 long, so a block per row spent most of its time in barriers).
extern "C" __global__ void __launch_bounds__(256)
sd_softmax_bf16_w(const float* S, uint16_t* P, int Tq, int Tk, int Tkp, long rows) {
    const long row = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
    const int l = threadIdx.x & 31;
    if (row >= rows) return;
    const float* s = S + row * Tk;
    uint16_t* p = P + row * Tkp;
    float mx = -INFINITY;
    for (int j = l; j < Tk; j += 32) mx = fmaxf(mx, s[j]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
    float sum = 0.0f;
    for (int j = l; j < Tk; j += 32) sum += expf(s[j] - mx);
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
    const float inv = 1.0f / sum;
    for (int j = l; j < Tkp; j += 32) p[j] = j < Tk ? f2bf(expf(s[j] - mx) * inv) : (uint16_t)0;
}

// ---- flash attention (head width 128, BF16 Q, K and transposed V): one block per (128 query rows, head), eight waves of 16 rows.
// The K and V tiles (64 keys) are shared by the eight waves; Q lives in registers (eight A fragments per wave). The scores of
// a 64-key tile stay in the WMMA accumulators; the row max and sum are reduced over the 16 lanes of a row with DPP (no LDS
// round trip); the exponentials are exp2 of the scores scaled into log2 units; P goes through the wave's own LDS rows as the
// next WMMA's A operand; O accumulates in eight 16-wide tiles per wave. Keys beyond Tk are masked; the value rows beyond Tk are
// zero (the host pads them). Shared memory 54 KB.
template <int CTRL>
__device__ __forceinline__ float fa_dpp(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, x), CTRL, 0xF, 0xF, true));
}
__device__ __forceinline__ float fa_row16_max(float x) {
    x = fmaxf(x, fa_dpp<0xB1>(x));
    x = fmaxf(x, fa_dpp<0x4E>(x));
    x = fmaxf(x, fa_dpp<0x141>(x));
    x = fmaxf(x, fa_dpp<0x140>(x));
    return x;
}
__device__ __forceinline__ float fa_row16_sum(float x) {
    x += fa_dpp<0xB1>(x);
    x += fa_dpp<0x4E>(x);
    x += fa_dpp<0x141>(x);
    x += fa_dpp<0x140>(x);
    return x;
}

extern "C" __global__ void __launch_bounds__(256)
sd_flash_attn(const uint16_t* __restrict__ qb, const uint16_t* __restrict__ kb, const uint16_t* __restrict__ vt, float* __restrict__ out,
      int Tq, int Tk, int Tkp, int C, float scale) {
    __shared__ __attribute__((aligned(16))) uint16_t Ks[64][136];
    __shared__ __attribute__((aligned(16))) uint16_t Vs[128][72];
    __shared__ __attribute__((aligned(16))) uint16_t Ps[8][16][72];
    const int t = threadIdx.x, w = t >> 5, l = t & 31, g = l >> 4, c16 = l & 15;
    const int q0 = blockIdx.x * 128, h = blockIdx.y;
    const uint16_t* Kp = kb + h * 128;
    const uint16_t* Vp = vt + (long)h * 128 * Tkp;
    bf16x16_t qa[8];
    {
        const int gm = q0 + w * 16 + c16;
        for (int ks = 0; ks < 8; ++ks) {
            if (gm < Tq) __builtin_memcpy(&qa[ks], qb + (long)gm * C + h * 128 + ks * 16, sizeof(bf16x16_t));
            else for (int e = 0; e < 16; ++e) qa[ks][e] = (__bf16)0.0f;
        }
    }
    f32x8_t O[8];
    for (int dt = 0; dt < 8; ++dt)
        for (int i = 0; i < 8; ++i) O[dt][i] = 0.0f;
    float mrow[8], lrow[8];
    for (int i = 0; i < 8; ++i) {
        mrow[i] = -INFINITY;
        lrow[i] = 0.0f;
    }
    for (int k0 = 0; k0 < Tkp; k0 += 64) {
        {  // K tile: thread t loads 32 values (of 128) of key t >> 2
            const int r = t >> 2, qq = (t & 3) * 32, gk = k0 + r;
            for (int j = 0; j < 4; ++j) {
                uint4 v = {0u, 0u, 0u, 0u};
                if (gk < Tk) v = *reinterpret_cast<const uint4*>(Kp + (long)gk * C + qq + 8 * j);
                *reinterpret_cast<uint4*>(&Ks[r][qq + 8 * j]) = v;
            }
        }
        {  // V tile (transposed: dims x keys): thread t loads 32 keys (of 64) of dim t >> 1
            const int r = t >> 1, hf = (t & 1) * 32;
            const uint16_t* src = Vp + (long)r * Tkp + k0 + hf;
            for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(&Vs[r][hf + 8 * j]) = *reinterpret_cast<const uint4*>(src + 8 * j);
        }
        __syncthreads();
        f32x8_t S[4];
        for (int nt = 0; nt < 4; ++nt)
            for (int i = 0; i < 8; ++i) S[nt][i] = 0.0f;
        for (int ks = 0; ks < 8; ++ks)
            for (int nt = 0; nt < 4; ++nt) {
                bf16x16_t b;
                __builtin_memcpy(&b, &Ks[nt * 16 + c16][ks * 16], sizeof(bf16x16_t));
                S[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(qa[ks], b, S[nt]);
            }
        for (int i = 0; i < 8; ++i) {
            float v[4];
            for (int nt = 0; nt < 4; ++nt) {
                v[nt] = S[nt][i] * (scale * 1.4426950408889634f);  // log2 domain
                if (k0 + nt * 16 + c16 >= Tk) v[nt] = -INFINITY;
            }
            float mx = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
            mx = fa_row16_max(mx);
            const float mnew = fmaxf(mrow[i], mx);
            const float alpha = exp2f(mrow[i] - mnew);
            float p[4], sum = 0.0f;
            for (int nt = 0; nt < 4; ++nt) {
                p[nt] = exp2f(v[nt] - mnew);
                sum += p[nt];
            }
            sum = fa_row16_sum(sum);
            lrow[i] = lrow[i] * alpha + sum;
            mrow[i] = mnew;
            for (int dt = 0; dt < 8; ++dt) O[dt][i] *= alpha;
            const int row = 2 * i + g;
            for (int nt = 0; nt < 4; ++nt) Ps[w][row][nt * 16 + c16] = f2bf(p[nt]);
        }
        __syncthreads();
        for (int ks = 0; ks < 4; ++ks) {
            bf16x16_t a;
            __builtin_memcpy(&a, &Ps[w][c16][ks * 16], sizeof(bf16x16_t));
            for (int dt = 0; dt < 8; ++dt) {
                bf16x16_t b;
                __builtin_memcpy(&b, &Vs[dt * 16 + c16][ks * 16], sizeof(bf16x16_t));
                O[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, O[dt]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 8; ++i) {
        const int gm = q0 + w * 16 + 2 * i + g;
        if (gm >= Tq) continue;
        const float inv = 1.0f / lrow[i];
        for (int dt = 0; dt < 8; ++dt) out[(long)gm * C + h * 128 + dt * 16 + c16] = O[dt][i] * inv;
    }
}

// The 256 x 128 tile with BF16 activations (linear_bf16 converts the F32 input once per call, sd_to_bf16): the activations are
// read at half the bytes of sd_gemm_wmma256's, which was the bottleneck (the output tiles re-read them 16 times). A 32-wide
// k-tile, LDS double-buffered, the next tile loaded into registers while the WMMAs run on the current one (one barrier per
// k-tile). The output is staged through LDS, so each lane stores 8 consecutive rows at once (cs_m == 1). Same operands,
// strides, batches and tails as sd_gemm_wmma; blocks of 256 threads; K a multiple of 32. Measured on 4096 x 4105 x 4096:
// 16.1 ms (the F32 tile) -> 8.8 ms; the outputs are bit-identical to the BF16 tile without the staging.
#define G256B_PAD 40

// The 16 x 16 accumulator tiles of one wave through its own staging area (16 x 17 floats at ep): row = column of the output
// (gn), so the lanes write gm contiguously. Tails: rows and columns beyond M and N are not stored.
__device__ __forceinline__ void gemm256_store(f32x8_t (&acc)[4][4], float* ep_base, int w, int l, int wm, int wn, int bm, int bn,
                                              float* __restrict__ C, long cs_m, long cs_n, const float* __restrict__ biasf,
                                              float alpha, float beta, int M, int N) {
    float* ep = ep_base + w * (16 * 17);
    for (int mt = 0; mt < 4; ++mt)
        for (int nt = 0; nt < 4; ++nt) {
            for (int i = 0; i < 8; ++i) ep[(l & 15) * 17 + 2 * i + (l >> 4)] = alpha * acc[mt][nt][i];
            __builtin_amdgcn_wave_barrier();
            const int gl = l >> 1, half = (l & 1) * 8;
            const int gn = bn + wn * 64 + nt * 16 + gl, gm0 = bm + wm * 64 + mt * 16 + half;
            if (gn < N && gm0 + 8 <= M && cs_m == 1) {  // a full 8 rows: two float4 stores (the rows are 16-byte aligned)
                float v[8];
                for (int q = 0; q < 8; ++q) v[q] = ep[gl * 17 + half + q];
                if (biasf)
                    for (int q = 0; q < 8; ++q) v[q] += biasf[gm0 + q];
                float* dst = C + (long)gn * cs_n + gm0;
                if (beta != 0.0f)
                    for (int q = 0; q < 8; ++q) v[q] += beta * dst[q];
                *reinterpret_cast<float4*>(dst) = {v[0], v[1], v[2], v[3]};
                *reinterpret_cast<float4*>(dst + 4) = {v[4], v[5], v[6], v[7]};
            } else if (gn < N) {
                for (int q = 0; q < 8; ++q) {
                    const int gm = gm0 + q;
                    if (gm < M) {
                        float v = ep[gl * 17 + half + q];
                        if (biasf) v += biasf[gm];
                        const long off = (long)gm * cs_m + (long)gn * cs_n;
                        if (beta != 0.0f) v += beta * C[off];
                        C[off] = v;
                    }
                }
            }
            __builtin_amdgcn_wave_barrier();
        }
}

extern "C" __global__ void __launch_bounds__(256)
sd_gemm_wmma256b(const uint16_t* __restrict__ A0, long lda, const uint16_t* __restrict__ B0, long sbn, float* __restrict__ C0,
                 long cs_m, long cs_n, const float* __restrict__ biasf, float alpha, float beta, int M, int N, int K, long bsA,
                 long bsB, long bsC) {
    const uint16_t* A = A0 + blockIdx.z * bsA;
    const uint16_t* B = B0 + blockIdx.z * bsB;
    float* C = C0 + blockIdx.z * bsC;
    __shared__ __attribute__((aligned(16))) uint16_t As[2][256][G256B_PAD];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[2][128][G256B_PAD];
    const int t = threadIdx.x, w = t >> 5, l = t & 31;
    const int wm = w >> 1, wn = w & 1;  // wm 0..3 (64 rows each), wn 0..1 (64 columns each)
    const int bm = blockIdx.y * 256, bn = blockIdx.x * 128;
    const int bn_row = t >> 1, bh = (t & 1) * 16;  // B: thread t loads 16 values of column t >> 1
    uint4 ra[4], rb[2];
    f32x8_t acc[4][4];
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < 8; ++i) acc[a][b][i] = 0.0f;
    auto loadA = [&](int k0) {
        const int gm = bm + t;
        for (int j = 0; j < 4; ++j) {
            ra[j] = {0u, 0u, 0u, 0u};
            if (gm < M) ra[j] = *reinterpret_cast<const uint4*>(A + (long)gm * lda + k0 + 8 * j);
        }
    };
    auto loadB = [&](int k0) {
        const int gn = bn + bn_row;
        for (int j = 0; j < 2; ++j) {
            rb[j] = {0u, 0u, 0u, 0u};
            if (gn < N) rb[j] = *reinterpret_cast<const uint4*>(B + (long)gn * sbn + k0 + bh + 8 * j);
        }
    };
    auto storeA = [&](int buf) {
        for (int j = 0; j < 4; ++j) *reinterpret_cast<uint4*>(&As[buf][t][8 * j]) = ra[j];
    };
    auto storeB = [&](int buf) {
        for (int j = 0; j < 2; ++j) *reinterpret_cast<uint4*>(&Bs[buf][bn_row][bh + 8 * j]) = rb[j];
    };
    loadA(0);
    loadB(0);
    storeA(0);
    storeB(0);
    __syncthreads();
    int cur = 0;
    for (int k0 = 0; k0 < K; k0 += 32) {
        const bool more = k0 + 32 < K;
        if (more) {
            loadA(k0 + 32);
            loadB(k0 + 32);
        }
        for (int ks = 0; ks < 2; ++ks) {
            bf16x16_t af[4], bfr[4];
            for (int mt = 0; mt < 4; ++mt) {
                const int row = wm * 64 + mt * 16 + (l & 15);
                __builtin_memcpy(&af[mt], &As[cur][row][ks * 16], sizeof(bf16x16_t));
            }
            for (int nt = 0; nt < 4; ++nt) {
                const int col = wn * 64 + nt * 16 + (l & 15);
                __builtin_memcpy(&bfr[nt], &Bs[cur][col][ks * 16], sizeof(bf16x16_t));
            }
            for (int mt = 0; mt < 4; ++mt)
                for (int nt = 0; nt < 4; ++nt)
                    acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[mt], bfr[nt], acc[mt][nt]);
        }
        if (more) {
            storeA(cur ^ 1);
            storeB(cur ^ 1);
        }
        __syncthreads();
        cur ^= 1;
    }
    __syncthreads();  // the last k-tile's reads of the staging memory (the LDS below is reused for the output)
    gemm256_store(acc, reinterpret_cast<float*>(&As[0][0]), w, l, wm, wn, bm, bn, C, cs_m, cs_n, biasf, alpha, beta, M, N);
}

// Qwen-Image's linear layers (FP8 E4M3 weights read in place, F32 activations, F32 accumulation): y[t][o] = wscale * sum_i x[t][i]
// W[o][i] + bias[o], the weights decoded in the load (no conversion copy). A block is 256 threads in a line; the tile is 64 tokens by 64
// outputs with a k-step of 16, each thread 4 x 4 outputs. bias_kind: 0 none, 1 FP8 E4M3, 2 BF16 bits, 3 F32. in % 16 == 0.
__device__ __forceinline__ float qi_e4m3(uint8_t v) {
    const uint32_t s = (uint32_t)(v & 0x80) << 24;
    const int e = (v >> 3) & 0xF;
    const int m = v & 0x7;
    if (e == 0) {  // subnormal: m * 2^-9
        const float mag = (float)m * (1.0f / 512.0f);
        return (v & 0x80) ? -mag : mag;
    }
    if (e == 15 && m == 7) return __uint_as_float(0x7fc00000u);  // NaN (E4M3fn has no infinity)
    return __uint_as_float(s | ((uint32_t)(e - 7 + 127) << 23) | ((uint32_t)m << 20));
}

extern "C" __global__ void __launch_bounds__(256)
qi_linear_f8(const float* __restrict__ x, int T, const uint8_t* __restrict__ W, float wscale, int in, int out,
             const void* __restrict__ bias, int bias_kind, float* __restrict__ y) {
    __shared__ float As[16][64 + 1];
    __shared__ float Bs[16][64 + 1];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int t0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < in; k0 += 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = tid + j * 256;      // 1024 elements per tile: 64 rows x 16 k
            const int r = e >> 4, k = e & 15;
            const int t = t0 + r, o = o0 + r;
            As[k][r] = (t < T) ? x[(size_t)t * in + k0 + k] : 0.0f;
            Bs[k][r] = (o < out) ? qi_e4m3(W[(size_t)o * in + k0 + k]) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int o = o0 + tx * 4 + j;
        if (o >= out) continue;
        float bv = 0.0f;
        if (bias_kind == 1) bv = qi_e4m3(static_cast<const uint8_t*>(bias)[o]);
        else if (bias_kind == 2) bv = __uint_as_float((uint32_t)static_cast<const uint16_t*>(bias)[o] << 16);
        else if (bias_kind == 3) bv = static_cast<const float*>(bias)[o];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = t0 + ty * 4 + i;
            if (t < T) y[(size_t)t * out + o] = wscale * acc[i][j] + bv;
        }
    }
}

// Qwen-Image's RMS norm with E4M3 weights (the q/k norms of 128, the text norm of 3584; no scale): one block per row of x [T][C]
extern "C" __global__ void __launch_bounds__(SD_THREADS) qi_rms_norm_f8(const float* x, int C, const uint8_t* w, float eps, float* y) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    float* yr = y + (long)blockIdx.x * C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) q += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) yr[i] = xr[i] * inv * qi_e4m3(w[i]);
}

// x[t][c] = x[t][c] (1 + s[c]) + b[c] in place (the adaLN modulation: scale and shift per channel); n = T * C
extern "C" __global__ void __launch_bounds__(SD_THREADS) qi_affine(float* x, const float* s, const float* b, int C, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    const int c = (int)(i % C);
    x[i] = x[i] * (1.0f + s[c]) + b[c];
}

// GELU with the tanh approximation, in place: x = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
extern "C" __global__ void __launch_bounds__(SD_THREADS) qi_gelu_tanh(float* x, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// Qwen-Image's VAE decoder (token-major activations x[h][w][c]). The RMS norm over the channels of each pixel: y = x / max(|x|, 1e-12)
// sqrt(C) gamma (one block per pixel row of C).
extern "C" __global__ void __launch_bounds__(SD_THREADS) qv_rms_tok(const float* x, int C, const uint16_t* g, float* y) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    float* yr = y + (long)blockIdx.x * C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) q += xr[i] * xr[i];
    const float nrm = sqrtf(block_sum(q, red));
    const float s = sqrtf((float)C) / fmaxf(nrm, 1e-12f);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) yr[i] = xr[i] * s * bf2f(g[i]);
}

// The patches of the convolution as a matrix, one row per output pixel p = p0 + r of the H x W grid (zero outside the image):
// mode 1 (a 2D 3 x 3 kernel): K = 9 C, index c * 9 + kh * 3 + kw; mode 3 (a causal 3 x 3 x 3 kernel on one frame: the history is
// zero, so only the current frame's tap kt = 2 contributes): K = 27 C, index c * 27 + kt * 9 + kh * 3 + kw with kt != 2 zero.
// The weights of the file ([cout][cin][kt][kh][kw]) are read as they are (no repacking).
extern "C" __global__ void __launch_bounds__(SD_THREADS)
qv_patch(const float* x, int H, int W, int C, int mode, int p0, int n, float* P) {
    const int K = (mode == 3 ? 27 : 9) * C;
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= (long)n * K) return;
    const int r = (int)(e / K), j = (int)(e % K);
    const int p = p0 + r, h = p / W, w = p % W;
    const int c = mode == 3 ? j / 27 : j / 9;
    const int rem = mode == 3 ? j % 27 : j % 9;
    float v = 0.0f;
    if (mode == 3) {
        const int kt = rem / 9;
        if (kt == 2) {
            const int rr = rem % 9, ih = h + rr / 3 - 1, iw = w + rr % 3 - 1;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = x[((long)ih * W + iw) * C + c];
        }
    } else {
        const int ih = h + rem / 3 - 1, iw = w + rem % 3 - 1;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = x[((long)ih * W + iw) * C + c];
    }
    P[e] = v;
}

// y[r][c] += b[c] (BF16 bias), rows x C
extern "C" __global__ void __launch_bounds__(SD_THREADS) qv_bias_tok(float* y, const uint16_t* b, int C, long n) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= n) return;
    y[e] += bf2f(b[e % C]);
}

// nearest-exact 2x upsampling of x[h][w][c] to y[2h + a][2w + b][c]
extern "C" __global__ void __launch_bounds__(SD_THREADS) qv_upsample2_tok(const float* x, int H, int W, int C, float* y) {
    const long n = 4L * H * W * C;
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= n) return;
    const int c = (int)(e % C);
    const long pix = e / C;
    const int ox = (int)(pix % (2 * W)), oy = (int)(pix / (2 * W));
    y[e] = x[((long)(oy / 2) * W + ox / 2) * C + c];
}

// Qwen-Image 2.1's ConvRot layers (int8 weights in place, symmetric per output row: W = q s[o], q in [-127, 127]). The activations
// are rotated first by the same orthonormal regular Hadamard of 256 in blocks of the input dimension (qi_hadamard256). Y = X R Q^T:
// the GEMM is the qi_linear_f8 tile with the int8 weights and the row scale applied to the accumulator.
extern "C" __global__ void __launch_bounds__(256)
qi_hadamard256(const float* __restrict__ x, int T, int K, const float* __restrict__ H, float* __restrict__ y) {
    const int t = blockIdx.y;
    const int k = blockIdx.x * 256 + threadIdx.x;  /* the output column; grid.x = K / 256 blocks of 256 threads */
    if (t >= T || k >= K) return;
    const int b0 = (k / 256) * 256, c = k - b0;
    const float* xr = x + (size_t)t * K + b0;
    float acc = 0.0f;
    for (int j = 0; j < 256; ++j) acc += xr[j] * H[j * 256 + c];
    y[(size_t)t * K + k] = acc;
}

extern "C" __global__ void __launch_bounds__(256)
qi_linear_i8(const float* __restrict__ x, int T, const int8_t* __restrict__ W, const float* __restrict__ wscale, int in, int out,
             float* __restrict__ y) {
    __shared__ float As[16][64 + 1];
    __shared__ float Bs[16][64 + 1];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int t0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < in; k0 += 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = tid + j * 256;
            const int r = e >> 4, k = e & 15;
            const int t = t0 + r, o = o0 + r;
            As[k][r] = (t < T) ? x[(size_t)t * in + k0 + k] : 0.0f;
            Bs[k][r] = (o < out) ? (float)W[(size_t)o * in + k0 + k] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Bs[k][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int o = o0 + tx * 4 + j;
        if (o >= out) continue;
        const float s = wscale[o];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = t0 + ty * 4 + i;
            if (t < T) y[(size_t)t * out + o] = s * acc[i][j];
        }
    }
}

// Qwen-Image 2.1: the zero-centred RMS norm of the text projection (y = x rsqrt(mean(x^2) + eps) (w + 1), w BF16); one block per row
extern "C" __global__ void __launch_bounds__(SD_THREADS) qi21_zc_rms(const float* x, int C, const uint16_t* w, float eps, float* y) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    float* yr = y + (long)blockIdx.x * C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) q += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) yr[i] = xr[i] * inv * (bf2f(w[i]) + 1.0f);
}

// mod_tok[j][c] = mod[sel[j]][c] with sel[j] = 0 (the sampled timestep's row) or 1 (t = 0's row); mod is [2][C]
extern "C" __global__ void __launch_bounds__(SD_THREADS) qi21_gather_mod(const float* mod, const float* sel, int C, long n, float* out) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= n) return;
    const long j = e / C;
    const int c = (int)(e % C);
    out[e] = mod[(sel[j] > 0.5f ? C : 0) + c];
}

// x[e] *= (1 + s[e]) elementwise (the adaLN scale; the 2.1 variant has no shift)
extern "C" __global__ void __launch_bounds__(SD_THREADS) qi21_mul_1p(float* x, const float* s, long n) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= n) return;
    x[e] *= (1.0f + s[e]);
}

// h[e] += tanh(g[e]) * y[e]
extern "C" __global__ void __launch_bounds__(SD_THREADS) qi21_gated_add(float* h, const float* g, const float* y, long n) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= n) return;
    h[e] += tanhf(g[e]) * y[e];
}

// The text queries of Qwen-Image 2.1's attention: query i (i < L) sees the keys 0..i (causal among the text). q, k, v, out: [J][heads][128]
// (only the first L rows are read and written). One thread per (query, head): L * heads threads, a few keys each.
extern "C" __global__ void __launch_bounds__(SD_THREADS)
qi21_attn(const float* q, const float* k, const float* v, int J, int L, int heads, float* out) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= (long)L * heads) return;
    const int i = (int)(e / heads), h = (int)(e % heads);
    const int D = 128;
    const float scale = rsqrtf((float)D);
    const int kmax = i + 1;
    const float* qi = q + ((long)i * heads + h) * D;
    float m = -3.0e38f;
    for (int j = 0; j < kmax; ++j) {
        const float* kj = k + ((long)j * heads + h) * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += qi[d] * kj[d];
        m = fmaxf(m, s * scale);
    }
    float denom = 0.0f;
    float acc[128];
    for (int d = 0; d < D; ++d) acc[d] = 0.0f;
    for (int j = 0; j < kmax; ++j) {
        const float* kj = k + ((long)j * heads + h) * D;
        const float* vj = v + ((long)j * heads + h) * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += qi[d] * kj[d];
        const float p = expf(s * scale - m);
        denom += p;
        for (int d = 0; d < D; ++d) acc[d] += p * vj[d];
    }
    float* oi = out + ((long)i * heads + h) * D;
    for (int d = 0; d < D; ++d) oi[d] = acc[d] / denom;
}

// Qwen-Image 2.1's BF16 linears with FP32 activations (the weights BF16 bits read in place, decoded in the load; no rounding of the
// activations): y[t][o] = sum_i x[t][i] W[o][i]. The qi_linear_f8 tile with a BF16 decode. in % 16 == 0. Block: 256 threads in a line.
extern "C" __global__ void __launch_bounds__(256)
qi21_linear_bf16(const float* __restrict__ x, int T, const uint16_t* __restrict__ W, int in, int out, float* __restrict__ y) {
    __shared__ float As[16][64 + 1];
    __shared__ float Bs[16][64 + 1];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int t0 = blockIdx.y * 64, o0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < in; k0 += 16) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int e = tid + j * 256;
            const int r = e >> 4, k = e & 15;
            const int t = t0 + r, o = o0 + r;
            As[k][r] = (t < T) ? x[(size_t)t * in + k0 + k] : 0.0f;
            Bs[k][r] = (o < out) ? __uint_as_float((uint32_t)W[(size_t)o * in + k0 + k] << 16) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) b[jj] = Bs[k][tx * 4 + jj];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) acc[i][jj] += a[i] * b[jj];
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int o = o0 + tx * 4 + j;
        if (o >= out) continue;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int t = t0 + ty * 4 + i;
            if (t < T) y[(size_t)t * out + o] = acc[i][j];
        }
    }
}

// Qwen-Image 2.1's residual up block shortcut (DupUp3D, one frame, the first_chunk slice): y[oy][ox][o] = x[h][w][c] with h = oy / 2,
// w = ox / 2, the channel c = (o * factor + ft * 4 + (oy & 1) * 2 + (ox & 1)) / repeats, factor = factor_t * 4, repeats = out * factor /
// in, ft = factor_t - 1 (the time slot kept). Token-major [H][W][in] -> [2H][2W][out].
extern "C" __global__ void __launch_bounds__(SD_THREADS)
qi21_dupup(const float* x, int H, int W, int in, int out, int factor_t, float* y) {
    const long n = 4L * H * W * out;
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= n) return;
    const int o = (int)(e % out);
    const long pix = e / out;
    const int ox = (int)(pix % (2 * W)), oy = (int)(pix / (2 * W));
    const int factor = factor_t * 4, repeats = out * factor / in;
    const int ft = factor_t - 1;
    const int k = o * factor + ft * 4 + (oy & 1) * 2 + (ox & 1);
    const int c = k / repeats;
    y[e] = x[((long)(oy >> 1) * W + (ox >> 1)) * in + c];
}

// ---- AudioLDM 2's GPT-2 (the language model): fp32 kernels, simple on purpose (the stage check comes first) ----
// y[t][o] = b[o] + sum_k x[t][k] W[k][o]: x [T][in], W [in][out] (GPT-2's Conv1D layout), y [T][out]
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_gemm_f32(const float* __restrict__ x, int T, int in, const float* __restrict__ W,
                                                                        const float* __restrict__ b, int out, float* __restrict__ y) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= (long)T * out) return;
    const int o = (int)(i % out), t = (int)(i / out);
    float acc = b[o];
    const float* xr = x + (long)t * in;
    for (int k = 0; k < in; ++k) acc += xr[k] * W[(long)k * out + o];
    y[i] = acc;
}

// y[t][c] = (x[t][c] - mean) / sqrt(var + eps) * w[c] + b[c]: one block per row
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_ln_affine_f32(const float* __restrict__ x, int C, const float* __restrict__ w,
                                                                             const float* __restrict__ b, float eps, float* __restrict__ y) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    float* yr = y + (long)blockIdx.x * C;
    float s = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) s += xr[i];
    const float mean = block_sum(s, red) / (float)C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

// Causal multi-head attention with 64-wide heads: qkv [T][3 C] (q, k, v in the three parts, C = H * 64), out [T][C].
// Grid (T, H), 64 threads (one per head dimension); the scores of a query live in shared memory (dynamic: (64 + T) floats).
extern "C" __global__ void __launch_bounds__(64) audio_causal_attn_f32(const float* __restrict__ qkv, int T, int H, float* __restrict__ out, float scale) {
    extern __shared__ float sm[];
    float* qs = sm;        // 64: the query, then (thread 0) the normalizer
    float* sc = sm + 64;   // T: the scores, then the probabilities
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int C = H * 64, ld = 3 * C;
    qs[d] = qkv[(long)t * ld + h * 64 + d];
    __syncthreads();
    for (int j = d; j <= t; j += 64) {
        float acc = 0.0f;
        const float* kr = qkv + (long)j * ld + C + h * 64;
        for (int e = 0; e < 64; ++e) acc += qs[e] * kr[e];
        sc[j] = acc * scale;
    }
    __syncthreads();
    if (d == 0) {
        float m = sc[0];
        for (int j = 1; j <= t; ++j) m = fmaxf(m, sc[j]);
        float sum = 0.0f;
        for (int j = 0; j <= t; ++j) {
            sc[j] = expf(sc[j] - m);
            sum += sc[j];
        }
        qs[0] = 1.0f / sum;
    }
    __syncthreads();
    float acc = 0.0f;
    for (int j = 0; j <= t; ++j) acc += sc[j] * qkv[(long)j * ld + 2 * C + h * 64 + d];
    out[(long)t * C + h * 64 + d] = acc * qs[0];
}

// Multi-head attention for AudioLDM 2's models: q [Tq][C], k, v [Tk][C] with row stride ld (C = H * D, D <= 128); causal (keys up to the
// query, Tq = Tk) or not; kmask (Tk floats, 0 for a padding key) or null; scores * scale + bias[h][t][j] (bias null: none). out [Tq][C]. Grid (Tq, H),
// 128 threads (the head dimension's threads and the rest strided); the query and the scores in shared memory.
extern "C" __global__ void __launch_bounds__(128) audio_attn_f32(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                                                int ld, int Tq, int Tk, int H, int D, int causal, const float* __restrict__ kmask,
                                                                float scale, const float* __restrict__ bias, float* __restrict__ out) {
    extern __shared__ float sm[];
    float* qs = sm;        /* 128: the query (then, thread 0, the normalizer in qs[0]) */
    float* sc = sm + 128;  /* Tk: the scores, then the probabilities */
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int C = H * D;
    if (d < D) qs[d] = q[(long)t * ld + h * D + d];
    __syncthreads();
    const int jmax = causal ? t : Tk - 1;
    for (int j = d; j <= jmax; j += 128) {
        if (kmask && kmask[j] == 0.0f) {
            sc[j] = -INFINITY;
        } else {
            float acc = 0.0f;
            const float* kr = k + (long)j * ld + h * D;
            for (int e = 0; e < D; ++e) acc += qs[e] * kr[e];
            sc[j] = acc * scale + (bias ? bias[((long)h * Tq + t) * Tk + j] : 0.0f);
        }
    }
    __syncthreads();
    if (d == 0) {
        float m = -INFINITY;
        for (int j = 0; j <= jmax; ++j) m = fmaxf(m, sc[j]);
        float sum = 0.0f;
        for (int j = 0; j <= jmax; ++j) {
            sc[j] = expf(sc[j] - m);
            sum += sc[j];
        }
        qs[0] = 1.0f / sum;
    }
    __syncthreads();
    if (d < D) {
        float acc = 0.0f;
        for (int j = 0; j <= jmax; ++j) acc += sc[j] * v[(long)j * ld + h * D + d];
        out[(long)t * C + h * D + d] = acc * qs[0];
    }
}

// GELU (the exact erf form, as transformers' "gelu"), in place
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_gelu_erf_f32(float* x, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}

// RMS norm (T5's: no mean, no bias): y = x / sqrt(mean(x^2) + eps) * w, one block per row
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_rms_f32(const float* __restrict__ x, int C, const float* __restrict__ w, float eps,
                                                                       float* __restrict__ y) {
    __shared__ float red[SD_THREADS];
    const float* xr = x + (long)blockIdx.x * C;
    float* yr = y + (long)blockIdx.x * C;
    float q = 0.0f;
    for (int i = threadIdx.x; i < C; i += SD_THREADS) q += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(q, red) / (float)C + eps);
    for (int i = threadIdx.x; i < C; i += SD_THREADS) yr[i] = xr[i] * inv * w[i];
}

// a = gelu_new(a) * b (T5's gated GELU: the tanh approximation), in place
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_gelu_tanh_mul_f32(float* a, const float* b, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    const float v = a[i];
    a[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) * b[i];
}

// h[c][p] += t[c] over the HW positions of each channel (the time embedding of a resnet), in place
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_add_channel_f32(float* h, const float* t, int C, int HW) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= (long)C * HW) return;
    h[i] += t[i / HW];
}

// dst[i] = src[i]
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_copy_f32(float* dst, const float* src, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// 1-D convolution (stride 1, dilation dil, zero padding pad): x [cin][T], w [cout][cin][k], b [cout] -> y [cout][Tout]
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_conv1d_f32(const float* __restrict__ x, const float* __restrict__ w,
                                                                          const float* __restrict__ b, int cin, int cout, int T, int k,
                                                                          int pad, int dil, int Tout, float* __restrict__ y) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= (long)cout * Tout) return;
    const int co = (int)(i / Tout), t = (int)(i % Tout);
    float acc = b[co];
    for (int ci = 0; ci < cin; ++ci)
        for (int j = 0; j < k; ++j) {
            const int ti = t - pad + j * dil;
            if (ti >= 0 && ti < T) acc += x[(long)ci * T + ti] * w[((long)co * cin + ci) * k + j];
        }
    y[i] = acc;
}

// 1-D transposed convolution (stride s, zero padding pad): x [cin][T], w [cin][cout][k], b [cout] -> y [cout][Tout]
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_conv_transpose1d_f32(const float* __restrict__ x, const float* __restrict__ w,
                                                                                    const float* __restrict__ b, int cin, int cout, int T,
                                                                                    int k, int stride, int pad, int Tout, float* __restrict__ y) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= (long)cout * Tout) return;
    const int co = (int)(i / Tout), o = (int)(i % Tout);
    float acc = b[co];
    for (int j = 0; j < k; ++j) {
        const int num = o + pad - j;
        if (num < 0 || num % stride != 0) continue;
        const int t = num / stride;
        if (t >= T) continue;
        for (int ci = 0; ci < cin; ++ci) acc += x[(long)ci * T + t] * w[((long)ci * cout + co) * k + j];
    }
    y[i] = acc;
}

// x = leaky_relu(x, slope), in place
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_leaky_relu_f32(float* x, float slope, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n && x[i] < 0.0f) x[i] *= slope;
}

// x = x * s, in place
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_scale_f32(float* x, float s, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) x[i] *= s;
}

// x = tanh(x), in place
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_tanh_f32(float* x, long n) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i < n) x[i] = tanhf(x[i]);
}

// nearest resampling of x [C][H][W] to [C][Ho][Wo] (PyTorch's nearest: the source index floor(dst * H / Ho))
extern "C" __global__ void __launch_bounds__(SD_THREADS) audio_upsample_nearest_f32(const float* __restrict__ x, int C, int H, int W, int Ho, int Wo,
                                                                                     float* __restrict__ y) {
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= (long)C * Ho * Wo) return;
    const int ox = (int)(i % Wo), oy = (int)((i / Wo) % Ho), c = (int)(i / ((long)Wo * Ho));
    const float sy = (float)H / (float)Ho, sx = (float)W / (float)Wo;
    int iy = (int)floorf((float)oy * sy), ix = (int)floorf((float)ox * sx);
    iy = iy < H - 1 ? iy : H - 1;
    ix = ix < W - 1 ? ix : W - 1;
    y[i] = x[((long)c * H + iy) * W + ix];
}

// ---- TRELLIS.2 sparse-structure flow (src/trellis_ss.cpp): x = x * s + sh per channel over [T][C]; h += g * y per channel;
// per token and head the q/k RMS norm of the DiT (x = normalize(x) * sqrt(D) * gamma, F.normalize over the head's D values),
// then the RoPE pairs (2i, 2i+1) rotated by phases[t][i] = (cos, sin) when phases is not null. D = 128, one block per (t, h).
extern "C" __global__ void __launch_bounds__(SD_THREADS) ss_affine(float* x, long n, int C, const float* s, const float* sh) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= n) return;
    const int c = (int)(e % C);
    x[e] = x[e] * s[c] + sh[c];
}

extern "C" __global__ void __launch_bounds__(SD_THREADS) ss_gate_add(float* h, long n, int C, const float* g, const float* y) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= n) return;
    h[e] += g[(int)(e % C)] * y[e];
}

extern "C" __global__ void __launch_bounds__(128) ss_head_norm_rope(float* x, int ld, int H, int D, const float* gamma, const float* phases) {
    __shared__ float v[128];
    __shared__ float red[128];
    const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float* row = x + (long)t * ld + (long)h * D;
    const float a = row[d];
    red[d] = a * a;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (d < s) red[d] += red[d + s];
        __syncthreads();
    }
    const float scale = sqrtf((float)D) / fmaxf(sqrtf(red[0]), 1e-12f);
    float y = a * scale * gamma[h * D + d];
    if (phases) {
        v[d] = y;
        __syncthreads();
        const int i = d >> 1;
        const float c = phases[((long)t * (D / 2) + i) * 2], sn = phases[((long)t * (D / 2) + i) * 2 + 1];
        const float re = v[2 * i], im = v[2 * i + 1];
        y = (d & 1) ? (re * sn + im * c) : (re * c - im * sn);
    }
    row[d] = y;
}

// ---- TRELLIS.2 sparse-structure decoder (src/trellis_ss_dec.cpp): token-major activations [V][C], V = n^3 voxels
// im2col of a 3x3x3 convolution with padding 1: P[v][c*27 + t] = x[neighbour t of v][c] (zero outside); x is [V][C]
extern "C" __global__ void __launch_bounds__(SD_THREADS) ss_im2col3(const float* x, int C, int n, float* P) {
    const long K = (long)C * 27, V = (long)n * n * n;
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= V * K) return;
    const long v = e / K;
    const int k = (int)(e % K);
    const int c = k / 27, t = k % 27;
    const int i = t / 9, j = (t / 3) % 3, q = t % 3;
    const int xa = (int)(v / ((long)n * n)), yb = (int)((v / n) % n), zc = (int)(v % n);
    const int xi = xa + i - 1, yi = yb + j - 1, zi = zc + q - 1;
    float val = 0.0f;
    if (xi >= 0 && xi < n && yi >= 0 && yi < n && zi >= 0 && zi < n)
        val = x[(((long)xi * n + yi) * n + zi) * C + c];
    P[e] = val;
}

// y[v][o] = b[o] + sum_k P[v][k] * W[o][k] for v in [v0, v0 + nv), o in [o0, o0 + no); W is fp16 (read in place)
extern "C" __global__ void __launch_bounds__(256) ss_gemm_f16w(const float* P, long K, const uint16_t* W, const float* b,
                                                              int Cout, long Vtot, float* y, int v0, int nv, int o0, int no) {
    const int v = v0 + blockIdx.x * 16 + threadIdx.x;
    const int o = o0 + blockIdx.y * 16 + threadIdx.y;
    if (v >= v0 + nv || o >= o0 + no) return;
    const float* p = P + (long)v * K;
    const uint16_t* w = W + (long)o * K;
    float acc = b[o];
    for (long k = 0; k < K; ++k) acc += p[k] * h2f(w[k]);
    y[(long)v * Cout + o] = acc;
}

// the 3D pixel shuffle, token-major: x [V_in][Cout * 8] -> y [V_out][Cout] with V_out = (2n)^3;
// y[(X,Y,Z)][c] = x[(X/2,Y/2,Z/2)][c*8 + (X&1)*4 + (Y&1)*2 + (Z&1)] (the python reshape/permute of pixel_shuffle_3d)
extern "C" __global__ void __launch_bounds__(SD_THREADS) ss_pixel_shuffle3(const float* x, int Cout, int n, float* y) {
    const long n2 = 2L * n;
    const long Vo = n2 * n2 * n2;
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= Vo * Cout) return;
    const long v = e / Cout;
    const int c = (int)(e % Cout);
    const int X = (int)(v / (n2 * n2)), Y = (int)((v / n2) % n2), Z = (int)(v % n2);
    const int i = X & 1, j = Y & 1, q = Z & 1;
    const long vin = ((long)(X >> 1) * n + (Y >> 1)) * n + (Z >> 1);
    y[e] = x[vin * (Cout * 8L) + c * 8 + i * 4 + j * 2 + q];
}

// ---- sparse voxels (src/sparse.c): active voxels as int32 coords [N][4] = (batch, x, y, z), plain C-style kernels.
// A hash table of 64-bit keys (open addressing, linear probing, power-of-two size) maps a voxel's key to its index.
#define SP_EMPTY 0xFFFFFFFFFFFFFFFFull
#define SP_BIAS (1 << 19)

__device__ __forceinline__ unsigned long long sp_pack(int b, int x, int y, int z) {
    return ((unsigned long long)b << 60) | ((unsigned long long)(x + SP_BIAS) << 40) |
           ((unsigned long long)(y + SP_BIAS) << 20) | (unsigned long long)(z + SP_BIAS);
}

__device__ __forceinline__ unsigned int sp_hash(unsigned long long k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return (unsigned int)k;
}

// insert voxel i with key pack(coords[i]) -> index i (coords are unique)
extern "C" __global__ void __launch_bounds__(SD_THREADS) sp_hash_insert(const int* coords, int n, unsigned long long* keys,
                                                                       int* vals, unsigned int mask) {
    const int i = blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    const unsigned long long key = sp_pack(coords[4 * i], coords[4 * i + 1], coords[4 * i + 2], coords[4 * i + 3]);
    unsigned int h = sp_hash(key) & mask;
    for (;;) {
        const unsigned long long old = atomicCAS(&keys[h], SP_EMPTY, key);
        if (old == SP_EMPTY) {
            vals[h] = i;
            return;
        }
        h = (h + 1) & mask;
    }
}

// nb[i][t] = index of the voxel at offset t of voxel i (t = 9 * dx + 3 * dy + dz, each in {-1, 0, 1}), or -1 when absent
extern "C" __global__ void __launch_bounds__(SD_THREADS) sp_neighbor_map(const int* coords, int n, const unsigned long long* keys,
                                                                        const int* vals, unsigned int mask, int* nb) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= (long)n * 27) return;
    const int i = (int)(e / 27), t = (int)(e % 27);
    const int dx = t / 9 - 1, dy = (t / 3) % 3 - 1, dz = t % 3 - 1;
    const unsigned long long key = sp_pack(coords[4 * i], coords[4 * i + 1] + dx, coords[4 * i + 2] + dy, coords[4 * i + 3] + dz);
    unsigned int h = sp_hash(key) & mask;
    int found = -1;
    for (;;) {
        const unsigned long long k = keys[h];
        if (k == SP_EMPTY) break;
        if (k == key) {
            found = vals[h];
            break;
        }
        h = (h + 1) & mask;
    }
    nb[e] = found;
}

// submanifold 3x3x3 convolution, one thread per (voxel, output channel): y[i][o] = b[o] + sum_t sum_c x[nb[i][t]][c] W[o][c][t]
// with W fp16 in the checkpoint's [Cout][Cin][3][3][3] layout (t = 9 * kx + 3 * ky + kz). y is [N][Cout].
extern "C" __global__ void __launch_bounds__(256) sp_subm_conv_f16w(const float* x, int n, int cin, const int* nb,
                                                                   const unsigned short* W, const float* b, int cout,
                                                                   float* y, int o0, int no, int layout) {
    const int i = blockIdx.x * 16 + threadIdx.x;
    const int o = o0 + blockIdx.y * 16 + threadIdx.y;
    if (i >= n || o >= o0 + no) return;
    float acc = b[o];
    for (int t = 0; t < 27; ++t) {
        const int j = nb[(long)i * 27 + t];
        if (j < 0) continue;
        const float* xr = x + (long)j * cin;
        if (layout == 0) {  // [Cout][Cin][27] (the PyTorch layout)
            const unsigned short* wr = W + ((long)o * cin) * 27 + t;
            for (int c = 0; c < cin; ++c) acc += xr[c] * h2f(wr[(long)c * 27]);
        } else {            // [Cout][27][Cin] (FlexGEMM's layout, the shape decoder's checkpoint)
            const unsigned short* wr = W + ((long)o * 27 + t) * cin;
            for (int c = 0; c < cin; ++c) acc += xr[c] * h2f(wr[c]);
        }
    }
    y[(long)i * cout + o] = acc;
}

// ---- channel-to-spatial upsampling (factor 2, src/sparse.c): each voxel's children where the subdivision logit is > 0, in
// increasing child index (k = x + 2 y + 4 z bits), so the output order is the python op's order.
extern "C" __global__ void __launch_bounds__(SD_THREADS) sp_c2s_mask(const float* logit, int n, unsigned char* bits, int* cnt) {
    const int i = blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    unsigned int b = 0;
    for (int k = 0; k < 8; ++k)
        if (logit[i * 8 + k] > 0.0f) b |= 1u << k;
    bits[i] = (unsigned char)b;
    cnt[i] = __popc(b);
}

// exclusive prefix sum of cnt[0..n) into off[0..n], off[n] = total (one thread: fine for the sizes of one decoder level)
extern "C" __global__ void sp_exclusive_scan_serial(const int* cnt, int n, int* off) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int s = 0;
    for (int i = 0; i < n; ++i) {
        off[i] = s;
        s += cnt[i];
    }
    off[n] = s;
}

// new voxel p for each set child bit of voxel i: coords (b, 2x + dx, 2y + dy, 2z + dz); idx[p] = i; subidx[p] = k
extern "C" __global__ void __launch_bounds__(SD_THREADS) sp_c2s_fill(const int* coords, const unsigned char* bits, const int* off,
                                                                    int n, int* new_coords, int* idx, int* subidx) {
    const int i = blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    const unsigned int b = bits[i];
    int p = off[i];
    for (int k = 0; k < 8; ++k) {
        if (!((b >> k) & 1u)) continue;
        new_coords[4 * p + 0] = coords[4 * i + 0];
        new_coords[4 * p + 1] = 2 * coords[4 * i + 1] + (k & 1);
        new_coords[4 * p + 2] = 2 * coords[4 * i + 2] + ((k >> 1) & 1);
        new_coords[4 * p + 3] = 2 * coords[4 * i + 3] + ((k >> 2) & 1);
        idx[p] = i;
        subidx[p] = k;
        ++p;
    }
}

// out[p][j] = src[(idx[p] * 8 + subidx[p]) * cc + j / rep] for j < cc * rep: the child's channels (rep = 1), or the skip's
// channels repeated (the python repeat_interleave of out_channels // (channels // 8))
extern "C" __global__ void __launch_bounds__(SD_THREADS) sp_c2s_gather(const float* src, const int* idx, const int* subidx, long M,
                                                                      int cc, int rep, float* out) {
    const long e = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    const int cout = cc * rep;
    if (e >= M * cout) return;
    const long p = e / cout;
    const int j = (int)(e % cout);
    out[e] = src[((long)idx[p] * 8 + subidx[p]) * cc + j / rep];
}

// ---- the mesh step of the shape decoder (src/trellis_mesh.cpp): the dual grid to a mesh, plain C-style kernels.
// The decoder's features per voxel: 0..2 the dual vertex (the margin-sigmoid), 3..5 the intersection flags, 6 the split weight.
// Edge offsets: the four voxels around the x, y or z edge (flexible_dual_grid_to_mesh's edge_neighbor_voxel_offset).
__device__ __forceinline__ void mesh_edge_offset(int a, int k, int* d) {
    const int tab[3][4][3] = {{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                              {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
                              {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}};
    d[0] = tab[a][k][0];
    d[1] = tab[a][k][1];
    d[2] = tab[a][k][2];
}

// the vertex of each voxel: (coords + dual) * voxel + aabb0, dual = (1 + 2 margin) sigmoid(f) - margin
extern "C" __global__ void __launch_bounds__(SD_THREADS) mesh_vertices(const int* coords, const float* feats, int n, float vox,
                                                                      float aabb0, float margin, float* v) {
    const int i = blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    for (int c = 0; c < 3; ++c) {
        const float s = 1.0f / (1.0f + expf(-feats[i * 7 + c]));
        const float dual = (1.0f + 2.0f * margin) * s - margin;
        v[i * 3 + c] = ((float)coords[i * 4 + 1 + c] + dual) * vox + aabb0;
    }
}

// the four voxels of each (voxel, axis) edge: q[((i * 3 + a) * 4 + k)] = coords[i] + offset(a, k)
extern "C" __global__ void __launch_bounds__(SD_THREADS) mesh_edge_queries(const int* coords, int n, int* q) {
    const int e = blockIdx.x * SD_THREADS + threadIdx.x;  // over n * 12
    if (e >= n * 12) return;
    const int i = e / 12, ak = e % 12, a = ak / 4, k = ak % 4;
    int d[3];
    mesh_edge_offset(a, k, d);
    q[e * 4 + 0] = coords[i * 4 + 0];
    q[e * 4 + 1] = coords[i * 4 + 1] + d[0];
    q[e * 4 + 2] = coords[i * 4 + 2] + d[1];
    q[e * 4 + 3] = coords[i * 4 + 3] + d[2];
}

// lookup of nq coordinates in the table: out[k] = the voxel's index, or -1
extern "C" __global__ void __launch_bounds__(SD_THREADS) sp_lookup(const int* q, int nq, const unsigned long long* keys,
                                                                  const int* vals, unsigned int mask, int* out) {
    const int k = blockIdx.x * SD_THREADS + threadIdx.x;
    if (k >= nq) return;
    const unsigned long long key = sp_pack(q[4 * k], q[4 * k + 1], q[4 * k + 2], q[4 * k + 3]);
    unsigned int h = sp_hash(key) & mask;
    int found = -1;
    for (;;) {
        const unsigned long long kk = keys[h];
        if (kk == SP_EMPTY) break;
        if (kk == key) {
            found = vals[h];
            break;
        }
        h = (h + 1) & mask;
    }
    out[k] = found;
}

// an edge is a quad when its intersection flag is set and its four voxels exist: valid[e] = 1 then
extern "C" __global__ void __launch_bounds__(SD_THREADS) mesh_edge_valid(const float* feats, int n, const int* look, int* valid) {
    const int e = blockIdx.x * SD_THREADS + threadIdx.x;  // over n * 3
    if (e >= n * 3) return;
    const int i = e / 3, a = e % 3;
    int ok = feats[i * 7 + 3 + a] > 0.0f;
    for (int k = 0; k < 4; ++k) ok = ok && look[e * 4 + k] >= 0;
    valid[e] = ok ? 1 : 0;
}

// the quads, in the order of the edges: quads[pos[e]] = the four voxels, for the valid edges
extern "C" __global__ void __launch_bounds__(SD_THREADS) mesh_quads(const int* valid, const int* pos, const int* look, int ne,
                                                                   int* quads) {
    const int e = blockIdx.x * SD_THREADS + threadIdx.x;
    if (e >= ne) return;
    if (!valid[e]) return;
    for (int k = 0; k < 4; ++k) quads[pos[e] * 4 + k] = look[e * 4 + k];
}

// two triangles per quad: the split with the larger product of opposite split weights (softplus of feature 6)
extern "C" __global__ void __launch_bounds__(SD_THREADS) mesh_triangles(const int* quads, int L, const float* feats, int* faces) {
    const int l = blockIdx.x * SD_THREADS + threadIdx.x;
    if (l >= L) return;
    int q[4];
    float w[4];
    for (int k = 0; k < 4; ++k) {
        q[k] = quads[l * 4 + k];
        const float x = feats[q[k] * 7 + 6];
        w[k] = x > 20.0f ? x : log1pf(expf(x));  // softplus
    }
    const float s02 = w[0] * w[2], s13 = w[1] * w[3];
    if (s02 > s13) {  // split 1: (0, 1, 2), (0, 2, 3)
        faces[l * 6 + 0] = q[0]; faces[l * 6 + 1] = q[1]; faces[l * 6 + 2] = q[2];
        faces[l * 6 + 3] = q[0]; faces[l * 6 + 4] = q[2]; faces[l * 6 + 5] = q[3];
    } else {          // split 2: (0, 1, 3), (3, 1, 2)
        faces[l * 6 + 0] = q[0]; faces[l * 6 + 1] = q[1]; faces[l * 6 + 2] = q[3];
        faces[l * 6 + 3] = q[3]; faces[l * 6 + 4] = q[1]; faces[l * 6 + 5] = q[2];
    }
}

// ---- DINOv3's rotary position embedding on the patch tokens (dinov3.cpp): x[t][h * D + d] for the tokens t >= prefix; the pair
// (d, d + D / 2) is rotated together: out_a = a cos_a - b sin_a, out_b = b cos_b + a sin_b (rotate_half), with cos and sin [P][D].
extern "C" __global__ void __launch_bounds__(SD_THREADS) dino_rope(float* x, int T, int prefix, int C, int H, int D, const float* cs,
                                                                   const float* sn) {
    const long n = (long)(T - prefix) * H * (D / 2);
    const long i = (long)blockIdx.x * SD_THREADS + threadIdx.x;
    if (i >= n) return;
    const int d = (int)(i % (D / 2));
    const long th = i / (D / 2);
    const int h = (int)(th % H);
    const int t = prefix + (int)(th / H);
    const int p = t - prefix;
    float* row = x + (long)t * C + h * D;
    const float a = row[d], b = row[d + D / 2];
    row[d] = a * cs[p * D + d] - b * sn[p * D + d];
    row[d + D / 2] = b * cs[p * D + d + D / 2] + a * sn[p * D + d + D / 2];
}

// ---- mesh simplification (trellis_simplify.cpp; CuMesh's simplify.cu in plain C): the quadric error of each vertex, the cost of each
// edge collapse, the cheapest collapse of each face (packed cost and edge id, atomic minimum), then the collapses that are the cheapest
// around both their ends, the vertices and faces compacted by an exclusive scan on the host driver. Layouts: vertices [V][3] floats,
// faces [F][3] ints, edges as (min << 32 | max) with the sorted unique edges, vertex-face lists in CSR, QEM [V][10] (the upper triangle).
__device__ __forceinline__ void simp_add_plane(float* e, float a, float b, float c, float d) {
    e[0] += a * a;
    e[1] += a * b;
    e[2] += a * c;
    e[3] += a * d;
    e[4] += b * b;
    e[5] += b * c;
    e[6] += b * d;
    e[7] += c * c;
    e[8] += c * d;
    e[9] += d * d;
}

__device__ __forceinline__ float simp_eval(const float* e, float x, float y, float z) {
    const float w = 1.0f;
    float res = 0.0f;
    res += e[0] * x * x;
    res += 2.0f * e[1] * x * y;
    res += 2.0f * e[2] * x * z;
    res += 2.0f * e[3] * x * w;
    res += e[4] * y * y;
    res += 2.0f * e[5] * y * z;
    res += 2.0f * e[6] * y * w;
    res += e[7] * z * z;
    res += 2.0f * e[8] * z * w;
    res += e[9] * w * w;
    return res;
}

extern "C" __global__ void __launch_bounds__(SD_THREADS) simp_qem(const float* v, const int* f, const int* v2f, const int* v2f_off, int V,
                                                                  float* qem) {
    const int tid = blockIdx.x * SD_THREADS + threadIdx.x;
    if (tid >= V) return;
    float e[10];
    for (int i = 0; i < 10; ++i) e[i] = 0.0f;
    for (int k = v2f_off[tid]; k < v2f_off[tid + 1]; ++k) {
        const int fi = v2f[k];
        const int a = f[3 * fi], b = f[3 * fi + 1], c = f[3 * fi + 2];
        const float e1x = v[3 * b] - v[3 * a], e1y = v[3 * b + 1] - v[3 * a + 1], e1z = v[3 * b + 2] - v[3 * a + 2];
        const float e2x = v[3 * c] - v[3 * a], e2y = v[3 * c + 1] - v[3 * a + 1], e2z = v[3 * c + 2] - v[3 * a + 2];
        float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        const float inv = rsqrtf(nx * nx + ny * ny + nz * nz);
        nx *= inv;
        ny *= inv;
        nz *= inv;
        const float d = -(nx * v[3 * a] + ny * v[3 * a + 1] + nz * v[3 * a + 2]);
        simp_add_plane(e, nx, ny, nz, d);
    }
    for (int i = 0; i < 10; ++i) qem[10 * tid + i] = e[i];
}

// the triangle's cost term when the vertex `keep` moves to vnew (the other vertex of the edge is removed): 0 when the triangle would
// flip, 1 otherwise (its skinny term added to *skinny)
__device__ __forceinline__ int simp_tri(int tri, int keep, int other, const float* v, const int* f, const float* vnew, float* skinny,
                                        int* ntri) {
    const int ia = f[3 * tri], ib = f[3 * tri + 1], ic = f[3 * tri + 2];
    if (ia == other || ib == other || ic == other) return 1;
    float a[3] = {v[3 * ia], v[3 * ia + 1], v[3 * ia + 2]};
    float b[3] = {v[3 * ib], v[3 * ib + 1], v[3 * ib + 2]};
    float c[3] = {v[3 * ic], v[3 * ic + 1], v[3 * ic + 2]};
    const float* na = (ia == keep) ? vnew : a;
    const float* nb = (ib == keep) ? vnew : b;
    const float* nc = (ic == keep) ? vnew : c;
    const float o1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const float o2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    const float on[3] = {o1[1] * o2[2] - o1[2] * o2[1], o1[2] * o2[0] - o1[0] * o2[2], o1[0] * o2[1] - o1[1] * o2[0]};
    const float e1[3] = {nb[0] - na[0], nb[1] - na[1], nb[2] - na[2]};
    const float e2[3] = {nc[0] - na[0], nc[1] - na[1], nc[2] - na[2]};
    const float nn[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const float new_area = 0.5f * sqrtf(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
    if (on[0] * nn[0] + on[1] * nn[1] + on[2] * nn[2] < 0.0f) return 0;
    const float e0v[3] = {nc[0] - nb[0], nc[1] - nb[1], nc[2] - nb[2]};
    float denom = (e0v[0] * e0v[0] + e0v[1] * e0v[1] + e0v[2] * e0v[2]) + (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]) +
                  (e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
    if (denom < 1e-12f) denom = 1e-12f;
    const float shape = 4.0f * sqrtf(3.0f) * new_area / denom;
    *skinny += 1.0f - fminf(fmaxf(shape, 0.0f), 1.0f);
    *ntri += 1;
    return 1;
}

extern "C" __global__ void __launch_bounds__(SD_THREADS) simp_cost(const float* v, const int* f, const int* v2f, const int* v2f_off,
                                                                   const unsigned long long* edges, const unsigned char* vb, const float* qem,
                                                                   int E, float lam_len, float lam_sk, float* cost) {
    const int tid = blockIdx.x * SD_THREADS + threadIdx.x;
    if (tid >= E) return;
    const unsigned long long ed = edges[tid];
    const int e0 = (int)(ed >> 32), e1 = (int)(ed & 0xFFFFFFFFu);
    float w0 = 0.5f;
    if (vb[e0] && !vb[e1]) w0 = 1.0f;
    else if (!vb[e0] && vb[e1]) w0 = 0.0f;
    float vn[3];
    for (int c = 0; c < 3; ++c) vn[c] = v[3 * e0 + c] * w0 + v[3 * e1 + c] * (1.0f - w0);
    float q[10];
    for (int i = 0; i < 10; ++i) q[i] = qem[10 * e0 + i] + qem[10 * e1 + i];
    float c_total = simp_eval(q, vn[0], vn[1], vn[2]);
    const float dx = v[3 * e1] - v[3 * e0], dy = v[3 * e1 + 1] - v[3 * e0 + 1], dz = v[3 * e1 + 2] - v[3 * e0 + 2];
    const float len2 = dx * dx + dy * dy + dz * dz;
    c_total += lam_len * len2;
    float skinny = 0.0f;
    int ntri = 0;
    for (int k = v2f_off[e0]; k < v2f_off[e0 + 1]; ++k)
        if (!simp_tri(v2f[k], e0, e1, v, f, vn, &skinny, &ntri)) {
            cost[tid] = INFINITY;
            return;
        }
    for (int k = v2f_off[e1]; k < v2f_off[e1 + 1]; ++k)
        if (!simp_tri(v2f[k], e1, e0, v, f, vn, &skinny, &ntri)) {
            cost[tid] = INFINITY;
            return;
        }
    if (ntri > 0) skinny /= (float)ntri;
    c_total += lam_sk * skinny * len2;
    cost[tid] = c_total;
}

extern "C" __global__ void __launch_bounds__(SD_THREADS) simp_propagate(const unsigned long long* edges, const int* v2f, const int* v2f_off,
                                                                        const float* cost, int E, unsigned long long* prop) {
    const int tid = blockIdx.x * SD_THREADS + threadIdx.x;
    if (tid >= E) return;
    const unsigned long long ed = edges[tid];
    const int e0 = (int)(ed >> 32), e1 = (int)(ed & 0xFFFFFFFFu);
    const unsigned long long packed = ((unsigned long long)__float_as_uint(cost[tid]) << 32) | (unsigned int)tid;
    for (int k = v2f_off[e0]; k < v2f_off[e0 + 1]; ++k) atomicMin(&prop[v2f[k]], packed);
    for (int k = v2f_off[e1]; k < v2f_off[e1 + 1]; ++k) atomicMin(&prop[v2f[k]], packed);
}

extern "C" __global__ void __launch_bounds__(SD_THREADS) simp_collapse(float* v, int* f, const unsigned long long* edges, const int* v2f,
                                                                       const int* v2f_off, const float* cost,
                                                                       const unsigned long long* prop, const unsigned char* vb, int E,
                                                                       float thresh, int* vkept, int* fkept) {
    const int tid = blockIdx.x * SD_THREADS + threadIdx.x;
    if (tid >= E) return;
    const float c = cost[tid];
    if (c > thresh) return;
    const unsigned long long ed = edges[tid];
    const int e0 = (int)(ed >> 32), e1 = (int)(ed & 0xFFFFFFFFu);
    const unsigned long long packed = ((unsigned long long)__float_as_uint(c) << 32) | (unsigned int)tid;
    // the edge is collapsed only when it is the cheapest around both its ends
    for (int k = v2f_off[e0]; k < v2f_off[e0 + 1]; ++k)
        if (prop[v2f[k]] != packed) return;
    for (int k = v2f_off[e1]; k < v2f_off[e1 + 1]; ++k)
        if (prop[v2f[k]] != packed) return;
    float w0 = 0.5f;
    if (vb[e0] && !vb[e1]) w0 = 1.0f;
    else if (!vb[e0] && vb[e1]) w0 = 0.0f;
    for (int c2 = 0; c2 < 3; ++c2) v[3 * e0 + c2] = v[3 * e0 + c2] * w0 + v[3 * e1 + c2] * (1.0f - w0);
    vkept[e1] = 0;
    for (int k = v2f_off[e0]; k < v2f_off[e0 + 1]; ++k) {
        const int fid = v2f[k];
        if (f[3 * fid] == e1 || f[3 * fid + 1] == e1 || f[3 * fid + 2] == e1) fkept[fid] = 0;
    }
    for (int k = v2f_off[e1]; k < v2f_off[e1 + 1]; ++k) {
        const int fid = v2f[k];
        if (f[3 * fid] == e1) f[3 * fid] = e0;
        else if (f[3 * fid + 1] == e1) f[3 * fid + 1] = e0;
        else if (f[3 * fid + 2] == e1) f[3 * fid + 2] = e0;
    }
}

extern "C" __global__ void __launch_bounds__(SD_THREADS) simp_compress_v(const int* vmap, const float* oldv, int V, float* newv) {
    const int tid = blockIdx.x * SD_THREADS + threadIdx.x;
    if (tid >= V) return;
    const int nid = vmap[tid];
    if (vmap[tid + 1] == nid + 1) {
        newv[3 * nid] = oldv[3 * tid];
        newv[3 * nid + 1] = oldv[3 * tid + 1];
        newv[3 * nid + 2] = oldv[3 * tid + 2];
    }
}

extern "C" __global__ void __launch_bounds__(SD_THREADS) simp_compress_f(const int* fmap, const int* vmap, const int* oldf, int F, int* newf) {
    const int tid = blockIdx.x * SD_THREADS + threadIdx.x;
    if (tid >= F) return;
    const int nid = fmap[tid];
    if (fmap[tid + 1] == nid + 1) {
        newf[3 * nid] = vmap[oldf[3 * tid]];
        newf[3 * nid + 1] = vmap[oldf[3 * tid + 1]];
        newf[3 * nid + 2] = vmap[oldf[3 * tid + 2]];
    }
}

// ---- TRELLIS.2 mesh charts (src/trellis_charts.cpp): CuMesh's compute_charts (cumesh-ssh/src/atlas.cu) as device steps, one thread per
// face, edge or chart. The host drives the loops and builds the sorted chart pairs and the chart-face lists in CuMesh's order.
// The cone of a chart: axis [4 floats: the axis and the half angle] (cn); its area (carea).

extern "C" __global__ void __launch_bounds__(SD_THREADS) chart_face_geom(const float* v, const int* f, int F, float* fn, float* fa) {
    const int t = blockIdx.x * SD_THREADS + threadIdx.x;
    if (t >= F) return;
    const int a = f[3 * t], b = f[3 * t + 1], c = f[3 * t + 2];
    const float e1x = v[3 * b] - v[3 * a], e1y = v[3 * b + 1] - v[3 * a + 1], e1z = v[3 * b + 2] - v[3 * a + 2];
    const float e2x = v[3 * c] - v[3 * a], e2y = v[3 * c + 1] - v[3 * a + 1], e2z = v[3 * c + 2] - v[3 * a + 2];
    const float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    const float s = nx * nx + ny * ny + nz * nz;
    fa[t] = 0.5f * sqrtf(s);
    const float inv = s > 0.0f ? rsqrtf(s) : 0.0f;  // a degenerate face has no normal (CuMesh's would be NaN; python removes them)
    fn[3 * t] = nx * inv;
    fn[3 * t + 1] = ny * inv;
    fn[3 * t + 2] = nz * inv;
}

// the cone of each chart: the normalised sum of its face normals, and the largest angle from the axis to a face normal; the area
extern "C" __global__ void __launch_bounds__(SD_THREADS) chart_cones(const int* coff, const int* cface, const float* fn, const float* fa, int C,
                                                                     float* cn, float* carea) {
    const int c = blockIdx.x * SD_THREADS + threadIdx.x;
    if (c >= C) return;
    float area = 0.0f, sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (int k = coff[c]; k < coff[c + 1]; ++k) {
        const int f = cface[k];
        area += fa[f];
        sx += fn[3 * f];
        sy += fn[3 * f + 1];
        sz += fn[3 * f + 2];
    }
    const float norm = sqrtf(sx * sx + sy * sy + sz * sz);
    if (norm > 0.0f) {
        sx /= norm;
        sy /= norm;
        sz /= norm;
    }
    float half = 0.0f;
    for (int k = coff[c]; k < coff[c + 1]; ++k) {
        const int f = cface[k];
        const float d = sx * fn[3 * f] + sy * fn[3 * f + 1] + sz * fn[3 * f + 2];
        half = fmaxf(half, acosf(fmaxf(fminf(d, 1.0f), -1.0f)));
    }
    cn[4 * c] = sx;
    cn[4 * c + 1] = sy;
    cn[4 * c + 2] = sz;
    cn[4 * c + 3] = half;
    carea[c] = area;
}

// the perimeter of each chart: the lengths of its chart-graph edges
extern "C" __global__ void __launch_bounds__(SD_THREADS) chart_perims(const int* c2e_off, const int* c2e, const float* elen, int C, float* perim) {
    const int c = blockIdx.x * SD_THREADS + threadIdx.x;
    if (c >= C) return;
    float s = 0.0f;
    for (int k = c2e_off[c]; k < c2e_off[c + 1]; ++k) s += elen[c2e[k]];
    perim[c] = s;
}

// the cost of each chart-graph edge: the half angle of the merged cone, plus the area and the perimeter-area penalties
extern "C" __global__ void __launch_bounds__(SD_THREADS) chart_costs(const unsigned long long* ek, const float* elen, const float* cn,
                                                                     const float* carea, const float* perim, float aw, float pw, int E,
                                                                     float* cost) {
    const int t = blockIdx.x * SD_THREADS + threadIdx.x;
    if (t >= E) return;
    const unsigned long long key = ek[t];
    const int c0 = (int)(key >> 32), c1 = (int)(key & 0xFFFFFFFFu);
    const float a0x = cn[4 * c0], a0y = cn[4 * c0 + 1], a0z = cn[4 * c0 + 2], h0 = cn[4 * c0 + 3];
    const float a1x = cn[4 * c1], a1y = cn[4 * c1 + 1], a1z = cn[4 * c1 + 2], h1 = cn[4 * c1 + 3];
    const float cosA = a0x * a1x + a0y * a1y + a0z * a1z;
    const float axisA = acosf(fmaxf(fminf(cosA, 1.0f), -1.0f));
    const float low = fminf(-h0, axisA - h1), high = fmaxf(h0, axisA + h1);
    float c = (high - low) * 0.5f;
    const float newArea = carea[c0] + carea[c1];
    c += aw * newArea;
    const float newPerim = perim[c0] + perim[c1] - 2 * elen[t];
    c += pw * (newPerim * newPerim / newArea);
    cost[t] = c;
}

// the cheapest edge of each chart (by cost, then edge id), packed as (cost bits << 32 | edge id) for the collapse
extern "C" __global__ void __launch_bounds__(SD_THREADS) chart_propagate(const int* c2e_off, const int* c2e, const float* cost, int C,
                                                                        unsigned long long* prop) {
    const int c = blockIdx.x * SD_THREADS + threadIdx.x;
    if (c >= C) return;
    int min_e = -1;
    float min_c = 3.402823466e+38f;
    for (int k = c2e_off[c]; k < c2e_off[c + 1]; ++k) {
        const int e = c2e[k];
        const float v = cost[e];
        if (v < min_c || (v == min_c && e < min_e)) {
            min_e = e;
            min_c = v;
        }
    }
    prop[c] = ((unsigned long long)__float_as_uint(min_c) << 32) | (unsigned int)min_e;
}

// the chart-graph edges that are the cheapest at both their charts and under the threshold collapse: the second chart is mapped to the
// first (cmap, the identity before), and the first chart's cone takes both cones
extern "C" __global__ void __launch_bounds__(SD_THREADS) chart_collapse(const unsigned long long* ek, const float* cost,
                                                                       const unsigned long long* prop, float thresh, int E, int* cmap,
                                                                       float* cn, int* flag) {
    const int t = blockIdx.x * SD_THREADS + threadIdx.x;
    if (t >= E) return;
    const float c = cost[t];
    if (c > thresh) return;
    const unsigned long long key = ek[t];
    const int c0 = (int)(key >> 32), c1 = (int)(key & 0xFFFFFFFFu);
    const unsigned long long pack = ((unsigned long long)__float_as_uint(c) << 32) | (unsigned int)t;
    if (prop[c0] < pack || prop[c1] < pack) return;
    cmap[c1] = c0;
    const float a0x = cn[4 * c0], a0y = cn[4 * c0 + 1], a0z = cn[4 * c0 + 2], h0 = cn[4 * c0 + 3];
    const float a1x = cn[4 * c1], a1y = cn[4 * c1 + 1], a1z = cn[4 * c1 + 2], h1 = cn[4 * c1 + 3];
    const float cosA = a0x * a1x + a0y * a1y + a0z * a1z;
    const float axisA = acosf(fmaxf(fminf(cosA, 1.0f), -1.0f));
    const float low = fminf(-h0, axisA - h1), high = fmaxf(h0, axisA + h1);
    const float nh = (high - low) * 0.5f;
    float nx = a0x, ny = a0y, nz = a0z;
    if (axisA >= 1e-3f) {
        const float ang = (high + low) * 0.5f;
        float ux = a1x - a0x * cosA, uy = a1y - a0y * cosA, uz = a1z - a0z * cosA;
        const float iu = rsqrtf(ux * ux + uy * uy + uz * uz);
        ux *= iu;
        uy *= iu;
        uz *= iu;
        const float cs = cosf(ang), sn = sinf(ang);
        nx = a0x * cs + ux * sn;
        ny = a0y * cs + uy * sn;
        nz = a0z * cs + uz * sn;
        const float in = rsqrtf(nx * nx + ny * ny + nz * nz);
        nx *= in;
        ny *= in;
        nz *= in;
    }
    cn[4 * c0] = nx;
    cn[4 * c0 + 1] = ny;
    cn[4 * c0 + 2] = nz;
    cn[4 * c0 + 3] = nh;
    *flag = 0;  // not the end of the iteration
}

// the refinement: each face takes the chart of its neighbours (across its edges) whose cone holds its normal best, with the edge-length
// weighted smoothness (CuMesh's refine_charts_kernel; at most four candidate charts per face)
extern "C" __global__ void __launch_bounds__(SD_THREADS) chart_refine(const float* cn, const float* fn, const float* v, const unsigned long long* ek,
                                                                     const int* f2e, const int* e2f, const int* e2f_off, int F, float lam,
                                                                     const int* ids, int* out) {
    const int t = blockIdx.x * SD_THREADS + threadIdx.x;
    if (t >= F) return;
    const int cur = ids[t];
    const float nx = fn[3 * t], ny = fn[3 * t + 1], nz = fn[3 * t + 2];
    int cand[4];
    float score[4];
    int num = 1;
    cand[0] = cur;
    score[0] = 0.0f;
    for (int i = 0; i < 3; ++i) {
        const int e = f2e[3 * t + i];
        const unsigned long long key = ek[e];
        const int a = (int)(key >> 32), b = (int)(key & 0xFFFFFFFFu);
        const float dx = v[3 * b] - v[3 * a], dy = v[3 * b + 1] - v[3 * a + 1], dz = v[3 * b + 2] - v[3 * a + 2];
        const float len = sqrtf(dx * dx + dy * dy + dz * dz);
        for (int j = e2f_off[e]; j < e2f_off[e + 1]; ++j) {
            const int nb = e2f[j];
            if (nb == t) continue;
            const int nc = ids[nb];
            int idx = -1;
            for (int k = 0; k < num; ++k)
                if (cand[k] == nc) {
                    idx = k;
                    break;
                }
            if (idx == -1 && num < 4) {
                idx = num++;
                cand[idx] = nc;
                score[idx] = 0.0f;
            }
            if (idx != -1) score[idx] += len;
        }
    }
    int best_c = cur;
    float best = -1e9f;
    for (int i = 0; i < num; ++i) {
        const int c = cand[i];
        const float geo = cn[4 * c] * nx + cn[4 * c + 1] * ny + cn[4 * c + 2] * nz;
        if (geo <= 0.0f) continue;
        const float total = geo + score[i] * lam;
        if (c == cur && best == -1e9f) {
            best = total;
            best_c = c;
        }
        const float diff = total - best;
        const float eps = 1e-5f;
        if (diff > eps) {
            best = total;
            best_c = c;
        } else if (fabsf(diff) <= eps) {
            if (c < best_c) {
                best = total;
                best_c = c;
            }
        }
    }
    out[t] = best_c;
}
