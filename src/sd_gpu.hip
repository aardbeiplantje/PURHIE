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
sd_gemm_wmma(const uint16_t* __restrict__ A, long lda, const float* __restrict__ B, long sbn, float* __restrict__ C, long cs_m,
             long cs_n, const float* __restrict__ biasf, float alpha, float beta, int M, int N, int K) {
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
