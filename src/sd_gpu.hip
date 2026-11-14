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

// ---- flash attention (head width 128, BF16 Q, K and transposed V): one block per (64 query rows, head), four waves of 16 rows.
// The scores of a 64-key tile stay in registers (WMMA accumulators), with an online softmax per row (the row max and sum reduced
// over the half-wave's 16 lanes); P goes through LDS as the next WMMA's A operand; O accumulates in eight 16-wide tiles per wave.
// Keys beyond Tk are masked; the value rows beyond Tk are zero (the host pads them). Shared memory 61 KB.
extern "C" __global__ void __launch_bounds__(128)
sd_flash_attn(const uint16_t* __restrict__ qb, const uint16_t* __restrict__ kb, const uint16_t* __restrict__ vt, float* __restrict__ out,
              int Tq, int Tk, int Tkp, int C, float scale) {
    __shared__ __attribute__((aligned(16))) uint16_t Qs[64][136];
    __shared__ __attribute__((aligned(16))) uint16_t Ks[64][136];
    __shared__ __attribute__((aligned(16))) uint16_t Vs[128][72];
    __shared__ __attribute__((aligned(16))) uint16_t Ps[64][72];
    const int t = threadIdx.x, w = t >> 5, l = t & 31, g = l >> 4, c16 = l & 15;
    const int q0 = blockIdx.x * 64, h = blockIdx.y;
    const uint16_t* Qp = qb + h * 128;
    const uint16_t* Kp = kb + h * 128;
    const uint16_t* Vp = vt + (long)h * 128 * Tkp;
    {  // the query tile: thread t loads 64 values of row t / 2
        const int r = t >> 1, hf = (t & 1) * 64, gm = q0 + r;
        for (int j = 0; j < 8; ++j) {
            uint4 v = {0u, 0u, 0u, 0u};
            if (gm < Tq) v = *reinterpret_cast<const uint4*>(Qp + (long)gm * C + hf + 8 * j);
            *reinterpret_cast<uint4*>(&Qs[r][hf + 8 * j]) = v;
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
        {  // the key tile (rows of d, 64 per key): thread t loads 64 values of key t / 2; keys beyond Tk are zero
            const int r = t >> 1, hf = (t & 1) * 64, gk = k0 + r;
            for (int j = 0; j < 8; ++j) {
                uint4 v = {0u, 0u, 0u, 0u};
                if (gk < Tk) v = *reinterpret_cast<const uint4*>(Kp + (long)gk * C + hf + 8 * j);
                *reinterpret_cast<uint4*>(&Ks[r][hf + 8 * j]) = v;
            }
        }
        {  // the value tile, transposed: thread t loads the 64 keys of value dimension t
            const uint16_t* src = Vp + (long)t * Tkp + k0;
            for (int j = 0; j < 8; ++j) *reinterpret_cast<uint4*>(&Vs[t][8 * j]) = *reinterpret_cast<const uint4*>(src + 8 * j);
        }
        __syncthreads();
        // S = Q K^T for this wave's 16 query rows and the tile's 64 keys
        f32x8_t S[4];
        for (int nt = 0; nt < 4; ++nt)
            for (int i = 0; i < 8; ++i) S[nt][i] = 0.0f;
        for (int ks = 0; ks < 8; ++ks) {
            bf16x16_t a;
            __builtin_memcpy(&a, &Qs[w * 16 + c16][ks * 16], sizeof(bf16x16_t));
            for (int nt = 0; nt < 4; ++nt) {
                bf16x16_t b;
                __builtin_memcpy(&b, &Ks[nt * 16 + c16][ks * 16], sizeof(bf16x16_t));
                S[nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S[nt]);
            }
        }
        // the online softmax per row (row = 2 i + g of this wave's 16 rows)
        for (int i = 0; i < 8; ++i) {
            float v[4];
            for (int nt = 0; nt < 4; ++nt) {
                v[nt] = S[nt][i] * scale;
                if (k0 + nt * 16 + c16 >= Tk) v[nt] = -INFINITY;
            }
            float mx = fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3]));
            for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor(mx, off, 32));
            const float mnew = fmaxf(mrow[i], mx);
            const float alpha = expf(mrow[i] - mnew);
            float p[4], sum = 0.0f;
            for (int nt = 0; nt < 4; ++nt) {
                p[nt] = expf(v[nt] - mnew);
                sum += p[nt];
            }
            for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor(sum, off, 32);
            lrow[i] = lrow[i] * alpha + sum;
            mrow[i] = mnew;
            for (int dt = 0; dt < 8; ++dt) O[dt][i] *= alpha;
            const int row = w * 16 + 2 * i + g;
            for (int nt = 0; nt < 4; ++nt) Ps[row][nt * 16 + c16] = f2bf(p[nt]);
        }
        __syncthreads();
        // O += P V: k = the tile's 64 keys (four steps), n = the 128 value dimensions (eight tiles)
        for (int ks = 0; ks < 4; ++ks) {
            bf16x16_t a;
            __builtin_memcpy(&a, &Ps[w * 16 + c16][ks * 16], sizeof(bf16x16_t));
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
