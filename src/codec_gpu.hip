// The Qwen3-TTS codec decoder's convolutions on the device (qwen3tts_codec.cpp: the device path of decode_stream, PLAN.md
// item 9): float32 weights read in place from the GGUF mapping, float32 activations in registered memory, channel-major
// rows [ch][columns] with a row stride. cd_pad_act builds a layer's padded input (its history, then the activation of the
// new columns) and the next history; cd_conv and cd_tconv are tiled products: a block is 32 output rows by 128 columns,
// each thread holds 2 rows by 8 columns in registers, and the weights' and the inputs' tiles are staged in shared memory
// per chunk of input channels. Every output sums its terms in one order (the bias, then the input channels in order, each
// channel's taps in order) whatever its place in a tile, so the stream's pieces give the same bits as one piece.
#include <hip/hip_runtime.h>
#include <stdint.h>

#define CD_TM 32   // output rows of a tile
#define CD_TN 128  // output columns of a tile

// SnakeBeta: x + ib * sin^2(x * ea), with ea = exp(alpha) and ib = 1 / (exp(beta) + 1e-9) per channel (the host's values)
__device__ __forceinline__ float cd_snake(float x, float ea, float ib) {
    const float v = sinf(x * ea);
    return x + ib * v * v;
}

// dst [ch][g + L] (row stride g + L): the layer's history tail [ch][g], then the new columns src [ch][L] (row stride lds)
// through the activation (act 1: SnakeBeta with ea / ib [ch]; 0: none). tail_out [ch][g] gets dst's last g columns: the
// next history. Grid: (ceil((g + L) / 256), ch) blocks of 256 threads.
extern "C" __global__ void __launch_bounds__(256)
cd_pad_act(const float *src, int lds, const float *tail, float *dst, float *tail_out, int ch, int L, int g, int act,
           const float *ea, const float *ib) {
    const int c = blockIdx.y, col = blockIdx.x * 256 + threadIdx.x;
    if (c >= ch || col >= g + L) return;
    float v;
    if (col < g) {
        v = tail[(size_t)c * g + col];
    } else {
        v = src[(size_t)c * lds + (col - g)];
        if (act) v = cd_snake(v, ea[c], ib[c]);
    }
    dst[(size_t)c * (g + L) + col] = v;
    if (col >= L) tail_out[(size_t)c * g + (col - L)] = v;
}

// Y [cout][ldy] (columns 0 .. T - 1): the causal convolution of X [cin][ldx] (padded: output column t reads the columns
// t + j * dil, j < k) with W [cout][cin][k] and bias [cout] (or null). mode 0: Y = the result; 1: Y += the result (the
// residual add); 2: Y = SnakeBeta(the result) with ea / ib [cout]. KC: the input channels of a chunk (a multiple of 16).
// Grid: (ceil(cout / 32), ceil(T / 128)); dynamic shared memory: (32 * KC * k + KC * (128 + (k - 1) * dil)) floats.
extern "C" __global__ void __launch_bounds__(256)
cd_conv(const float *X, int ldx, const float *W, const float *bias, float *Y, int ldy, int cin, int cout, int k, int dil,
        int T, int KC, int mode, const float *ea, const float *ib) {
    extern __shared__ float sh[];
    const int tid = threadIdx.x;
    const int o0 = blockIdx.x * CD_TM, t0 = blockIdx.y * CD_TN;
    const int r0 = 2 * (tid / 16), c0 = 8 * (tid % 16);  // this thread's rows r0, r0 + 1 and columns c0 .. c0 + 7 of the tile
    const int halo = (k - 1) * dil, XW = CD_TN + halo, KW = KC * k;
    float *Ws = sh, *Xs = sh + CD_TM * KW;
    float a0[8], a1[8];
    {
        const int oa = o0 + r0 < cout ? o0 + r0 : cout - 1, ob = o0 + r0 + 1 < cout ? o0 + r0 + 1 : cout - 1;
        const float b0 = bias ? bias[oa] : 0.0f, b1 = bias ? bias[ob] : 0.0f;
        for (int u = 0; u < 8; ++u) {
            a0[u] = b0;
            a1[u] = b1;
        }
    }
    const int wr = tid / 8, wc = tid % 8;    // the weights' staging: row wr, the channels wc, wc + 8, ...
    const int xc = tid % 16, xo = tid / 16;  // the inputs' staging: the channels xc, xc + 16, ..., the columns xo, xo + 16, ...
    for (int cb = 0; cb < cin; cb += KC) {
        for (int c = wc; c < KC; c += 8) {
            const bool ok = o0 + wr < cout && cb + c < cin;  // (the loads are unconditional: a row past the end reads row 0)
            const float *wp = W + (ok ? ((size_t)(o0 + wr) * cin + cb + c) * k : 0);
            for (int j = 0; j < k; ++j) {
                const float w = wp[j];
                Ws[wr * KW + c * k + j] = ok ? w : 0.0f;
            }
        }
        for (int c = xc; c < KC; c += 16) {
            const bool okc = cb + c < cin;
            const float *xp = X + (okc ? (size_t)(cb + c) * ldx : 0);
            for (int col = xo; col < XW; col += 16) {
                const int tc = t0 + col;
                const bool ok = okc && tc < T + halo;
                const float x = xp[ok ? tc : 0];
                Xs[c * XW + col] = ok ? x : 0.0f;
            }
        }
        __syncthreads();
        const float *w0p = Ws + r0 * KW, *w1p = w0p + KW;
        for (int c = 0; c < KC; ++c) {
            const float *xr = Xs + c * XW + c0;
            for (int j = 0; j < k; ++j) {
                const float w0 = w0p[c * k + j], w1 = w1p[c * k + j];
                const float *xp = xr + j * dil;
                for (int u = 0; u < 8; ++u) {
                    const float x = xp[u];
                    a0[u] += w0 * x;
                    a1[u] += w1 * x;
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) {
        const int o = o0 + r0 + i;
        if (o >= cout) continue;
        const float *a = i ? a1 : a0;
        float *yp = Y + (size_t)o * ldy + t0 + c0;
        for (int u = 0; u < 8; ++u) {
            if (t0 + c0 + u >= T) break;
            float v = a[u];
            if (mode == 1)
                v = yp[u] + v;
            else if (mode == 2)
                v = cd_snake(v, ea[o], ib[o]);
            yp[u] = v;
        }
    }
}

// The transposed convolution (stride s, kernel k = s or 2 s, the output trimmed to T * s): Y [cout][ldy] with
// Y[o][q * s + p] = bias[o] + the sum over the input channels ci of W[ci][o][p] * X[ci][g + q] (and, when k = 2 s, of
// W[ci][o][p + s] * X[ci][g + q - 1]); X [cin][ldx] is padded with g = k / s - 1 history columns, W is [cin][cout][k].
// A tile's rows are the pairs (o, p), r = o * s + p. KC: the input channels of a chunk (a multiple of 16).
// Grid: (ceil(cout * s / 32), ceil(T / 128)); dynamic shared memory: KC * ((32 / s + 2) * k + 128 + g) floats.
extern "C" __global__ void __launch_bounds__(256)
cd_tconv(const float *X, int ldx, const float *W, const float *bias, float *Y, int ldy, int cin, int cout, int k, int s,
         int T, int KC) {
    extern __shared__ float sh[];
    const int tid = threadIdx.x;
    const int rb = blockIdx.x * CD_TM, t0 = blockIdx.y * CD_TN;
    const int r0 = rb + 2 * (tid / 16), c0 = 8 * (tid % 16);
    const int g = k / s - 1, ntap = g + 1, XW = CD_TN + g;
    const int ob = rb / s, SW = (CD_TM / s + 2) * k;  // the strip of each input channel's [cout * k] row that the tile reads
    float *Ws = sh, *Xs = sh + KC * SW;
    const int oA = r0 / s, pA = r0 % s, oB = (r0 + 1) / s, pB = (r0 + 1) % s;
    const int wiA = (oA - ob) * k + pA, wiB = (oB - ob) * k + pB;  // the rows' first taps in the strip
    float a0[8], a1[8];
    {
        const float b0 = bias ? bias[oA < cout ? oA : cout - 1] : 0.0f, b1 = bias ? bias[oB < cout ? oB : cout - 1] : 0.0f;
        for (int u = 0; u < 8; ++u) {
            a0[u] = b0;
            a1[u] = b1;
        }
    }
    const int xc = tid % 16, xo = tid / 16;  // the staging: the channels xc, xc + 16, ..., the elements xo, xo + 16, ...
    for (int cb = 0; cb < cin; cb += KC) {
        for (int c = xc; c < KC; c += 16) {
            const bool okc = cb + c < cin;
            const float *wp = W + (okc ? (size_t)(cb + c) * cout * k : 0);
            for (int e = xo; e < SW; e += 16) {
                const int idx = ob * k + e;
                const bool ok = okc && idx < cout * k;
                const float w = wp[ok ? idx : 0];
                Ws[c * SW + e] = ok ? w : 0.0f;
            }
            const float *xp = X + (okc ? (size_t)(cb + c) * ldx : 0);
            for (int col = xo; col < XW; col += 16) {
                const int tc = t0 + col;
                const bool ok = okc && tc < T + g;
                const float x = xp[ok ? tc : 0];
                Xs[c * XW + col] = ok ? x : 0.0f;
            }
        }
        __syncthreads();
        for (int c = 0; c < KC; ++c) {
            const float *wr = Ws + c * SW;
            const float *xr = Xs + c * XW + c0 + g;
            for (int tap = 0; tap < ntap; ++tap) {
                const float w0 = wr[wiA + tap * s], w1 = wr[wiB + tap * s];
                const float *xp = xr - tap;
                for (int u = 0; u < 8; ++u) {
                    const float x = xp[u];
                    a0[u] += w0 * x;
                    a1[u] += w1 * x;
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) {
        const int o = i ? oB : oA, p = i ? pB : pA;
        if (o >= cout) continue;
        const float *a = i ? a1 : a0;
        float *yp = Y + (size_t)o * ldy + p;
        for (int u = 0; u < 8; ++u) {
            const int q = t0 + c0 + u;
            if (q >= T) break;
            yp[(size_t)q * s] = a[u];
        }
    }
}
