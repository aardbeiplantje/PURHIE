// quantize.cpp -- weight encoders (see quantize.h). Ports of the reference encoders, checked bit for bit against
// them by tests/quantize_verify.cpp: ggml's quantize_row_q8_0_ref, ggml_compute_fp32_to_fp16 / _bf16 (MIT), and
// ROCmFPX's rocmfp4.c (rocmfp4_quantize_row_q4_0_ref / _fast_ref: Codebook10, UE4M3 scales, exhaustive search).

#include "quantize.h"
#include "quant_types.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#if defined(__FAST_MATH__)
#error "quantize.cpp must be compiled without fast math (the scale search must round as the reference does)"
#endif
#pragma STDC FP_CONTRACT OFF

// ---------------------------------------------------------------- fp16 / bf16 (ggml-impl.h)

static inline uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

uint16_t qz_fp32_to_fp16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    const uint32_t w = f2u(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    base = u2f((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = f2u(base);
    const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
    const uint32_t mantissa_bits = bits & 0x00000FFFu;
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
}

uint16_t qz_fp32_to_bf16(float f) {
    const uint32_t u = f2u(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 64);  // NaN: force quiet
    return (uint16_t)((u + (0x7fffu + ((u >> 16) & 1))) >> 16);
}

// ---------------------------------------------------------------- Q8_0 (ggml quantize_row_q8_0_ref)

void qz_row_q8_0(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, int8 qs[32]
    for (int64_t i = 0; i < k / 32; ++i, x += 32, out += 34) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = amax > fabsf(x[j]) ? amax : fabsf(x[j]);  // ggml's MAX: a NaN wins
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t h = qz_fp32_to_fp16(d);
        memcpy(out, &h, 2);
        for (int j = 0; j < 32; ++j) out[2 + j] = (uint8_t)(int8_t)roundf(x[j] * id);
    }
}

// ---------------------------------------------------------------- the legacy 4- and 5-bit types (ggml's reference encoders)
// Q4_0: block of 32, fp16 d, qs[16] (value i in the low nibble, i + 16 in the high). d = max / -8 with max the value of
// largest magnitude (sign kept); q = min(15, (int)(x / d + 8.5)).
void qz_row_q4_0(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, qs[16]
    for (int64_t i = 0; i < k / 32; ++i, x += 32, out += 18) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < 32; ++j)
            if (fabsf(x[j]) > amax) {
                amax = fabsf(x[j]);
                max = x[j];
            }
        const float d = max / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t h = qz_fp32_to_fp16(d);
        memcpy(out, &h, 2);
        for (int j = 0; j < 16; ++j) {
            const int a = std::min(15, (int)(x[j] * id + 8.5f)), b = std::min(15, (int)(x[16 + j] * id + 8.5f));
            out[2 + j] = (uint8_t)(a | (b << 4));
        }
    }
}

// Q4_1: fp16 d, fp16 m (the minimum), qs[16]; d = (max - min) / 15, q = min(15, (int)((x - min) / d + 0.5))
void qz_row_q4_1(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, fp16 m, qs[16]
    for (int64_t i = 0; i < k / 32; ++i, x += 32, out += 20) {
        float mn = FLT_MAX, mx = -FLT_MAX;
        for (int j = 0; j < 32; ++j) {
            mn = std::min(mn, x[j]);
            mx = std::max(mx, x[j]);
        }
        const float d = (mx - mn) / 15.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t hd = qz_fp32_to_fp16(d), hm = qz_fp32_to_fp16(mn);
        memcpy(out, &hd, 2);
        memcpy(out + 2, &hm, 2);
        for (int j = 0; j < 16; ++j) {
            const int a = std::min(15, (int)((x[j] - mn) * id + 0.5f)), b = std::min(15, (int)((x[16 + j] - mn) * id + 0.5f));
            out[4 + j] = (uint8_t)(a | (b << 4));
        }
    }
}

// Q5_0: fp16 d, qh[4] (the fifth bit of value i at bit i, of value i + 16 at bit i + 16), qs[16]; d = max / -16,
// q = min(31, (int)(x / d + 16.5))
void qz_row_q5_0(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, qh[4], qs[16]
    for (int64_t i = 0; i < k / 32; ++i, x += 32, out += 22) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < 32; ++j)
            if (fabsf(x[j]) > amax) {
                amax = fabsf(x[j]);
                max = x[j];
            }
        const float d = max / -16.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t h = qz_fp32_to_fp16(d);
        memcpy(out, &h, 2);
        uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const int a = std::min(31, (int)(x[j] * id + 16.5f)), b = std::min(31, (int)(x[16 + j] * id + 16.5f));
            out[6 + j] = (uint8_t)((a & 0x0F) | ((b & 0x0F) << 4));
            qh |= (uint32_t)((a & 0x10) >> 4) << j;
            qh |= (uint32_t)((b & 0x10) >> 4) << (j + 16);
        }
        memcpy(out + 2, &qh, 4);
    }
}

// Q5_1: fp16 d, fp16 m, qh[4], qs[16]; d = (max - min) / 31, q = min(31, (int)((x - min) / d + 0.5))
void qz_row_q5_1(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, fp16 m, qh[4], qs[16]
    for (int64_t i = 0; i < k / 32; ++i, x += 32, out += 24) {
        float mn = FLT_MAX, mx = -FLT_MAX;
        for (int j = 0; j < 32; ++j) {
            mn = std::min(mn, x[j]);
            mx = std::max(mx, x[j]);
        }
        const float d = (mx - mn) / 31.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t hd = qz_fp32_to_fp16(d), hm = qz_fp32_to_fp16(mn);
        memcpy(out, &hd, 2);
        memcpy(out + 2, &hm, 2);
        uint32_t qh = 0;
        for (int j = 0; j < 16; ++j) {
            const int a = std::min(31, (int)((x[j] - mn) * id + 0.5f)), b = std::min(31, (int)((x[16 + j] - mn) * id + 0.5f));
            out[8 + j] = (uint8_t)((a & 0x0F) | ((b & 0x0F) << 4));
            qh |= (uint32_t)((a & 0x10) >> 4) << j;
            qh |= (uint32_t)((b & 0x10) >> 4) << (j + 16);
        }
        memcpy(out + 4, &qh, 4);
    }
}

// ---------------------------------------------------------------- ROCmFP4 (rocmfp4.c, without an importance matrix)

// Unsigned E4M3 scale bytes 0..126 as half-scale values: subnormals e * 2^-10, then (8 + m) * 2^(E - 11)
static float g_scale_half[127];
static const bool g_scale_init = [] {
    for (int e = 0; e < 127; ++e)
        g_scale_half[e] = e < 8 ? ldexpf((float)e, -10) : ldexpf((float)(8 + (e & 7)), (e >> 3) - 11);
    return true;
}();

static inline float scale_half(int e) { return e <= 0x7e ? g_scale_half[e] : 0.0f; }

// Codebook10 code for x / scale: 0, +-1, +-2, +-3, +-4, +-6, +-8, +-10; ties to the lower magnitude
static inline uint8_t best_index_finite(float x, float inv) {
    const float a = fabsf(x * inv);
    if (a <= 0.5f) return 0;
    const bool neg = x < 0.0f;
    if (a <= 1.5f) return neg ? 9 : 1;
    if (a <= 2.5f) return neg ? 10 : 2;
    if (a <= 3.5f) return neg ? 11 : 3;
    if (a <= 5.0f) return neg ? 12 : 4;
    if (a <= 7.0f) return neg ? 13 : 5;
    if (a <= 9.0f) return neg ? 14 : 6;
    return neg ? 15 : 7;
}

static inline uint8_t best_index(float x, float inv) { return std::isfinite(x) ? best_index_finite(x, inv) : 0; }

static inline float decoded_mag_finite(float x, float inv) {
    const float a = fabsf(x * inv);
    float mag;
    if (a <= 0.5f) mag = 0.0f;
    else if (a <= 1.5f) mag = 1.0f;
    else if (a <= 2.5f) mag = 2.0f;
    else if (a <= 3.5f) mag = 3.0f;
    else if (a <= 5.0f) mag = 4.0f;
    else if (a <= 7.0f) mag = 6.0f;
    else if (a <= 9.0f) mag = 8.0f;
    else mag = 10.0f;
    return x < 0.0f ? -mag : mag;
}

static inline float decoded_mag(float x, float inv) { return std::isfinite(x) ? decoded_mag_finite(x, inv) : 0.0f; }

// Squared error of the block at scale byte e, stopping once it exceeds best_err
template <bool FINITE>
static float block_mse(const float *x, int n, int e, float best_err) {
    const float s = scale_half(e);
    const float inv = 1.0f / s;
    float err = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float y = (FINITE ? decoded_mag_finite(x[i], inv) : decoded_mag(x[i], inv)) * s;
        const float d = x[i] - y;
        err += d * d;
        if (err > best_err) return err;
    }
    return err;
}

static int nearest_scale(float target) {
    if (!(target > 0.0f) || !std::isfinite(target)) return 1;
    int lo = 1, hi = 126;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (scale_half(mid) < target) lo = mid + 1;
        else hi = mid;
    }
    if (lo == 1) return 1;
    const float hs = scale_half(lo), ls = scale_half(lo - 1);
    return (target - ls <= hs - target) ? lo - 1 : lo;  // midpoint ties keep the lower byte
}

// The scale byte with the least squared error: outward from the nearest to max|x| / 10, both directions
static uint8_t choose_scale(const float *x, int n) {
    float max_abs = 0.0f;
    bool finite = true;
    for (int i = 0; i < n; ++i) {
        finite = finite && std::isfinite(x[i]);
        const float ax = fabsf(x[i]);
        if (ax > max_abs) max_abs = ax;
    }
    if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) return 0;

    const int start = nearest_scale(max_abs / 10.0f);
    int best_e = 0;
    float best_err = FLT_MAX;
    bool lower_done = false;
    for (int delta = 0; delta <= 125; ++delta) {
        const int e0 = start - delta;
        if (!lower_done && e0 >= 1 && e0 <= 126) {
            const float clip = max_abs - 10.0f * scale_half(e0);
            if (clip > 0.0f && clip * clip > best_err) {
                lower_done = true;  // smaller scales clip the largest value by more than the best error
            } else {
                const float err = finite ? block_mse<true>(x, n, e0, best_err) : block_mse<false>(x, n, e0, best_err);
                if (err < best_err || (err == best_err && e0 < best_e)) { best_err = err; best_e = e0; }
            }
        }
        const int e1 = start + delta;
        if (delta != 0 && e1 >= 1 && e1 <= 126) {
            const float err = finite ? block_mse<true>(x, n, e1, best_err) : block_mse<false>(x, n, e1, best_err);
            if (err < best_err || (err == best_err && e1 < best_e)) { best_err = err; best_e = e1; }
        }
        if ((lower_done || e0 <= 1) && e1 >= 126) break;
    }
    return (uint8_t)best_e;
}

static inline float inv_of(uint8_t e) {
    const float s = scale_half(e);
    return s > 0.0f ? 1.0f / s : 0.0f;
}

// Q4_0_ROCMFP4: qs[16] (low nibble = value j, high = value j + 16), then one scale byte per 16-value half
void qz_row_rocmfp4(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;
    for (int64_t ib = 0; ib < k / 32; ++ib, x += 32, out += 18) {
        const uint8_t e0 = choose_scale(x, 16), e1 = choose_scale(x + 16, 16);
        const float i0 = inv_of(e0), i1 = inv_of(e1);
        for (int j = 0; j < 16; ++j) out[j] = (uint8_t)(best_index(x[j], i0) | (best_index(x[j + 16], i1) << 4));
        out[16] = e0;
        out[17] = e1;
    }
}

// Q4_0_ROCMFP4_FAST: qs[16], one scale byte for the 32 values
void qz_row_rocmfp4_fast(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;
    for (int64_t ib = 0; ib < k / 32; ++ib, x += 32, out += 17) {
        const uint8_t e = choose_scale(x, 32);
        const float inv = inv_of(e);
        for (int j = 0; j < 16; ++j) out[j] = (uint8_t)(best_index(x[j], inv) | (best_index(x[j + 16], inv) << 4));
        out[16] = e;
    }
}

// ---------------------------------------------------------------- rows

int qz_supported(int t) {
    return t == QT_F32 || t == QT_F16 || t == QT_BF16 || t == QT_Q8_0 || t == QT_Q4_0_ROCMFP4 ||
           t == QT_Q4_0_ROCMFP4_FAST || t == QT_Q4_0 || t == QT_Q4_1 || t == QT_Q5_0 || t == QT_Q5_1;
}

static void encode_row(int t, const float *x, void *y, int64_t n) {
    switch (t) {
    case QT_F32: memcpy(y, x, (size_t)n * 4); break;
    case QT_F16: for (int64_t i = 0; i < n; ++i) ((uint16_t *)y)[i] = qz_fp32_to_fp16(x[i]); break;
    case QT_BF16: for (int64_t i = 0; i < n; ++i) ((uint16_t *)y)[i] = qz_fp32_to_bf16(x[i]); break;
    case QT_Q8_0: qz_row_q8_0(x, y, n); break;
    case QT_Q4_0: qz_row_q4_0(x, y, n); break;
    case QT_Q4_1: qz_row_q4_1(x, y, n); break;
    case QT_Q5_0: qz_row_q5_0(x, y, n); break;
    case QT_Q5_1: qz_row_q5_1(x, y, n); break;
    case QT_Q4_0_ROCMFP4: qz_row_rocmfp4(x, y, n); break;
    case QT_Q4_0_ROCMFP4_FAST: qz_row_rocmfp4_fast(x, y, n); break;
    }
}

// Rows 0 .. nrows of ncols values decoded by dec(row, floats) and encoded as dst_type, rows on n_threads threads
template <typename Dec>
static bool run_rows(int dst_type, void *dst, int64_t nrows, int64_t ncols, int n_threads, const Dec &dec) {
    const size_t dst_row = qt_row_size(dst_type, ncols);
    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    n_threads = (int)std::min<int64_t>(n_threads, std::max<int64_t>(1, nrows));
    // Rows in chunks of ~64K values a thread takes at a time
    const int64_t chunk = std::max<int64_t>(1, 65536 / ncols);
    std::atomic<int64_t> next{0};
    std::atomic<int> bad{0};
    auto work = [&] {
        std::vector<float> f((size_t)ncols);
        for (;;) {
            const int64_t r0 = next.fetch_add(chunk);
            if (r0 >= nrows) break;
            const int64_t r1 = std::min(nrows, r0 + chunk);
            for (int64_t r = r0; r < r1; ++r) {
                if (!dec(r, f.data())) {
                    bad = 1;
                    return;
                }
                encode_row(dst_type, f.data(), (uint8_t *)dst + (size_t)r * dst_row, ncols);
            }
        }
    };
    if (n_threads == 1) {
        work();
    } else {
        std::vector<std::thread> th;
        for (int i = 0; i < n_threads; ++i) th.emplace_back(work);
        for (auto &t : th) t.join();
    }
    return !bad;
}

static bool check_dst(int dst_type, int64_t ncols, char *err, size_t err_len) {
    if (qz_supported(dst_type) && qt_row_size(dst_type, ncols)) return true;
    const qt_traits *tr = qt_get_traits(dst_type);
    snprintf(err, err_len, "quantize: cannot encode type %d%s%s rows of %lld", dst_type, tr ? " " : "", tr ? tr->name : "",
             (long long)ncols);
    return false;
}

int qz_quantize(int dst_type, void *dst, int src_type, const void *src, int64_t nrows, int64_t ncols, int n_threads,
                char *err, size_t err_len) {
    const size_t src_row = qt_row_size(src_type, ncols);
    if (nrows < 0 || ncols <= 0 || !dst || !src) {
        snprintf(err, err_len, "quantize: bad arguments (%lld rows of %lld)", (long long)nrows, (long long)ncols);
        return -1;
    }
    if (!src_row) {
        const qt_traits *tr = qt_get_traits(src_type);
        snprintf(err, err_len, "quantize: cannot decode type %d%s%s rows of %lld", src_type, tr ? " " : "",
                 tr ? tr->name : "", (long long)ncols);
        return -1;
    }
    if (src_type == dst_type) {
        memcpy(dst, src, src_row * (size_t)nrows);
        return 0;
    }
    if (!check_dst(dst_type, ncols, err, err_len)) return -1;
    const uint8_t *s = (const uint8_t *)src;
    if (!run_rows(dst_type, dst, nrows, ncols, n_threads,
                  [&](int64_t r, float *f) { return qt_to_float(src_type, s + (size_t)r * src_row, f, ncols) == 0; })) {
        snprintf(err, err_len, "quantize: decoding type %d failed", src_type);
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------- FP8 (OCP e4m3fn) with block scales

// e4m3fn: sign, 4 exponent bits (bias 7), 3 mantissa bits; no infinities, S.1111.111 is NaN
float qz_fp8_e4m3_to_fp32(uint8_t v) {
    const int e = (v >> 3) & 15, m = v & 7;
    float f;
    if ((v & 0x7f) == 0x7f) f = NAN;
    else if (e == 0) f = ldexpf((float)m, -9);
    else f = ldexpf((float)(8 + m), e - 10);
    return (v & 0x80) ? -f : f;
}

static float g_fp8[256];
static const bool g_fp8_init = [] {
    for (int i = 0; i < 256; ++i) g_fp8[i] = qz_fp8_e4m3_to_fp32((uint8_t)i);
    return true;
}();

int qz_quantize_fp8_blocks(int dst_type, void *dst, const uint8_t *src, const float *scale, int64_t nrows, int64_t ncols,
                           int block_rows, int block_cols, int64_t scale_cols, int n_threads, char *err, size_t err_len) {
    if (nrows < 0 || ncols <= 0 || !dst || !src || !scale || block_rows <= 0 || block_cols <= 0 ||
        scale_cols < (ncols + block_cols - 1) / block_cols) {
        snprintf(err, err_len, "quantize fp8: bad arguments (%lld rows of %lld, blocks %d x %d, %lld scale columns)",
                 (long long)nrows, (long long)ncols, block_rows, block_cols, (long long)scale_cols);
        return -1;
    }
    if (!check_dst(dst_type, ncols, err, err_len)) return -1;
    (void)g_fp8_init;
    run_rows(dst_type, dst, nrows, ncols, n_threads, [&](int64_t r, float *f) {
        const uint8_t *w = src + (size_t)r * (size_t)ncols;
        const float *sc = scale + (size_t)(r / block_rows) * (size_t)scale_cols;
        for (int64_t j = 0; j < ncols; ++j) f[j] = g_fp8[w[j]] * sc[j / block_cols];  // one rounding, as torch's
        return true;
    });
    return 0;
}
