// quantize.cpp -- weight encoders (see quantize.h). Ports of the reference encoders, checked bit for bit against
// them by tests/quantize_verify.cpp: ggml's quantize_row_q8_0_ref, ggml_compute_fp32_to_fp16 / _bf16 (MIT), and
// ROCmFPX's rocmfp4.c (rocmfp4_quantize_row_q4_0_ref / _fast_ref: Codebook10, UE4M3 scales, exhaustive search).

#include "quantize.h"
#include "quantize_iq.h"
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

// ---------------------------------------------------------------- the K-quants (ggml's reference encoders)
// nearest_int (ggml): round to nearest, ties to even, by adding the magic 1.5 * 2^23
static inline int kq_nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// make_qkx2_quants (ggml): the scale and the minimum (*the_min = -min) of n values with weights, nmax levels, a search of
// nstep steps; L receives the levels. use_mad selects |diff| instead of diff^2 for the error.
static float kq_make_qkx2(int n, int nmax, const float *x, const float *weights, uint8_t *L, float *the_min, uint8_t *Laux,
                          float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        const float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = kq_nearest_int(iscale * (x[i] - min));
        L[i] = (uint8_t)std::max(0, std::min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        const float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = kq_nearest_int(iscale * (x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = (uint8_t)l;
            const float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                const float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// get_scale_min_k4 (ggml): the 6-bit scale and minimum of sub-block j from the 12 packed bytes
static inline void kq_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// The scales and minimums of a super-block of 256 (ggml's quantize_row_q4_K_ref / q5_K_ref, shared): eight sub-blocks, each
// searched with make_qkx2 (weights av_x + |x|); the 6-bit scales and minimums packed in scl[12] (ggml's layout), the
// super-block's fp16 d and dmin, and the levels L (0 .. lmax) of every value, refined on the packed scales.
// search: the levels of the sub-block search (15 for 4-bit, 31 for 5-bit); rmin, nstep: make_qkx2's search.
static void kq_super_block(const float *x, int search, float rmin, int nstep, int lmax, uint8_t *L, uint8_t *scl,
                           uint16_t *hd, uint16_t *hm) {
    uint8_t Laux[32];
    float weights[32], mins[8], scales[8];
    float max_scale = 0, max_min = 0;  // scales are positive: the minimum is deducted
    for (int j = 0; j < 8; ++j) {
        float sum_x2 = 0;
        for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
        const float av_x = sqrtf(sum_x2 / 32);
        for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
        scales[j] = kq_make_qkx2(32, search, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, rmin, 0.1f, nstep, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }
    const float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    const float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    memset(scl, 0, 12);
    for (int j = 0; j < 8; ++j) {
        uint8_t ls = (uint8_t)kq_nearest_int(inv_scale * scales[j]);
        uint8_t lm = (uint8_t)kq_nearest_int(inv_min * mins[j]);
        ls = std::min<uint8_t>(63, ls);
        lm = std::min<uint8_t>(63, lm);
        if (j < 4) {
            scl[j] = ls;
            scl[j + 4] = lm;
        } else {
            scl[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scl[j - 4] |= (uint8_t)((ls >> 4) << 6);
            scl[j - 0] |= (uint8_t)((lm >> 4) << 6);
        }
    }
    *hd = qz_fp32_to_fp16(max_scale / 63.f);
    *hm = qz_fp32_to_fp16(max_min / 63.f);
    const float fd = qt_fp16_to_fp32(*hd), fm = qt_fp16_to_fp32(*hm);
    for (int j = 0; j < 8; ++j) {
        uint8_t sc, m;
        kq_get_scale_min_k4(j, scl, &sc, &m);
        const float d = fd * sc;
        if (!d) continue;
        const float dm = fm * m;
        for (int ii = 0; ii < 32; ++ii) {
            const int l = kq_nearest_int((x[32 * j + ii] + dm) / d);
            L[32 * j + ii] = (uint8_t)std::max(0, std::min(lmax, l));
        }
    }
}

// Q4_K: super-block of 256: fp16 d, fp16 dmin, scales[12], qs[128] (the low nibble of byte i holds value i of its 64, the high
// nibble value i + 32).
void qz_row_q4_K(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, fp16 dmin, scales[12], qs[128]
    for (int64_t b = 0; b < k / 256; ++b, x += 256, out += 144) {
        uint8_t L[256];
        uint16_t hd, hm;
        kq_super_block(x, 15, -1.f, 20, 15, L, out + 4, &hd, &hm);
        memcpy(out, &hd, 2);
        memcpy(out + 2, &hm, 2);
        uint8_t *q = out + 16;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
            q += 32;
        }
    }
}

// Q5_K: as Q4_K with 31-level sub-blocks and levels 0 .. 31: qh[32] holds the fifth bit (of a level above 15: bit 0 for the
// first 32 values of a 64-group, bit 1 for the next 32, and so on, 2 bits per group), qs[128] the low four bits.
void qz_row_q5_K(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, fp16 dmin, scales[12], qh[32], qs[128]
    for (int64_t b = 0; b < k / 256; ++b, x += 256, out += 176) {
        uint8_t L[256];
        uint16_t hd, hm;
        kq_super_block(x, 31, -0.5f, 15, 31, L, out + 4, &hd, &hm);
        memcpy(out, &hd, 2);
        memcpy(out + 2, &hm, 2);
        uint8_t *qh = out + 16, *ql = out + 48;
        memset(qh, 0, 32);
        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < 256; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j];
                if (l1 > 15) {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                int l2 = L[n + j + 32];
                if (l2 > 15) {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[j] = (uint8_t)(l1 | (l2 << 4));
            }
            m1 <<= 2;
            m2 <<= 2;
            ql += 32;
        }
    }
}

// make_qx_quants (ggml): the symmetric scale of n values for nmax levels (levels -nmax .. nmax - 1, stored as L + nmax);
// rmse_type selects the weights of the error search (1: x^2, 2: 1, 3: |x|, else sqrt|x|), qw overrides them; a negative
// rmse_type returns early (the blend of the two scales). Returns the scale.
static float kq_make_qx(int n, int nmax, const float *x, int8_t *L, int rmse_type, const float *qw) {
    const float kGroupMaxEps = 1e-15f;
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < kGroupMaxEps) {  // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            const int l = kq_nearest_int(iscale * x[i]);
            L[i] = (int8_t)(nmax + std::max(-nmax, std::min(nmax - 1, l)));
        }
        return 1 / iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    auto weight = [&](int i) {
        return qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1.f : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
    };
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = kq_nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = (int8_t)(l + nmax);
        const float w = weight(i);
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = kq_nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax - 1, l));
            const float w = weight(i);
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                const int l = kq_nearest_int(iscale * x[i]);
                L[i] = (int8_t)(nmax + std::max(-nmax, std::min(nmax - 1, l)));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

// Q6_K: super-block of 256: ql[128] (the low four bits), qh[64] (the top two bits, four values per byte), scales[16] (int8),
// fp16 d at byte 208. Levels -32 .. 31 (stored + 32); the scales are symmetric with d = -128 / max scale.
void qz_row_q6_K(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: ql[128], qh[64], scales[16], fp16 d
    for (int64_t b = 0; b < k / 256; ++b, x += 256, out += 210) {
        int8_t L[256];
        float scales[16];
        float max_scale = 0, max_abs_scale = 0;
        for (int ib = 0; ib < 16; ++ib) {
            const float scale = kq_make_qx(16, 32, x + 16 * ib, L + 16 * ib, 1, nullptr);
            scales[ib] = scale;
            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }
        memset(out, 0, 210);
        if (max_abs_scale < 1e-15f) continue;  // an all-zero super-block: the zero block
        const float iscale = -128.f / max_scale;
        const uint16_t hd = qz_fp32_to_fp16(1 / iscale);
        memcpy(out + 208, &hd, 2);
        int8_t *sc = (int8_t *)(out + 192);
        for (int ib = 0; ib < 16; ++ib) sc[ib] = (int8_t)std::min(127, kq_nearest_int(iscale * scales[ib]));
        const float fd = qt_fp16_to_fp32(hd);
        for (int j = 0; j < 16; ++j) {
            const float d = fd * sc[j];
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) {
                const int l = kq_nearest_int(x[16 * j + ii] / d);
                L[16 * j + ii] = (int8_t)(std::max(-32, std::min(31, l)) + 32);
            }
        }
        uint8_t *ql = out, *qh = out + 128;
        for (int j = 0; j < 256; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l + 0] & 0xF, q2 = L[j + l + 32] & 0xF, q3 = L[j + l + 64] & 0xF, q4 = L[j + l + 96] & 0xF;
                ql[l + 0] = (uint8_t)(q1 | (q3 << 4));
                ql[l + 32] = (uint8_t)(q2 | (q4 << 4));
                qh[l] = (uint8_t)((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6));
            }
            ql += 64;
            qh += 32;
        }
    }
}

// Q2_K: super-block of 256 in 16 groups of 16: scales[16] (the low nibble the 4-bit scale, the high the 4-bit minimum), qs[64]
// (the 2-bit level of value i in bits 2 * (i / 32) of byte i mod 32 ... as the reference's packing), fp16 d at 80, dmin at 82.
void qz_row_q2_K(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: scales[16], qs[64], fp16 d, fp16 dmin
    for (int64_t b = 0; b < k / 256; ++b, x += 256, out += 84) {
        uint8_t L[256], Laux[16];
        float weights[16], mins[16], scales[16];
        float max_scale = 0, max_min = 0;
        for (int j = 0; j < 16; ++j) {
            for (int l = 0; l < 16; ++l) weights[l] = fabsf(x[16 * j + l]);
            scales[j] = kq_make_qkx2(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux, -0.5f, 0.1f, 15, true);
            if (scales[j] > max_scale) max_scale = scales[j];
            if (mins[j] > max_min) max_min = mins[j];
        }
        const float q4scale = 15.f;
        uint8_t *sc = out;
        if (max_scale > 0) {
            const float iscale = q4scale / max_scale;
            for (int j = 0; j < 16; ++j) sc[j] = (uint8_t)kq_nearest_int(iscale * scales[j]);
            const uint16_t hd = qz_fp32_to_fp16(max_scale / q4scale);
            memcpy(out + 80, &hd, 2);
        } else {
            const uint16_t hd = qz_fp32_to_fp16(0.f);
            memcpy(out + 80, &hd, 2);
        }
        if (max_min > 0) {
            const float iscale = q4scale / max_min;
            for (int j = 0; j < 16; ++j) sc[j] |= (uint8_t)(kq_nearest_int(iscale * mins[j]) << 4);
            const uint16_t hm = qz_fp32_to_fp16(max_min / q4scale);
            memcpy(out + 82, &hm, 2);
        } else {
            const uint16_t hm = qz_fp32_to_fp16(0.f);
            memcpy(out + 82, &hm, 2);
        }
        uint16_t hd, hm;
        memcpy(&hd, out + 80, 2);
        memcpy(&hm, out + 82, 2);
        const float fd = qt_fp16_to_fp32(hd), fm = qt_fp16_to_fp32(hm);
        for (int j = 0; j < 16; ++j) {
            const float d = fd * (sc[j] & 0xF);
            if (!d) continue;
            const float dm = fm * (sc[j] >> 4);
            for (int ii = 0; ii < 16; ++ii) {
                const int l = kq_nearest_int((x[16 * j + ii] + dm) / d);
                L[16 * j + ii] = (uint8_t)std::max(0, std::min(3, l));
            }
        }
        uint8_t *qs = out + 16;
        for (int j = 0; j < 256; j += 128)
            for (int l = 0; l < 32; ++l)
                qs[j / 4 + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
    }
}

// make_q3_quants (ggml, the do_rmse path): the scale of n values for nmax levels (-nmax .. nmax - 1, stored as L + nmax):
// the rounded levels, then a coordinate refinement of the least-squares error (weights x^2), five passes at most.
static float kq_make_q3(int n, int nmax, const float *x, int8_t *L) {
    const float kGroupMaxEps = 1e-15f;
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < kGroupMaxEps) {  // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    const float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = kq_nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax - 1, l));
        L[i] = (int8_t)l;
        const float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = x[i] * x[i];
            float slx = sumlx - w * x[i] * L[i];
            if (slx > 0) {
                float sl2 = suml2 - w * L[i] * L[i];
                int new_l = kq_nearest_int(x[i] * sl2 / slx);
                new_l = std::max(-nmax, std::min(nmax - 1, new_l));
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (int8_t)new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    for (int i = 0; i < n; ++i) L[i] += (int8_t)nmax;
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

// Q3_K: super-block of 256 in 16 groups: hmask[32] (the fourth bit of value i at bit i / 32 of byte i mod 32), qs[64] (the low
// two bits), scales[12] (sixteen 6-bit scales, offset by 32, packed), fp16 d at 108.
void qz_row_q3_K(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: hmask[32], qs[64], scales[12], fp16 d
    for (int64_t b = 0; b < k / 256; ++b, x += 256, out += 110) {
        int8_t L[256];
        float scales[16];
        float max_scale = 0, amax = 0;
        for (int j = 0; j < 16; ++j) {
            scales[j] = kq_make_q3(16, 4, x + 16 * j, L + 16 * j);
            const float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale;
                max_scale = scales[j];
            }
        }
        uint8_t *sc = out + 96;
        memset(sc, 0, 12);
        uint16_t hd;
        if (max_scale) {
            const float iscale = -32.f / max_scale;
            for (int j = 0; j < 16; ++j) {
                int8_t l = (int8_t)kq_nearest_int(iscale * scales[j]);
                l = (int8_t)(std::max(-32, std::min(31, (int)l)) + 32);
                if (j < 8) sc[j] = (uint8_t)(l & 0xF);
                else sc[j - 8] |= (uint8_t)((l & 0xF) << 4);
                l >>= 4;
                sc[j % 4 + 8] |= (uint8_t)(l << (2 * (j / 4)));
            }
            hd = qz_fp32_to_fp16(1 / iscale);
        } else {
            hd = qz_fp32_to_fp16(0.f);
        }
        memcpy(out + 108, &hd, 2);
        const float fd = qt_fp16_to_fp32(hd);
        for (int j = 0; j < 16; ++j) {
            int sv = j < 8 ? (sc[j] & 0xF) : (sc[j - 8] >> 4);
            sv = (sv | (((sc[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32;
            const float d = fd * sv;
            if (!d) continue;
            for (int ii = 0; ii < 16; ++ii) {
                int l = kq_nearest_int(x[16 * j + ii] / d);
                l = std::max(-4, std::min(3, l));
                L[16 * j + ii] = (int8_t)(l + 4);
            }
        }
        memset(out, 0, 32);  // hmask
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < 256; ++j) {
            if (L[j] > 3) {
                out[m] |= hm;
                L[j] -= 4;
            }
            if (++m == 32) {
                m = 0;
                hm <<= 1;
            }
        }
        uint8_t *qs = out + 32;
        for (int j = 0; j < 256; j += 128)
            for (int l = 0; l < 32; ++l)
                qs[j / 4 + l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6));
    }
}

// ---------------------------------------------------------------- MXFP4 and NVFP4 (ggml's reference encoders)
// The E2M1 codebook of the two (twice the values: the scale is taken at half), as ggml's kvalues_mxfp4
static const int8_t kq_kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// GGML_E8M0_TO_FP32_HALF: 2^(e - 128)
static inline float kq_e8m0_half(uint8_t x) {
    const uint32_t bits = x < 2 ? 0x00200000u << x : (uint32_t)(x - 1) << 23;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ggml_ue4m3_to_fp32 (the half convention: value * 0.5 of the UE4M3 value)
static inline float kq_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int ex = (x >> 3) & 0xF, man = x & 0x7;
    const float raw = ex == 0 ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, ex - 7);
    return raw * 0.5f;
}

// ggml_fp32_to_ue4m3 (ggml's rounding: nearest, the subnormals as m * 2^-9)
static inline uint8_t kq_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) return 0;
    if (x > 448.0f) x = 448.0f;
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp = (int)((bits >> 23) & 0xFF) - 127;
    const int fp32_man = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        int man = (int)(x * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 1) return 0;
        return (uint8_t)man;
    }
    if (ue4m3_exp >= 15) return 0x7E;
    const int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) return 0x7E;
    }
    return (uint8_t)((ue4m3_exp << 3) | ue4m3_man);
}

// best_index_mxfp4 (ggml): the codebook entry whose value times e is nearest x (the first on a tie)
static inline uint8_t kq_best_index_mxfp4(float x, float e) {
    int best_index = 0;
    float best_err = fabsf(kq_kvalues_mxfp4[0] * e - x);
    for (int i = 1; i < 16; i++) {
        const float err = fabsf(kq_kvalues_mxfp4[i] * e - x);
        if (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    return (uint8_t)best_index;
}

// MXFP4: block of 32: e (the E8M0 exponent of the block's maximum), qs[16] (value j in the low nibble of byte j, j + 16 in the high)
void qz_row_mxfp4(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: e, qs[16]
    for (int64_t i = 0; i < k / 32; ++i, x += 32, out += 17) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float v = x[j];
            if (amax < fabsf(v)) amax = fabsf(v);
        }
        const uint8_t e = amax > 0.0f ? (uint8_t)(floorf(log2f(amax)) - 2 + 127) : 0;
        const float d = kq_e8m0_half(e);
        out[0] = e;
        for (int j = 0; j < 16; ++j) {
            const uint8_t x0 = kq_best_index_mxfp4(x[j], d);
            const uint8_t x1 = kq_best_index_mxfp4(x[16 + j], d);
            out[1 + j] = (uint8_t)(x0 | (x1 << 4));
        }
    }
}

// NVFP4: block of 64 in four sub-blocks of 16: d[4] (the UE4M3 scale of each, amax / 6), qs[32] (sub-block s in bytes 8s .. 8s + 7:
// value j in the low nibble of byte j, j + 8 in the high)
void qz_row_nvfp4(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: d[4], qs[32]
    for (int64_t i = 0; i < k / 64; ++i, x += 64, out += 36) {
        for (int sb = 0; sb < 4; ++sb) {
            const float *xb = x + 16 * sb;
            float amax = 0.0f;
            for (int j = 0; j < 16; ++j)
                if (amax < fabsf(xb[j])) amax = fabsf(xb[j]);
            const uint8_t ue = kq_fp32_to_ue4m3(amax / 6.0f);
            out[sb] = ue;
            const float d = kq_ue4m3_to_fp32(ue);
            for (int j = 0; j < 8; ++j) {
                const uint8_t x0 = kq_best_index_mxfp4(xb[j], d);
                const uint8_t x1 = kq_best_index_mxfp4(xb[8 + j], d);
                out[4 + 8 * sb + j] = (uint8_t)(x0 | (x1 << 4));
            }
        }
    }
}

// ---------------------------------------------------------------- the ternary types (ggml's reference encoders)
// TQ1_0: super-block of 256: qs[48] (5 trits per byte, base 3, the trits in the order of the reference's loops), qh[4] (4
// trits per byte, the first one as the most significant), fp16 d at 52 (the maximum magnitude). A trit t is round(x / d) + 1.
void qz_row_tq1_0(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: qs[48], qh[4], fp16 d
    for (int64_t b = 0; b < k / 256; ++b, x += 256, out += 54) {
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) amax = amax > fabsf(x[j]) ? amax : fabsf(x[j]);  // ggml's MAX: a NaN wins
        const float d = amax;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t hd = qz_fp32_to_fp16(d);
        memcpy(out + 52, &hd, 2);
        const float *xp = x;
        // 5 trits per byte, along 32 bytes
        for (int j = 0; j < 32; j += 32) {
            for (int m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (int n = 0; n < 5; ++n) {
                    const int xi = (int)lroundf(xp[m + n * 32] * id) + 1;
                    q = (uint8_t)(q * 3);
                    q = (uint8_t)(q + xi);
                }
                q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
                out[j + m] = q;
            }
            xp += 5 * 32;
        }
        // along 16 bytes
        for (int j = 32; j < 48; j += 16) {
            for (int m = 0; m < 16; ++m) {
                uint8_t q = 0;
                for (int n = 0; n < 5; ++n) {
                    const int xi = (int)lroundf(xp[m + n * 16] * id) + 1;
                    q = (uint8_t)(q * 3);
                    q = (uint8_t)(q + xi);
                }
                q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
                out[j + m] = q;
            }
            xp += 5 * 16;
        }
        // 4 trits per byte
        for (int j = 0; j < 4; ++j) {
            uint8_t q = 0;
            for (int m = 0; m < 4; ++m) {
                const int xi = (int)lroundf(xp[j + m * 4] * id) + 1;
                q = (uint8_t)(q * 3);
                q = (uint8_t)(q + xi);
            }
            q = (uint8_t)(q * 3);
            q = (uint8_t)(((uint16_t)q * 256 + (243 - 1)) / 243);
            out[48 + j] = q;
        }
    }
}

// TQ2_0: super-block of 256: qs[64] (2 bits per trit, trit t = round(x / d) + 1, the value of the group of 32 at bits 2 l), fp16 d
// at 64.
void qz_row_tq2_0(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: qs[64], fp16 d
    for (int64_t b = 0; b < k / 256; ++b, x += 256, out += 66) {
        float amax = 0.0f;
        for (int j = 0; j < 256; ++j) amax = amax > fabsf(x[j]) ? amax : fabsf(x[j]);  // ggml's MAX: a NaN wins
        const float d = amax;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t hd = qz_fp32_to_fp16(d);
        memcpy(out + 64, &hd, 2);
        const float *xp = x;
        for (int j = 0; j < 64; j += 32) {
            for (int m = 0; m < 32; ++m) {
                uint8_t q = 0;
                for (int n = 0; n < 4; ++n) {
                    const int xi = (int)lroundf(xp[m + n * 32] * id) + 1;
                    q = (uint8_t)(q + ((xi & 3) << (2 * n)));
                }
                out[j + m] = q;
            }
            xp += 4 * 32;
        }
    }
}

// ---------------------------------------------------------------- IQ4_NL and IQ4_XS (ggml's reference encoders, no importance matrix)
static const int8_t kq_kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// best_index_int8 (ggml): the nearest of the sorted values (a binary search, the lower one on a tie)
static inline int kq_best_index_int8(int n, const int8_t *val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n - 1]) return n - 1;
    int ml = 0, mu = n - 1;
    while (mu - ml > 1) {
        const int mav = (ml + mu) / 2;
        if (x < val[mav]) mu = mav;
        else ml = mav;
    }
    return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

// quantize_row_iq4_nl_impl (ggml) without quant_weights: the weights are x^2. One block of sbs values in blocks of bs; with more
// than one block (IQ4_XS), the blocks' scales are quantized to six bits around -max / 32 (scales_h: two bits per block above
// the low four, in 16-bit words; scales_l: the low four bits, two blocks per byte).
static void kq_iq4_impl(int sbs, int bs, const float *x, uint16_t *dh, uint8_t *q4, uint16_t *scales_h, uint8_t *scales_l,
                        float *scales, float *weight, uint8_t *L, const int8_t *values, int ntry) {
    memset(q4, 0, sbs / 2);
    dh[0] = qz_fp32_to_fp16(0.f);
    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < sbs / bs; ++ib) {
        const float *xb = x + ib * bs;
        uint8_t *Lb = L + ib * bs;
        for (int j = 0; j < bs; ++j) weight[j] = xb[j] * xb[j];
        float amax = 0, max = 0;
        for (int j = 0; j < bs; ++j) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax < 1e-15f) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max / values[0] : max / values[0];
        float id = 1 / d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < bs; ++j) {
            const float al = id * xb[j];
            const int l = kq_best_index_int8(16, values, al);
            Lb[j] = (uint8_t)l;
            const float q = values[l];
            const float w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = sumq2 > 0 ? sumqx / sumq2 : 0.f;
        float best = d * sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0]) / max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < bs; ++j) {
                const float al = id * xb[j];
                const int l = kq_best_index_int8(16, values, al);
                const float q = values[l];
                const float w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        const float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d;
            max_scale = d;
        }
    }
    if (sbs / bs > 1) {
        const int nb = sbs / bs;
        memset(scales_h, 0, ((nb + 7) / 8) * sizeof(uint16_t));
        const float d = -max_scale / 32;
        dh[0] = qz_fp32_to_fp16(d);
        const float id = d ? 1 / d : 0.f;
        for (int ib = 0; ib < nb; ++ib) {
            int l = kq_nearest_int(id * scales[ib]);
            l = std::max(-32, std::min(31, l));
            const float dl = d * l;
            const float idl = dl ? 1 / dl : 0.f;
            uint8_t *Lb = L + ib * bs;
            const float *xb = x + ib * bs;
            for (int j = 0; j < bs; ++j) Lb[j] = (uint8_t)kq_best_index_int8(16, values, idl * xb[j]);
            l += 32;
            const uint8_t l_l = l & 0xf, l_h = (uint8_t)(l >> 4);
            if (ib % 2 == 0) scales_l[ib / 2] = l_l;
            else scales_l[ib / 2] |= (uint8_t)(l_l << 4);
            scales_h[ib / 8] |= (uint16_t)(l_h << (2 * (ib % 8)));
        }
    } else {
        dh[0] = qz_fp32_to_fp16(scales[0]);
        if (ntry > 0) {
            const float id = scales[0] ? 1 / scales[0] : 0;
            for (int j = 0; j < sbs; ++j) L[j] = (uint8_t)kq_best_index_int8(16, values, id * x[j]);
        }
    }
    for (int i = 0; i < sbs / 32; ++i)
        for (int j = 0; j < 16; ++j) q4[16 * i + j] = (uint8_t)(L[32 * i + j] | (L[32 * i + 16 + j] << 4));
}

// IQ4_NL: block of 32: fp16 d, qs[16] (the non-linear levels kvalues_iq4nl of the low and high nibbles)
void qz_row_iq4_nl(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, qs[16]
    float weight[32], scales[1];
    uint8_t L[32];
    for (int64_t b = 0; b < k / 32; ++b, x += 32, out += 18) {
        uint16_t dh;
        uint8_t q4[16];
        // (the search of ggml's quantize_iq4_nl, the chunk API llama-quantize uses: seven retries)
        kq_iq4_impl(32, 32, x, &dh, q4, nullptr, nullptr, scales, weight, L, kq_kvalues_iq4nl, 7);
        memcpy(out, &dh, 2);
        memcpy(out + 2, q4, 16);
    }
}

// IQ4_XS: super-block of 256 in eight blocks of 32: fp16 d, scales_h (two bits of the six-bit scales), scales_l[4] (their low four
// bits), qs[128]
void qz_row_iq4_xs(const float *x, void *y, int64_t k) {
    uint8_t *out = (uint8_t *)y;  // block: fp16 d, scales_h (uint16), scales_l[4], qs[128]
    float weight[32], scales[8];
    uint8_t L[256];
    for (int64_t b = 0; b < k / 256; ++b, x += 256, out += 136) {
        uint16_t dh = 0, sh = 0;
        uint8_t sl[4] = {0, 0, 0, 0}, q4[128];
        kq_iq4_impl(256, 32, x, &dh, q4, &sh, sl, scales, weight, L, kq_kvalues_iq4nl, 7);
        memcpy(out, &dh, 2);
        memcpy(out + 2, &sh, 2);
        memcpy(out + 4, sl, 4);
        memcpy(out + 8, q4, 128);
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
           t == QT_Q4_0_ROCMFP4_FAST || t == QT_Q4_0 || t == QT_Q4_1 || t == QT_Q5_0 || t == QT_Q5_1 || t == QT_Q4_K || t == QT_Q5_K || t == QT_Q6_K || t == QT_Q2_K || t == QT_Q3_K || t == QT_MXFP4 || t == QT_NVFP4 || t == QT_TQ1_0 || t == QT_TQ2_0 || t == QT_IQ4_NL || t == QT_IQ4_XS || qz_iq_supported(t);
}

static void encode_row(int t, const float *x, void *y, int64_t n, const float *qw) {
    switch (t) {
    case QT_F32: memcpy(y, x, (size_t)n * 4); break;
    case QT_F16: for (int64_t i = 0; i < n; ++i) ((uint16_t *)y)[i] = qz_fp32_to_fp16(x[i]); break;
    case QT_BF16: for (int64_t i = 0; i < n; ++i) ((uint16_t *)y)[i] = qz_fp32_to_bf16(x[i]); break;
    case QT_Q8_0: qz_row_q8_0(x, y, n); break;
    case QT_Q4_0: qz_row_q4_0(x, y, n); break;
    case QT_Q4_1: qz_row_q4_1(x, y, n); break;
    case QT_Q5_0: qz_row_q5_0(x, y, n); break;
    case QT_Q5_1: qz_row_q5_1(x, y, n); break;
    case QT_Q4_K: qz_row_q4_K(x, y, n); break;
    case QT_Q5_K: qz_row_q5_K(x, y, n); break;
    case QT_Q6_K: qz_row_q6_K(x, y, n); break;
    case QT_Q2_K: qz_row_q2_K(x, y, n); break;
    case QT_Q3_K: qz_row_q3_K(x, y, n); break;
    case QT_MXFP4: qz_row_mxfp4(x, y, n); break;
    case QT_NVFP4: qz_row_nvfp4(x, y, n); break;
    case QT_TQ1_0: qz_row_tq1_0(x, y, n); break;
    case QT_TQ2_0: qz_row_tq2_0(x, y, n); break;
    case QT_IQ4_NL: qz_row_iq4_nl(x, y, n); break;
    case QT_IQ4_XS: qz_row_iq4_xs(x, y, n); break;
    case QT_IQ2_XXS: case QT_IQ2_XS: case QT_IQ2_S: case QT_IQ3_XXS: case QT_IQ3_S: case QT_IQ1_S: case QT_IQ1_M: qz_iq_encode(t, x, y, n, qw); break;
    case QT_Q4_0_ROCMFP4: qz_row_rocmfp4(x, y, n); break;
    case QT_Q4_0_ROCMFP4_FAST: qz_row_rocmfp4_fast(x, y, n); break;
    }
}

// Rows 0 .. nrows of ncols values decoded by dec(row, floats) and encoded as dst_type, rows on n_threads threads
template <typename Dec>
static bool run_rows(int dst_type, void *dst, int64_t nrows, int64_t ncols, int n_threads, const float *qw, const Dec &dec) {
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
                encode_row(dst_type, f.data(), (uint8_t *)dst + (size_t)r * dst_row, ncols, qw);
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
    return qz_quantize_im(dst_type, dst, src_type, src, nrows, ncols, nullptr, n_threads, err, err_len);
}

int qz_quantize_im(int dst_type, void *dst, int src_type, const void *src, int64_t nrows, int64_t ncols,
                   const float *imatrix, int n_threads, char *err, size_t err_len) {
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
    // the IQ types' weights: the importance matrix, or uniform weights (ggml requires one for some of them)
    const std::vector<float> ones(imatrix ? 0 : (size_t)ncols, 1.0f);
    const float *qw = imatrix ? imatrix : ones.data();
    if (!run_rows(dst_type, dst, nrows, ncols, n_threads, qw,
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
    run_rows(dst_type, dst, nrows, ncols, n_threads, nullptr, [&](int64_t r, float *f) {
        const uint8_t *w = src + (size_t)r * (size_t)ncols;
        const float *sc = scale + (size_t)(r / block_rows) * (size_t)scale_cols;
        for (int64_t j = 0; j < ncols; ++j) f[j] = g_fp8[w[j]] * sc[j / block_cols];  // one rounding, as torch's
        return true;
    });
    return 0;
}
