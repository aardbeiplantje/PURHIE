// quantize_iq.cpp -- see quantize_iq.h. ggml's reference encoders for the IQ types (MIT License, Copyright (c) 2023-2026 The
// ggml authors; the algorithms re-implemented here, the codebooks copied into iq_grids.h / iq_kgrids.h by the generators).

#include "quantize_iq.h"

#include "iq_kgrids.h"
#include "quant_types.h"
#include "quantize.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#define NGRID_IQ1S 2048
#define IQ_TABLE(type, name, size) static const type name[size]
#include "iq_grids.h"
#undef IQ_TABLE

namespace {

const float kGroupMaxEps = 1e-15f;

inline int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

inline float max_of(float a, float b) { return a > b ? a : b; }  // ggml's MAX

// make_qp_quants (ggml): the levels 0 .. nmax of n values (weights quant_weights): the scale, refined over a grid of scales
float make_qp_quants(int n, int nmax, const float *x, uint8_t *L, const float *quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) max = max_of(max, x[i]);
    if (max < kGroupMaxEps) {  // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) L[i] = (uint8_t)nearest_int(iscale * x[i]);
    float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        const float diff = x[i] - scale * L[i];
        const float w = quant_weights[i];
        best_mse += w * diff * diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        const float iscale_is = (0.1f * is + nmax) / max;
        const float scale_is = 1 / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is * x[i]);
            l = std::min(nmax, l);
            const float diff = x[i] - scale_is * l;
            const float w = quant_weights[i];
            mse += w * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::min(nmax, l);
        L[i] = (uint8_t)l;
        const float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = quant_weights[i];
            float slx = sumlx - w * x[i] * L[i];
            float sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = std::min(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (uint8_t)new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

// ---- the lookup tables of the 2-bit codebooks (ggml's iq2xs_init_impl): the grid as 8 signed bytes (2l + 1), the map from
// a 2-bit-per-value index to the grid entry (or to the neighbour list, -(k + 1)), and the neighbour lists (nearest first).
struct Iq2Tab {
    std::vector<uint64_t> grid;
    std::vector<int> map;
    std::vector<uint16_t> nb;
};

int iq2_compare(const void *left, const void *right) {  // (distance, index) order, as ggml's qsort
    const int *l = (const int *)left;
    const int *r = (const int *)right;
    return l[0] < r[0] ? -1 : l[0] > r[0] ? 1 : l[1] < r[1] ? -1 : l[1] > r[1] ? 1 : 0;
}

Iq2Tab build_iq2_tab(const uint16_t *kgrid, int grid_size, int nwant) {
    const int kmap_size = 43692;
    Iq2Tab t;
    t.grid.resize(grid_size);
    for (int k = 0; k < grid_size; ++k) {
        int8_t *pos = (int8_t *)(t.grid.data() + k);
        for (int i = 0; i < 8; ++i) {
            const int l = (kgrid[k] >> 2 * i) & 0x3;
            pos[i] = (int8_t)(2 * l + 1);
        }
    }
    t.map.assign(kmap_size, -1);
    for (int i = 0; i < grid_size; ++i) {
        uint64_t aux64 = t.grid[i];
        uint8_t *aux8 = (uint8_t *)&aux64;
        uint16_t index = 0;
        for (int k = 0; k < 8; ++k) {
            const uint16_t q = (uint16_t)((aux8[k] - 1) / 2);
            index |= (uint16_t)(q << 2 * k);
        }
        t.map[index] = i;
    }
    std::vector<int> dist2(2 * (size_t)grid_size);
    auto point_dist = [&](int i, int8_t *pos) {
        for (int k = 0; k < 8; ++k) {
            const int l = (i >> 2 * k) & 0x3;
            pos[k] = (int8_t)(2 * l + 1);
        }
        for (int j = 0; j < grid_size; ++j) {
            const int8_t *pg = (const int8_t *)(t.grid.data() + j);
            int d2 = 0;
            for (int k = 0; k < 8; ++k) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2.data(), grid_size, 2 * sizeof(int), iq2_compare);
    };
    int8_t pos[8];
    int num_neighbors = 0, num_not_in_map = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (t.map[i] >= 0) continue;
        ++num_not_in_map;
        point_dist(i, pos);
        int n = 0;
        int d2 = dist2[0];
        int nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2 * j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2 * j];
                ++nhave;
            }
            ++n;
        }
        num_neighbors += n;
    }
    t.nb.assign((size_t)num_neighbors + num_not_in_map, 0);
    int counter = 0;
    for (int i = 0; i < kmap_size; ++i) {
        if (t.map[i] >= 0) continue;
        point_dist(i, pos);
        t.map[i] = -(counter + 1);
        const int start = counter++;
        int d2 = dist2[0];
        int n = 0, nhave = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2 * j] > d2) {
                if (nhave == nwant) break;
                d2 = dist2[2 * j];
                ++nhave;
            }
            t.nb[counter++] = (uint16_t)dist2[2 * j + 1];
            ++n;
        }
        t.nb[start] = (uint16_t)n;
    }
    return t;
}

// The tables of one codebook, built on first use (thread-safe): 0 IQ2_XXS (256 entries, two neighbours), 1 IQ2_XS (512, two),
// 2 IQ1 (2048, three), 3 IQ2_S (1024, one)
const Iq2Tab &iq2_tab(int which) {
    static const Iq2Tab t0 = build_iq2_tab(kgrid_2bit_256, 256, 2);
    static const Iq2Tab t1 = build_iq2_tab(kgrid_2bit_512, 512, 2);
    static const Iq2Tab t2 = build_iq2_tab(kgrid_1bit_2048, NGRID_IQ1S, 3);
    static const Iq2Tab t3 = build_iq2_tab(kgrid_2bit_1024, 1024, 1);
    return which == 0 ? t0 : which == 1 ? t1 : which == 2 ? t2 : t3;
}

// iq2_find_best_neighbour (ggml): the grid entry among the neighbours of a point nearest to xval (weighted), its levels to L
int iq2_find_best_neighbour(const Iq2Tab &t, int base, const float *xval, const float *weight, float scale, int8_t *L) {
    const int num_neighbors = t.nb[base];
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t *pg = (const int8_t *)(t.grid.data() + t.nb[base + j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            const float q = pg[i];
            const float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = t.nb[base + j];
        }
    }
    const int8_t *pg = (const int8_t *)(t.grid.data() + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (int8_t)((pg[i] - 1) / 2);
    return grid_index;
}

// the grid index of a point of levels u (2 bits per value): the codebook entry, or the neighbour list's base (negative: -(base + 1))
inline int iq2_lookup(const Iq2Tab &t, uint16_t u, bool *exact) {
    const int g = t.map[u];
    *exact = g >= 0;
    return g;
}
inline int iq2_neighbours_base(const Iq2Tab &t, uint16_t u) { return -t.map[u] - 1; }

// IQ2_XXS: block of 256: fp16 d, qs[32] (uint16 words: four grid bytes, then four 7-bit sign groups and a 4-bit scale per 32)
void iq2_xxs_row(const float *x, uint8_t *out, int64_t n, const float *quant_weights) {
    const Iq2Tab &t = iq2_tab(0);
    const int kMaxQ = 3;
    const int64_t nbl = n / 256;
    for (int64_t ibl = 0; ibl < nbl; ++ibl, out += 66) {
        float scales[8];
        float weight[32], xval[32], waux[32];
        int8_t L[32], Laux[32];
        uint8_t block_signs[4];
        uint32_t q2[2 * (256 / 32)];
        const uint16_t zero_h = qz_fp32_to_fp16(0.f);
        memcpy(out, &zero_h, 2);
        memset(q2, 0, 64);
        float max_scale = 0;
        const float *xbl = x + 256 * ibl;
        float sumx2 = 0;
        for (int i = 0; i < 256; ++i) sumx2 += xbl[i] * xbl[i];
        const float sigma2 = sumx2 / 256;
        for (int ib = 0; ib < 8; ++ib) {
            const float *xb = xbl + 32 * ib;
            const float *qw = quant_weights + 256 * ibl + 32 * ib;
            for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 4; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        ++nflip;
                        s |= (uint8_t)(1 << i);
                    }
                }
                if (nflip % 2) {
                    int imin = 0;
                    float min = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
                    for (int i = 1; i < 8; ++i) {
                        const float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (uint8_t)(1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = max_of(max, xval[i]);
            if (max < kGroupMaxEps) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float scale = make_qp_quants(32, kMaxQ + 1, xval, (uint8_t *)L, weight);
            const float eff_max = scale * kMaxQ;
            if (eff_max <= 0) {
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            for (int is = -6; is <= 6; ++is) {
                const float id = (2 * kMaxQ - 1 + is * 0.1f) / eff_max;
                const float this_scale = 1 / id;
                for (int k = 0; k < 4; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        const int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        Laux[8 * k + i] = (int8_t)std::max(0, std::min(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (uint16_t)(Laux[8 * k + i] << 2 * i);
                    bool exact;
                    if (!(iq2_lookup(t, u, &exact) >= 0))
                        iq2_find_best_neighbour(t, iq2_neighbours_base(t, u), xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    const float w = weight[i];
                    const float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    memcpy(L, Laux, 32);
                }
            }
            if (scale > 0) {
                const float id = 1 / scale;
                for (int k = 0; k < 4; ++k) {
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        l = std::max(0, std::min(kMaxQ - 1, l));
                        u |= (uint16_t)(l << 2 * i);
                    }
                    bool exact;
                    int grid_index = iq2_lookup(t, u, &exact);
                    if (grid_index < 0) grid_index = iq2_find_best_neighbour(t, iq2_neighbours_base(t, u), xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                    const int8_t *pg = (const int8_t *)(t.grid.data() + grid_index);
                    for (int i = 0; i < 8; ++i) L[8 * k + i] = (int8_t)((pg[i] - 1) / 2);
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    const float w = weight[i];
                    const float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (uint16_t)(L[8 * k + i] << 2 * i);
                const int grid_index = t.map[u];
                if (grid_index < 0) throw std::runtime_error("iq2_xxs: a point is not on the grid (ggml aborts here too)");
                q2[2 * ib + 0] |= ((uint32_t)grid_index) << 8 * k;
                q2[2 * ib + 1] |= ((uint32_t)block_signs[k]) << 7 * k;
            }
            scales[ib] = scale;
            max_scale = max_of(max_scale, scale);
        }
        if (!max_scale) {
            memset(out + 2, 0, 64);
            continue;
        }
        const float d = max_scale / 31;
        const uint16_t hd = qz_fp32_to_fp16(d);
        memcpy(out, &hd, 2);
        const float id = 1 / d;
        for (int ib = 0; ib < 8; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = std::max(0, std::min(15, l));
            q2[2 * ib + 1] |= ((uint32_t)l) << 28;
        }
        memcpy(out + 2, q2, 64);
    }
}

// IQ2_XS: block of 256: fp16 d, qs[32] (uint16: a 9-bit grid index and 7 sign bits), scales[8] (a 4-bit scale per 32 values)
void iq2_xs_row(const float *x, uint8_t *out, int64_t n, const float *quant_weights) {
    const Iq2Tab &t = iq2_tab(1);
    const int kMaxQ = 3;
    const int64_t nbl = n / 256;
    for (int64_t ibl = 0; ibl < nbl; ++ibl, out += 74) {
        float scales[16];
        float weight[16], xval[16], waux[16];
        int8_t L[16], Laux[16];
        bool is_on_grid[2], is_on_grid_aux[2];
        uint8_t block_signs[2];
        uint16_t q2[2 * (256 / 16)];
        const uint16_t zero_h = qz_fp32_to_fp16(0.f);
        memcpy(out, &zero_h, 2);
        memset(q2, 0, 64);
        memset(out + 66, 0, 8);  // scales
        float max_scale = 0;
        const float *xbl = x + 256 * ibl;
        float sumx2 = 0;
        for (int i = 0; i < 256; ++i) sumx2 += xbl[i] * xbl[i];
        const float sigma2 = sumx2 / 256;
        for (int ib = 0; ib < 16; ++ib) {
            const float *xb = xbl + 16 * ib;
            const float *qw = quant_weights + 256 * ibl + 16 * ib;
            for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                int nflip = 0;
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        ++nflip;
                        s |= (uint8_t)(1 << i);
                    }
                }
                if (nflip % 2) {
                    int imin = 0;
                    float min = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
                    for (int i = 1; i < 8; ++i) {
                        const float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if (ax < min) {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= (uint8_t)(1 << imin);
                }
                block_signs[k] = s & 127;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = max_of(max, xval[i]);
            memset(L, 0, 16);
            if (max < kGroupMaxEps) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max / (2 * kMaxQ - 1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                const float id = (2 * kMaxQ - 1 + is * 0.1f) / max;
                const float this_scale = 1 / id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        const int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        Laux[8 * k + i] = (int8_t)std::max(0, std::min(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (uint16_t)(Laux[8 * k + i] << 2 * i);
                    bool exact;
                    const int grid_index = iq2_lookup(t, u, &exact);
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        iq2_find_best_neighbour(t, iq2_neighbours_base(t, u), xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    const float w = weight[i];
                    const float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k < 2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k)
                if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                const float id = 1 / scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        l = std::max(0, std::min(kMaxQ - 1, l));
                        u |= (uint16_t)(l << 2 * i);
                        L[8 * k + i] = (int8_t)l;
                    }
                    bool exact;
                    const int grid_index = iq2_lookup(t, u, &exact);
                    if (grid_index < 0) iq2_find_best_neighbour(t, iq2_neighbours_base(t, u), xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    const float w = weight[i];
                    const float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (uint16_t)(L[8 * k + i] << 2 * i);
                const int grid_index = t.map[u];
                if (grid_index < 0) throw std::runtime_error("iq2_xs: a point is not on the grid (ggml aborts here too)");
                q2[2 * ib + k] = (uint16_t)(grid_index | (block_signs[k] << 9));
            }
            scales[ib] = scale;
            max_scale = max_of(max_scale, scale);
        }
        if (!max_scale) {
            memset(out + 2, 0, 64);
            continue;
        }
        const float d = max_scale / 31;
        const uint16_t hd = qz_fp32_to_fp16(d);
        memcpy(out, &hd, 2);
        const float id = 1 / d;
        uint8_t *scl = out + 66;
        for (int ib = 0; ib < 16; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = std::max(0, std::min(15, l));
            if (ib % 2 == 0) scl[ib / 2] = (uint8_t)l;
            else scl[ib / 2] |= (uint8_t)(l << 4);
        }
        memcpy(out + 2, q2, 64);
    }
}

// IQ2_S: block of 256: fp16 d (scaled by 0.9875), qs[32] (the low eight bits of each grid index), signs[32] (eight sign bits per
// group of eight values), qh[8] (the two high bits of the indices), scales[8] (a 4-bit scale per 32 values)
void iq2_s_row(const float *x, uint8_t *out, int64_t n, const float *quant_weights) {
    const Iq2Tab &t = iq2_tab(3);
    const int kMaxQ = 3;
    const int64_t nbl = n / 256;
    for (int64_t ibl = 0; ibl < nbl; ++ibl, out += 82) {
        memset(out, 0, 82);  // (the zero block, as the reference's memset)
        float scales[16];
        float weight[16], xval[16], waux[16];
        int8_t L[16], Laux[16];
        bool is_on_grid[2], is_on_grid_aux[2];
        uint8_t block_signs[2];
        float max_scale = 0;
        const float *xbl = x + 256 * ibl;
        float sumx2 = 0;
        for (int i = 0; i < 256; ++i) sumx2 += xbl[i] * xbl[i];
        const float sigma2 = 2 * sumx2 / 256;
        for (int ib = 0; ib < 16; ++ib) {
            const float *xb = xbl + 16 * ib;
            const float *qw = quant_weights + 256 * ibl + 16 * ib;
            for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < 2; ++k) {
                uint8_t sg = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8 * k + i] >= 0) {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        sg |= (uint8_t)(1 << i);
                    }
                }
                block_signs[k] = sg;
            }
            float max = xval[0];
            for (int i = 1; i < 16; ++i) max = max_of(max, xval[i]);
            memset(L, 0, 16);
            if (max < 1e-8f) {  // (GROUP_MAX_EPS_IQ2_S)
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max / (2 * kMaxQ - 1);
            is_on_grid[0] = is_on_grid[1] = true;
            for (int is = -9; is <= 9; ++is) {
                const float id = (2 * kMaxQ - 1 + is * 0.1f) / max;
                const float this_scale = 1 / id;
                for (int k = 0; k < 2; ++k) {
                    for (int i = 0; i < 8; ++i) {
                        const int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        Laux[8 * k + i] = (int8_t)std::max(0, std::min(kMaxQ - 1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) u |= (uint16_t)(Laux[8 * k + i] << 2 * i);
                    const int grid_index = t.map[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        iq2_find_best_neighbour(t, iq2_neighbours_base(t, u), xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    const float w = weight[i];
                    const float q = 2 * Laux[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                    for (int k = 0; k < 2; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < 2; ++k)
                if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                const float id = 1 / scale;
                for (int k = 0; k < 2; ++k) {
                    if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 8; ++i) {
                        int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                        l = std::max(0, std::min(kMaxQ - 1, l));
                        u |= (uint16_t)(l << 2 * i);
                        L[8 * k + i] = (int8_t)l;
                    }
                    const int grid_index = t.map[u];
                    if (grid_index < 0) iq2_find_best_neighbour(t, iq2_neighbours_base(t, u), xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 16; ++i) {
                    const float w = weight[i];
                    const float q = 2 * L[i] + 1;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if (sumq2 > 0) scale = sumqx / sumq2;
            }
            if (scale < 0) {
                scale = -scale;
                for (int k = 0; k < 2; ++k) block_signs[k] = (uint8_t)~block_signs[k];
            }
            for (int k = 0; k < 2; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (uint16_t)(L[8 * k + i] << 2 * i);
                const int grid_index = t.map[u];
                if (grid_index < 0) throw std::runtime_error("iq2_s: a point is not on the grid (ggml aborts here too)");
                const int i8 = 2 * ib + k;
                out[2 + i8] = (uint8_t)(grid_index & 255);
                out[66 + i8 / 4] |= (uint8_t)((grid_index >> 8) << 2 * (i8 % 4));
                out[2 + 32 + i8] = block_signs[k];
            }
            scales[ib] = scale;
            max_scale = max_of(max_scale, scale);
        }
        if (!max_scale) continue;
        const float d = max_scale / 31;
        const uint16_t hd = qz_fp32_to_fp16(d * 0.9875f);
        memcpy(out, &hd, 2);
        const float id = 1 / d;
        uint8_t *scl = out + 74;
        for (int ib = 0; ib < 16; ++ib) {
            int l = nearest_int(0.5f * (id * scales[ib] - 1));
            l = std::max(0, std::min(15, l));
            if (ib % 2 == 0) scl[ib / 2] = (uint8_t)l;
            else scl[ib / 2] |= (uint8_t)(l << 4);
        }
    }
}

}  // namespace

int qz_iq_supported(int t) { return t == QT_IQ2_XXS || t == QT_IQ2_XS || t == QT_IQ2_S; }

void qz_iq_encode(int t, const float *x, void *y, int64_t n, const float *qw) {
    if (t == QT_IQ2_XXS) iq2_xxs_row(x, (uint8_t *)y, n, qw);
    else if (t == QT_IQ2_XS) iq2_xs_row(x, (uint8_t *)y, n, qw);
    else if (t == QT_IQ2_S) iq2_s_row(x, (uint8_t *)y, n, qw);
    else throw std::runtime_error("qz_iq_encode: type not written yet");
}
