// A closed solid from the TRELLIS.2 mesh (trellis_solid.h): the device path (sd_gpu.hip solid_* kernels, the scans and the pieces
// on the host) and its host reference (the same steps in plain loops, for tests/solid_verify.cpp).
#include "trellis_solid.h"

#include "sd_device.h"
#include "trellis_solid_math.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace trellis {
namespace {

// the plug's place: rc = close + kPlug in the field's second term. Small enough that the term is not positive behind a wall (the
// distance to the outside goes straight through it: about close + the wall's 1.2 cells), large enough that it is positive on the
// surface iso out from the mesh, where the outside comes within about close of it
constexpr float kPlug = 0.5f;

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The grid: gx * gy * gz points (x fastest), the point (0, 0, 0) at (ox, oy, oz) in the mesh's units, cells of h; a border of pad
// cells around the mesh's box, so the fill starts all around it and no crossed cell touches the grid's edge.
struct Grid {
    int gx = 0, gy = 0, gz = 0;
    float ox = 0, oy = 0, oz = 0, h = 0;
    long n() const { return (long)gx * gy * gz; }
    long rows() const { return (long)(gy - 1) * (gz - 1); }  // the rows of cells along x
};

bool make_grid(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg, Grid& g, std::string& error) {
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    if (V == 0 || F == 0) {
        error = "no mesh";
        return false;
    }
    for (size_t i = 0; i < 3 * F; ++i)
        if (faces[i] < 0 || (size_t)faces[i] >= V) {
            error = "a face names a vertex the mesh does not have";
            return false;
        }
    if (cfg.grid < 64 || cfg.grid > 1024 || !(cfg.iso > 0.5f) || !(cfg.close >= 1.0f) || !(cfg.band >= cfg.iso + 1.8f)) {
        error = "the solid's settings: grid 64 to 1024, iso over 0.5, close 1 or more, band at least iso + 1.8";
        return false;
    }
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i < V; ++i)
        for (int k = 0; k < 3; ++k) {
            const float x = verts[3 * i + k];
            if (!std::isfinite(x)) {
                error = "a vertex is not a finite number";
                return false;
            }
            lo[k] = std::min(lo[k], x);
            hi[k] = std::max(hi[k], x);
        }
    const float ext = std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
    if (!(ext > 0.0f)) {
        error = "the mesh has no size";
        return false;
    }
    g.h = ext / (float)cfg.grid;
    const int pad = (int)std::ceil(cfg.close + 1.5f + cfg.band) + 2;
    int* dims[3] = {&g.gx, &g.gy, &g.gz};
    float* org[3] = {&g.ox, &g.oy, &g.oz};
    for (int k = 0; k < 3; ++k) {
        *dims[k] = (int)std::ceil((hi[k] - lo[k]) / g.h) + 1 + 2 * pad;
        *org[k] = lo[k] - (float)pad * g.h;
    }
    return true;
}

// an exclusive scan of n counts: off[0 .. n], off[n] the total (false when it does not fit an int)
bool exclusive_scan(const int* cnt, size_t n, std::vector<int32_t>& off) {
    off.resize(n + 1);
    long s = 0;
    for (size_t i = 0; i < n; ++i) {
        off[i] = (int32_t)s;
        s += cnt[i];
        if (s > std::numeric_limits<int32_t>::max()) return false;
    }
    off[n] = (int32_t)s;
    return true;
}

// The pieces of the surface (faces joined through their vertices: on a closed manifold the same as through their edges) and their
// signed volumes; those with less than min_piece of the largest one's |volume| are dropped (specks, small voids), the vertices
// compacted in their order.
void drop_small_pieces(std::vector<float>& v, std::vector<int32_t>& f, double min_piece, SolidStats& st) {
    const size_t V = v.size() / 3, F = f.size() / 3;
    std::vector<int32_t> parent(V);
    for (size_t i = 0; i < V; ++i) parent[i] = (int32_t)i;
    auto find = [&](int32_t x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    };
    for (size_t t = 0; t < F; ++t)
        for (int k = 1; k < 3; ++k) {
            const int32_t a = find(f[3 * t]), b = find(f[3 * t + k]);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
        }
    std::vector<int32_t> piece(V, -1);
    int np = 0;
    for (size_t i = 0; i < V; ++i) {
        const int32_t r = find((int32_t)i);
        if (piece[r] < 0) piece[r] = np++;
        piece[i] = piece[r];
    }
    std::vector<double> vol(np, 0.0);
    for (size_t t = 0; t < F; ++t) {
        const float* a = &v[3 * f[3 * t]];
        const float* b = &v[3 * f[3 * t + 1]];
        const float* c = &v[3 * f[3 * t + 2]];
        vol[piece[f[3 * t]]] += ((double)a[0] * ((double)b[1] * c[2] - (double)b[2] * c[1]) +
                                 (double)a[1] * ((double)b[2] * c[0] - (double)b[0] * c[2]) +
                                 (double)a[2] * ((double)b[0] * c[1] - (double)b[1] * c[0])) / 6.0;
    }
    double largest = 0;
    for (double x : vol) largest = std::max(largest, std::fabs(x));
    std::vector<char> keep(np);
    st.pieces = np;
    st.pieces_dropped = 0;
    for (int p = 0; p < np; ++p) {
        keep[p] = std::fabs(vol[p]) >= min_piece * largest;
        st.pieces_dropped += !keep[p];
    }
    std::vector<int32_t> map(V, -1);
    std::vector<float> nv;
    std::vector<int32_t> nf;
    nf.reserve(f.size());
    st.volume = 0;
    for (size_t t = 0; t < F; ++t) {
        if (!keep[piece[f[3 * t]]]) continue;
        for (int k = 0; k < 3; ++k) {
            const int32_t a = f[3 * t + k];
            if (map[a] < 0) {
                map[a] = (int32_t)(nv.size() / 3);
                nv.insert(nv.end(), &v[3 * a], &v[3 * a] + 3);
            }
            nf.push_back(map[a]);
        }
    }
    for (int p = 0; p < np; ++p)
        if (keep[p]) st.volume += vol[p];
    st.area = 0;
    for (size_t t = 0; t < nf.size() / 3; ++t) {
        const float* a = &nv[3 * nf[3 * t]];
        const float* b = &nv[3 * nf[3 * t + 1]];
        const float* c = &nv[3 * nf[3 * t + 2]];
        const double ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], wx = c[0] - a[0], wy = c[1] - a[1], wz = c[2] - a[2];
        const double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        st.area += 0.5 * std::sqrt(nx * nx + ny * ny + nz * nz);
    }
    v.swap(nv);
    f.swap(nf);
    st.faces = (int)(f.size() / 3);
    st.verts = (int)(v.size() / 3);
}

// ---- the host reference: the kernels' loops

void ref_edt_line(float* d, long start, long stride, int n, std::vector<float>& g, std::vector<int>& vv, std::vector<float>& zz) {
    g.resize(n);
    vv.resize(n);
    zz.resize(n + 1);
    int k = -1;
    for (int q = 0; q < n; ++q) {
        const float gq = d[start + q * stride];
        g[q] = gq;
        if (std::isinf(gq)) continue;
        if (k < 0) {
            k = 0;
            vv[0] = q;
            zz[0] = -INFINITY;
            zz[1] = INFINITY;
            continue;
        }
        float s;
        for (;;) {
            const int vk = vv[k];
            s = ((gq + (float)q * q) - (g[vk] + (float)vk * vk)) / (2.0f * (float)(q - vk));
            if (s <= zz[k]) --k;
            else break;
        }
        ++k;
        vv[k] = q;
        zz[k] = s;
        zz[k + 1] = INFINITY;
    }
    if (k < 0) return;
    k = 0;
    for (int q = 0; q < n; ++q) {
        while (zz[k + 1] < (float)q) ++k;
        const int vk = vv[k];
        d[start + q * stride] = (float)(q - vk) * (float)(q - vk) + g[vk];
    }
}

void line_of(const Grid& g, int axis, long line, long& start, long& stride, int& n) {
    if (axis == 0) {
        start = line * g.gx, stride = 1, n = g.gx;
    } else if (axis == 1) {
        const long x = line % g.gx, z = line / g.gx;
        start = x + (long)g.gx * g.gy * z, stride = g.gx, n = g.gy;
    } else {
        start = line, stride = (long)g.gx * g.gy, n = g.gz;
    }
}

long lines_of(const Grid& g, int axis) {
    return axis == 0 ? (long)g.gy * g.gz : axis == 1 ? (long)g.gx * g.gz : (long)g.gx * g.gy;
}

void ref_edt(float* d, const Grid& g) {
    std::vector<float> a, z;
    std::vector<int> v;
    for (int axis = 0; axis < 3; ++axis)
        for (long l = 0; l < lines_of(g, axis); ++l) {
            long start, stride;
            int n;
            line_of(g, axis, l, start, stride, n);
            ref_edt_line(d, start, stride, n, a, v, z);
        }
}

int ref_flood(uint8_t* st, const Grid& g) {
    int rounds = 0;
    for (;;) {
        bool changed = false;
        for (int axis = 0; axis < 3; ++axis)
            for (long l = 0; l < lines_of(g, axis); ++l) {
                long start, stride;
                int n;
                line_of(g, axis, l, start, stride, n);
                for (int dir = 0; dir < 2; ++dir) {
                    bool carry = false;
                    for (int i = 0; i < n; ++i) {
                        const int q = dir == 0 ? i : n - 1 - i;
                        uint8_t& s = st[start + q * stride];
                        if (s == 1 && carry) s = 2, changed = true;
                        carry = s == 2;
                    }
                }
            }
        ++rounds;
        if (!changed) break;
    }
    return rounds;
}

void corners_of(const float* f, long base, long gx, long gxy, float* fc) {
    fc[0] = f[base], fc[1] = f[base + 1], fc[2] = f[base + gx], fc[3] = f[base + gx + 1];
    fc[4] = f[base + gxy], fc[5] = f[base + gxy + 1], fc[6] = f[base + gxy + gx], fc[7] = f[base + gxy + gx + 1];
}

// the surface of the field: the crossed cells, their edges' vertices, the triangles (the kernels' steps in order)
bool ref_surface(const float* f, const Grid& g, std::vector<float>& ov, std::vector<int32_t>& of, std::string& error) {
    const long gx = g.gx, gxy = (long)g.gx * g.gy, rows = g.rows();
    const int cy = g.gy - 1;
    std::vector<int> cnt(rows, 0), cx;
    std::vector<int32_t> row_off;
    std::vector<int> crow;
    for (long row = 0; row < rows; ++row) {
        const long y = row % cy, z = row / cy;
        for (int x = 0; x < g.gx - 1; ++x) {
            float fc[8];
            corners_of(f, x + gx * y + gxy * z, gx, gxy, fc);
            int neg = 0;
            for (int k = 0; k < 8; ++k) neg += fc[k] < 0.0f;
            if (neg == 0 || neg == 8) continue;
            cx.push_back(x);
            crow.push_back((int)row);
            cnt[row]++;
        }
    }
    if (!exclusive_scan(cnt.data(), rows, row_off)) {
        error = "too many crossed cells";
        return false;
    }
    const size_t na = cx.size();
    std::vector<uint8_t> emask(na);
    std::vector<int> ecnt(na);
    for (size_t i = 0; i < na; ++i) {
        const long y = crow[i] % cy, z = crow[i] / cy;
        float fc[8];
        corners_of(f, cx[i] + gx * y + gxy * z, gx, gxy, fc);
        unsigned m = 0;
        for (int c = 1; c < 8; ++c)
            if ((fc[0] < 0.0f) != (fc[c] < 0.0f)) m |= 1u << (c - 1);
        emask[i] = (uint8_t)m;
        ecnt[i] = __builtin_popcount(m);
    }
    std::vector<int32_t> vbase;
    if (!exclusive_scan(ecnt.data(), na, vbase)) {
        error = "too many vertices";
        return false;
    }
    ov.assign(3 * (size_t)vbase[na], 0.0f);
    for (size_t i = 0; i < na; ++i) {
        const long y = crow[i] % cy, z = crow[i] / cy, x = cx[i];
        float fc[8];
        corners_of(f, x + gx * y + gxy * z, gx, gxy, fc);
        int w = vbase[i];
        for (int c = 1; c < 8; ++c) {
            if (!(emask[i] >> (c - 1) & 1)) continue;
            const float tt = fc[0] / (fc[0] - fc[c]);
            ov[3 * w] = g.ox + g.h * ((float)x + tt * (float)(c & 1));
            ov[3 * w + 1] = g.oy + g.h * ((float)y + tt * (float)(c >> 1 & 1));
            ov[3 * w + 2] = g.oz + g.h * ((float)z + tt * (float)(c >> 2 & 1));
            ++w;
        }
    }
    of.clear();
    bool missing = false;
    for (size_t i = 0; i < na; ++i) {
        const int y = crow[i] % cy, z = crow[i] / cy, x = cx[i];
        float fc[8];
        corners_of(f, x + gx * y + gxy * z, gx, gxy, fc);
        auto vid = [&](int lo, int hi) -> int32_t {
            const int qx = x + (lo & 1), qy = y + (lo >> 1 & 1), qz = z + (lo >> 2 & 1);
            const long row = (long)qy + (long)cy * qz;
            const auto b = cx.begin() + row_off[row], e = cx.begin() + row_off[row + 1];
            const auto it = std::lower_bound(b, e, qx);
            if (it == e || *it != qx) return -1;
            const size_t j = (size_t)(it - cx.begin());
            return vbase[j] + __builtin_popcount((unsigned)emask[j] & ((1u << ((lo ^ hi) - 1)) - 1u));
        };
        solid_cell_triangles(fc, [&](int a0, int a1, int b0, int b1, int c0, int c1) {
            const int32_t t[3] = {vid(a0, a1), vid(b0, b1), vid(c0, c1)};
            for (int k = 0; k < 3; ++k) {
                missing |= t[k] < 0;
                of.push_back(t[k]);
            }
        });
    }
    if (missing) {
        error = "a vertex of the surface is missing (a crossed cell at the grid's border)";
        return false;
    }
    return true;
}

// ---- the device path

template <typename T>
T* dev(size_t count) {
    return static_cast<T*>(sddev::alloc(sizeof(T) * (count ? count : 1), false));
}
template <typename T>
void undev(T* p, size_t count) {
    sddev::release(p, sizeof(T) * (count ? count : 1));
}

// the three passes of the distance transform, the lines in launches of at most 128 MB of scratch
void dev_edt(float* d, const Grid& g) {
    const int dims[3] = {g.gx, g.gy, g.gz};
    for (int axis = 0; axis < 3; ++axis) {
        const long lines = lines_of(g, axis);
        const long per = (long)(3 * dims[axis] + 1);
        const long L = std::max(1L, std::min(lines, (128L << 20) / (per * 4)));
        float* scr = dev<float>((size_t)(per * L));
        for (long l0 = 0; l0 < lines; l0 += L) sddev::solid_edt(d, g.gx, g.gy, g.gz, axis, l0, (int)std::min(L, lines - l0), scr);
        undev(scr, (size_t)(per * L));
    }
}

// the fill's rounds of six sweeps (each axis both ways) until nothing changes
int dev_flood(uint8_t* sv, const Grid& g) {
    int* changed = dev<int>(1);
    int rounds = 0;
    for (;;) {
        *changed = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const long lines = lines_of(g, axis);
            for (long l0 = 0; l0 < lines; l0 += 65536)
                sddev::solid_flood_sweep(sv, g.gx, g.gy, g.gz, axis, l0, (int)std::min(65536L, lines - l0), changed);
        }
        ++rounds;
        if (!*changed) break;
    }
    undev(changed, 1);
    return rounds;
}

bool run_device(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg, SolidGrids* grids,
                std::vector<float>* out_v, std::vector<int32_t>* out_f, SolidStats& st, std::string& error) {
    const auto t_all = std::chrono::steady_clock::now();
    st = SolidStats();
    Grid g;
    if (!make_grid(verts, faces, cfg, g, error)) return false;
    if (!sddev::enabled()) {
        error = "no device";
        return false;
    }
    st.dims[0] = g.gx, st.dims[1] = g.gy, st.dims[2] = g.gz;
    st.cell = g.h;
    const long n = g.n();
    const size_t V = verts.size() / 3, F = faces.size() / 3;

    // 1. the distance to the mesh near it (the vertices in grid units)
    auto t0 = std::chrono::steady_clock::now();
    float* dv = dev<float>(3 * V);
    const float inv = 1.0f / g.h, org[3] = {g.ox, g.oy, g.oz};
    for (size_t i = 0; i < V; ++i)
        for (int k = 0; k < 3; ++k) dv[3 * i + k] = (verts[3 * i + k] - org[k]) * inv;
    int* df = dev<int>(3 * F);
    std::memcpy(df, faces.data(), sizeof(int) * 3 * F);
    unsigned int* udf = dev<unsigned int>((size_t)n);
    sddev::solid_fill_u32(udf, n, 0x7f800000u);
    for (size_t f0 = 0; f0 < F; f0 += 65536)
        sddev::solid_udf(dv, df, (int)f0, (int)std::min<size_t>(65536, F - f0), g.gx, g.gy, g.gz, cfg.band, udf);
    undev(dv, 3 * V);
    undev(df, 3 * F);
    st.seconds_distance = seconds_since(t0);

    // 2. the distance to the wall
    t0 = std::chrono::steady_clock::now();
    float* dd = dev<float>((size_t)n);
    sddev::solid_sources(udf, nullptr, n, cfg.iso, 0, dd);
    dev_edt(dd, g);
    st.seconds_transform = seconds_since(t0);

    // 3. the fill from the border, then the distance to the outside
    t0 = std::chrono::steady_clock::now();
    uint8_t* sv = dev<uint8_t>((size_t)n);
    sddev::solid_flood_init(dd, g.gx, g.gy, g.gz, cfg.close * cfg.close, sv);
    st.flood_rounds = dev_flood(sv, g);
    for (long i = 0; i < n; ++i) st.outside += sv[i] == 2;
    if (grids) {
        grids->udf.resize(n);
        std::memcpy(grids->udf.data(), udf, sizeof(float) * n);
        grids->d1.assign(dd, dd + n);
        grids->state.assign(sv, sv + n);
    }
    sddev::solid_sources(nullptr, sv, n, 0.0f, 1, dd);
    dev_edt(dd, g);
    st.seconds_fill = seconds_since(t0);

    // 4. the field, in place of the distance to the mesh; then the second fill: only the positive points the outside reaches stay
    // outside (no pocket behind a wall, no void)
    t0 = std::chrono::steady_clock::now();
    sddev::solid_field(udf, dd, n, cfg.iso, cfg.close + kPlug);
    undev(dd, (size_t)n);
    sddev::solid_reach_init(udf, n, sv);
    st.flood_rounds += dev_flood(sv, g);
    sddev::solid_unreached(udf, sv, n);
    undev(sv, (size_t)n);
    const float* f = reinterpret_cast<const float*>(udf);
    if (grids) grids->field.assign(f, f + n);
    if (!out_v) {
        undev(udf, (size_t)n);
        return true;
    }

    // 5. the surface: the crossed cells by rows, their edges' vertices, the triangles
    const long rows = g.rows();
    int* cnt = dev<int>((size_t)rows);
    for (long r0 = 0; r0 < rows; r0 += 65536)
        sddev::solid_row_cells(f, g.gx, g.gy, r0, (int)std::min(65536L, rows - r0), 0, cnt, nullptr, nullptr);
    std::vector<int32_t> row_off;
    const bool fits = exclusive_scan(cnt, (size_t)rows, row_off);
    undev(cnt, (size_t)rows);
    if (!fits) {
        undev(udf, (size_t)n);
        error = "too many crossed cells";
        return false;
    }
    const size_t na = (size_t)row_off[rows];
    int* doff = dev<int>(row_off.size());
    std::memcpy(doff, row_off.data(), sizeof(int) * row_off.size());
    int* cx = dev<int>(na);
    for (long r0 = 0; r0 < rows; r0 += 65536)
        sddev::solid_row_cells(f, g.gx, g.gy, r0, (int)std::min(65536L, rows - r0), 1, nullptr, doff, cx);
    int* crow = dev<int>(na);
    for (long r = 0; r < rows; ++r)
        for (int i = row_off[r]; i < row_off[r + 1]; ++i) crow[i] = (int)r;
    uint8_t* em = dev<uint8_t>(na);
    int* ecnt = dev<int>(na);
    std::vector<int32_t> vb, tb;
    bool ok = na > 0;
    if (ok) {
        sddev::solid_cell_edges(f, g.gx, g.gy, cx, crow, (int)na, em, ecnt);
        ok = exclusive_scan(ecnt, na, vb);
    }
    if (!ok) error = na ? "too many vertices" : "the solid has no surface";
    int* dvb = nullptr;
    int* dtb = nullptr;
    float* ov = nullptr;
    int* of = nullptr;
    size_t nv = 0, nt = 0;
    if (ok) {
        nv = (size_t)vb[na];
        dvb = dev<int>(na + 1);
        std::memcpy(dvb, vb.data(), sizeof(int) * (na + 1));
        ov = dev<float>(3 * nv);
        sddev::solid_edge_verts(f, g.gx, g.gy, cx, crow, em, dvb, (int)na, g.ox, g.oy, g.oz, g.h, ov);
        sddev::solid_cell_tris(f, g.gx, g.gy, cx, crow, doff, em, dvb, nullptr, (int)na, 0, ecnt, nullptr);
        ok = exclusive_scan(ecnt, na, tb);
        if (!ok) error = "too many faces";
    }
    if (ok) {
        nt = (size_t)tb[na];
        dtb = dev<int>(na + 1);
        std::memcpy(dtb, tb.data(), sizeof(int) * (na + 1));
        of = dev<int>(3 * nt);
        sddev::solid_cell_tris(f, g.gx, g.gy, cx, crow, doff, em, dvb, dtb, (int)na, 1, nullptr, of);
        out_v->assign(ov, ov + 3 * nv);
        out_f->assign(of, of + 3 * nt);
        for (size_t i = 0; i < 3 * nt && ok; ++i)
            if ((*out_f)[i] < 0) {
                ok = false;
                error = "a vertex of the surface is missing (a crossed cell at the grid's border)";
            }
    }
    undev(udf, (size_t)n);
    undev(doff, row_off.size());
    undev(cx, na);
    undev(crow, na);
    undev(em, na);
    undev(ecnt, na);
    if (dvb) undev(dvb, na + 1);
    if (dtb) undev(dtb, na + 1);
    if (ov) undev(ov, 3 * nv);
    if (of) undev(of, 3 * nt);
    if (!ok) return false;
    drop_small_pieces(*out_v, *out_f, cfg.min_piece, st);
    st.seconds_surface = seconds_since(t0);
    st.seconds = seconds_since(t_all);
    return true;
}

bool reference_once(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg, std::vector<float>& out_verts,
                    std::vector<int32_t>& out_faces, SolidStats& st, std::string& error, SolidGrids* grids);

// the runs with close doubled while the result is a thin shell (trellis_solid.h): once(cfg, verts, faces, stats, error) is one run
template <typename Once>
bool escalate(const SolidConfig& cfg, Once once, std::vector<float>& out_v, std::vector<int32_t>& out_f, SolidStats& st, std::string& error) {
    SolidConfig c = cfg;
    std::vector<float> first_v;
    std::vector<int32_t> first_f;
    SolidStats first_st;
    bool have_first = false;
    double seconds = 0;
    int attempts = 0;
    for (;;) {
        std::vector<float> v;
        std::vector<int32_t> f;
        SolidStats s;
        if (!once(c, v, f, s, error)) return false;
        ++attempts;
        seconds += s.seconds;
        s.close = c.close;
        s.thickness = s.area > 0 && s.cell > 0 ? 2.0 * std::fabs(s.volume) / s.area / s.cell : 0.0;
        const bool thick = s.thickness >= cfg.min_thickness;
        if (thick || c.close * 2.0f > cfg.close_max) {
            if (!thick && have_first) {  // thin at every close: a thin object, kept with the least closing
                out_v.swap(first_v);
                out_f.swap(first_f);
                st = first_st;
            } else {
                out_v.swap(v);
                out_f.swap(f);
                st = s;
            }
            break;
        }
        if (!have_first) {
            first_v = std::move(v);
            first_f = std::move(f);
            first_st = s;
            have_first = true;
        }
        c.close *= 2.0f;
    }
    st.attempts = attempts;
    st.seconds = seconds;
    return true;
}

}  // namespace

bool solidify(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg, std::vector<float>& out_verts,
              std::vector<int32_t>& out_faces, SolidStats& stats, std::string& error) {
    return escalate(
        cfg,
        [&](const SolidConfig& c, std::vector<float>& v, std::vector<int32_t>& f, SolidStats& s, std::string& e) {
            return run_device(verts, faces, c, nullptr, &v, &f, s, e);
        },
        out_verts, out_faces, stats, error);
}

bool solidify_grids(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg, SolidGrids& grids,
                    std::string& error) {
    SolidStats st;
    return run_device(verts, faces, cfg, &grids, nullptr, nullptr, st, error);
}

bool solidify_reference(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg,
                        std::vector<float>& out_verts, std::vector<int32_t>& out_faces, SolidStats& st, std::string& error,
                        SolidGrids* grids) {
    if (grids) return reference_once(verts, faces, cfg, out_verts, out_faces, st, error, grids);  // one run, its grids
    return escalate(
        cfg,
        [&](const SolidConfig& c, std::vector<float>& v, std::vector<int32_t>& f, SolidStats& s, std::string& e) {
            return reference_once(verts, faces, c, v, f, s, e, nullptr);
        },
        out_verts, out_faces, st, error);
}

namespace {

bool reference_once(const std::vector<float>& verts, const std::vector<int32_t>& faces, const SolidConfig& cfg, std::vector<float>& out_verts,
                    std::vector<int32_t>& out_faces, SolidStats& st, std::string& error, SolidGrids* grids) {
    const auto t_all = std::chrono::steady_clock::now();
    st = SolidStats();
    Grid g;
    if (!make_grid(verts, faces, cfg, g, error)) return false;
    st.dims[0] = g.gx, st.dims[1] = g.gy, st.dims[2] = g.gz;
    st.cell = g.h;
    const long n = g.n();
    const size_t V = verts.size() / 3, F = faces.size() / 3;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> gv(3 * V);
    const float inv = 1.0f / g.h, org[3] = {g.ox, g.oy, g.oz};
    for (size_t i = 0; i < V; ++i)
        for (int k = 0; k < 3; ++k) gv[3 * i + k] = (verts[3 * i + k] - org[k]) * inv;
    std::vector<float> udf((size_t)n, INFINITY);
    const int dims[3] = {g.gx, g.gy, g.gz};
    const float b2 = cfg.band * cfg.band;
    for (size_t t = 0; t < F; ++t) {
        const float* a = &gv[3 * faces[3 * t]];
        const float* b = &gv[3 * faces[3 * t + 1]];
        const float* c = &gv[3 * faces[3 * t + 2]];
        int lo[3], hi[3];
        for (int k = 0; k < 3; ++k) {
            const float mn = std::fmin(a[k], std::fmin(b[k], c[k])) - cfg.band, mx = std::fmax(a[k], std::fmax(b[k], c[k])) + cfg.band;
            lo[k] = std::max(0, (int)std::ceil(mn));
            hi[k] = std::min(dims[k] - 1, (int)std::floor(mx));
        }
        for (int z = lo[2]; z <= hi[2]; ++z)
            for (int y = lo[1]; y <= hi[1]; ++y)
                for (int x = lo[0]; x <= hi[0]; ++x) {
                    const float d2 = solid_tri_dist2((float)x, (float)y, (float)z, a, b, c);
                    if (!(d2 < b2)) continue;
                    float& u = udf[(size_t)x + (size_t)g.gx * ((size_t)y + (size_t)g.gy * z)];
                    u = std::min(u, std::sqrt(d2));
                }
    }
    st.seconds_distance = seconds_since(t0);

    t0 = std::chrono::steady_clock::now();
    std::vector<float> dd((size_t)n);
    for (long i = 0; i < n; ++i) dd[i] = udf[i] < cfg.iso ? 0.0f : INFINITY;
    ref_edt(dd.data(), g);
    st.seconds_transform = seconds_since(t0);

    t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> sv((size_t)n);
    const float r2 = cfg.close * cfg.close;
    for (long i = 0; i < n; ++i) {
        const int x = (int)(i % g.gx), y = (int)((i / g.gx) % g.gy), z = (int)(i / ((long)g.gx * g.gy));
        uint8_t s = dd[i] >= r2 ? 1 : 0;
        if (s && (x == 0 || y == 0 || z == 0 || x == g.gx - 1 || y == g.gy - 1 || z == g.gz - 1)) s = 2;
        sv[i] = s;
    }
    st.flood_rounds = ref_flood(sv.data(), g);
    for (long i = 0; i < n; ++i) st.outside += sv[i] == 2;
    if (grids) {
        grids->udf = udf;
        grids->d1 = dd;
        grids->state = sv;
    }
    for (long i = 0; i < n; ++i) dd[i] = sv[i] == 2 ? 0.0f : INFINITY;
    ref_edt(dd.data(), g);
    st.seconds_fill = seconds_since(t0);

    t0 = std::chrono::steady_clock::now();
    std::vector<float> fld((size_t)n);
    const float rc = cfg.close + kPlug;
    for (long i = 0; i < n; ++i) {
        const float t1 = std::isinf(udf[i]) ? INFINITY : udf[i] - cfg.iso;
        float fv = std::fmin(t1, rc - std::sqrt(dd[i]));
        fld[i] = fv >= 0.0f ? std::fmax(fv, 1e-3f) : std::fmin(fv, -1e-3f);
    }
    for (long i = 0; i < n; ++i)
        if (sv[i] != 2) sv[i] = fld[i] > 0.0f ? 1 : 0;
    st.flood_rounds += ref_flood(sv.data(), g);
    for (long i = 0; i < n; ++i)
        if (sv[i] == 1) fld[i] = -1e-3f;
    if (grids) grids->field = fld;
    if (!ref_surface(fld.data(), g, out_verts, out_faces, error)) return false;
    drop_small_pieces(out_verts, out_faces, cfg.min_piece, st);
    st.seconds_surface = seconds_since(t0);
    st.seconds = seconds_since(t_all);
    return true;
}

}  // namespace

void check_mesh(const std::vector<float>& verts, const std::vector<int32_t>& faces, MeshCheck& out) {
    out = MeshCheck();
    const size_t F = faces.size() / 3;
    struct DE {
        uint64_t key;
        int32_t face;
        int8_t fwd;
    };
    std::vector<DE> de(3 * F);
    for (size_t t = 0; t < F; ++t) {
        const int32_t* p = &faces[3 * t];
        for (int k = 0; k < 3; ++k) {
            const int32_t a = p[k], b = p[(k + 1) % 3];
            de[3 * t + k] = {((uint64_t)(uint32_t)std::min(a, b) << 32) | (uint32_t)std::max(a, b), (int32_t)t, (int8_t)(a < b)};
        }
        const float* a = &verts[3 * p[0]];
        const float* b = &verts[3 * p[1]];
        const float* c = &verts[3 * p[2]];
        const double ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2], wx = c[0] - a[0], wy = c[1] - a[1], wz = c[2] - a[2];
        const double nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        if (p[0] == p[1] || p[1] == p[2] || p[0] == p[2] || nx * nx + ny * ny + nz * nz == 0.0) out.degenerate++;
        out.volume += ((double)a[0] * ((double)b[1] * c[2] - (double)b[2] * c[1]) + (double)a[1] * ((double)b[2] * c[0] - (double)b[0] * c[2]) +
                       (double)a[2] * ((double)b[0] * c[1] - (double)b[1] * c[0])) / 6.0;
    }
    std::sort(de.begin(), de.end(), [](const DE& x, const DE& y) { return x.key < y.key || (x.key == y.key && x.face < y.face); });
    std::vector<int32_t> parent(F);
    for (size_t t = 0; t < F; ++t) parent[t] = (int32_t)t;
    auto find = [&](int32_t x) {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    };
    for (size_t i = 0; i < de.size();) {
        size_t j = i + 1;
        while (j < de.size() && de[j].key == de[i].key) ++j;
        out.edges++;
        if (j - i == 1) {
            out.boundary++;
        } else if (j - i == 2) {
            if (de[i].fwd == de[i + 1].fwd) out.misoriented++;
            const int32_t a = find(de[i].face), b = find(de[i + 1].face);
            if (a != b) parent[std::max(a, b)] = std::min(a, b);
        } else {
            out.nonmanifold++;
        }
        i = j;
    }
    for (size_t t = 0; t < F; ++t) out.pieces += find((int32_t)t) == (int32_t)t;
}

}  // namespace trellis
