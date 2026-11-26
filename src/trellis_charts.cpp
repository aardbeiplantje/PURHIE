// CuMesh's chart clustering on the device (trellis_charts.h). The host holds the structure of the mesh (the sorted unique edges, the faces
// of each edge, the manifold face pairs), sorts the chart pairs and lists the faces of each chart, and runs the loops of compute_charts;
// the arithmetic is in the sd_gpu.hip chart_* kernels (the cones, the costs, the cheapest edge of each chart, the collapse, the refinement).
#include "trellis_charts.h"

#include "sd_device.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

namespace trellis {
namespace {

using ull = unsigned long long;

// the device buffers of one scope (sddev::alloc memory, mapped for the GPU and the host), released at its end
struct Scope {
    std::vector<std::pair<void*, size_t>> bufs;
    template <typename T>
    T* make(size_t n) {
        const size_t bytes = sizeof(T) * (n ? n : 1);
        void* p = sddev::alloc(bytes, false);
        bufs.push_back({p, bytes});
        return static_cast<T*>(p);
    }
    template <typename T>
    T* upload(const std::vector<T>& h) {
        T* d = make<T>(h.size());
        if (!h.empty()) std::memcpy(d, h.data(), sizeof(T) * h.size());
        return d;
    }
    ~Scope() {
        for (auto& b : bufs) sddev::release(b.first, b.second);
    }
};

// The structure of the mesh: the sorted unique edges (min << 32 | max), the faces of each edge (ascending), the three edges of each face,
// the manifold face pairs (the two faces of each edge used by two faces, in edge order) and their shared edge lengths.
struct Structure {
    std::vector<ull> ek;
    std::vector<int> e2f_off, e2f, f2e;
    std::vector<int> pairs;  // [M][2]
    std::vector<float> plen;  // [M]
};

float edge_length(const std::vector<float>& v, int a, int b) {
    const float dx = v[3 * a] - v[3 * b], dy = v[3 * a + 1] - v[3 * b + 1], dz = v[3 * a + 2] - v[3 * b + 2];
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

Structure build_structure(const std::vector<float>& verts, const std::vector<int32_t>& faces) {
    Structure S;
    const size_t F = faces.size() / 3;
    struct Inc {
        ull key;
        int f, k;
    };
    std::vector<Inc> inc;
    inc.reserve(3 * F);
    for (size_t f = 0; f < F; ++f)
        for (int k = 0; k < 3; ++k) {
            const int a = faces[3 * f + k], b = faces[3 * f + (k + 1) % 3];
            inc.push_back({((ull)std::min(a, b) << 32) | (unsigned)std::max(a, b), (int)f, k});
        }
    std::stable_sort(inc.begin(), inc.end(), [](const Inc& x, const Inc& y) { return x.key < y.key; });
    S.f2e.assign(3 * F, -1);
    for (size_t i = 0; i < inc.size();) {
        const int e = (int)S.ek.size();
        S.ek.push_back(inc[i].key);
        S.e2f_off.push_back((int)S.e2f.size());
        size_t j = i;
        for (; j < inc.size() && inc[j].key == inc[i].key; ++j) {
            S.e2f.push_back(inc[j].f);
            S.f2e[3 * inc[j].f + inc[j].k] = e;
        }
        i = j;
    }
    S.e2f_off.push_back((int)S.e2f.size());
    for (size_t e = 0; e + 1 < S.e2f_off.size(); ++e) {
        if (S.e2f_off[e + 1] - S.e2f_off[e] != 2) continue;  // a manifold edge: two faces
        const int a = (int)(S.ek[e] >> 32), b = (int)(S.ek[e] & 0xFFFFFFFFu);
        S.pairs.push_back(S.e2f[S.e2f_off[e]]);
        S.pairs.push_back(S.e2f[S.e2f_off[e] + 1]);
        S.plen.push_back(edge_length(verts, a, b));
    }
    return S;
}

// the faces of each chart (ascending faces): the offsets [C+1] and the list [F]
void chart_faces(const int* ids, size_t F, int C, std::vector<int>& off, std::vector<int>& list) {
    off.assign(C + 1, 0);
    for (size_t f = 0; f < F; ++f) off[ids[f] + 1]++;
    for (int c = 0; c < C; ++c) off[c + 1] += off[c];
    list.resize(F);
    std::vector<int> cur(off.begin(), off.end() - 1);
    for (size_t f = 0; f < F; ++f) list[cur[ids[f]]++] = (int)f;
}

// the rank of each value in 0..range-1 among the values present (CuMesh's compress_ids): rank[v] for the present v, -1 otherwise
int rank_values(const int* vals, size_t n, int range, std::vector<int>& rank) {
    std::vector<char> present(range, 0);
    for (size_t i = 0; i < n; ++i) present[vals[i]] = 1;
    rank.assign(range, -1);
    int count = 0;
    for (int v = 0; v < range; ++v)
        if (present[v]) rank[v] = count++;
    return count;
}

}  // namespace

bool compute_charts(const std::vector<float>& verts, const std::vector<int32_t>& faces, const ChartConfig& cfg, std::vector<int32_t>& chart_ids,
                    ChartStats& stats, std::string& error) {
    const auto t0 = std::chrono::steady_clock::now();
    stats = ChartStats();
    chart_ids.clear();
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    if (F == 0 || V == 0 || verts.size() % 3 != 0 || faces.size() % 3 != 0) {
        error = "charts: no mesh";
        return false;
    }
    for (int32_t v : faces)
        if (v < 0 || (size_t)v >= V) {
            error = "charts: a face index is out of range";
            return false;
        }
    if (!sddev::enabled()) {
        error = "charts: no device";
        return false;
    }

    const Structure S = build_structure(verts, faces);
    const size_t M = S.plen.size();
    Scope st;
    float* dv = st.upload(verts);
    int* df = st.upload(faces);
    float* dfn = st.make<float>(3 * F);
    float* dfa = st.make<float>(F);
    sddev::chart_face_geom(dv, df, (int)F, dfn, dfa);
    ull* dek = st.upload(S.ek);
    int* df2e = st.upload(S.f2e);
    int* de2f = st.upload(S.e2f);
    int* de2f_off = st.upload(S.e2f_off);
    int* dids = st.make<int>(F);
    int* dout = st.make<int>(F);
    for (size_t f = 0; f < F; ++f) dids[f] = (int)f;  // one chart per face
    int C = (int)F;
    std::vector<int> off, list, rank;

    for (int g = 0; g < cfg.global_iterations; ++g) {
        // collapse the chart graph until no edge collapses
        for (;;) {
            Scope pass;
            // the chart pairs of the manifold faces (CuMesh's init_chart_adj and the sort and reduce): the pairs of two charts with their lengths
            std::vector<std::pair<ull, float>> pr;
            pr.reserve(M);
            for (size_t m = 0; m < M; ++m) {
                const int c0 = dids[S.pairs[2 * m]], c1 = dids[S.pairs[2 * m + 1]];
                if (c0 == c1) continue;
                pr.push_back({((ull)std::min(c0, c1) << 32) | (unsigned)std::max(c0, c1), S.plen[m]});
            }
            std::stable_sort(pr.begin(), pr.end(), [](const std::pair<ull, float>& a, const std::pair<ull, float>& b) { return a.first < b.first; });
            std::vector<ull> ck;
            std::vector<float> cl;
            for (size_t i = 0; i < pr.size();) {
                float sum = 0.0f;
                size_t j = i;
                for (; j < pr.size() && pr[j].first == pr[i].first; ++j) sum += pr[j].second;
                ck.push_back(pr[i].first);
                cl.push_back(sum);
                i = j;
            }
            const size_t E = ck.size();
            if (E == 0) break;

            // the chart graph: the edges of each chart (CuMesh's CSR, in edge order)
            std::vector<int> c2o(C + 1, 0), c2e(2 * E);
            for (size_t e = 0; e < E; ++e) {
                c2o[(ck[e] >> 32) + 1]++;
                c2o[(ck[e] & 0xFFFFFFFFu) + 1]++;
            }
            for (int c = 0; c < C; ++c) c2o[c + 1] += c2o[c];
            {
                std::vector<int> cur(c2o.begin(), c2o.end() - 1);
                for (size_t e = 0; e < E; ++e) {
                    c2e[cur[ck[e] >> 32]++] = (int)e;
                    c2e[cur[ck[e] & 0xFFFFFFFFu]++] = (int)e;
                }
            }
            chart_faces(dids, F, C, off, list);

            // the device: the cones, the perimeters, the costs, the cheapest edge of each chart, the collapse
            float* dcn = pass.make<float>(4 * C);
            float* dca = pass.make<float>(C);
            int* dcoff = pass.upload(off);
            int* dcface = pass.upload(list);
            sddev::chart_cones(dcoff, dcface, dfn, dfa, C, dcn, dca);
            ull* dck = pass.upload(ck);
            float* dcl = pass.upload(cl);
            int* dc2o = pass.upload(c2o);
            int* dc2e = pass.upload(c2e);
            float* dper = pass.make<float>(C);
            sddev::chart_perims(dc2o, dc2e, dcl, C, dper);
            float* dcost = pass.make<float>(E);
            sddev::chart_costs(dck, dcl, dcn, dca, dper, cfg.area_penalty_weight, cfg.perimeter_area_ratio_weight, (int)E, dcost);
            ull* dprop = pass.make<ull>(C);
            sddev::chart_propagate(dc2o, dc2e, dcost, C, dprop);
            int* dcmap = pass.make<int>(C);
            for (int c = 0; c < C; ++c) dcmap[c] = c;
            int* dflag = pass.make<int>(1);
            *dflag = 1;
            sddev::chart_collapse(dck, dcost, dprop, cfg.threshold_cone_half_angle_rad, (int)E, dcmap, dcn, dflag);
            stats.collapse_passes++;
            if (*dflag == 1) break;  // no edge collapsed: the end of the iteration

            // the collapsed charts take their targets; the charts are the ranks of the targets (CuMesh's compress and index)
            const int newC = rank_values(dcmap, C, C, rank);
            for (size_t f = 0; f < F; ++f) dids[f] = rank[dcmap[dids[f]]];
            C = newC;
        }

        // refine the faces onto the charts of their neighbours
        for (int j = 0; j < cfg.refine_iterations; ++j) {
            Scope rs;
            chart_faces(dids, F, C, off, list);
            float* dcn = rs.make<float>(4 * C);
            float* dca = rs.make<float>(C);
            int* dcoff = rs.upload(off);
            int* dcface = rs.upload(list);
            sddev::chart_cones(dcoff, dcface, dfn, dfa, C, dcn, dca);
            sddev::chart_refine(dcn, dfn, dv, dek, df2e, de2f, de2f_off, (int)F, cfg.smooth_strength, dids, dout);
            const int newC = rank_values(dout, F, C, rank);
            for (size_t f = 0; f < F; ++f) dids[f] = rank[dout[f]];
            C = newC;
        }

        // a chart that became disconnected splits: the faces joined across manifold edges within a chart form the new charts, labelled by
        // their smallest face (CuMesh's hook and compress, which converge to the same labels)
        std::vector<int> par(F);
        std::iota(par.begin(), par.end(), 0);
        auto find = [&par](int x) {
            while (par[x] != x) {
                par[x] = par[par[x]];
                x = par[x];
            }
            return x;
        };
        for (size_t m = 0; m < M; ++m) {
            const int a = S.pairs[2 * m], b = S.pairs[2 * m + 1];
            if (dids[a] != dids[b]) continue;
            const int ra = find(a), rb = find(b);
            if (ra != rb) par[std::max(ra, rb)] = std::min(ra, rb);
        }
        std::vector<int> label(F);
        for (size_t f = 0; f < F; ++f) label[f] = find((int)f);
        const int newC = rank_values(label.data(), F, (int)F, rank);
        for (size_t f = 0; f < F; ++f) dids[f] = rank[label[f]];
        C = newC;
    }

    chart_ids.assign(dids, dids + F);
    stats.charts = C;
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace trellis
