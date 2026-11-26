// The mesh simplification on the device (trellis_simplify.h): the per-pass structures on the host (the vertex-to-face lists in CSR, the
// sorted unique edges as (min << 32 | max) with their boundary flags), the collapses and the compaction in the sd_gpu.hip simp_* kernels.
#include "trellis_simplify.h"

#include "sd_device.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

namespace trellis {
namespace {

// device-visible copies of the host arrays (the kernels read them through the mapping)
template <typename T>
T* upload(const std::vector<T>& h) {
    T* d = static_cast<T*>(sddev::alloc(sizeof(T) * (h.empty() ? 1 : h.size()), false));
    if (!h.empty()) std::memcpy(d, h.data(), sizeof(T) * h.size());
    return d;
}

}  // namespace

// an exclusive scan of the kept flags into map[0 .. n] (map[n] the count), on the host: the flags are in shared memory, and one
// device thread over millions of them would be a long launch
static void scan_kept(const int* kept, size_t n, int* map) {
    int s = 0;
    for (size_t i = 0; i < n; ++i) {
        map[i] = s;
        s += kept[i];
    }
    map[n] = s;
}

bool simplify_to_faces(std::vector<float>& verts, std::vector<int32_t>& faces, int target, SimplifyStats& stats, std::string& error,
                       bool manifold) {
    stats = SimplifyStats();
    const auto t0 = std::chrono::steady_clock::now();
    size_t V = verts.size() / 3, F = faces.size() / 3;
    stats.faces_before = (int)F;
    stats.faces_after = (int)F;
    if (target <= 0) {
        error = "the target face count must be positive";
        return false;
    }
    if (F <= (size_t)target) return true;
    if (!sddev::enabled()) {
        error = "no device";
        return false;
    }

    // the mesh on the device: vertices [V][3], faces [F][3]
    float* dv = upload(verts);
    int* df = upload(faces);
    float thresh = 1e-8f;
    const float lam_len = 1e-2f, lam_sk = 1e-3f;
    size_t num_face = F;
    for (;;) {
        // the structures of this pass on the host: the vertex-to-face lists (CSR, faces in order), the sorted unique edges and the
        // boundary vertices (the vertices of edges used by one face)
        std::vector<int32_t> off(V + 1, 0), list(3 * F);
        for (size_t f = 0; f < F; ++f)
            for (int k = 0; k < 3; ++k) off[df[3 * f + k] + 1]++;
        for (size_t v = 0; v < V; ++v) off[v + 1] += off[v];
        {
            std::vector<int32_t> fill_at(off.begin(), off.end() - 1);
            for (size_t f = 0; f < F; ++f)
                for (int k = 0; k < 3; ++k) list[fill_at[df[3 * f + k]]++] = (int32_t)f;
        }
        std::vector<uint64_t> edges;
        std::vector<uint8_t> vb(V, 0);
        if (manifold) {
            // closed and wound one way: each edge is run upwards (a < b) by exactly one of its two faces
            edges.reserve(3 * F / 2);
            for (size_t f = 0; f < F; ++f)
                for (int k = 0; k < 3; ++k) {
                    const int32_t a = df[3 * f + k], b = df[3 * f + (k + 1) % 3];
                    if (a < b) edges.push_back(((uint64_t)a << 32) | (uint32_t)b);
                }
            if (2 * edges.size() != 3 * F) {
                error = "the mesh is not closed and consistently wound (manifold simplification)";
                sddev::release(dv, sizeof(float) * 3 * (V ? V : 1));
                sddev::release(df, sizeof(int) * 3 * (F ? F : 1));
                return false;
            }
        } else {
            std::vector<uint64_t> keys(3 * F);
            for (size_t f = 0; f < F; ++f) {
                const int32_t a = df[3 * f], b = df[3 * f + 1], c = df[3 * f + 2];
                const int32_t pairs[3][2] = {{a, b}, {b, c}, {c, a}};
                for (int k = 0; k < 3; ++k)
                    keys[3 * f + k] = ((uint64_t)std::min(pairs[k][0], pairs[k][1]) << 32) | (uint32_t)std::max(pairs[k][0], pairs[k][1]);
            }
            std::sort(keys.begin(), keys.end());
            for (size_t i = 0; i < keys.size();) {
                size_t j = i + 1;
                while (j < keys.size() && keys[j] == keys[i]) ++j;
                edges.push_back(keys[i]);
                if (j - i == 1) {  // one face: a boundary edge
                    vb[keys[i] >> 32] = 1;
                    vb[keys[i] & 0xFFFFFFFFu] = 1;
                }
                i = j;
            }
        }
        const size_t E = edges.size();

        // the pass on the device
        int* dv2f = upload(list);
        int* doff = upload(off);
        unsigned long long* de = reinterpret_cast<unsigned long long*>(upload(edges));
        uint8_t* dvb = upload(vb);
        float* dqem = static_cast<float*>(sddev::alloc(sizeof(float) * 10 * V, false));
        sddev::simp_qem(dv, df, dv2f, doff, (int)V, dqem);
        float* dcost = static_cast<float*>(sddev::alloc(sizeof(float) * E, false));
        sddev::simp_cost(dv, df, dv2f, doff, de, dvb, dqem, (int)E, lam_len, lam_sk, dcost, manifold);
        unsigned long long* dprop = static_cast<unsigned long long*>(sddev::alloc(sizeof(uint64_t) * F, false));
        std::fill(dprop, dprop + F, std::numeric_limits<unsigned long long>::max());
        sddev::simp_propagate(de, dv2f, doff, dcost, (int)E, dprop);
        int* dvk = static_cast<int*>(sddev::alloc(sizeof(int) * V, false));
        int* dfk = static_cast<int*>(sddev::alloc(sizeof(int) * F, false));
        std::fill(dvk, dvk + V, 1);
        std::fill(dfk, dfk + F, 1);
        sddev::simp_collapse(dv, df, de, dv2f, doff, dcost, dprop, dvb, (int)E, thresh, dvk, dfk);
        int* dvmap = static_cast<int*>(sddev::alloc(sizeof(int) * (V + 1), false));
        scan_kept(dvk, V, dvmap);
        const size_t nV = (size_t)dvmap[V];
        int* dfmap = static_cast<int*>(sddev::alloc(sizeof(int) * (F + 1), false));
        scan_kept(dfk, F, dfmap);
        const size_t nF = (size_t)dfmap[F];
        float* nv = static_cast<float*>(sddev::alloc(sizeof(float) * 3 * (nV ? nV : 1), false));
        int* nf = static_cast<int*>(sddev::alloc(sizeof(int) * 3 * (nF ? nF : 1), false));
        sddev::simp_compress_v(dvmap, dv, (int)V, nv);
        sddev::simp_compress_f(dfmap, dvmap, df, (int)F, nf);

        // the pass's temporaries are released; the compacted mesh replaces the previous one
        sddev::release(dv2f, sizeof(int) * list.size());
        sddev::release(doff, sizeof(int) * off.size());
        sddev::release(de, sizeof(uint64_t) * E);
        sddev::release(dvb, sizeof(uint8_t) * (V ? V : 1));
        sddev::release(dqem, sizeof(float) * 10 * V);
        sddev::release(dcost, sizeof(float) * E);
        sddev::release(dprop, sizeof(uint64_t) * F);
        sddev::release(dvk, sizeof(int) * V);
        sddev::release(dfk, sizeof(int) * F);
        sddev::release(dvmap, sizeof(int) * (V + 1));
        sddev::release(dfmap, sizeof(int) * (F + 1));
        sddev::release(dv, sizeof(float) * 3 * (V ? V : 1));
        sddev::release(df, sizeof(int) * 3 * (F ? F : 1));
        dv = nv;
        df = nf;
        V = nV;
        F = nF;
        stats.passes++;

        // python's rule: stop at the target; the threshold grows tenfold while a pass removes less than 1% of the faces
        if (F <= (size_t)target) break;
        const size_t del = num_face - F;
        // no collapse is left at any cost (the threshold past every finite one): the target can not be reached, stop short of it
        if (del == 0 && !(thresh < 3e38f)) break;
        if ((double)del / (double)num_face < 1e-2) thresh *= 10.0f;
        num_face = F;
        if (F == 0 || V == 0) {
            error = "the simplification removed the whole mesh";
            return false;
        }
    }

    // back to the host vectors
    verts.assign(dv, dv + 3 * V);
    faces.assign(df, df + 3 * F);
    sddev::release(dv, sizeof(float) * 3 * (V ? V : 1));
    sddev::release(df, sizeof(int) * 3 * (F ? F : 1));
    stats.faces_after = (int)F;
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

}  // namespace trellis
