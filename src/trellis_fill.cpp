// The hole filling of the TRELLIS.2 mesh (trellis_fill.h), following CuMesh's fill_holes and its loop rule (connectivity.cu:
// get_manifold_boundary_adjacency, get_boundary_connected_components, is_bound_conn_comp_loop_kernel). The rule, as the code does it:
//  - a boundary edge is an edge of one face; the boundary vertex list (vert2bound) is the CSR of each vertex's boundary edges;
//  - a vertex is manifold when it has at most two boundary edges and no edge of more than two faces; the manifold boundary vertices
//    each pair the first two entries of the CSR array at their offset (for a vertex with one boundary edge, the second entry belongs
//    to the next vertex: CuMesh's pairing, kept as it is);
//  - the boundary edges are joined by those pairs (union-find, the components); a component is a loop when, for each of its edges,
//    both endpoints have another edge of the same component;
//  - the loops with a perimeter (the sum of their edges' lengths) below the limit are closed with a fan: a new vertex at the mean of
//    the edges' midpoints, and a triangle per edge.
// The faces of the fan keep the edge's direction from its face, reversed, so the surface stays consistently oriented.
#include "trellis_fill.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace trellis {
namespace {

struct Dir {
    uint32_t lo, hi;  // the undirected edge (lo < hi)
    int32_t u, v;     // the directed edge as its face has it
};

int32_t find_root(std::vector<int32_t>& parent, int32_t x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

}  // namespace

void fill_small_holes(std::vector<float>& verts, std::vector<int32_t>& faces, float max_perimeter, FillStats& stats) {
    stats = FillStats();
    const size_t V = verts.size() / 3, F = faces.size() / 3;
    if (V == 0 || F == 0) return;

    // 1. the directed edges grouped by their undirected edge: a group of one is a boundary edge; a group of more than two marks its vertices
    std::vector<Dir> edges;
    edges.reserve(3 * F);
    for (size_t f = 0; f < F; ++f) {
        const int32_t a = faces[3 * f], b = faces[3 * f + 1], c = faces[3 * f + 2];
        const int32_t tri[3][2] = {{a, b}, {b, c}, {c, a}};
        for (const auto& e : tri) {
            const uint32_t lo = (uint32_t)std::min(e[0], e[1]), hi = (uint32_t)std::max(e[0], e[1]);
            edges.push_back({lo, hi, e[0], e[1]});
        }
    }
    std::sort(edges.begin(), edges.end(), [](const Dir& x, const Dir& y) { return x.lo != y.lo ? x.lo < y.lo : x.hi < y.hi; });
    std::vector<Dir> bnd;  // the boundary edges, in (lo, hi) order
    std::vector<char> nonmanifold_vertex(V, 0);
    for (size_t i = 0; i < edges.size();) {
        size_t j = i + 1;
        while (j < edges.size() && edges[j].lo == edges[i].lo && edges[j].hi == edges[i].hi) ++j;
        if (j - i == 1) bnd.push_back(edges[i]);
        if (j - i > 2) {
            nonmanifold_vertex[edges[i].lo] = 1;
            nonmanifold_vertex[edges[i].hi] = 1;
        }
        i = j;
    }
    const size_t nb = bnd.size();
    stats.boundary_edges = (int)nb;
    if (nb == 0) return;

    // 2. the vert2bound CSR: each vertex's boundary edges, in increasing boundary index
    std::vector<int32_t> deg(V, 0);
    for (const Dir& d : bnd) {
        deg[d.lo]++;
        deg[d.hi]++;
    }
    std::vector<int64_t> offs(V + 1, 0);
    for (size_t v = 0; v < V; ++v) offs[v + 1] = offs[v] + deg[v];
    std::vector<int32_t> csr((size_t)offs[V]);
    std::vector<int64_t> fill_at(offs.begin(), offs.end() - 1);
    for (size_t k = 0; k < nb; ++k) {
        csr[fill_at[bnd[k].lo]++] = (int32_t)k;
        csr[fill_at[bnd[k].hi]++] = (int32_t)k;
    }

    // 3. the components: each manifold boundary vertex joins the two entries at its offset
    std::vector<int32_t> parent(nb);
    for (size_t k = 0; k < nb; ++k) parent[k] = (int32_t)k;
    for (size_t v = 0; v < V; ++v) {
        if (deg[v] == 0) continue;
        if (deg[v] > 2 || nonmanifold_vertex[v]) continue;
        const int64_t o = offs[v];
        const int32_t b0 = csr[o];
        const int32_t b1 = (o + 1 < (int64_t)csr.size()) ? csr[o + 1] : b0;
        const int32_t ra = find_root(parent, b0), rb = find_root(parent, b1);
        if (ra != rb) parent[ra] = rb;
    }
    std::vector<int32_t> comp(nb);
    for (size_t k = 0; k < nb; ++k) comp[k] = find_root(parent, (int32_t)k);

    // 4. the loops: a component whose every edge has another edge of its component at both endpoints
    std::vector<std::vector<int32_t>> members(nb);
    for (size_t k = 0; k < nb; ++k) members[comp[k]].push_back((int32_t)k);
    auto other_in_comp = [&](uint32_t vert, int32_t k) {
        for (int64_t i = offs[vert]; i < offs[vert + 1]; ++i) {
            const int32_t b = csr[i];
            if (b != k && comp[b] == comp[k]) return true;
        }
        return false;
    };
    for (size_t r = 0; r < nb; ++r) {
        const std::vector<int32_t>& ks = members[r];
        if (ks.empty()) continue;
        bool loop = true;
        for (int32_t k : ks) {
            if (!other_in_comp(bnd[k].lo, k) || !other_in_comp(bnd[k].hi, k)) {
                loop = false;
                break;
            }
        }
        if (!loop) continue;
        stats.loops++;

        // the perimeter (the sum of the edges' lengths) and the centre (the mean of their midpoints)
        double perimeter = 0, cx = 0, cy = 0, cz = 0;
        for (int32_t k : ks) {
            const uint32_t a = bnd[k].lo, b = bnd[k].hi;
            const double dx = (double)verts[3 * b] - verts[3 * a], dy = (double)verts[3 * b + 1] - verts[3 * a + 1],
                         dz = (double)verts[3 * b + 2] - verts[3 * a + 2];
            perimeter += std::sqrt(dx * dx + dy * dy + dz * dz);
            cx += 0.5 * ((double)verts[3 * a] + verts[3 * b]);
            cy += 0.5 * ((double)verts[3 * a + 1] + verts[3 * b + 1]);
            cz += 0.5 * ((double)verts[3 * a + 2] + verts[3 * b + 2]);
        }
        if (!(perimeter < (double)max_perimeter)) {
            stats.too_long++;
            continue;
        }
        const double n = (double)ks.size();
        const int32_t c = (int32_t)(verts.size() / 3);
        verts.push_back((float)(cx / n));
        verts.push_back((float)(cy / n));
        verts.push_back((float)(cz / n));
        for (int32_t k : ks) {
            faces.push_back(bnd[k].v);
            faces.push_back(bnd[k].u);
            faces.push_back(c);
        }
        stats.filled++;
    }
}

}  // namespace trellis
