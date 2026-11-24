// The mesh step of the TRELLIS.2 shape decoder on the device (trellis_mesh.h). The order of the faces is that of o-voxel's
// flexible_dual_grid_to_mesh: the valid quads in (voxel, axis) order, then the two triangles of each quad.
#include "trellis_mesh.h"

#include "sd_device.h"

#include <cstring>
#include <utility>

namespace trellis {

bool dual_grid_to_mesh(const int32_t* coords, const float* feats, int n, int grid, float lo, float hi, float margin,
                       std::vector<float>& verts, std::vector<int32_t>& faces, std::string& error) {
    verts.clear();
    faces.clear();
    if (!sddev::enabled()) {
        error = "no device";
        return false;
    }
    if (n <= 0 || grid <= 0) {
        error = "no voxels";
        return false;
    }
    std::vector<std::pair<void*, size_t>> bufs;
    struct Guard {
        std::vector<std::pair<void*, size_t>>& b;
        ~Guard() {
            for (auto& o : b) sddev::release(o.first, o.second);
        }
    } guard{bufs};
    auto A = [&](size_t bytes) {
        void* p = sddev::alloc(bytes, false);
        bufs.push_back({p, bytes});
        return p;
    };
    auto F = [&](size_t cnt) { return static_cast<float*>(A(sizeof(float) * cnt)); };
    auto I = [&](size_t cnt) { return static_cast<int*>(A(sizeof(int) * cnt)); };

    // the inputs in device-visible memory: a kernel must not read the caller's plain host arrays
    int* vc = I((size_t)n * 4);
    float* fe = F((size_t)n * 7);
    std::memcpy(vc, coords, sizeof(int) * (size_t)n * 4);
    std::memcpy(fe, feats, sizeof(float) * (size_t)n * 7);

    // the vertices: one per voxel, (coords + dual) * voxel + lo
    const float vox = (hi - lo) / (float)grid;
    float* vert = F((size_t)n * 3);
    sddev::mesh_vertices(vc, fe, n, vox, lo, margin, vert);

    // the voxels in a hash table (the value is the voxel's index); the 4 corners of each of the 3 edges of each voxel
    unsigned int tsize = 1;
    while (tsize < 2u * (unsigned)n) tsize <<= 1;
    unsigned long long* keys = static_cast<unsigned long long*>(A(sizeof(unsigned long long) * tsize));
    int* vals = I(tsize);
    sddev::sp_hash_build(vc, n, keys, vals, tsize);
    const int nq = n * 12;
    int* q = I((size_t)nq * 4);
    sddev::mesh_edge_queries(vc, n, q);
    int* look = I(nq);
    sddev::sp_lookup(q, nq, keys, vals, tsize, look);

    // an edge is a quad when it is intersected and all four of its voxels exist; the quads in edge order
    const int ne = n * 3;
    int* valid = I(ne);
    sddev::mesh_edge_valid(fe, n, look, valid);
    int* off = I((size_t)ne + 1);
    sddev::sp_exclusive_scan(valid, ne, off);
    const int L = off[ne];

    verts.assign(vert, vert + (size_t)n * 3);
    if (L == 0) return true;
    int* quads = I((size_t)L * 4);
    sddev::mesh_quads(valid, off, look, ne, quads);
    int* tri = I((size_t)L * 6);
    sddev::mesh_triangles(quads, L, fe, tri);
    faces.assign(tri, tri + (size_t)L * 6);
    return true;
}

}  // namespace trellis
