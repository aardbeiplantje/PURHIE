// Check of the engine's mesh step (src/trellis_mesh.cpp) against o-voxel (tools/mesh_oracle.py): the shape decoder's output
// features and coordinates (sdec_out_feats.f32 [M][7], sdec_out_coords.i32 [M][4]) give the vertices and faces of
// flexible_dual_grid_to_mesh (mesh_vertices.f32 [V][3], mesh_faces.i32 [T][3]). Usage: mesh_verify [oracle_dir]
#include "trellis_mesh.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool read_bytes(const std::string& path, std::vector<char>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)bytes);
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    return got == v.size();
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_sparse";
    std::vector<char> ff, cf, vf, tf;
    if (!read_bytes(dir + "/sdec_out_feats.f32", ff) || !read_bytes(dir + "/sdec_out_coords.i32", cf) ||
        !read_bytes(dir + "/mesh_vertices.f32", vf) || !read_bytes(dir + "/mesh_faces.i32", tf)) {
        std::fprintf(stderr, "cannot read the oracle in %s\n", dir.c_str());
        return 1;
    }
    const int n = (int)(cf.size() / (4 * sizeof(int32_t)));
    if (ff.size() != (size_t)n * 7 * sizeof(float)) {
        std::fprintf(stderr, "the features do not match the coordinates (%d voxels)\n", n);
        return 1;
    }
    const size_t nv_ref = vf.size() / (3 * sizeof(float));
    const size_t nt_ref = tf.size() / (3 * sizeof(int32_t));
    const float* ref_v = reinterpret_cast<const float*>(vf.data());
    const int32_t* ref_t = reinterpret_cast<const int32_t*>(tf.data());

    std::vector<float> verts;
    std::vector<int32_t> faces;
    std::string err;
    if (!trellis::dual_grid_to_mesh(reinterpret_cast<const int32_t*>(cf.data()), reinterpret_cast<const float*>(ff.data()), n,
                                    256, -0.5f, 0.5f, 0.5f, verts, faces, err)) {
        std::fprintf(stderr, "mesh: %s\n", err.c_str());
        return 1;
    }

    const bool nv_same = verts.size() == nv_ref * 3;
    double max_err = 0;
    if (nv_same)
        for (size_t i = 0; i < verts.size(); ++i) max_err = std::fmax(max_err, std::fabs((double)verts[i] - (double)ref_v[i]));
    const bool nt_same = faces.size() == nt_ref * 3;
    size_t differ = 0;
    if (nt_same)
        for (size_t i = 0; i < faces.size(); ++i) differ += faces[i] != ref_t[i] ? 1 : 0;

    std::printf("vertices %zu (oracle %zu), max abs err %.3e\n", verts.size() / 3, nv_ref, max_err);
    std::printf("faces %zu (oracle %zu), index entries differing %zu\n", faces.size() / 3, nt_ref, differ);
    const bool ok = nv_same && max_err < 1e-5 && nt_same && differ == 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
