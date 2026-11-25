// Check of the engine's textured output (src/trellis_bake.cpp) on the python pipeline's mesh and texture: the 512 shape oracle's mesh after
// CuMesh's fill (oracle_fill: the filled mesh the python path has before baking) and its texture voxels (oracle_tex: tex_out and its coords,
// the 512 grid). Checks: the xatlas unwrap (every face kept, uvs in [0, 1]); the bake's coverage; the mean base colour of the texels
// against python's voxel colours; the GLB header. Writes the GLB and its two PNGs' source (the .glb itself) for external decoding.
// Usage: bake_verify [oracle_fill] [oracle_tex] [out.glb]
#include "trellis_bake.h"

#include <cmath>
#include <cstdio>
#include <cstring>
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
    const std::string fdir = argc > 1 ? argv[1] : "/data/trellis2/oracle_fill";
    const std::string tdir = argc > 2 ? argv[2] : "/data/trellis2/oracle_tex";
    const std::string out = argc > 3 ? argv[3] : "/data/trellis2/out/textured_512_bake.glb";
    std::vector<char> vb, fb, tb, cb;
    if (!read_bytes(fdir + "/filled_vertices.f32", vb) || !read_bytes(fdir + "/filled_faces.i32", fb) ||
        !read_bytes(tdir + "/tex_out.f32", tb) || !read_bytes(tdir + "/tex_out_coords.i32", cb)) {
        std::fprintf(stderr, "cannot read the oracles in %s and %s\n", fdir.c_str(), tdir.c_str());
        return 1;
    }
    std::vector<float> verts(vb.size() / sizeof(float));
    std::memcpy(verts.data(), vb.data(), vb.size());
    std::vector<int32_t> faces(fb.size() / sizeof(int32_t));
    std::memcpy(faces.data(), fb.data(), fb.size());
    std::vector<float> tex(tb.size() / sizeof(float));
    std::memcpy(tex.data(), tb.data(), tb.size());
    std::vector<int32_t> tcoords(cb.size() / sizeof(int32_t));
    std::memcpy(tcoords.data(), cb.data(), cb.size());
    const size_t D = tcoords.size() / 4;
    bool ok = true;

    // the python texture voxels as the attribute volume (the 512 grid; six channels, already x * 0.5 + 0.5)
    trellis::VoxelAttributes attr;
    attr.coords = tcoords;
    attr.feats = tex;
    attr.channels = 6;
    attr.res = 512;

    // 1. the unwrap
    trellis::UvMesh uv;
    std::string error;
    if (!trellis::unwrap_uv(verts, faces, 1024, uv, error)) {
        std::fprintf(stderr, "unwrap: %s\n", error.c_str());
        return 1;
    }
    float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
    for (size_t i = 0; i < uv.uvs.size() / 2; ++i) {
        umin = std::fmin(umin, uv.uvs[2 * i]);
        umax = std::fmax(umax, uv.uvs[2 * i]);
        vmin = std::fmin(vmin, uv.uvs[2 * i + 1]);
        vmax = std::fmax(vmax, uv.uvs[2 * i + 1]);
    }
    const size_t F_in = faces.size() / 3, F_uv = uv.faces.size() / 3;
    const bool uv_ok = umin >= -1e-6f && umax <= 1 + 1e-6f && vmin >= -1e-6f && vmax <= 1 + 1e-6f && F_uv == F_in;
    std::printf("unwrap: %zu vertices (from %zu), %zu faces (from %zu), %d charts, atlas %dx%d, uv in [%.3f, %.3f] x [%.3f, %.3f]: %s\n",
                uv.positions.size() / 3, verts.size() / 3, F_uv, F_in, uv.charts, uv.atlas_width, uv.atlas_height, umin, umax, vmin, vmax,
                uv_ok ? "ok" : "BAD");
    ok = ok && uv_ok;

    // 2. the bake: 1024 texels, the attributes sampled trilinearly at each covered texel's position
    trellis::BakedTextures bt;
    if (!trellis::bake_textures(uv, attr, 1024, bt, error)) {
        std::fprintf(stderr, "bake: %s\n", error.c_str());
        return 1;
    }
    double mean[3] = {0, 0, 0};
    size_t n = 0;
    for (size_t i = 0; i < (size_t)bt.size * bt.size; ++i) {
        if (bt.base_rgba[i * 4 + 3] == 0 && bt.base_rgba[i * 4] == 0 && bt.base_rgba[i * 4 + 1] == 0 && bt.base_rgba[i * 4 + 2] == 0) continue;
        for (int c = 0; c < 3; ++c) mean[c] += bt.base_rgba[i * 4 + c] / 255.0;
        n++;
    }
    double pm[3] = {0, 0, 0};
    for (size_t i = 0; i < D; ++i)
        for (int c = 0; c < 3; ++c) {
            const float v = tex[i * 6 + c];
            pm[c] += v < 0 ? 0 : (v > 1 ? 1 : v);
        }
    std::printf("bake: coverage %.3f, base colour mean over the baked texels %.3f %.3f %.3f, python voxels %.3f %.3f %.3f\n", bt.coverage,
                mean[0] / n, mean[1] / n, mean[2] / n, pm[0] / D, pm[1] / D, pm[2] / D);
    bool colour_ok = bt.coverage > 0.05;
    for (int c = 0; c < 3; ++c) colour_ok = colour_ok && std::fabs(mean[c] / n - pm[c] / D) <= 0.15;
    ok = ok && colour_ok;

    // 3. the GLB
    const std::vector<float> normals = trellis::vertex_normals(verts, faces);
    if (!trellis::write_glb(out, uv, normals, bt, error)) {
        std::fprintf(stderr, "glb: %s\n", error.c_str());
        return 1;
    }
    std::vector<char> glb;
    read_bytes(out, glb);
    const bool glb_ok = glb.size() > 20 && std::memcmp(glb.data(), "glTF", 4) == 0 &&
                        *reinterpret_cast<const uint32_t*>(glb.data() + 8) == glb.size();
    std::printf("glb: %s, %zu bytes: %s\n", out.c_str(), glb.size(), glb_ok ? "ok" : "BAD");
    ok = ok && glb_ok;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
