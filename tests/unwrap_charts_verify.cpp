// The unwrap with CuMesh's charts (src/trellis_bake.cpp unwrap_uv_by_charts) on the filled shape mesh (oracle_fill) and its chart ids
// (chart_ids.i32, CuMesh's compute_charts): the checks of bake_verify (every face kept, uvs in [0, 1], the bake's coverage and the mean base
// colour against python's voxels from oracle_tex) and the time. With --default, also the xatlas default unwrap (about 70 s) for comparison.
// Usage: unwrap_charts_verify [oracle_fill] [oracle_tex] [--default]
#include "trellis_bake.h"

#include <chrono>
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

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    std::string fdir = "/data/trellis2/oracle_fill", tdir = "/data/trellis2/oracle_tex";
    bool with_default = false;
    int positional = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--default") == 0) {
            with_default = true;
        } else if (positional == 0) {
            fdir = argv[i];
            positional++;
        } else {
            tdir = argv[i];
            positional++;
        }
    }
    std::vector<char> vb, fb, cb, tb, tcb;
    if (!read_bytes(fdir + "/filled_vertices.f32", vb) || !read_bytes(fdir + "/filled_faces.i32", fb) ||
        !read_bytes(fdir + "/chart_ids.i32", cb) || !read_bytes(tdir + "/tex_out.f32", tb) || !read_bytes(tdir + "/tex_out_coords.i32", tcb)) {
        std::fprintf(stderr, "cannot read the oracles in %s and %s\n", fdir.c_str(), tdir.c_str());
        return 1;
    }
    std::vector<float> verts(vb.size() / sizeof(float));
    std::memcpy(verts.data(), vb.data(), vb.size());
    std::vector<int32_t> faces(fb.size() / sizeof(int32_t)), charts(cb.size() / sizeof(int32_t));
    std::memcpy(faces.data(), fb.data(), fb.size());
    std::memcpy(charts.data(), cb.data(), cb.size());
    std::vector<float> tex(tb.size() / sizeof(float));
    std::memcpy(tex.data(), tb.data(), tb.size());
    std::vector<int32_t> tcoords(tcb.size() / sizeof(int32_t));
    std::memcpy(tcoords.data(), tcb.data(), tcb.size());
    const size_t F_in = faces.size() / 3, D = tcoords.size() / 4;
    std::string error;
    bool ok = true;

    // 1. the unwrap by CuMesh's charts
    trellis::UvMesh uv;
    auto t0 = std::chrono::steady_clock::now();
    if (!trellis::unwrap_uv_by_charts(verts, faces, charts, 1024, uv, error)) {
        std::fprintf(stderr, "by charts: %s\n", error.c_str());
        return 1;
    }
    const double t_charts = seconds_since(t0);
    float umin = 1e9f, umax = -1e9f, vmin = 1e9f, vmax = -1e9f;
    for (size_t i = 0; i < uv.uvs.size() / 2; ++i) {
        umin = std::fmin(umin, uv.uvs[2 * i]);
        umax = std::fmax(umax, uv.uvs[2 * i]);
        vmin = std::fmin(vmin, uv.uvs[2 * i + 1]);
        vmax = std::fmax(vmax, uv.uvs[2 * i + 1]);
    }
    const size_t F_uv = uv.faces.size() / 3;
    const bool uv_ok = umin >= -1e-6f && umax <= 1 + 1e-6f && vmin >= -1e-6f && vmax <= 1 + 1e-6f && F_uv == F_in;
    std::printf("charts unwrap (%d charts, %.2f s): %zu vertices (from %zu), %zu faces (from %zu), atlas %dx%d, uv in [%.3f, %.3f] x [%.3f, %.3f]: %s\n",
                uv.charts, t_charts, uv.positions.size() / 3, verts.size() / 3, F_uv, F_in, uv.atlas_width, uv.atlas_height, umin, umax, vmin,
                vmax, uv_ok ? "ok" : "BAD");
    ok = ok && uv_ok;

    // 2. the bake (1024 texels) and python's voxel colours
    trellis::VoxelAttributes attr;
    attr.coords = tcoords;
    attr.feats = tex;
    attr.channels = 6;
    attr.res = 512;
    trellis::BakedTextures bt;
    t0 = std::chrono::steady_clock::now();
    if (!trellis::bake_textures(uv, attr, 1024, bt, error)) {
        std::fprintf(stderr, "bake: %s\n", error.c_str());
        return 1;
    }
    const double t_bake = seconds_since(t0);
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
    std::printf("bake (%.2f s): coverage %.3f, base colour mean over the baked texels %.3f %.3f %.3f, python voxels %.3f %.3f %.3f\n", t_bake,
                bt.coverage, mean[0] / n, mean[1] / n, mean[2] / n, pm[0] / D, pm[1] / D, pm[2] / D);
    bool colour_ok = bt.coverage > 0.05 && n > 0;
    for (int c = 0; c < 3 && n > 0; ++c) colour_ok = colour_ok && std::fabs(mean[c] / n - pm[c] / D) <= 0.15;
    std::printf("colours: %s\n", colour_ok ? "ok" : "BAD");
    ok = ok && colour_ok;

    // 3. optionally the xatlas default, for the time and the chart count
    if (with_default) {
        trellis::UvMesh def;
        t0 = std::chrono::steady_clock::now();
        if (!trellis::unwrap_uv(verts, faces, 1024, def, error)) {
            std::fprintf(stderr, "default: %s\n", error.c_str());
            return 1;
        }
        std::printf("xatlas default (%d charts, %.2f s): %zu faces, atlas %dx%d\n", def.charts, seconds_since(t0), def.faces.size() / 3,
                    def.atlas_width, def.atlas_height);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
