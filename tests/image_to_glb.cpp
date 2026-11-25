// The engine's image-to-3D run on a real image: the PNG decoded, the image conditioning (the DINOv3 encoder from ComfyUI's local weights),
// the textured pipeline (512, or the 1024 cascade), and the textured GLB. The noise is a seeded host Gaussian (not the python generator's).
// Usage: image_to_glb image.png out.glb [512|1024] [seed]
#include "dinov3.h"
#include "image.h"
#include "trellis_bake.h"
#include "trellis_charts.h"
#include "trellis_image.h"
#include "trellis_shape.h"
#include "trellis_simplify.h"
#include "trellis_tex.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
    if (argc < 3) {
        std::fprintf(stderr, "usage: image_to_glb image.png out.glb [512|1024] [seed]\n");
        return 2;
    }
    const std::string image = argv[1], out = argv[2];
    const int res = argc > 3 ? std::atoi(argv[3]) : 512;
    const unsigned seed = argc > 4 ? (unsigned)std::atoi(argv[4]) : 1u;
    const bool cascade = res == 1024;
    const std::string ckpt = "/data/trellis2/TRELLIS.2-4B/ckpts/";
    std::vector<char> png, shape_norm_b, tex_norm_b;
    if (!read_bytes(image, png) || !read_bytes("/data/trellis2/oracle_shape/shape_norm.f32", shape_norm_b) ||
        !read_bytes("/data/trellis2/oracle_tex/tex_norm.f32", tex_norm_b)) {
        std::fprintf(stderr, "cannot read %s or the normalisation files\n", image.c_str());
        return 1;
    }
    const float* snorm = reinterpret_cast<const float*>(shape_norm_b.data());
    const float* tnorm = reinterpret_cast<const float*>(tex_norm_b.data());

    unsigned char* rgb = nullptr;
    int w = 0, h = 0;
    if (image_decode_rgb(reinterpret_cast<const unsigned char*>(png.data()), png.size(), &rgb, &w, &h) != 0) {
        std::fprintf(stderr, "cannot decode %s\n", image.c_str());
        return 1;
    }
    auto t0 = std::chrono::steady_clock::now();
    trellis::DinoV3 enc("/comfyui/clip_vision/dino_v3_L_naf_fp32.safetensors");
    if (!enc.ok()) {
        std::fprintf(stderr, "image encoder: %s\n", enc.error().c_str());
        return 1;
    }
    std::string error;
    std::vector<float> cond_lr, cond_hr;
    if (!trellis::image_condition(enc, rgb, w, h, 512, cond_lr, error)) {
        std::fprintf(stderr, "condition 512: %s\n", error.c_str());
        return 1;
    }
    if (cascade && !trellis::image_condition(enc, rgb, w, h, 1024, cond_hr, error)) {
        std::fprintf(stderr, "condition 1024: %s\n", error.c_str());
        return 1;
    }
    image_free(rgb);
    const double seconds_cond = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("image %dx%d: conditioning %.2f s (%zu tokens)\n", w, h, seconds_cond, cond_lr.size() / 1024);

    trellis::ShapePipeline shape(ckpt + "ss_flow_img_dit_1_3B_64_bf16.safetensors",
                                 "/data/trellis2/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                                 ckpt + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", ckpt + "shape_dec_next_dc_f16c32_fp16.safetensors",
                                 cascade ? ckpt + "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors" : std::string());
    trellis::TexturePipeline tp(ckpt + "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", ckpt + "tex_dec_next_dc_f16c32_fp16.safetensors",
                                cascade ? ckpt + "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors" : std::string());
    if (!shape.ok() || !tp.ok()) {
        std::fprintf(stderr, "open: %s%s\n", shape.error().c_str(), tp.error().c_str());
        return 1;
    }
    trellis::ShapeConfig scfg;
    scfg.ss.steps = 12;
    scfg.ss.guidance = 7.5f;
    scfg.ss.guidance_rescale = 0.7f;
    scfg.ss.rescale_t = 5.0;
    scfg.slat.steps = 12;
    scfg.slat.guidance = 7.5f;
    scfg.slat.guidance_rescale = 0.5f;
    scfg.slat.rescale_t = 3.0;
    for (int c = 0; c < 32; ++c) {
        scfg.slat_mean[c] = snorm[c];
        scfg.slat_std[c] = snorm[32 + c];
    }
    scfg.mesh_grid = res;
    trellis::TexConfig tcfg;
    tcfg.sampler.steps = 12;
    tcfg.sampler.guidance = 1.0f;
    tcfg.sampler.guidance_rescale = 0.0f;
    tcfg.sampler.rescale_t = 3.0;
    tcfg.sampler.interval_lo = 0.6;
    tcfg.sampler.interval_hi = 0.9;
    for (int c = 0; c < 32; ++c) {
        tcfg.shape_mean[c] = snorm[c];
        tcfg.shape_std[c] = snorm[32 + c];
        tcfg.tex_mean[c] = tnorm[c];
        tcfg.tex_std[c] = tnorm[32 + c];
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    trellis::NoiseFn noise = [&](float* dst, size_t count) {
        for (size_t i = 0; i < count; ++i) dst[i] = gauss(rng);
    };
    trellis::TexturedMesh tm;
    bool run_ok;
    if (cascade)
        run_ok = trellis::run_textured_cascade(shape, tp, cond_lr.data(), (int)(cond_lr.size() / 1024), cond_hr.data(),
                                               (int)(cond_hr.size() / 1024), noise, scfg, tcfg, tm, error);
    else
        run_ok = trellis::run_textured(shape, tp, cond_lr.data(), (int)(cond_lr.size() / 1024), noise, scfg, tcfg, tm, error);
    if (!run_ok) {
        std::fprintf(stderr, "run: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "run done: %zu vertices, %zu faces, %zu texture voxels, %d holes filled\n", tm.shape.verts.size() / 3,
                 tm.shape.faces.size() / 3, tm.tex.coords.size() / 4, tm.shape.holes_filled);
    trellis::VoxelAttributes attr;
    attr.coords = tm.tex.coords;
    attr.feats = tm.tex.out;
    attr.channels = 6;
    attr.res = res;
    // the export in its steps, each timed (stderr, unbuffered): the simplification to python's decimation target (1,000,000 faces), the
    // unwrap, the bake, the GLB
    std::vector<float> sv = tm.shape.verts;
    std::vector<int32_t> sf = tm.shape.faces;
    auto t1 = std::chrono::steady_clock::now();
    trellis::SimplifyStats sst;
    if (!trellis::simplify_to_faces(sv, sf, 1000000, sst, error)) {
        std::fprintf(stderr, "simplify: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "simplify: %.1f s, %d -> %d faces in %d passes\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(),
                 sst.faces_before, sst.faces_after, sst.passes);
    t1 = std::chrono::steady_clock::now();
    std::vector<int32_t> chart_ids;
    trellis::ChartStats cst;
    if (!trellis::compute_charts(sv, sf, trellis::ChartConfig(), chart_ids, cst, error)) {
        std::fprintf(stderr, "charts: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "charts: %.1f s, %d charts, %d collapse passes\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(),
                 cst.charts, cst.collapse_passes);
    t1 = std::chrono::steady_clock::now();
    trellis::UvMesh uvm;
    if (!trellis::unwrap_uv_by_charts(sv, sf, chart_ids, 1024, uvm, error)) {
        std::fprintf(stderr, "unwrap: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "unwrap: %.1f s, %zu vertices, %zu faces, %d charts\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(), uvm.positions.size() / 3, uvm.faces.size() / 3,
                 uvm.charts);
    t1 = std::chrono::steady_clock::now();
    trellis::BakedTextures bt;
    if (!trellis::bake_textures(uvm, attr, 1024, bt, error)) {
        std::fprintf(stderr, "bake: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "bake: %.1f s, coverage %.3f\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count(),
                 bt.coverage);
    t1 = std::chrono::steady_clock::now();
    if (!trellis::write_glb(out, uvm, trellis::vertex_normals(sv, sf), bt, error)) {
        std::fprintf(stderr, "glb: %s\n", error.c_str());
        return 1;
    }
    std::fprintf(stderr, "glb: %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
    std::printf("%d: %zu vertices, %zu faces, %d holes filled, %s\n", res, tm.shape.verts.size() / 3, tm.shape.faces.size() / 3,
                tm.shape.holes_filled, out.c_str());
    return 0;
}
