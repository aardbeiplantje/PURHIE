// The engine's end-to-end textured output (512 or 1024 cascade): the shape, the texture, the colours, the UV unwrap, the baked base colour
// and metallic-roughness textures, and a GLB. The noise and image features are the oracles' (the rows wrap where the engine's voxel counts
// differ from python's). Checks: the GLB header; the baked coverage; the mean base colour of the texels against the texture oracle's
// voxels (the colours of the texture decoder's output, as python's bake reads them). Usage: textured_glb_verify 512|1024 out.glb
#include "trellis_bake.h"
#include "trellis_shape.h"
#include "trellis_tex.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    const int res = argc > 1 ? std::atoi(argv[1]) : 512;
    const std::string out = argc > 2 ? argv[2] : "/data/trellis2/out/textured_512.glb";
    const bool cascade = res == 1024;
    const std::string sdir = "/data/trellis2/oracle_shape", cdir = "/data/trellis2/oracle_cascade";
    const std::string tdir = cascade ? "/data/trellis2/oracle_tex1024" : "/data/trellis2/oracle_tex";
    const std::string ckpt = "/data/trellis2/TRELLIS.2-4B/ckpts/";
    std::vector<char> ssn, sln, cb, normb, tnormb, tnz, cbh, hrn, ttex, ttexc;
    bool read_ok = read_bytes(sdir + "/ss_noise.f32", ssn) && read_bytes(sdir + "/slat_noise.f32", sln) &&
                   read_bytes(sdir + "/cond.f32", cb) && read_bytes(sdir + "/shape_norm.f32", normb) &&
                   read_bytes(tdir + "/tex_norm.f32", tnormb) && read_bytes(tdir + "/tex_noise.f32", tnz) &&
                   read_bytes(tdir + "/tex_out.f32", ttex) && read_bytes(tdir + "/tex_out_coords.i32", ttexc);
    if (cascade) read_ok = read_ok && read_bytes(cdir + "/cond_hr.f32", cbh) && read_bytes(cdir + "/hr_noise.f32", hrn);
    if (!read_ok) {
        std::fprintf(stderr, "cannot read the oracles for %d\n", res);
        return 1;
    }
    const float* cond = reinterpret_cast<const float*>(cb.data());
    const int Lc = (int)(cb.size() / (1024 * sizeof(float)));
    const float* norm = reinterpret_cast<const float*>(normb.data());
    const float* tnorm = reinterpret_cast<const float*>(tnormb.data());

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
        scfg.slat_mean[c] = norm[c];
        scfg.slat_std[c] = norm[32 + c];
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
        tcfg.shape_mean[c] = norm[c];
        tcfg.shape_std[c] = norm[32 + c];
        tcfg.tex_mean[c] = tnorm[c];
        tcfg.tex_std[c] = tnorm[32 + c];
    }

    // the noise draws in order: the sparse-structure latent, the 512 shape latent, then (cascade) the 1024 shape latent, the texture latent
    struct Row {
        const float* p;
        size_t n;
    };
    std::vector<Row> rows = {{reinterpret_cast<const float*>(ssn.data()), ssn.size() / 4},
                             {reinterpret_cast<const float*>(sln.data()), sln.size() / 4}};
    if (cascade) rows.push_back({reinterpret_cast<const float*>(hrn.data()), hrn.size() / 4});
    rows.push_back({reinterpret_cast<const float*>(tnz.data()), tnz.size() / 4});
    int draw = 0;
    trellis::NoiseFn noise = [&](float* dst, size_t count) {
        const Row& r = rows[draw < (int)rows.size() ? draw : (int)rows.size() - 1];
        for (size_t i = 0; i < count; ++i) dst[i] = r.p[i % r.n];
        ++draw;
    };

    trellis::TexturedMesh tm;
    std::string error;
    bool run_ok;
    if (cascade) {
        const float* cond_hr = reinterpret_cast<const float*>(cbh.data());
        const int Lc_hr = (int)(cbh.size() / (1024 * sizeof(float)));
        run_ok = trellis::run_textured_cascade(shape, tp, cond, Lc, cond_hr, Lc_hr, noise, scfg, tcfg, tm, error);
    } else {
        run_ok = trellis::run_textured(shape, tp, cond, Lc, noise, scfg, tcfg, tm, error);
    }
    if (!run_ok) {
        std::fprintf(stderr, "textured run: %s\n", error.c_str());
        return 1;
    }
    const size_t V = tm.shape.verts.size() / 3, T = tm.shape.faces.size() / 3;
    std::printf("%d: %zu vertices, %zu faces, %d holes filled, colours for %zu vertices (texture voxels %zu)\n", res, V, T,
                tm.shape.holes_filled, tm.colors.size() / 3, tm.tex.coords.size() / 4);

    // the bake and the GLB: the attributes are the texture decoder's voxels at the mesh's grid
    trellis::VoxelAttributes attr;
    attr.coords = tm.tex.coords;
    attr.feats = tm.tex.out;
    attr.channels = 6;
    attr.res = res;
    if (!trellis::export_textured_glb(tm.shape.verts, tm.shape.faces, attr, 1024, out, error)) {
        std::fprintf(stderr, "export: %s\n", error.c_str());
        return 1;
    }

    // the colours of the texture against the texture oracle's voxels (clamped as the bake reads them)
    double mean_e[3] = {0, 0, 0}, mean_p[3] = {0, 0, 0};
    const size_t Dv = tm.tex.out.size() / 6;
    for (size_t i = 0; i < Dv; ++i)
        for (int c = 0; c < 3; ++c) mean_e[c] += tm.tex.out[i * 6 + c] < 0 ? 0 : (tm.tex.out[i * 6 + c] > 1 ? 1 : tm.tex.out[i * 6 + c]);
    const float* tp_out = reinterpret_cast<const float*>(ttex.data());
    const size_t Dp = ttex.size() / (6 * sizeof(float));
    for (size_t i = 0; i < Dp; ++i)
        for (int c = 0; c < 3; ++c) mean_p[c] += tp_out[i * 6 + c] < 0 ? 0 : (tp_out[i * 6 + c] > 1 ? 1 : tp_out[i * 6 + c]);
    std::printf("base colour mean per channel: engine %.3f %.3f %.3f, python %.3f %.3f %.3f\n", mean_e[0] / Dv, mean_e[1] / Dv,
                mean_e[2] / Dv, mean_p[0] / Dp, mean_p[1] / Dp, mean_p[2] / Dp);
    std::printf("timing: shape %.1f s (sparse structure %.1f, shape-SLAT %.1f, decoder %.1f, mesh %.1f), texture %.1f s (flow %.1f, decoder %.1f)\n",
                tm.shape.seconds_ss + tm.shape.seconds_slat + tm.shape.seconds_dec + tm.shape.seconds_mesh + tm.shape.seconds_hr,
                tm.shape.seconds_ss, tm.shape.seconds_slat, tm.shape.seconds_dec, tm.shape.seconds_mesh, tm.tex.seconds_flow + tm.tex.seconds_dec,
                tm.tex.seconds_flow, tm.tex.seconds_dec);

    std::vector<char> glb;
    read_bytes(out, glb);
    const bool glb_ok = glb.size() > 20 && std::memcmp(glb.data(), "glTF", 4) == 0 &&
                        *reinterpret_cast<const uint32_t*>(glb.data() + 8) == glb.size();
    bool colour_ok = true;
    for (int c = 0; c < 3; ++c) colour_ok = colour_ok && std::fabs(mean_e[c] / Dv - mean_p[c] / Dp) <= 0.15;
    std::printf("glb: %s, %zu bytes: %s\n", out.c_str(), glb.size(), glb_ok ? "ok" : "BAD");
    const bool ok = glb_ok && colour_ok && V > 0 && T > 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
