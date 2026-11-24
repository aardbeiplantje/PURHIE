// Check of the engine's 1024 cascade (src/trellis_shape.cpp run_cascade, src/trellis_sdec.cpp upsample_times) against the python
// pipeline (tools/trellis_cascade_oracle.py, which continues the 512 shape oracle in oracle_shape). Stage checks, exact: the decoder's
// upsampled coordinates from the 512 voxels and latent, and the 64^3 voxels from them. End to end (a smoke test, as shape_verify):
// the cascade's occupied counts and the 1024 mesh against the oracle's. Usage: cascade_verify [oracle_dir] [shape_oracle_dir]
#include "trellis_sdec.h"
#include "trellis_shape.h"

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

// the symmetric Chamfer distance between two point sets (xyz floats), each subsampled to at most `cap` points
static double chamfer(const float* p, size_t np, const float* q, size_t nq, size_t cap) {
    auto one_way = [&](const float* A, size_t na, const float* B, size_t nb) {
        const size_t sa = na > cap ? na / cap : 1, sb = nb > cap ? nb / cap : 1;
        double sum = 0;
        size_t cnt = 0;
        for (size_t i = 0; i < na; i += sa) {
            double best = 1e30;
            for (size_t j = 0; j < nb; j += sb) {
                const double dx = A[3 * i] - B[3 * j], dy = A[3 * i + 1] - B[3 * j + 1], dz = A[3 * i + 2] - B[3 * j + 2];
                best = std::fmin(best, dx * dx + dy * dy + dz * dz);
            }
            sum += std::sqrt(best);
            ++cnt;
        }
        return cnt ? sum / (double)cnt : 0.0;
    };
    return 0.5 * (one_way(p, np, q, nq) + one_way(q, nq, p, np));
}

static size_t differ_i32(const int32_t* got, size_t got_n, const int32_t* ref, size_t ref_n) {
    if (got_n != ref_n) return (size_t)-1;
    size_t d = 0;
    for (size_t i = 0; i < got_n; ++i) d += got[i] != ref[i] ? 1 : 0;
    return d;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_cascade";
    const std::string sdir = argc > 2 ? argv[2] : "/data/trellis2/oracle_shape";
    const std::string ckpt = "/data/trellis2/TRELLIS.2-4B/ckpts/";
    std::vector<char> c32, s32, c512b, hrb, cb, cbh, ssn, sln, hrn, df, dc, mv, mf, normb;
    const bool read_ok = read_bytes(sdir + "/coords.i32", c32) && read_bytes(sdir + "/slat.f32", s32) &&
                         read_bytes(dir + "/coords512.i32", c512b) && read_bytes(dir + "/hr_coords.i32", hrb) &&
                         read_bytes(sdir + "/cond.f32", cb) && read_bytes(dir + "/cond_hr.f32", cbh) &&
                         read_bytes(sdir + "/ss_noise.f32", ssn) && read_bytes(sdir + "/slat_noise.f32", sln) &&
                         read_bytes(dir + "/hr_noise.f32", hrn) && read_bytes(dir + "/dec_feats.f32", df) &&
                         read_bytes(dir + "/dec_coords.i32", dc) && read_bytes(dir + "/mesh_vertices.f32", mv) &&
                         read_bytes(dir + "/mesh_faces.i32", mf) && read_bytes(sdir + "/shape_norm.f32", normb);
    if (!read_ok) {
        std::fprintf(stderr, "cannot read the oracles in %s and %s\n", dir.c_str(), sdir.c_str());
        return 1;
    }
    const int N = (int)(c32.size() / (4 * sizeof(int32_t)));
    const int32_t* coords32 = reinterpret_cast<const int32_t*>(c32.data());
    const float* slat32 = reinterpret_cast<const float*>(s32.data());
    const std::vector<int32_t> ref512(reinterpret_cast<const int32_t*>(c512b.data()),
                                      reinterpret_cast<const int32_t*>(c512b.data()) + c512b.size() / sizeof(int32_t));
    const std::vector<int32_t> ref64(reinterpret_cast<const int32_t*>(hrb.data()),
                                     reinterpret_cast<const int32_t*>(hrb.data()) + hrb.size() / sizeof(int32_t));
    bool ok = true;

    // 1. the upsampled coordinates (the decoder's first four levels) from the 512 voxels and latent: exact
    trellis::ShapeDecoder dec(ckpt + "shape_dec_next_dc_f16c32_fp16.safetensors");
    if (!dec.ok()) {
        std::fprintf(stderr, "decoder: %s\n", dec.error().c_str());
        return 1;
    }
    std::vector<int32_t> up((size_t)N * 4096 * 4);
    int M512 = 0;
    if (!dec.forward(coords32, N, slat32, nullptr, up.data(), &M512, N * 4096, nullptr, 4)) {
        std::fprintf(stderr, "upsample: %s\n", dec.error().c_str());
        return 1;
    }
    up.resize((size_t)M512 * 4);
    const size_t d512 = differ_i32(up.data(), up.size(), ref512.data(), ref512.size());
    std::printf("upsampled voxels at 512: %d (oracle %zu), coords differing %zu\n", M512, ref512.size() / 4, d512);
    ok = ok && d512 == 0;

    // 2. the 64^3 voxels from them: exact
    std::vector<int32_t> hr;
    trellis::cascade_voxels(ref512, hr);
    const size_t dhr = differ_i32(hr.data(), hr.size(), ref64.data(), ref64.size());
    std::printf("64^3 voxels (from the oracle's 512 voxels): %zu (oracle %zu), coords differing %zu\n", hr.size() / 4, ref64.size() / 4,
                dhr);
    ok = ok && dhr == 0;

    // 3. the whole cascade: the 512 stages as the pipeline makes them, the 1024 flow and decoder, the 1024 mesh (smoke test)
    trellis::ShapePipeline pipe(ckpt + "ss_flow_img_dit_1_3B_64_bf16.safetensors",
                                "/data/trellis2/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                                ckpt + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", ckpt + "shape_dec_next_dc_f16c32_fp16.safetensors",
                                ckpt + "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors");
    if (!pipe.ok()) {
        std::fprintf(stderr, "open: %s\n", pipe.error().c_str());
        return 1;
    }
    const float* norm = reinterpret_cast<const float*>(normb.data());
    trellis::ShapeConfig cfg;
    cfg.ss.steps = 12;
    cfg.ss.guidance = 7.5f;
    cfg.ss.guidance_rescale = 0.7f;
    cfg.ss.rescale_t = 5.0;
    cfg.slat.steps = 12;
    cfg.slat.guidance = 7.5f;
    cfg.slat.guidance_rescale = 0.5f;
    cfg.slat.rescale_t = 3.0;
    for (int c = 0; c < 32; ++c) {
        cfg.slat_mean[c] = norm[c];
        cfg.slat_std[c] = norm[32 + c];
    }
    cfg.mesh_grid = 1024;

    // the noise: the sparse-structure latent is the oracle's; the two shape latents replay the oracle's rows in order (the voxel
    // counts of the engine's samples differ from python's, so only the geometric checks apply to them)
    int draw = 0;
    trellis::NoiseFn noise = [&](float* dst, size_t count) {
        const float* rows = reinterpret_cast<const float*>(draw == 0 ? ssn.data() : draw == 1 ? sln.data() : hrn.data());
        const size_t have = (draw == 0 ? ssn.size() : draw == 1 ? sln.size() : hrn.size()) / sizeof(float);
        for (size_t i = 0; i < count; ++i) dst[i] = rows[i % have];
        ++draw;
    };
    const float* cond = reinterpret_cast<const float*>(cb.data());
    const float* cond_hr = reinterpret_cast<const float*>(cbh.data());
    trellis::ShapeResult r;
    if (!pipe.run_cascade(cond, (int)(cb.size() / (1024 * sizeof(float))), cond_hr, (int)(cbh.size() / (1024 * sizeof(float))), noise,
                          cfg, r)) {
        std::fprintf(stderr, "run_cascade: %s\n", pipe.error().c_str());
        return 1;
    }
    const int M = (int)(r.dec_coords.size() / 4);
    const size_t D_ref = dc.size() / (4 * sizeof(int32_t));
    const size_t V_ref = mv.size() / (3 * sizeof(float));
    const size_t T_ref = mf.size() / (3 * sizeof(int32_t));
    const double cham = chamfer(r.verts.data(), r.verts.size() / 3, reinterpret_cast<const float*>(mv.data()), V_ref, 4096);
    const double dratio = (double)M / (double)D_ref;
    std::printf("cascade: %zu voxels at 64^3 (oracle %zu), %d decoder voxels at 1024 (oracle %zu, ratio %.3f)\n", r.hr_coords.size() / 4,
                ref64.size() / 4, M, D_ref, dratio);
    std::printf("mesh: %zu vertices (oracle %zu), %zu faces (oracle %zu), Chamfer distance %.4f (box units)\n", r.verts.size() / 3,
                V_ref, r.faces.size() / 3, T_ref, cham);
    std::printf("timing: sparse-structure sampler %.1f s, shape-SLAT (512) %.1f s, shape-SLAT (1024) %.1f s, decoder %.1f s, mesh %.1f s\n",
                r.seconds_ss, r.seconds_slat, r.seconds_hr, r.seconds_dec, r.seconds_mesh);
    ok = ok && dratio >= 0.5 && dratio <= 2.0 && cham <= 0.1;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
