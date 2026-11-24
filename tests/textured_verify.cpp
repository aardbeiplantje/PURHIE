// The end-to-end textured run on the engine (src/trellis_tex.cpp run_textured): the 512 shape and its texture from one set of
// noise and one image-feature set, checked against the python pipeline (oracle_shape: the shape; oracle_tex: the texture). The
// checks are smoke tests, as the shape and cascade checks: the voxel count and the mesh's Chamfer distance against the shape oracle,
// and the vertices' mean base colour per channel against the texture oracle's. The noise replays the oracles' rows in order; the
// engine's voxel counts differ from python's, so the rows wrap around past the end. Usage: textured_verify [oracle_shape] [oracle_tex]
// [out.ply] (the coloured mesh written as ASCII PLY, when given).
#include "trellis_shape.h"
#include "trellis_tex.h"

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

// IoU of two sets of occupied voxels (coords [n][4], batch 0, in 0..31): the bitsets over the 32^3 grid
static double occupancy_iou(const int32_t* a, int na, const int32_t* b, int nb) {
    std::vector<char> A(32 * 32 * 32, 0), B(32 * 32 * 32, 0);
    for (int i = 0; i < na; ++i) A[((size_t)a[4 * i + 1] * 32 + a[4 * i + 2]) * 32 + a[4 * i + 3]] = 1;
    for (int i = 0; i < nb; ++i) B[((size_t)b[4 * i + 1] * 32 + b[4 * i + 2]) * 32 + b[4 * i + 3]] = 1;
    size_t inter = 0, uni = 0;
    for (size_t i = 0; i < A.size(); ++i) {
        inter += (A[i] && B[i]) ? 1 : 0;
        uni += (A[i] || B[i]) ? 1 : 0;
    }
    return uni ? (double)inter / (double)uni : 1.0;
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

static void write_ply(const std::string& path, const std::vector<float>& verts, const std::vector<float>& colors,
                      const std::vector<int32_t>& faces) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    const size_t V = verts.size() / 3, T = faces.size() / 3;
    std::fprintf(f, "ply\nformat ascii 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\nelement face %zu\n"
                    "property list uchar int vertex_indices\nend_header\n", V, T);
    for (size_t i = 0; i < V; ++i) {
        int rgb[3];
        for (int c = 0; c < 3; ++c) rgb[c] = (int)std::lround(colors[i * 3 + c] * 255.0f);
        std::fprintf(f, "%.6f %.6f %.6f %d %d %d\n", verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2], rgb[0], rgb[1], rgb[2]);
    }
    for (size_t t = 0; t < T; ++t) std::fprintf(f, "3 %d %d %d\n", faces[t * 3], faces[t * 3 + 1], faces[t * 3 + 2]);
    std::fclose(f);
}

int main(int argc, char** argv) {
    const std::string sdir = argc > 1 ? argv[1] : "/data/trellis2/oracle_shape";
    const std::string tdir = argc > 2 ? argv[2] : "/data/trellis2/oracle_tex";
    const std::string ply = argc > 3 ? argv[3] : "";
    const std::string ckpt = "/data/trellis2/TRELLIS.2-4B/ckpts/";
    std::vector<char> ssn, sln, cb, normb, tnormb, tnz, ref_c, ref_v, ref_f, ref_tout;
    const bool read_ok = read_bytes(sdir + "/ss_noise.f32", ssn) && read_bytes(sdir + "/slat_noise.f32", sln) &&
                         read_bytes(sdir + "/cond.f32", cb) && read_bytes(sdir + "/shape_norm.f32", normb) &&
                         read_bytes(sdir + "/coords.i32", ref_c) && read_bytes(sdir + "/mesh_vertices.f32", ref_v) &&
                         read_bytes(sdir + "/mesh_faces.i32", ref_f) && read_bytes(tdir + "/tex_noise.f32", tnz) &&
                         read_bytes(tdir + "/tex_norm.f32", tnormb) && read_bytes(tdir + "/tex_out.f32", ref_tout);
    if (!read_ok) {
        std::fprintf(stderr, "cannot read the oracles in %s and %s\n", sdir.c_str(), tdir.c_str());
        return 1;
    }
    const float* cond = reinterpret_cast<const float*>(cb.data());
    const int Lc = (int)(cb.size() / (1024 * sizeof(float)));
    const float* norm = reinterpret_cast<const float*>(normb.data());
    const float* tnorm = reinterpret_cast<const float*>(tnormb.data());
    const int32_t* rc = reinterpret_cast<const int32_t*>(ref_c.data());
    const int N_ref = (int)(ref_c.size() / (4 * sizeof(int32_t)));
    const size_t V_ref = ref_v.size() / (3 * sizeof(float));
    const float* rv = reinterpret_cast<const float*>(ref_v.data());
    const std::vector<float> tex_rows(reinterpret_cast<const float*>(ref_tout.data()),
                                      reinterpret_cast<const float*>(ref_tout.data()) + ref_tout.size() / sizeof(float));
    const size_t M_ref = tex_rows.size() / 6;

    trellis::ShapePipeline shape(ckpt + "ss_flow_img_dit_1_3B_64_bf16.safetensors",
                                 "/data/trellis2/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
                                 ckpt + "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors", ckpt + "shape_dec_next_dc_f16c32_fp16.safetensors");
    trellis::TexturePipeline tex(ckpt + "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", ckpt + "tex_dec_next_dc_f16c32_fp16.safetensors");
    if (!shape.ok() || !tex.ok()) {
        std::fprintf(stderr, "open: %s%s\n", shape.error().c_str(), tex.error().c_str());
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
    scfg.mesh_grid = 512;

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

    // the three draws: the sparse-structure latent, the shape latent, the texture latent (the oracles' rows; wrap past the end)
    const float* rows[3] = {reinterpret_cast<const float*>(ssn.data()), reinterpret_cast<const float*>(sln.data()),
                            reinterpret_cast<const float*>(tnz.data())};
    const size_t have[3] = {ssn.size() / sizeof(float), sln.size() / sizeof(float), tnz.size() / sizeof(float)};
    int draw = 0;
    trellis::NoiseFn noise = [&](float* dst, size_t count) {
        const int d = draw < 3 ? draw : 2;
        for (size_t i = 0; i < count; ++i) dst[i] = rows[d][i % have[d]];
        ++draw;
    };

    trellis::TexturedMesh tm;
    std::string error;
    if (!trellis::run_textured(shape, tex, cond, Lc, noise, scfg, tcfg, tm, error)) {
        std::fprintf(stderr, "run_textured: %s\n", error.c_str());
        return 1;
    }

    // the shape against the shape oracle (smoke bounds, as shape_verify)
    const int N = (int)(tm.shape.coords.size() / 4);
    const double count_ratio = (double)N / (double)N_ref;
    const double iou = occupancy_iou(tm.shape.coords.data(), N, rc, N_ref);
    const size_t V = tm.shape.verts.size() / 3;
    const double cham = chamfer(tm.shape.verts.data(), V, rv, V_ref, 4096);
    std::printf("shape: %d occupied voxels (oracle %d, ratio %.3f), IoU %.3f; %zu vertices (oracle %zu), Chamfer %.4f (box units)\n",
                N, N_ref, count_ratio, iou, V, V_ref, cham);
    bool ok = count_ratio >= 0.5 && count_ratio <= 2.0 && iou >= 0.5 && cham <= 0.1;

    // the texture: one colour per vertex, the mean base colour per channel against the texture oracle's (clamped the same way)
    double mean_e[3] = {0, 0, 0}, mean_p[3] = {0, 0, 0};
    for (size_t i = 0; i < V; ++i)
        for (int c = 0; c < 3; ++c) mean_e[c] += tm.colors[i * 3 + c];
    for (size_t i = 0; i < M_ref; ++i)
        for (int c = 0; c < 3; ++c) {
            const float v = tex_rows[i * 6 + c];
            mean_p[c] += (v < 0 ? 0 : (v > 1 ? 1 : v));
        }
    std::printf("base colour, mean per channel: engine %.3f %.3f %.3f, python %.3f %.3f %.3f (%zu voxels)\n", mean_e[0] / V,
                mean_e[1] / V, mean_e[2] / V, mean_p[0] / M_ref, mean_p[1] / M_ref, mean_p[2] / M_ref, M_ref);
    std::printf("timing: sparse-structure %.1f s, shape-SLAT %.1f s, decoder %.1f s, mesh %.1f s, texture flow %.1f s, texture decoder %.1f s\n",
                tm.shape.seconds_ss, tm.shape.seconds_slat, tm.shape.seconds_dec, tm.shape.seconds_mesh, tm.tex.seconds_flow,
                tm.tex.seconds_dec);
    for (int c = 0; c < 3; ++c) ok = ok && std::fabs(mean_e[c] / V - mean_p[c] / M_ref) <= 0.15;

    if (!ply.empty()) {
        write_ply(ply, tm.shape.verts, tm.colors, tm.shape.faces);
        std::printf("wrote %s: %zu vertices, %zu faces\n", ply.c_str(), V, tm.shape.faces.size() / 3);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
