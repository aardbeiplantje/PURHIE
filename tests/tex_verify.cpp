// Check of the engine's texture path (src/trellis_tex.cpp) against the python pipeline (tools/trellis_tex_oracle.py, the texture of the
// 512 pipeline, fp32 with the shipped weights upcast; it continues the shape oracle in oracle_shape). Stage checks: the texture
// flow's single forward on the 64-channel input; the shape decoder's subdivisions at its four levels (exact voxels, logits); the
// texture decoder on the engine's subdivisions (its six outputs and voxels, against python). The texture sampler from the oracle's
// noise: its latent. Usage: tex_verify [oracle_tex] [shape_oracle]
#include "trellis_sdec.h"
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

static double rel_l2(const float* got, const float* ref, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = (double)got[i] - (double)ref[i];
        num += e * e;
        den += (double)ref[i] * ref[i];
    }
    return std::sqrt(num / (den > 0 ? den : 1.0));
}

static std::vector<float> floats(const std::vector<char>& b) {
    std::vector<float> v(b.size() / sizeof(float));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(float));
    return v;
}

static std::vector<int32_t> ints(const std::vector<char>& b) {
    std::vector<int32_t> v(b.size() / sizeof(int32_t));
    std::memcpy(v.data(), b.data(), v.size() * sizeof(int32_t));
    return v;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_tex";
    const std::string sdir = argc > 2 ? argv[2] : "/data/trellis2/oracle_shape";
    const std::string ckpt = "/data/trellis2/TRELLIS.2-4B/ckpts/";
    std::vector<char> fx, fo, nz, tslat, traw, tout, tcoords, cb, c32b, s32b, normb, tnormb;
    std::vector<char> sc[4], sf[4];
    bool read_ok = read_bytes(dir + "/fwd_x.f32", fx) && read_bytes(dir + "/fwd_out.f32", fo) &&
                   read_bytes(dir + "/tex_noise.f32", nz) && read_bytes(dir + "/tex_slat.f32", tslat) &&
                   read_bytes(dir + "/tex_dec_raw.f32", traw) && read_bytes(dir + "/tex_out_coords.i32", tcoords) &&
                   read_bytes(dir + "/tex_norm.f32", tnormb) && read_bytes(sdir + "/coords.i32", c32b) &&
                   read_bytes(sdir + "/slat.f32", s32b) && read_bytes(sdir + "/cond.f32", cb) &&
                   read_bytes(sdir + "/shape_norm.f32", normb);
    for (int i = 0; i < 4; ++i)
        read_ok = read_ok && read_bytes(dir + "/subs" + std::to_string(i) + "_coords.i32", sc[i]) &&
                  read_bytes(dir + "/subs" + std::to_string(i) + "_feats.f32", sf[i]);
    if (!read_ok) {
        std::fprintf(stderr, "cannot read the oracles in %s and %s\n", dir.c_str(), sdir.c_str());
        return 1;
    }
    const std::vector<int32_t> coords = ints(c32b);
    const int N = (int)(coords.size() / 4);
    const std::vector<float> slat = floats(s32b);
    const std::vector<float> cond = floats(cb);
    const int Lc = (int)(cb.size() / (1024 * sizeof(float)));
    const std::vector<float> shape_norm = floats(normb), tex_norm = floats(tnormb);
    const std::vector<float> ref_fo = floats(fo), ref_tslat = floats(tslat), ref_raw = floats(traw);
    const std::vector<int32_t> ref_tcoords = ints(tcoords);
    const int M_ref = (int)(ref_tcoords.size() / 4);
    bool ok = true;

    trellis::TexturePipeline tp(ckpt + "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", ckpt + "tex_dec_next_dc_f16c32_fp16.safetensors");
    if (!tp.ok()) {
        std::fprintf(stderr, "open: %s\n", tp.error().c_str());
        return 1;
    }

    // 1. the texture flow's single forward on [noise | normalised shape latent] at t = 0.5
    std::vector<float> out32((size_t)N * 32);
    if (!tp.flow_forward(coords.data(), N, floats(fx).data(), cond.data(), Lc, 0.5f, out32.data())) {
        std::fprintf(stderr, "flow: %s\n", tp.error().c_str());
        return 1;
    }
    const double rf = rel_l2(out32.data(), ref_fo.data(), out32.size());
    std::printf("texture flow, one forward (64 inputs): rel L2 %.3e\n", rf);
    ok = ok && rf < 1e-2;

    // 2. the shape decoder's subdivisions (the engine's shape decoder on the 512 latent): the voxels exact, the logits
    trellis::ShapeDecoder shape(ckpt + "shape_dec_next_dc_f16c32_fp16.safetensors");
    if (!shape.ok()) {
        std::fprintf(stderr, "shape decoder: %s\n", shape.error().c_str());
        return 1;
    }
    const int cap = 1024 * N;  // the shape decoder's output at 512 has ~400 voxels per input voxel here
    std::vector<float> sfeat((size_t)cap * 7);
    std::vector<int32_t> scoord((size_t)cap * 4);
    int M_shape = 0;
    std::vector<trellis::Subdivision> subs;
    if (!shape.forward(coords.data(), N, slat.data(), sfeat.data(), scoord.data(), &M_shape, cap, nullptr, 0, &subs)) {
        std::fprintf(stderr, "shape decoder: %s\n", shape.error().c_str());
        return 1;
    }
    for (int i = 0; i < 4; ++i) {
        const std::vector<int32_t> rc = ints(sc[i]);
        const std::vector<float> rl = floats(sf[i]);
        const bool same = subs.size() > (size_t)i && subs[i].coords.size() == rc.size() &&
                          std::memcmp(subs[i].coords.data(), rc.data(), rc.size() * sizeof(int32_t)) == 0;
        const double rl2 = same ? rel_l2(subs[i].logits.data(), rl.data(), rl.size()) : -1.0;
        std::printf("subdivision level %d: %zu voxels, voxels %s, logits rel L2 %.3e\n", i, rc.size() / 4, same ? "exact" : "DIFFER", rl2);
        ok = ok && same && rl2 >= 0 && rl2 < 1e-3;
    }

    // 3. the texture decoder on the oracle's texture latent, guided by the engine's subdivisions: the six outputs and the voxels
    trellis::TexResult dec;
    if (!tp.decode(coords.data(), N, ref_tslat.data(), subs, dec)) {
        std::fprintf(stderr, "decode: %s\n", tp.error().c_str());
        return 1;
    }
    const bool dsame = dec.coords.size() == ref_tcoords.size() &&
                       std::memcmp(dec.coords.data(), ref_tcoords.data(), ref_tcoords.size() * sizeof(int32_t)) == 0;
    const double rd = dsame ? rel_l2(dec.raw.data(), ref_raw.data(), dec.raw.size()) : -1.0;
    std::printf("texture decoder: %d voxels (oracle %d), voxels %s, raw outputs rel L2 %.3e\n", (int)dec.coords.size() / 4, M_ref,
                dsame ? "exact" : "DIFFER", rd);
    ok = ok && dsame && rd >= 0 && rd < 1e-2;

    // 4. the texture sampler from the oracle's noise (the shape latent and the subdivisions the engine made): the texture latent
    trellis::TexConfig cfg;
    cfg.sampler.steps = 12;
    cfg.sampler.guidance = 1.0f;
    cfg.sampler.guidance_rescale = 0.0f;
    cfg.sampler.rescale_t = 3.0;
    cfg.sampler.interval_lo = 0.6;
    cfg.sampler.interval_hi = 0.9;
    for (int c = 0; c < 32; ++c) {
        cfg.shape_mean[c] = shape_norm[c];
        cfg.shape_std[c] = shape_norm[32 + c];
        cfg.tex_mean[c] = tex_norm[c];
        cfg.tex_std[c] = tex_norm[32 + c];
    }
    const std::vector<float> noise_rows = floats(nz);
    trellis::NoiseFn noise = [&](float* dst, size_t count) {
        if (count == noise_rows.size()) std::memcpy(dst, noise_rows.data(), count * sizeof(float));
        else for (size_t i = 0; i < count; ++i) dst[i] = noise_rows[i % noise_rows.size()];
    };
    trellis::TexResult smp;
    if (!tp.run(coords.data(), N, slat.data(), subs, cond.data(), Lc, noise, cfg, smp)) {
        std::fprintf(stderr, "run: %s\n", tp.error().c_str());
        return 1;
    }
    const double rs = rel_l2(smp.tex_slat.data(), ref_tslat.data(), smp.tex_slat.size());
    std::printf("texture latent from the sampler: rel L2 %.3e (flow %.1f s, decoder %.1f s)\n", rs, smp.seconds_flow, smp.seconds_dec);
    ok = ok && rs < 5e-2;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
