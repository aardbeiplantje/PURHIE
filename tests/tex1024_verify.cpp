// Check of the engine's 1024 texture pass (src/trellis_tex.cpp, TexturePipeline with the 1024 flow) against the python pipeline
// (tools/trellis_tex1024_oracle.py, which continues the cascade oracle: the 64^3 voxels and the 1024 shape latent of oracle_cascade).
// Stage checks as tex_verify: the flow's forward on the 64-channel input; the 1024 shape decoder's subdivisions (voxels exact, logits);
// the texture decoder on the oracle's texture latent, guided by the engine's subdivisions (voxels exact, outputs); the sampler from the
// oracle's noise (its latent). Usage: tex1024_verify [oracle_tex1024] [oracle_cascade] [oracle_shape]
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
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_tex1024";
    const std::string cdir = argc > 2 ? argv[2] : "/data/trellis2/oracle_cascade";
    const std::string sdir = argc > 3 ? argv[3] : "/data/trellis2/oracle_shape";
    const std::string ckpt = "/data/trellis2/TRELLIS.2-4B/ckpts/";
    std::vector<char> fx, fo, nz, tslat, traw, tcoords, hrc, hrs, cbh, normb, tnormb;
    std::vector<char> sc[4], sf[4];
    bool read_ok = read_bytes(dir + "/fwd_x.f32", fx) && read_bytes(dir + "/fwd_out.f32", fo) &&
                   read_bytes(dir + "/tex_noise.f32", nz) && read_bytes(dir + "/tex_slat.f32", tslat) &&
                   read_bytes(dir + "/tex_dec_raw.f32", traw) && read_bytes(dir + "/tex_out_coords.i32", tcoords) &&
                   read_bytes(dir + "/tex_norm.f32", tnormb) && read_bytes(cdir + "/hr_coords.i32", hrc) &&
                   read_bytes(cdir + "/hr_slat.f32", hrs) && read_bytes(cdir + "/cond_hr.f32", cbh) &&
                   read_bytes(sdir + "/shape_norm.f32", normb);
    for (int i = 0; i < 4; ++i)
        read_ok = read_ok && read_bytes(dir + "/subs" + std::to_string(i) + "_coords.i32", sc[i]) &&
                  read_bytes(dir + "/subs" + std::to_string(i) + "_feats.f32", sf[i]);
    if (!read_ok) {
        std::fprintf(stderr, "cannot read the oracles in %s, %s and %s\n", dir.c_str(), cdir.c_str(), sdir.c_str());
        return 1;
    }
    const std::vector<int32_t> coords = ints(hrc);
    const int K = (int)(coords.size() / 4);
    const std::vector<float> slat = floats(hrs);
    const std::vector<float> cond = floats(cbh);
    const int Lc = (int)(cbh.size() / (1024 * sizeof(float)));
    const std::vector<float> shape_norm = floats(normb), tex_norm = floats(tnormb);
    const std::vector<float> ref_fo = floats(fo), ref_tslat = floats(tslat), ref_raw = floats(traw);
    const std::vector<int32_t> ref_tcoords = ints(tcoords);
    const int D_ref = (int)(ref_tcoords.size() / 4);
    bool ok = true;

    trellis::TexturePipeline tp(ckpt + "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors", ckpt + "tex_dec_next_dc_f16c32_fp16.safetensors",
                                ckpt + "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors");
    if (!tp.ok()) {
        std::fprintf(stderr, "open: %s\n", tp.error().c_str());
        return 1;
    }

    // 1. the 1024 texture flow's single forward on [noise | normalised shape latent] (64 inputs) at t = 0.5
    std::vector<float> out32((size_t)K * 32);
    if (!tp.flow_forward(coords.data(), K, floats(fx).data(), cond.data(), Lc, 0.5f, out32.data(), true)) {
        std::fprintf(stderr, "flow: %s\n", tp.error().c_str());
        return 1;
    }
    const double rf = rel_l2(out32.data(), ref_fo.data(), out32.size());
    std::printf("1024 texture flow, one forward (64 inputs): rel L2 %.3e\n", rf);
    ok = ok && rf < 1e-2;

    // 2. the 1024 shape decoder's subdivisions on the 64^3 voxels
    trellis::ShapeDecoder shape(ckpt + "shape_dec_next_dc_f16c32_fp16.safetensors");
    if (!shape.ok()) {
        std::fprintf(stderr, "shape decoder: %s\n", shape.error().c_str());
        return 1;
    }
    const int cap = 1024 * K;
    std::vector<float> sfeat((size_t)cap * 7);
    std::vector<int32_t> scoord((size_t)cap * 4);
    int M_shape = 0;
    std::vector<trellis::Subdivision> subs;
    if (!shape.forward(coords.data(), K, slat.data(), sfeat.data(), scoord.data(), &M_shape, cap, nullptr, 0, &subs)) {
        std::fprintf(stderr, "shape decoder: %s\n", shape.error().c_str());
        return 1;
    }
    for (int i = 0; i < 4; ++i) {
        const std::vector<int32_t> rc = ints(sc[i]);
        const std::vector<float> rl = floats(sf[i]);
        const bool same = subs.size() > (size_t)i && subs[i].coords.size() == rc.size() &&
                          std::memcmp(subs[i].coords.data(), rc.data(), rc.size() * sizeof(int32_t)) == 0;
        const double rl2 = same ? rel_l2(subs[i].logits.data(), rl.data(), rl.size()) : -1.0;
        // the children kept: the positive logits (python's subdiv > 0); a sign that differs is a child whose logit is at zero
        size_t flips = 0;
        if (same)
            for (size_t k = 0; k < rl.size(); ++k) flips += ((subs[i].logits[k] > 0) != (rl[k] > 0)) ? 1 : 0;
        std::printf("1024 subdivision level %d: %zu voxels, voxels %s, logits rel L2 %.3e, kept-child flips %zu\n", i, rc.size() / 4,
                    same ? "exact" : "DIFFER", rl2, flips);
        ok = ok && same && rl2 >= 0 && rl2 < 1e-3;
    }

    // 3. the texture decoder on the oracle's texture latent, guided by python's subdivisions (the decoder alone; the engine's own
    //    subdivisions differ at one child whose logit is at zero, see the flips above)
    std::vector<trellis::Subdivision> psubs(4);
    for (int i = 0; i < 4; ++i) {
        psubs[i].coords = ints(sc[i]);
        psubs[i].logits = floats(sf[i]);
    }
    trellis::TexResult dec;
    if (!tp.decode(coords.data(), K, ref_tslat.data(), psubs, dec)) {
        std::fprintf(stderr, "decode: %s\n", tp.error().c_str());
        return 1;
    }
    const bool dsame = dec.coords.size() == ref_tcoords.size() &&
                       std::memcmp(dec.coords.data(), ref_tcoords.data(), ref_tcoords.size() * sizeof(int32_t)) == 0;
    const double rd = dsame ? rel_l2(dec.raw.data(), ref_raw.data(), dec.raw.size()) : -1.0;
    std::printf("1024 texture decoder (python subdivisions): %d voxels (oracle %d), voxels %s, raw outputs rel L2 %.3e\n",
                (int)dec.coords.size() / 4, D_ref,
                dsame ? "exact" : "DIFFER", rd);
    ok = ok && dsame && rd >= 0 && rd < 1e-2;

    // 4. the 1024 texture sampler from the oracle's noise: its latent
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
        for (size_t i = 0; i < count; ++i) dst[i] = noise_rows[i % noise_rows.size()];
    };
    trellis::TexResult smp;
    if (!tp.run(coords.data(), K, slat.data(), subs, cond.data(), Lc, noise, cfg, smp, true)) {
        std::fprintf(stderr, "run: %s\n", tp.error().c_str());
        return 1;
    }
    const double rs = rel_l2(smp.tex_slat.data(), ref_tslat.data(), smp.tex_slat.size());
    std::printf("1024 texture latent from the sampler: rel L2 %.3e (flow %.1f s, decoder %.1f s)\n", rs, smp.seconds_flow, smp.seconds_dec);
    ok = ok && rs < 5e-2;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
