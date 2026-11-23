/* AudioLDM 2's projection model (src/audioldm2_projection.cpp) against the diffusers oracle: tools/audioldm2_stage_projection.py
 * writes proj_clap.f32 (Tc x 512), proj_t5.f32 (Tt x 1024) and proj_out.f32 ((Tc + Tt + 4) x 768) to /data/audioldm2-stages.
 * Usage: audioldm2_projection_verify [projection.safetensors] [stages dir]. Exit 0 when the output matches. */
#include "audioldm2_projection.h"
#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool read_floats(const std::string& path, std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)bytes / sizeof(float));
    const size_t got = std::fread(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return got == v.size();
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : "/data/audioldm2/projection_model/diffusion_pytorch_model.safetensors";
    const std::string dir = argc > 2 ? argv[2] : "/data/audioldm2-stages";
    std::vector<float> clap, t5, ref;
    if (!read_floats(dir + "/proj_clap.f32", clap) || !read_floats(dir + "/proj_t5.f32", t5) || !read_floats(dir + "/proj_out.f32", ref)) {
        std::fprintf(stderr, "cannot read the stage files in %s\n", dir.c_str());
        return 1;
    }
    const int Tc = (int)(clap.size() / 512), Tt = (int)(t5.size() / 1024);
    zimage::SafeTensors st(model);
    if (!st.ok() || !st.register_device()) {
        std::fprintf(stderr, "the projection file: %s\n", st.ok() ? "registration failed" : st.error().c_str());
        return 1;
    }
    std::vector<float> got(ref.size());
    if (!audioldm2::projection_forward(st, clap.data(), Tc, t5.data(), Tt, got.data())) {
        std::fprintf(stderr, "the projection device path failed\n");
        return 1;
    }
    double num = 0, den = 0, maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)got[i] - ref[i];
        num += d * d; den += (double)ref[i] * ref[i];
        maxerr = std::fmax(maxerr, std::fabs(d));
    }
    const double rel = std::sqrt(num / den);
    const bool ok = rel <= 1e-3;
    std::printf("projection, Tc %d, Tt %d -> %d rows: rel L2 %.3e, max abs %.3e  %s\n", Tc, Tt, Tc + Tt + 4, rel, maxerr, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
