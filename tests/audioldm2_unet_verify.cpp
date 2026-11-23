/* AudioLDM 2's UNet (src/audioldm2_unet.cpp) against the diffusers oracle: tools/audioldm2_stage_unet.py writes unet_in.f32 (8 x 32 x 32),
 * unet_out.f32 (8 x 32 x 32) for timestep 500, encoder_hidden_states = gen_out.f32 (8 x 768), encoder_hidden_states_1 = t5_hidden.f32
 * (9 x 1024). Usage: audioldm2_unet_verify [unet/diffusion_pytorch_model.safetensors] [stages dir]. */
#include "audioldm2_unet.h"
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
    const std::string model = argc > 1 ? argv[1] : "/data/audioldm2/unet/diffusion_pytorch_model.safetensors";
    const std::string dir = argc > 2 ? argv[2] : "/data/audioldm2-stages";
    std::vector<float> x, gen, t5, ref;
    if (!read_floats(dir + "/unet_in.f32", x) || !read_floats(dir + "/gen_out.f32", gen) || !read_floats(dir + "/t5_hidden.f32", t5) ||
        !read_floats(dir + "/unet_out.f32", ref)) {
        std::fprintf(stderr, "cannot read the stage files in %s\n", dir.c_str());
        return 1;
    }
    zimage::SafeTensors st(model);
    if (!st.ok() || !st.register_device()) {
        std::fprintf(stderr, "the UNet file: %s\n", st.ok() ? "registration failed" : st.error().c_str());
        return 1;
    }
    const int H = 32, W = 32, T1 = (int)(gen.size() / 768), T2 = (int)(t5.size() / 1024);
    std::vector<float> got(ref.size());
    if (!audioldm2::unet_forward(st, x.data(), H, W, 500.0f, gen.data(), T1, t5.data(), T2, got.data())) {
        std::fprintf(stderr, "the UNet device path failed\n");
        return 1;
    }
    double num = 0, den = 0, maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)got[i] - ref[i];
        num += d * d; den += (double)ref[i] * ref[i];
        if (std::fabs(d) > maxerr) maxerr = std::fabs(d);
    }
    const double rel = std::sqrt(num / den);
    std::printf("UNet, latent 8 x %d x %d, t 500: rel L2 %.3e, max abs %.3e  %s\n", H, W, rel, maxerr, rel <= 1e-3 ? "ok" : "FAIL");
    return rel <= 1e-3 ? 0 : 1;
}
