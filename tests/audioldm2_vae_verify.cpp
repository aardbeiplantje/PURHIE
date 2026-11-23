/* AudioLDM 2's VAE decoder (src/audioldm2_vae.cpp) against the diffusers oracle: tools/audioldm2_stage_vae.py decodes unet_out.f32
 * (8 x 32 x 32) to vae_out.f32 (1 x 128 x 128). Usage: audioldm2_vae_verify [vae/diffusion_pytorch_model.safetensors] [stages dir]. */
#include "audioldm2_vae.h"
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
    const std::string model = argc > 1 ? argv[1] : "/data/audioldm2/vae/diffusion_pytorch_model.safetensors";
    const std::string dir = argc > 2 ? argv[2] : "/data/audioldm2-stages";
    std::vector<float> z, ref;
    if (!read_floats(dir + "/unet_out.f32", z) || !read_floats(dir + "/vae_out.f32", ref)) {
        std::fprintf(stderr, "cannot read the stage files in %s\n", dir.c_str());
        return 1;
    }
    zimage::SafeTensors st(model);
    if (!st.ok() || !st.register_device()) {
        std::fprintf(stderr, "the VAE file: %s\n", st.ok() ? "registration failed" : st.error().c_str());
        return 1;
    }
    const int h = 32, w = 32;
    std::vector<float> got(ref.size());
    if (!audioldm2::vae_decode(st, z.data(), h, w, got.data())) {
        std::fprintf(stderr, "the VAE device path failed\n");
        return 1;
    }
    double num = 0, den = 0, maxerr = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double)got[i] - ref[i];
        num += d * d; den += (double)ref[i] * ref[i];
        if (std::fabs(d) > maxerr) maxerr = std::fabs(d);
    }
    const double rel = std::sqrt(num / den);
    std::printf("VAE decoder, latent 8 x %d x %d -> mel 1 x %d x %d: rel L2 %.3e, max abs %.3e  %s\n", h, w, 4 * h, 4 * w, rel, maxerr,
                rel <= 1e-3 ? "ok" : "FAIL");
    return rel <= 1e-3 ? 0 : 1;
}
