// mm_vae_verify: MiniMax-H3-Image's VAE decoder in the engine (src/minimax_vae.cpp, PLAN.md M2) against diffusers 0.43.0's
// AutoencoderKLMiniMaxH3Image (float32 on the CPU; /data/minimax-ref/vae, mm_vae_ref.py): a 16 x 16 latent (one 256-px
// tile) and a 40 x 40 latent (three tiles per axis, blended), both with the reference's tiled decode.
//
//   mm_vae_verify <vae-decoder.gguf> <reference dir>
#include "minimax_vae.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool read_f32(const std::string& path, std::vector<float>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)n / sizeof(float));
    const size_t got = std::fread(v.data(), 1, (size_t)n, f);
    std::fclose(f);
    return got == (size_t)n;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <vae-decoder.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    mmvae_t* m = mmvae_open(argv[1]);
    if (!m) {
        std::fprintf(stderr, "open failed: %s\n", mmvae_last_error());
        return 1;
    }
    int failures = 0;
    for (const char* name : {"small", "tiled"}) {
        std::vector<float> z, want;
        if (!read_f32(dir + "/" + name + "_z.f32", z) || !read_f32(dir + "/" + name + "_out.f32", want)) {
            std::fprintf(stderr, "cannot read the %s reference in %s\n", name, dir.c_str());
            return 2;
        }
        const int hw = (int)std::lround(std::sqrt((double)z.size() / 24.0));
        std::vector<float> got(want.size());
        if (mmvae_decode(m, z.data(), hw, hw, got.data()) != 0) {
            std::fprintf(stderr, "decode failed: %s\n", mmvae_last_error());
            return 1;
        }
        double dot = 0, na = 0, nb = 0, peak = 0, worst = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            dot += (double)got[i] * want[i];
            na += (double)got[i] * got[i];
            nb += (double)want[i] * want[i];
            if (std::fabs(want[i]) > peak) peak = std::fabs(want[i]);
            const double d = std::fabs((double)got[i] - want[i]);
            if (d > worst) worst = d;
        }
        const double cos = dot / std::sqrt(na * nb);
        const bool ok = cos > 0.99999 && worst < 1e-3 * peak;
        if (!ok) ++failures;
        std::printf("%s: latent %d x %d -> image %d x %d: cosine %.7f, max |diff| %.3e of the peak %.2f  %s\n", name, hw, hw,
                    16 * hw, 16 * hw, cos, worst, peak, ok ? "ok" : "FAIL");
    }
    mmvae_close(m);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
