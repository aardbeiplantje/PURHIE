// sd_vae_verify: the Stable Diffusion engine's VAE decoder (src/sd.cpp, PLAN.md I3) against diffusers' AutoencoderKL decode
// (sd-turbo, float32) for the reference latents of the one-step run: the decoded planes [3][512][512] (the float output, not
// clamped). Reference directory (/data/image-ref/sd-turbo): latents_after_step.f32 (4 x 64 x 64), vae_out.f32 (3 x 512 x 512).
// Passes at a largest difference of 1e-3 of the output's largest magnitude and a cosine of at least 0.99999.
//
//   sd_vae_verify <sd.gguf> <reference dir>
#include "sd.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> x(b.size() / 4);
    std::memcpy(x.data(), b.data(), x.size() * 4);
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <sd.gguf> <reference dir>\n", argv[0]); return 2; }
    const std::string dir = argv[2];
    const std::vector<float> lat = read_f32(dir + "/latents_after_step.f32");
    const std::vector<float> want = read_f32(dir + "/vae_out.f32");
    sd_model_t* m = sd_open(argv[1]);
    if (!m) { std::fprintf(stderr, "sd_open: %s\n", sd_last_error()); return 1; }
    std::vector<float> got(want.size());
    auto t0 = std::chrono::steady_clock::now();
    if (sd_vae_decode(m, lat.data(), 64, 64, got.data()) != 0) {
        std::fprintf(stderr, "sd_vae_decode: %s\n", sd_last_error());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double peak = 0.0, worst = 0.0, dot = 0.0, gg = 0.0, ww = 0.0;
    size_t worst_at = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        peak = std::fmax(peak, std::fabs((double)want[i]));
        const double d = std::fabs((double)got[i] - want[i]);
        if (d > worst) { worst = d; worst_at = i; }
        dot += (double)got[i] * want[i];
        gg += (double)got[i] * got[i];
        ww += (double)want[i] * want[i];
    }
    const double cos = dot / std::sqrt(gg * ww);
    const bool ok = worst / peak <= 1e-3 && cos >= 0.99999;
    std::printf("VAE decode 3 x 512 x 512: cosine %.7f, max |diff| %.3e (at %zu) = %.2e of the peak %.2f, %.2f s  %s\n", cos,
                worst, worst_at, worst / peak, peak, std::chrono::duration<double>(t1 - t0).count(), ok ? "ok" : "FAIL");
    sd_close(m);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
