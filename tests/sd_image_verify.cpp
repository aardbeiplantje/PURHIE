// sd_image_verify: the Stable Diffusion engine end to end (src/sd.cpp, PLAN.md I5): the reference run's image (diffusers
// 0.35.2, sd-turbo, one step, seed 0, "a photo of a red fox in the snow", 512 x 512, float32). The pipeline was given the
// initial latents noise * sigma and scaled them once more inside (its prepare_latents), so the engine gets the same
// initial latents (noise * sigma * sigma) and must give the same pixels: at most 1 of 255 steps off, and most equal. The
// PNG the engine writes must decode (checked by the caller: python's PIL against image_u8.bin).
//
//   sd_image_verify <sd.gguf> <reference dir> [out.png]
#include "sd.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T> x(b.size() / sizeof(T));
    std::memcpy(x.data(), b.data(), x.size() * sizeof(T));
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) { std::fprintf(stderr, "usage: %s <sd.gguf> <reference dir> [out.png]\n", argv[0]); return 2; }
    const std::string dir = argv[2];
    const std::vector<float> noise = read_raw<float>(dir + "/noise.f32");
    const std::vector<uint8_t> want = read_raw<uint8_t>(dir + "/image_u8.bin");
    sd_model_t* m = sd_open(argv[1]);
    if (!m) { std::fprintf(stderr, "sd_open: %s\n", sd_last_error()); return 1; }
    float sigma = 0.0f;
    sd_init_sigma(m, &sigma);
    const bool sigma_ok = std::fabs(sigma - 14.614646911621094f) < 1e-3f;
    std::printf("sigma %.6f (the scheduler's 14.614647) %s\n", sigma, sigma_ok ? "ok" : "FAIL");
    std::vector<float> init(noise.size());
    for (size_t i = 0; i < noise.size(); ++i) init[i] = noise[i] * sigma * sigma;
    std::vector<uint8_t> got(want.size());
    if (sd_generate(m, "a photo of a red fox in the snow", init.data(), 512, 512, got.data()) != 0) {
        std::fprintf(stderr, "sd_generate: %s\n", sd_last_error());
        return 1;
    }
    size_t equal = 0, worst = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const int d = std::abs((int)got[i] - (int)want[i]);
        equal += d == 0;
        if ((size_t)d > worst) worst = (size_t)d;
    }
    const double frac = (double)equal / want.size();
    const bool ok = sigma_ok && worst <= 1 && frac >= 0.99;
    std::printf("image 512 x 512 x 3 against the pipeline's: %.4f%% equal, largest step %zu  %s\n", 100.0 * frac, worst,
                ok ? "ok" : "FAIL");
    if (argc == 4) {
        uint8_t* png = nullptr;
        size_t n = 0;
        if (sd_encode_png(got.data(), 512, 512, &png, &n) != 0) {
            std::fprintf(stderr, "sd_encode_png: %s\n", sd_last_error());
            return 1;
        }
        std::ofstream f(argv[3], std::ios::binary);
        f.write(reinterpret_cast<const char*>(png), (std::streamsize)n);
        sd_free_png(png);
        std::printf("wrote %s (%zu bytes)\n", argv[3], n);
    }
    sd_close(m);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
