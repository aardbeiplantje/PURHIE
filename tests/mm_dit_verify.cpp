// mm_dit_verify: MiniMax-H3-Image's DiT in the engine (src/minimax_dit.cpp, PLAN.md M3) against diffusers 0.43.0's
// transformer in bf16 on the GPU (mm_dit_ref.py, /data/minimax-ref/dit): the velocity of a 16 x 16 latent with the 9-token
// prompt, at sampling steps 0 and 5 (the step's PDD head), and the sigmas of the schedule.
//
//   mm_dit_verify <dit.gguf> <pdd-heads.gguf> <reference dir>
#include "minimax_dit.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <dit.gguf> <pdd-heads.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[3];
    mmdit_t* m = mmdit_open(argv[1], argv[2]);
    if (!m) {
        std::fprintf(stderr, "open failed: %s\n", mmdit_last_error());
        return 1;
    }
    const int h = 16, w = 16, T = 9;
    std::vector<float> x, text;
    if (!read_f32(dir + "/sample.f32", x) || !read_f32(dir + "/text.f32", text) || x.size() != 24 * h * w || text.size() != (size_t)T * 5120) {
        std::fprintf(stderr, "cannot read the inputs in %s\n", dir.c_str());
        return 2;
    }
    int failures = 0;
    const float want_sigma[] = {1.0f, 0.9675237f, 0.8875485f, 0.75f, 0.5327739f, 0.2320089f, 0.0f};
    for (int i = 0; i <= 6; ++i) {
        const float s = mmdit_sigma(m, i);
        if (std::fabs(s - want_sigma[i]) > 1e-6f) ++failures;
    }
    std::printf("sigmas: %s\n", failures ? "FAIL" : "ok");
    for (int step : {0, 5}) {
        std::vector<float> want;
        if (!read_f32(dir + "/velocity_step" + std::to_string(step) + ".f32", want)) {
            std::fprintf(stderr, "cannot read velocity_step%d\n", step);
            return 2;
        }
        std::vector<float> got(want.size());
        if (mmdit_velocity(m, text.data(), T, x.data(), h, w, step, got.data()) != 0) {
            std::fprintf(stderr, "velocity failed: %s\n", mmdit_last_error());
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
        // bf16 weights in the reference, its activations bf16 too: the engine's fp32 keeps closer; a cosine of 0.999 is the bar
        const bool ok = cos > 0.999;
        if (!ok) ++failures;
        std::printf("velocity step %d: cosine %.7f, max |diff| %.3e of the peak %.3f  %s\n", step, cos, worst, peak,
                    ok ? "ok" : "FAIL");
    }
    {  // the whole sampler: six steps from the same noise against the reference's sample_final
        std::vector<float> want, got(x.size());
        if (!read_f32(dir + "/sample_final.f32", want)) {
            std::fprintf(stderr, "cannot read sample_final in %s\n", dir.c_str());
            return 2;
        }
        if (mmdit_sample(m, text.data(), T, x.data(), h, w, got.data()) != 0) {
            std::fprintf(stderr, "sample failed: %s\n", mmdit_last_error());
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
        const bool ok = cos > 0.999;
        if (!ok) ++failures;
        std::printf("sampler (6 steps): cosine %.7f, max |diff| %.3e of the peak %.3f  %s\n", cos, worst, peak, ok ? "ok" : "FAIL");
    }
    if (std::getenv("MM_DIT_TIME")) {  // a 1024 x 1024 canvas (latent 64 x 64: 1024 image tokens), one velocity and the sampler
        const int H2 = 64, W2 = 64;
        std::vector<float> big((size_t)24 * H2 * W2, 0.1f), v((size_t)24 * H2 * W2);
        auto t0 = std::chrono::steady_clock::now();
        const int rc = mmdit_velocity(m, text.data(), T, big.data(), H2, W2, 0, v.data());
        auto t1 = std::chrono::steady_clock::now();
        std::printf("1024 x 1024 velocity: %s, %.2f s\n", rc == 0 ? "ok" : mmdit_last_error(),
                    std::chrono::duration<double>(t1 - t0).count());
    }
    mmdit_close(m);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
