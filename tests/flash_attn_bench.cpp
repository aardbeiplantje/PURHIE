// The fused fp32 attention (sd_device attention_f32_strided) on random tokens: n tokens, C = 1536, 12 heads of 128 (the shape-SLAT
// flow's self-attention). Prints the time per call and a checksum. Set SD_ATTN_FLASH=0 for the GEMM path; SD_TIME=1 for per-kernel times.
// Usage: flash_attn_bench [n] [repeats]
#include "sd_device.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 8192;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 2;
    const int C = 1536, H = 12;
    if (!sddev::enabled()) {
        std::fprintf(stderr, "no device\n");
        return 1;
    }
    const size_t bytes = sizeof(float) * (size_t)n * C;
    float* q = static_cast<float*>(sddev::alloc(bytes, false));
    float* k = static_cast<float*>(sddev::alloc(bytes, false));
    float* v = static_cast<float*>(sddev::alloc(bytes, false));
    float* out = static_cast<float*>(sddev::alloc(bytes, true));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (size_t i = 0; i < (size_t)n * C; ++i) {
        q[i] = u(rng);
        k[i] = u(rng);
        v[i] = u(rng);
    }
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        sddev::attention_f32_strided(q, C, n, k, C, v, C, n, C, H, out);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s < best) best = s;
        std::printf("call %d: %.3f s\n", r, s);
    }
    double sum = 0, asum = 0;
    for (size_t i = 0; i < (size_t)n * C; ++i) {
        sum += out[i];
        asum += std::fabs(out[i]);
    }
    std::printf("n %d: best %.3f s, checksum %.6e, |.| %.6e\n", n, best, sum, asum);
    sddev::release(q, bytes);
    sddev::release(k, bytes);
    sddev::release(v, bytes);
    sddev::release(out, bytes);
    return 0;
}
