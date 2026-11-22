/* The engine's BF16 linear layers (sddev::linear_bf16) at the Z-Image DiT's shapes (T = 4128 tokens at 1024 x 1024): the time
 * per call (the best of the repeats), the TFLOPS, and the error of sampled outputs against a CPU dot product (the activations
 * rounded to BF16 as the kernels do). Usage: sd_gemm_bench [repeats]. Needs the device path (the default). */
#include "sd_device.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

uint16_t f2bf_host(float f) {  // round to nearest, ties to even (finite inputs)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

float bf2f_host(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 3;
    if (!sddev::enabled()) {
        std::printf("[FAIL] the device path is off\n");
        return 1;
    }
    struct Shape {
        const char* name;
        int in, out, T;
    };
    const Shape shapes[] = {
        {"qkv and out   (3840 -> 3840)", 3840, 3840, 4128},
        {"mlp up        (3840 -> 10240)", 3840, 10240, 4128},
        {"mlp down      (10240 -> 3840)", 10240, 3840, 4128},
    };
    unsigned seed = 1;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (float)(seed >> 8) / 16777216.0f - 0.5f;
    };
    int rc = 0;
    for (const Shape& s : shapes) {
        const size_t nw = (size_t)s.out * s.in, nx = (size_t)s.T * s.in, ny = (size_t)s.T * s.out;
        uint16_t* W = static_cast<uint16_t*>(sddev::alloc(2 * nw, false));
        float* x = static_cast<float*>(sddev::alloc(4 * nx, false));
        float* y = static_cast<float*>(sddev::alloc(4 * ny, false));
        for (size_t i = 0; i < nw; ++i) W[i] = f2bf_host(0.1f * rnd());
        for (size_t i = 0; i < nx; ++i) x[i] = rnd();
        sddev::linear_bf16(x, s.T, W, s.in, s.out, nullptr, y);  /* warm-up: the kernel load and the first call */
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            sddev::linear_bf16(x, s.T, W, s.in, s.out, nullptr, y);
            best = std::fmin(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        /* sampled outputs against the CPU dot product (the activations rounded to BF16, the products in double) */
        double maxerr = 0, maxref = 0;
        for (int q = 0; q < 256; ++q) {
            const int t = (q * 977) % s.T, o = (q * 1237) % s.out;
            double ref = 0;
            for (int i = 0; i < s.in; ++i)
                ref += (double)bf2f_host(f2bf_host(x[(size_t)t * s.in + i])) * (double)bf2f_host(W[(size_t)o * s.in + i]);
            maxerr = std::fmax(maxerr, std::fabs((double)y[(size_t)t * s.out + o] - ref));
            maxref = std::fmax(maxref, std::fabs(ref));
        }
        const double tflops = 2.0 * s.T * s.in * s.out / best / 1e12;
        const bool ok = maxerr <= 1e-3 * maxref;
        std::printf("%s: %8.2f ms  %6.2f TFLOPS  max err %.2e of max %.2e  %s\n", s.name, best * 1e3, tflops, maxerr, maxref,
                    ok ? "ok" : "FAIL");
        if (!ok) rc = 1;
    }
    return rc;
}
