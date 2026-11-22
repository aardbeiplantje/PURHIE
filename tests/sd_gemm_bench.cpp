/* The engine's BF16 linear layers (sddev::linear_bf16) at the Z-Image DiT's shapes (T = 4128 tokens at 1024 x 1024): the time
 * per call (the best of the repeats), the TFLOPS, and the error of sampled outputs against a CPU dot product (the activations
 * rounded to BF16 as the kernels do). Usage: sd_gemm_bench [repeats]. Needs the device path (the default). */
#include "sd_device.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
        /* SD_BENCH_DEVICE=1: W, x and y in device memory (hipMalloc) instead of the engine's pinned host allocation */
        const bool devmem = std::getenv("SD_BENCH_DEVICE") != nullptr;
        /* SD_BENCH_REGISTERED=1: W as the engine has its weights: page-aligned memory registered read-only for the device
         * (sddev::register_weights, the safetensors mapping's path) */
        const bool regw = std::getenv("SD_BENCH_REGISTERED") != nullptr;
        uint16_t* W = nullptr;
        if (regw) {
            void* wp = std::aligned_alloc(4096, (2 * nw + 4095) & ~(size_t)4095);
            if (!wp || !sddev::register_weights(wp, 2 * nw)) {
                std::printf("[FAIL] the weights' registration\n");
                return 1;
            }
            W = static_cast<uint16_t*>(wp);
        } else {
            W = static_cast<uint16_t*>(devmem ? nullptr : sddev::alloc(2 * nw, false));
        }
        float* x = static_cast<float*>(devmem ? nullptr : sddev::alloc(4 * nx, false));
        float* y = static_cast<float*>(devmem ? nullptr : sddev::alloc(4 * ny, false));
        if (devmem) {
            void *dW, *dx, *dy;
            hipMalloc(&dW, 2 * nw);
            hipMalloc(&dx, 4 * nx);
            hipMalloc(&dy, 4 * ny);
            W = static_cast<uint16_t*>(dW);
            x = static_cast<float*>(dx);
            y = static_cast<float*>(dy);
        }
        std::vector<uint16_t> hW(nw);
        std::vector<float> hx(nx), hb(s.out);
        for (int o = 0; o < s.out; ++o) hb[o] = 0.5f * rnd();
        for (size_t i = 0; i < nw; ++i) hW[i] = f2bf_host(0.1f * rnd());
        for (size_t i = 0; i < nx; ++i) hx[i] = rnd();
        float* bias = static_cast<float*>(sddev::alloc(4 * (size_t)s.out, false));
        std::memcpy(bias, hb.data(), 4 * (size_t)s.out);
        if (devmem) {
            hipMemcpy(W, hW.data(), 2 * nw, hipMemcpyHostToDevice);
            hipMemcpy(x, hx.data(), 4 * nx, hipMemcpyHostToDevice);
        } else {
            std::memcpy(W, hW.data(), 2 * nw);
            std::memcpy(x, hx.data(), 4 * nx);
        }
        sddev::linear_bf16(x, s.T, W, s.in, s.out, bias, y);  /* warm-up: the kernel load and the first call */
        double best = 1e30;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            sddev::linear_bf16(x, s.T, W, s.in, s.out, bias, y);
            best = std::fmin(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        if (std::getenv("SD_BENCH_OUT16")) {
            /* the BF16 output (linear_bf16_out16): the input as BF16, the output rounded to BF16 by the library */
            uint16_t* xb = static_cast<uint16_t*>(sddev::alloc(2 * nx, false));
            uint16_t* y16 = static_cast<uint16_t*>(sddev::alloc(2 * ny, false));
            for (size_t i = 0; i < nx; ++i) xb[i] = f2bf_host(hx[i]);
            if (!sddev::linear_bf16_out16(xb, s.T, W, s.in, s.out, bias, y16)) {
                std::printf("%s: [FAIL] the BF16 output is not taken\n", s.name);
                return 1;
            }
            double best16 = 1e30;
            for (int r = 0; r < reps; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                sddev::linear_bf16_out16(xb, s.T, W, s.in, s.out, bias, y16);
                best16 = std::fmin(best16, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            double me = 0, mr = 0, signed_sum = 0;
            int same_rne = 0, total = 0;
            for (int q = 0; q < 256; ++q) {
                const int t = (q * 977) % s.T, o = (q * 1237) % s.out;
                double ref = hb[o];
                for (int i = 0; i < s.in; ++i)
                    ref += (double)bf2f_host(f2bf_host(hx[(size_t)t * s.in + i])) * (double)bf2f_host(hW[(size_t)o * s.in + i]);
                me = std::fmax(me, std::fabs((double)bf2f_host(y16[(size_t)t * s.out + o]) - ref));
                mr = std::fmax(mr, std::fabs(ref));
                signed_sum += (double)bf2f_host(y16[(size_t)t * s.out + o]) - ref;
                /* the library's bits against the round-to-nearest-even of the same fp32 value (in double: close, not exact) */
                same_rne += (y16[(size_t)t * s.out + o] == f2bf_host((float)ref));
                ++total;
            }
            std::printf("   mean signed error %.2e (bf16 ulp ~%.1e at this magnitude); bits equal to RNE of the reference: %d of %d\n",
                        signed_sum / total, mr / 256.0 / 128.0, same_rne, total);
            std::printf("%s: BF16 output %8.2f ms  %6.2f TFLOPS  max err %.2e of max %.2e (bf16 rounding: 4e-3)\n", s.name,
                        best16 * 1e3, 2.0 * s.T * s.in * s.out / best16 / 1e12, me, mr);
        }
        /* sampled outputs against the CPU dot product (the activations rounded to BF16, the products in double) */
        std::vector<float> hy(ny);
        if (devmem) hipMemcpy(hy.data(), y, 4 * ny, hipMemcpyDeviceToHost);
        else std::memcpy(hy.data(), y, 4 * ny);
        double maxerr = 0, maxref = 0;
        for (int q = 0; q < 256; ++q) {
            const int t = (q * 977) % s.T, o = (q * 1237) % s.out;
            double ref = hb[o];
            for (int i = 0; i < s.in; ++i)
                ref += (double)bf2f_host(f2bf_host(hx[(size_t)t * s.in + i])) * (double)bf2f_host(hW[(size_t)o * s.in + i]);
            maxerr = std::fmax(maxerr, std::fabs((double)hy[(size_t)t * s.out + o] - ref));
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
