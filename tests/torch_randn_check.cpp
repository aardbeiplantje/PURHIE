/* torch_randn (src/torch_randn.h) against torch.randn after manual_seed(42) on the CPU, shape (1, 16, 128, 128) (the reference
 * file /data/zimage-ref/z1024/noise_comfy42.f32, written by torch). PASS when the largest difference is 1e-6 or less (the
 * reference is float32 torch's normal_fill; the differences are last-bit rounding of log, sqrt and the trigonometric functions).
 * Build and run: g++ -O2 -std=c++17 -Isrc tests/torch_randn_check.cpp -o torch_randn_check && ./torch_randn_check */
#include "torch_randn.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const size_t n = 16 * 128 * 128;
    std::vector<float> mine(n), ref(n);
    torch_randn(42, mine.data(), n);
    FILE* f = std::fopen("/data/zimage-ref/z1024/noise_comfy42.f32", "rb");
    if (!f || std::fread(ref.data(), sizeof(float), n, f) != n) {
        std::printf("[FAIL] the reference noise (/data/zimage-ref/z1024/noise_comfy42.f32)\n");
        return 1;
    }
    std::fclose(f);
    double maxd = 0, sumd = 0;
    size_t exact = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double)mine[i] - ref[i]);
        if (d > maxd) maxd = d;
        sumd += d;
        exact += (mine[i] == ref[i]);
    }
    const bool ok = maxd <= 1e-6;
    std::printf("[%s] torch_randn(42) vs torch: max abs diff %.3g, mean %.3g, bit-exact %zu of %zu\n", ok ? "PASS" : "FAIL", maxd,
                sumd / n, exact, n);
    return ok ? 0 : 1;
}
