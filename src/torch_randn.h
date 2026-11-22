/* torch.randn on the CPU (the generator of torch.manual_seed(seed), float32): the initial noise of a latent is the same as the
 * one ComfyUI draws for the same seed. The Mersenne Twister (mt19937, seeded with the seed's low 32 bits); each uniform is the
 * low 24 bits of a 32-bit draw times 2^-24; for n >= 16 the normals are the Box-Muller pairs of 16-value blocks (the first 8
 * values' radii with the last 8 values' angles), the last block recomputed when n is not a multiple of 16 (torch's normal_fill).
 * Reference-only host code (the noise is written once per image). */
#ifndef TORCH_RANDN_H
#define TORCH_RANDN_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

inline void torch_randn(uint64_t seed, float* out, size_t n) {
    std::mt19937 gen(static_cast<uint32_t>(seed));
    auto uniform = [&gen]() -> float {
        const uint32_t r = gen();
        return static_cast<float>(r & ((1u << 24) - 1)) * (1.0f / 16777216.0f);
    };
    for (size_t i = 0; i < n; ++i) out[i] = uniform();
    if (n < 16) {  /* torch's small case: Box-Muller on the pairs, in order */
        for (size_t i = 0; i < n; i += 2) {
            const float u1 = 1.0f - out[i];
            const float u2 = i + 1 < n ? out[i + 1] : uniform();
            const float radius = std::sqrt(-2.0f * std::log(u1));
            const float theta = 2.0f * 3.14159265358979323846f * u2;
            out[i] = radius * std::cos(theta);
            if (i + 1 < n) out[i + 1] = radius * std::sin(theta);
        }
        return;
    }
    auto block16 = [](float* d) {
        for (int j = 0; j < 8; ++j) {
            const float u1 = 1.0f - d[j];
            const float u2 = d[j + 8];
            const float radius = std::sqrt(-2.0f * std::log(u1));
            const float theta = 2.0f * 3.14159265358979323846f * u2;
            d[j] = radius * std::cos(theta);
            d[j + 8] = radius * std::sin(theta);
        }
    };
    for (size_t i = 0; i + 16 <= n; i += 16) block16(out + i);
    if (n % 16 != 0) {  /* the last 16 values are drawn again and normalized as a block */
        float* d = out + n - 16;
        for (int i = 0; i < 16; ++i) d[i] = uniform();
        block16(d);
    }
}

#endif /* TORCH_RANDN_H */
