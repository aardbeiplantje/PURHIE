/* Qwen-Image's sampler (qwenimage_pipe.h): the schedule computed as FlowMatchEulerDiscreteScheduler does, the model's velocity from
 * the device DiT, and the Euler step x += (sigma_next - sigma) v. The state is kept on the host (small: tokens x 64). */
#include "qwenimage_pipe.h"

#include "qwenimage_dit.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace qwenimage {

namespace {

/* the nearest BF16 value (round to nearest even on the 16 high bits of the float) */
float bf16_round(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}

}  // namespace

void sigmas_for(int steps, int image_tokens, std::vector<float>& sig) {
    /* calculate_shift: mu = image_tokens * m + b with m = (0.9 - 0.5) / (8192 - 256) and b = 0.5 - 256 m */
    const double m = (0.9 - 0.5) / (8192.0 - 256.0), b = 0.5 - 256.0 * m;
    const double mu = image_tokens * m + b, e = std::exp(mu);
    std::vector<double> s((size_t)steps);
    for (int i = 0; i < steps; ++i) {
        const double base = steps > 1 ? 1.0 + i * (1.0 / steps - 1.0) / (steps - 1) : 1.0;  /* linspace(1, 1/steps, steps) */
        s[i] = e / (e + (1.0 / base - 1.0));                                                  /* the exponential time shift (sigma 1) */
    }
    /* the stretch to shift_terminal = 0.02: the last one-minus-sigma maps to 1 - 0.02 */
    const double scale = (1.0 - s[steps - 1]) / (1.0 - 0.02);
    for (int i = 0; i < steps; ++i) s[i] = 1.0 - (1.0 - s[i]) / scale;
    sig.assign(steps + 1, 0.0f);
    for (int i = 0; i < steps; ++i) sig[i] = (float)s[i];
    sig[steps] = 0.0f;
}

bool dev_sample(const zimage::SafeTensors& dit, const float* noise, int gh, int gw, const float* enc, int L, int steps, float* out,
                bool bf16_timestep) {
    const int N = gh * gw;
    std::vector<float> sig, x((size_t)N * 64), v((size_t)N * 64);
    sigmas_for(steps, N, sig);
    for (size_t i = 0; i < x.size(); ++i) x[i] = noise[i];
    for (int i = 0; i < steps; ++i) {
        /* the pipeline's timestep: sigma * 1000 held in the model's BF16, then divided by 1000 in BF16 (the model gets it in BF16) */
        const float t1000 = bf16_round(sig[i] * 1000.0f);
        const float tmodel = bf16_timestep ? bf16_round(t1000 / 1000.0f) : sig[i];
        if (!dev_dit_forward(dit, x.data(), gh, gw, enc, L, tmodel, v.data())) return false;
        const float dt = sig[i + 1] - sig[i];
        for (size_t k = 0; k < x.size(); ++k) x[k] += dt * v[k];
    }
    for (size_t k = 0; k < x.size(); ++k) out[k] = x[k];
    return true;
}

void unpack(const float* packed, int gh, int gw, float* lat) {
    /* packed[(h2 * gw + w2)][c * 4 + ph * 2 + pw] -> lat[c][h2 * 2 + ph][w2 * 2 + pw] */
    const int H = 2 * gh, W = 2 * gw;
    for (int c = 0; c < 16; ++c)
        for (int h2 = 0; h2 < gh; ++h2)
            for (int w2 = 0; w2 < gw; ++w2)
                for (int ph = 0; ph < 2; ++ph)
                    for (int pw = 0; pw < 2; ++pw)
                        lat[(size_t)c * H * W + (h2 * 2 + ph) * W + (w2 * 2 + pw)] =
                            packed[((size_t)h2 * gw + w2) * 64 + c * 4 + ph * 2 + pw];
}

}  // namespace qwenimage
