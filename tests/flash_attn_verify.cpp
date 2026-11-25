// The fused fp32 attention (sd_device attention_f32_strided, sd_gpu.hip attn_flash_*) against a double-precision CPU reference on sampled
// rows: self-attention (D = 128, 12 heads), cross-attention with lengths that are not multiples of the tiles (Tq 500, Tk 1029), and D = 64
// (24 heads). Checks the largest error relative to the largest output. Usage: flash_attn_verify
#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// one case: Tq queries, Tk keys, C = heads * D; checks rows sampled from every head
static bool check_case(const char* name, int Tq, int Tk, int C, int heads) {
    const int D = C / heads;
    const size_t qbytes = sizeof(float) * (size_t)Tq * C, kbytes = sizeof(float) * (size_t)Tk * C;
    float* q = static_cast<float*>(sddev::alloc(qbytes, false));
    float* k = static_cast<float*>(sddev::alloc(kbytes, false));
    float* v = static_cast<float*>(sddev::alloc(kbytes, false));
    float* out = static_cast<float*>(sddev::alloc(qbytes, true));
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (size_t i = 0; i < (size_t)Tq * C; ++i) q[i] = u(rng);
    for (size_t i = 0; i < (size_t)Tk * C; ++i) {
        k[i] = u(rng);
        v[i] = u(rng);
    }
    sddev::attention_f32_strided(q, C, Tq, k, C, v, C, Tk, C, heads, out);
    const double scale = 1.0 / std::sqrt((double)D);
    double max_err = 0, max_val = 0;
    std::mt19937 pick(5);
    std::vector<double> sc(Tk);
    for (int sample = 0; sample < 48; ++sample) {
        const int r = sample < 8 ? sample * (Tq / 8) : (int)(pick() % (unsigned)Tq);
        const int h = (int)(pick() % (unsigned)heads);
        double mx = -1e300;
        for (int j = 0; j < Tk; ++j) {
            double s = 0;
            for (int d = 0; d < D; ++d) s += (double)q[(size_t)r * C + h * D + d] * (double)k[(size_t)j * C + h * D + d];
            sc[j] = s * scale;
            if (sc[j] > mx) mx = sc[j];
        }
        double den = 0;
        for (int j = 0; j < Tk; ++j) {
            sc[j] = std::exp(sc[j] - mx);
            den += sc[j];
        }
        for (int d = 0; d < D; ++d) {
            double acc = 0;
            for (int j = 0; j < Tk; ++j) acc += sc[j] * (double)v[(size_t)j * C + h * D + d];
            acc /= den;
            const double err = std::fabs(acc - (double)out[(size_t)r * C + h * D + d]);
            if (err > max_err) max_err = err;
            if (std::fabs(acc) > max_val) max_val = std::fabs(acc);
        }
    }
    const double rel = max_err / (max_val > 0 ? max_val : 1.0);
    const bool ok = rel < 1e-4;
    std::printf("%-24s Tq %5d Tk %5d D %3d heads %2d: max error %.3e, relative %.3e: %s\n", name, Tq, Tk, D, heads, max_err, rel,
                ok ? "ok" : "BAD");
    sddev::release(q, qbytes);
    sddev::release(k, kbytes);
    sddev::release(v, kbytes);
    sddev::release(out, qbytes);
    return ok;
}

int main() {
    if (!sddev::enabled()) {
        std::fprintf(stderr, "no device\n");
        return 1;
    }
    bool ok = true;
    ok = check_case("self-attention", 3000, 3000, 1536, 12) && ok;
    ok = check_case("cross-attention", 500, 1029, 1536, 12) && ok;
    ok = check_case("self-attention, D 64", 2000, 2000, 1536, 24) && ok;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
