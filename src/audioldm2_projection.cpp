#include "audioldm2_projection.h"

#include "sd_device.h"

#include <cstdio>
#include <cstring>

namespace audioldm2 {
namespace {

constexpr int kDim = 768, kClap = 512, kT5 = 1024;

bool tensor(const zimage::SafeTensors& st, const char* name, size_t n, const float*& p) {
    std::vector<size_t> shape;
    if (!st.f32_in_place(name, p, shape)) {
        std::fprintf(stderr, "audioldm2: %s missing or not F32\n", name);
        return false;
    }
    size_t count = 1;
    for (size_t d : shape) count *= d;
    if (count != n) {
        std::fprintf(stderr, "audioldm2: %s has %zu values, not %zu\n", name, count, n);
        return false;
    }
    return true;
}

}  // namespace

bool projection_forward(const zimage::SafeTensors& st, const float* clap, int Tc, const float* t5, int Tt, const int32_t* t5_mask, float* out,
                        int32_t* out_mask) {
    if (!sddev::enabled()) { std::fprintf(stderr, "audioldm2: no device\n"); return false; }
    if (Tc < 1 || Tt < 1) { std::fprintf(stderr, "audioldm2: empty text embeddings\n"); return false; }
    const float *wp, *bp, *wp1, *bp1, *sos, *eos, *sos1, *eos1;
    if (!tensor(st, "projection.weight", (size_t)kDim * kClap, wp) || !tensor(st, "projection.bias", kDim, bp) ||
        !tensor(st, "projection_1.weight", (size_t)kDim * kT5, wp1) || !tensor(st, "projection_1.bias", kDim, bp1) ||
        !tensor(st, "sos_embed", kDim, sos) || !tensor(st, "eos_embed", kDim, eos) ||
        !tensor(st, "sos_embed_1", kDim, sos1) || !tensor(st, "eos_embed_1", kDim, eos1))
        return false;
    sddev::Floats Xc((size_t)Tc * kClap), Xt((size_t)Tt * kT5), Yc((size_t)Tc * kDim), Yt((size_t)Tt * kDim);
    std::memcpy(Xc.data(), clap, sizeof(float) * (size_t)Tc * kClap);
    std::memcpy(Xt.data(), t5, sizeof(float) * (size_t)Tt * kT5);
    /* the linears (PyTorch layout: W [out][in]) on the device */
    sddev::linear_f32(Xc.data(), Tc, wp, kClap, kDim, bp, Yc.data());
    sddev::linear_f32(Xt.data(), Tt, wp1, kT5, kDim, bp1, Yt.data());
    /* sos, the CLAP rows, eos, then the same for the T5 sequence (its padding rows stay, masked in the mask) */
    float* o = out;
    std::memcpy(o, sos, sizeof(float) * kDim); o += kDim;
    std::memcpy(o, Yc.data(), sizeof(float) * (size_t)Tc * kDim); o += (size_t)Tc * kDim;
    std::memcpy(o, eos, sizeof(float) * kDim); o += kDim;
    std::memcpy(o, sos1, sizeof(float) * kDim); o += kDim;
    std::memcpy(o, Yt.data(), sizeof(float) * (size_t)Tt * kDim); o += (size_t)Tt * kDim;
    std::memcpy(o, eos1, sizeof(float) * kDim);
    /* the mask: ones for the special tokens and the CLAP row, the T5 mask for its rows */
    if (out_mask) {
        int32_t* m = out_mask;
        *m++ = 1;
        for (int c = 0; c < Tc; ++c) *m++ = 1;
        *m++ = 1;
        *m++ = 1;
        for (int t = 0; t < Tt; ++t) *m++ = t5_mask ? t5_mask[t] : 1;
        *m++ = 1;
    }
    return true;
}

}  // namespace audioldm2
