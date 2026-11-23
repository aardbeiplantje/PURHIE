#include "audioldm2_t5.h"

#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace audioldm2 {
namespace {

constexpr int kDim = 1024, kHeads = 16, kFF = 2816, kLayers = 24, kVocab = 32128, kBuckets = 32, kMaxDist = 128;
constexpr float kEps = 1e-6f;

bool tensor(const zimage::SafeTensors& st, const std::string& name, size_t n, const float*& p) {
    std::vector<size_t> shape;
    if (!st.f32_in_place(name, p, shape)) {
        std::fprintf(stderr, "audioldm2: %s missing or not F32\n", name.c_str());
        return false;
    }
    size_t count = 1;
    for (size_t d : shape) count *= d;
    if (count != n) {
        std::fprintf(stderr, "audioldm2: %s has %zu values, not %zu\n", name.c_str(), count, n);
        return false;
    }
    return true;
}

/* T5's bidirectional relative position bucket of (key - query) */
int bucket(int rel) {
    const int nb = kBuckets / 2, max_exact = nb / 2;
    int ret = rel > 0 ? nb : 0;
    const int n = rel < 0 ? -rel : rel;
    if (n < max_exact) return ret + n;
    const int large = max_exact + (int)(std::log((double)n / max_exact) / std::log((double)kMaxDist / max_exact) * (nb - max_exact));
    return ret + (large < nb - 1 ? large : nb - 1);
}

}  // namespace

bool t5_encoder_forward(const zimage::SafeTensors& st, const int32_t* ids, const int32_t* mask, int T, float* hidden) {
    if (!sddev::enabled()) { std::fprintf(stderr, "audioldm2: no device\n"); return false; }
    if (T < 1 || T > 512) { std::fprintf(stderr, "audioldm2: the T5 encoder takes 1 to 512 tokens, not %d\n", T); return false; }
    const float *shared, *bias_table, *ln_final;
    if (!tensor(st, "shared.weight", (size_t)kVocab * kDim, shared) ||
        !tensor(st, "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", (size_t)kBuckets * kHeads, bias_table) ||
        !tensor(st, "encoder.final_layer_norm.weight", kDim, ln_final))
        return false;
    const size_t n = (size_t)T * kDim;
    sddev::Floats X(n), Hb(n), Q(n), K(n), V(n), A(n), O(n), G0((size_t)T * kFF), G1((size_t)T * kFF), Z((size_t)kFF);
    /* the embedding rows (host), and the relative position bias [H][T][T] from the layer-0 table */
    for (int t = 0; t < T; ++t) {
        if (ids[t] < 0 || ids[t] >= kVocab) { std::fprintf(stderr, "audioldm2: token id %d out of range\n", ids[t]); return false; }
        std::memcpy(X.data() + (size_t)t * kDim, shared + (size_t)ids[t] * kDim, sizeof(float) * kDim);
    }
    /* the padding keys (mask 0) are masked: the device reads the mask as floats */
    sddev::Floats KM((size_t)T);
    for (int t = 0; t < T; ++t) KM.data()[t] = mask ? (mask[t] ? 1.0f : 0.0f) : 1.0f;
    sddev::Floats BIAS((size_t)kHeads * T * T);
    for (int h = 0; h < kHeads; ++h)
        for (int i = 0; i < T; ++i)
            for (int j = 0; j < T; ++j)
                BIAS.data()[((size_t)h * T + i) * T + j] = bias_table[(size_t)bucket(j - i) * kHeads + h];
    for (int L = 0; L < kLayers; ++L) {
        const std::string p = "encoder.block." + std::to_string(L) + ".";
        const float *ln0, *wq, *wk, *wv, *wo, *ln1, *wi0, *wi1, *wdn;
        if (!tensor(st, p + "layer.0.layer_norm.weight", kDim, ln0) || !tensor(st, p + "layer.0.SelfAttention.q.weight", (size_t)kDim * kDim, wq) ||
            !tensor(st, p + "layer.0.SelfAttention.k.weight", (size_t)kDim * kDim, wk) || !tensor(st, p + "layer.0.SelfAttention.v.weight", (size_t)kDim * kDim, wv) ||
            !tensor(st, p + "layer.0.SelfAttention.o.weight", (size_t)kDim * kDim, wo) || !tensor(st, p + "layer.1.layer_norm.weight", kDim, ln1) ||
            !tensor(st, p + "layer.1.DenseReluDense.wi_0.weight", (size_t)kFF * kDim, wi0) ||
            !tensor(st, p + "layer.1.DenseReluDense.wi_1.weight", (size_t)kFF * kDim, wi1) ||
            !tensor(st, p + "layer.1.DenseReluDense.wo.weight", (size_t)kDim * kFF, wdn))
            return false;
        /* self-attention: RMS norm, q k v (no bias), the unscaled scores + the position bias, the output projection, the residual */
        sddev::audio_rms_norm_f32(X.data(), T, kDim, ln0, kEps, Hb.data());
        sddev::linear_f32(Hb.data(), T, wq, kDim, kDim, Z.data(), Q.data());
        sddev::linear_f32(Hb.data(), T, wk, kDim, kDim, Z.data(), K.data());
        sddev::linear_f32(Hb.data(), T, wv, kDim, kDim, Z.data(), V.data());
        sddev::audio_attention_f32(Q.data(), K.data(), V.data(), kDim, T, T, kHeads, 64, false, KM.data(), 1.0f, BIAS.data(), A.data());
        sddev::linear_f32(A.data(), T, wo, kDim, kDim, Z.data(), O.data());
        sddev::add(X.data(), O.data(), n);
        /* the feed-forward: RMS norm, gated GELU (wi_0 and wi_1), wo, the residual */
        sddev::audio_rms_norm_f32(X.data(), T, kDim, ln1, kEps, Hb.data());
        sddev::linear_f32(Hb.data(), T, wi0, kDim, kFF, Z.data(), G0.data());
        sddev::linear_f32(Hb.data(), T, wi1, kDim, kFF, Z.data(), G1.data());
        sddev::audio_gelu_tanh_mul(G0.data(), G1.data(), (size_t)T * kFF);
        sddev::linear_f32(G0.data(), T, wdn, kFF, kDim, Z.data(), O.data());
        sddev::add(X.data(), O.data(), n);
    }
    sddev::audio_rms_norm_f32(X.data(), T, kDim, ln_final, kEps, Hb.data());
    std::memcpy(hidden, Hb.data(), sizeof(float) * n);
    return true;
}

}  // namespace audioldm2
