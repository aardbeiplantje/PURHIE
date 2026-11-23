#include "audioldm2_clap.h"

#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace audioldm2 {
namespace {

constexpr int kDim = 768, kHeads = 12, kFF = 3072, kLayers = 12, kVocab = 50265, kPos = 514, kPad = 1, kProj = 512;
constexpr float kEps = 1e-12f;

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

}  // namespace

bool clap_text_forward(const zimage::SafeTensors& st, const int32_t* ids, const int32_t* mask, int T, float* hidden, float* pooled,
                       float* features) {
    if (!sddev::enabled()) { std::fprintf(stderr, "audioldm2: no device\n"); return false; }
    if (T < 1 || T > 512) { std::fprintf(stderr, "audioldm2: the CLAP text model takes 1 to 512 tokens, not %d\n", T); return false; }
    const std::string tm = "text_model.";
    const float *word, *posemb, *tt, *elnw, *elnb, *pw, *pb, *tp1w, *tp1b, *tp2w, *tp2b;
    if (!tensor(st, tm + "embeddings.word_embeddings.weight", (size_t)kVocab * kDim, word) ||
        !tensor(st, tm + "embeddings.position_embeddings.weight", (size_t)kPos * kDim, posemb) ||
        !tensor(st, tm + "embeddings.token_type_embeddings.weight", kDim, tt) ||
        !tensor(st, tm + "embeddings.LayerNorm.weight", kDim, elnw) || !tensor(st, tm + "embeddings.LayerNorm.bias", kDim, elnb) ||
        !tensor(st, tm + "pooler.dense.weight", (size_t)kDim * kDim, pw) || !tensor(st, tm + "pooler.dense.bias", kDim, pb) ||
        !tensor(st, "text_projection.linear1.weight", (size_t)kProj * kDim, tp1w) || !tensor(st, "text_projection.linear1.bias", kProj, tp1b) ||
        !tensor(st, "text_projection.linear2.weight", (size_t)kProj * kProj, tp2w) || !tensor(st, "text_projection.linear2.bias", kProj, tp2b))
        return false;
    const size_t n = (size_t)T * kDim;
    /* the embeddings: word + position (RoBERTa's position ids: the running count of the non-padding tokens, from padding_idx + 1)
     * + token type 0; the rows gathered from the mapping on the host */
    sddev::Floats X(n), Y(n), Q(n), K(n), V(n), A(n), O(n), Mk(T), I((size_t)T * kFF);
    int run = 0;
    for (int t = 0; t < T; ++t) {
        const int id = ids[t];
        if (id < 0 || id >= kVocab) { std::fprintf(stderr, "audioldm2: token id %d out of range\n", id); return false; }
        const int pos = (mask[t] ? ++run : 0) + kPad;
        Mk.data()[t] = mask[t] ? 1.0f : 0.0f;
        for (int c = 0; c < kDim; ++c) X.data()[(size_t)t * kDim + c] = word[(size_t)id * kDim + c] + posemb[(size_t)pos * kDim + c] + tt[c];
    }
    sddev::audio_layer_norm_f32(X.data(), T, kDim, elnw, elnb, kEps, Y.data());
    float* x = Y.data();      /* the current hidden state */
    float* xo = X.data();     /* the next one */
    for (int L = 0; L < kLayers; ++L) {
        const std::string p = tm + "encoder.layer." + std::to_string(L) + ".";
        const float *wq, *bq, *wk, *bk, *wv, *bv, *wo, *bo, *lnaw, *lnab, *wi, *bi, *wd, *bd, *lnbw, *lnbb;
        if (!tensor(st, p + "attention.self.query.weight", (size_t)kDim * kDim, wq) || !tensor(st, p + "attention.self.query.bias", kDim, bq) ||
            !tensor(st, p + "attention.self.key.weight", (size_t)kDim * kDim, wk) || !tensor(st, p + "attention.self.key.bias", kDim, bk) ||
            !tensor(st, p + "attention.self.value.weight", (size_t)kDim * kDim, wv) || !tensor(st, p + "attention.self.value.bias", kDim, bv) ||
            !tensor(st, p + "attention.output.dense.weight", (size_t)kDim * kDim, wo) || !tensor(st, p + "attention.output.dense.bias", kDim, bo) ||
            !tensor(st, p + "attention.output.LayerNorm.weight", kDim, lnaw) || !tensor(st, p + "attention.output.LayerNorm.bias", kDim, lnab) ||
            !tensor(st, p + "intermediate.dense.weight", (size_t)kFF * kDim, wi) || !tensor(st, p + "intermediate.dense.bias", kFF, bi) ||
            !tensor(st, p + "output.dense.weight", (size_t)kDim * kFF, wd) || !tensor(st, p + "output.dense.bias", kDim, bd) ||
            !tensor(st, p + "output.LayerNorm.weight", kDim, lnbw) || !tensor(st, p + "output.LayerNorm.bias", kDim, lnbb))
            return false;
        /* self-attention (the padding keys masked), the output projection, the residual, the post-LN */
        sddev::linear_f32(x, T, wq, kDim, kDim, bq, Q.data());
        sddev::linear_f32(x, T, wk, kDim, kDim, bk, K.data());
        sddev::linear_f32(x, T, wv, kDim, kDim, bv, V.data());
        sddev::audio_attention_f32(Q.data(), K.data(), V.data(), kDim, T, T, kHeads, 64, false, Mk.data(), 0.125f, nullptr, A.data());
        sddev::linear_f32(A.data(), T, wo, kDim, kDim, bo, O.data());
        sddev::add(O.data(), x, n);
        sddev::audio_layer_norm_f32(O.data(), T, kDim, lnaw, lnab, kEps, xo);
        /* the feed-forward: intermediate (exact GELU), the output projection, the residual, the post-LN */
        sddev::linear_f32(xo, T, wi, kDim, kFF, bi, I.data());
        sddev::audio_gelu_erf(I.data(), (size_t)T * kFF);
        sddev::linear_f32(I.data(), T, wd, kFF, kDim, bd, O.data());
        sddev::add(O.data(), xo, n);
        sddev::audio_layer_norm_f32(O.data(), T, kDim, lnbw, lnbb, kEps, x);
    }
    if (hidden) std::memcpy(hidden, x, sizeof(float) * n);
    if (!pooled && !features) return true;
    /* the pooler: tanh(dense(first token)) */
    sddev::Floats P(kDim), P2(kDim), Pin(kDim);
    sddev::linear_f32(x, 1, pw, kDim, kDim, pb, P.data());
    for (int c = 0; c < kDim; ++c) P.data()[c] = std::tanh(P.data()[c]);
    if (pooled) std::memcpy(pooled, P.data(), sizeof(float) * kDim);
    if (!features) return true;
    /* the text projection: Linear 768 -> 512, ReLU, Linear 512 -> 512 (the ReLU on the host, one row) */
    sddev::linear_f32(P.data(), 1, tp1w, kDim, kProj, tp1b, Pin.data());
    sddev::Floats H1(kProj), F(kProj);
    for (int c = 0; c < kProj; ++c) H1.data()[c] = Pin.data()[c] > 0.0f ? Pin.data()[c] : 0.0f;
    sddev::linear_f32(H1.data(), 1, tp2w, kProj, kProj, tp2b, F.data());
    /* get_text_features ends with F.normalize (L2, eps 1e-12): the pipeline's prompt embedding is the unit vector */
    double nrm = 0.0;
    for (int c = 0; c < kProj; ++c) nrm += (double)F.data()[c] * F.data()[c];
    const float inv = 1.0f / (float)std::fmax(std::sqrt(nrm), 1e-12);
    for (int c = 0; c < kProj; ++c) features[c] = F.data()[c] * inv;
    return true;
}

}  // namespace audioldm2
