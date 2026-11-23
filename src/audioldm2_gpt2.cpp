#include "audioldm2_gpt2.h"

#include "sd_device.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace audioldm2 {
namespace {

constexpr int kDim = 768, kHeads = 12, kFF = 3072, kLayers = 12;

/* the fp32 tensor of the mapping by name (its element count must be n) */
bool weight(const zimage::SafeTensors& st, const std::string& name, size_t n, const float*& p) {
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

bool gpt2_forward(const zimage::SafeTensors& st, const float* x, int T, float* hidden6, float* last) {
    if (!sddev::enabled()) { std::fprintf(stderr, "audioldm2: no device\n"); return false; }
    if (T < 1 || T > 1024) { std::fprintf(stderr, "audioldm2: the GPT-2 takes 1 to 1024 tokens, not %d\n", T); return false; }
    const size_t n = (size_t)T * kDim;
    sddev::Floats X(n), H(n), QKV((size_t)T * 3 * kDim), A(n), O(n), M((size_t)T * kFF);
    std::memcpy(X.data(), x, sizeof(float) * n);
    /* the position embeddings (wpe: the first T rows, contiguous in the mapping) are added to the input, as GPT2Model does */
    const float* wpe = nullptr;
    if (!weight(st, "wpe.weight", (size_t)1024 * kDim, wpe)) return false;
    sddev::add(X.data(), wpe, n);
    for (int L = 0; L < kLayers; ++L) {
        const std::string p = "h." + std::to_string(L) + ".";
        const float *ln1w, *ln1b, *wqkv, *bqkv, *wproj, *bproj, *ln2w, *ln2b, *wfc, *bfc, *wmp, *bmp;
        if (!weight(st, p + "ln_1.weight", kDim, ln1w) || !weight(st, p + "ln_1.bias", kDim, ln1b) ||
            !weight(st, p + "attn.c_attn.weight", (size_t)kDim * 3 * kDim, wqkv) || !weight(st, p + "attn.c_attn.bias", 3 * kDim, bqkv) ||
            !weight(st, p + "attn.c_proj.weight", (size_t)kDim * kDim, wproj) || !weight(st, p + "attn.c_proj.bias", kDim, bproj) ||
            !weight(st, p + "ln_2.weight", kDim, ln2w) || !weight(st, p + "ln_2.bias", kDim, ln2b) ||
            !weight(st, p + "mlp.c_fc.weight", (size_t)kDim * kFF, wfc) || !weight(st, p + "mlp.c_fc.bias", kFF, bfc) ||
            !weight(st, p + "mlp.c_proj.weight", (size_t)kFF * kDim, wmp) || !weight(st, p + "mlp.c_proj.bias", kDim, bmp))
            return false;
        /* attention: x += proj(attention(qkv(LN(x)))) */
        sddev::audio_layer_norm_f32(X.data(), T, kDim, ln1w, ln1b, 1e-5f, H.data());
        sddev::audio_gemm_f32(H.data(), T, kDim, wqkv, bqkv, 3 * kDim, QKV.data());
        sddev::audio_causal_attn_f32(QKV.data(), T, kHeads, A.data());
        sddev::audio_gemm_f32(A.data(), T, kDim, wproj, bproj, kDim, O.data());
        sddev::add(X.data(), O.data(), n);
        /* feed-forward: x += proj(gelu(fc(LN(x)))) */
        sddev::audio_layer_norm_f32(X.data(), T, kDim, ln2w, ln2b, 1e-5f, H.data());
        sddev::audio_gemm_f32(H.data(), T, kDim, wfc, bfc, kFF, M.data());
        sddev::gelu_tanh(M.data(), (size_t)T * kFF);
        sddev::audio_gemm_f32(M.data(), T, kFF, wmp, bmp, kDim, O.data());
        sddev::add(X.data(), O.data(), n);
        if (L == 5 && hidden6) std::memcpy(hidden6, X.data(), sizeof(float) * n);
    }
    if (last) {
        const float *lfw, *lfb;
        if (!weight(st, "ln_f.weight", kDim, lfw) || !weight(st, "ln_f.bias", kDim, lfb)) return false;
        /* the kernel writes device-accessible memory only: the result goes through a buffer and is copied out */
        sddev::Floats LN(n);
        sddev::audio_layer_norm_f32(X.data(), T, kDim, lfw, lfb, 1e-5f, LN.data());
        std::memcpy(last, LN.data(), sizeof(float) * n);
    }
    return true;
}

}  // namespace audioldm2
