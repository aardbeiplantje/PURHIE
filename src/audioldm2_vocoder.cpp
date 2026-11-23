#include "audioldm2_vocoder.h"

#include "sd_device.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace audioldm2 {
namespace {

constexpr int kCh = 1024, kMel = 64, kStages = 5;
constexpr int kRate[kStages] = {5, 4, 2, 2, 2};
constexpr int kUpK[kStages] = {16, 16, 8, 4, 4};
constexpr int kResK[3] = {3, 7, 11};
constexpr int kDil[3] = {1, 3, 5};
constexpr float kSlope = 0.1f;

bool get(const zimage::SafeTensors& st, const std::string& name, const float*& p) {
    std::vector<size_t> s;
    if (!st.f32_in_place(name, p, s)) {
        std::fprintf(stderr, "audioldm2 vocoder: %s missing or not F32\n", name.c_str());
        return false;
    }
    return true;
}

/* one residual block (kernel index j): the pairs (convs1[d], convs2[d]) with dilation kDil[d]; each pair takes the block's
 * running hidden state as its residual: h = conv2(lrelu(conv1(lrelu(h)))) + h */
bool residual_block(const zimage::SafeTensors& st, int block, int ch, int T, const float* in, float* out) {
    const int K = kResK[block % 3];
    sddev::Floats cur((size_t)ch * T), res((size_t)ch * T), t1((size_t)ch * T), a((size_t)ch * T), b((size_t)ch * T);
    sddev::audio_copy(cur.data(), in, (size_t)ch * T);
    for (int d = 0; d < 3; ++d) {
        const std::string p = "resblocks." + std::to_string(block) + ".";
        const float *w1, *b1, *w2, *b2;
        if (!get(st, p + "convs1." + std::to_string(d) + ".weight", w1) || !get(st, p + "convs1." + std::to_string(d) + ".bias", b1) ||
            !get(st, p + "convs2." + std::to_string(d) + ".weight", w2) || !get(st, p + "convs2." + std::to_string(d) + ".bias", b2))
            return false;
        const int dd = kDil[d];
        sddev::audio_copy(res.data(), cur.data(), (size_t)ch * T);
        sddev::audio_copy(t1.data(), cur.data(), (size_t)ch * T);
        sddev::audio_leaky_relu(t1.data(), kSlope, (size_t)ch * T);
        sddev::audio_conv1d(t1.data(), ch, T, w1, b1, ch, K, (K * dd - dd) / 2, dd, a.data(), T);
        sddev::audio_leaky_relu(a.data(), kSlope, (size_t)ch * T);
        sddev::audio_conv1d(a.data(), ch, T, w2, b2, ch, K, (K - 1) / 2, 1, b.data(), T);
        sddev::add(b.data(), res.data(), (size_t)ch * T);
        sddev::audio_copy(cur.data(), b.data(), (size_t)ch * T);
    }
    sddev::audio_copy(out, cur.data(), (size_t)ch * T);
    return true;
}

}  // namespace

bool vocoder_forward(const zimage::SafeTensors& st, const float* mel, int T0, std::vector<float>& wav) {
    if (!sddev::enabled()) { std::fprintf(stderr, "audioldm2 vocoder: no device\n"); return false; }
    if (T0 < 1) return false;
    /* the mel transposed to channel-major on the host, into a device buffer */
    sddev::Floats X((size_t)kMel * T0);
    for (int t = 0; t < T0; ++t)
        for (int c = 0; c < kMel; ++c) X.data()[(size_t)c * T0 + t] = mel[(size_t)t * kMel + c];
    const float *wpre, *bpre;
    if (!get(st, "conv_pre.weight", wpre) || !get(st, "conv_pre.bias", bpre)) return false;
    int T = T0;
    sddev::Floats H((size_t)kCh * T);
    sddev::audio_conv1d(X.data(), kMel, T, wpre, bpre, kCh, 7, 3, 1, H.data(), T);
    for (int i = 0; i < kStages; ++i) {
        const int cin = kCh >> i, cout = kCh >> (i + 1), k = kUpK[i], s = kRate[i], pad = (k - s) / 2;
        sddev::audio_leaky_relu(H.data(), kSlope, (size_t)cin * T);
        const float *wu, *bu;
        const std::string up = "upsampler." + std::to_string(i) + ".";
        if (!get(st, up + "weight", wu) || !get(st, up + "bias", bu)) return false;
        const int Tn = (T - 1) * s - 2 * pad + k;
        sddev::Floats U((size_t)cout * Tn);
        sddev::audio_conv_transpose1d(H.data(), cin, T, wu, bu, cout, k, s, pad, U.data(), Tn);
        H = std::move(U);
        T = Tn;
        /* the three residual blocks of this stage, averaged */
        sddev::Floats Acc((size_t)cout * T), Rb((size_t)cout * T);
        for (int j = 0; j < 3; ++j) {
            if (!residual_block(st, i * 3 + j, cout, T, H.data(), Rb.data())) return false;
            if (j == 0) sddev::audio_copy(Acc.data(), Rb.data(), (size_t)cout * T);
            else sddev::add(Acc.data(), Rb.data(), (size_t)cout * T);
        }
        sddev::audio_scale(Acc.data(), 1.0f / 3.0f, (size_t)cout * T);
        H = std::move(Acc);
    }
    /* the final leaky ReLU (torch's default slope, 0.01), conv_post (32 -> 1, 7), tanh */
    sddev::audio_leaky_relu(H.data(), 0.01f, (size_t)32 * T);
    const float *wpost, *bpost;
    if (!get(st, "conv_post.weight", wpost) || !get(st, "conv_post.bias", bpost)) return false;
    sddev::Floats W((size_t)T);
    sddev::audio_conv1d(H.data(), 32, T, wpost, bpost, 1, 7, 3, 1, W.data(), T);
    sddev::audio_tanh(W.data(), (size_t)T);
    wav.assign((size_t)T, 0.0f);
    std::memcpy(wav.data(), W.data(), sizeof(float) * (size_t)T);
    return true;
}

}  // namespace audioldm2
