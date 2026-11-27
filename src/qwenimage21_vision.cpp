/* Qwen-Image 2.1's condition-image encoder on the device: see qwenimage21_vision.h. Token-major activations [t][c] in merge order (the
 * processor's: 2 x 2 blocks of patches, raster over the blocks), the residual stream FP32; each product's input rounded once to BF16,
 * hipBLASLt's BF16 product with a BF16 output, widened with the bias and the residual (qv_from16_bias). The position embedding and the
 * rotary angles are computed on the host as transformers does (fast_pos_embed_interpolate: torch's float32 linspace, the four
 * bilinear terms summed in order; rot_pos_emb). The attention is the FP32 one over all patches of the image (one image: no
 * windows), from the fused q, k, v rows. */
#include "qwenimage21_vision.h"

#include "sd_device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwenimage21 {

namespace {

const int C = 1152, HEADS = 16, HD = 72, FF = 4304, DEPTH = 27, IN = 1536, MC = 4 * C, SIDE = 48;
const int DS_LAYER[VISION_DEEPSTACK] = {8, 16, 24};
const float EPS = 1e-6f;

/* a device buffer of n elements (FP32 from sddev::alloc; BF16 from device memory, sddev::dalloc: never read by the host) */
template <typename T>
class Buf {
public:
    Buf() = default;
    explicit Buf(size_t count) : n_(count) {
        p_ = static_cast<T*>(dev() ? sddev::dalloc(sizeof(T) * count) : sddev::alloc(sizeof(T) * count, false));
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() {
        if (!p_) return;
        if (dev()) sddev::drelease(p_, sizeof(T) * n_);
        else sddev::release(p_, sizeof(T) * n_);
    }
    T* get() const { return p_; }

private:
    static constexpr bool dev() { return sizeof(T) == 2; }
    T* p_ = nullptr;
    size_t n_ = 0;
};
using DevBuf = Buf<float>;

bool bf(const zimage::SafeTensors& st, const std::string& name, size_t count, const uint16_t*& p) {
    std::vector<size_t> s;
    size_t n = 1;
    if (!st.bf16_in_place(name, p, s)) {
        fprintf(stderr, "qwenimage21 vision: %s missing or not BF16\n", name.c_str());
        return false;
    }
    for (size_t d : s) n *= d;
    if (n != count) {
        fprintf(stderr, "qwenimage21 vision: %s has %zu elements, want %zu\n", name.c_str(), n, count);
        return false;
    }
    return true;
}

float bf2f(uint16_t b) {
    const uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/* a linear with a bias: W [out][in], b [out] (BF16, in place) */
struct Lin {
    const uint16_t* W = nullptr;
    const uint16_t* b = nullptr;
    int in = 0, out = 0;
};

bool load_lin(const zimage::SafeTensors& st, const std::string& base, int in, int out, Lin& L) {
    if (!bf(st, base + ".weight", (size_t)out * in, L.W) || !bf(st, base + ".bias", (size_t)out, L.b)) return false;
    L.in = in;
    L.out = out;
    return true;
}

/* y [T][out] = x [T][in] W^T + b (+ res, which may be y itself): the BF16 weights read in place, FP32 activations and accumulation
 * (qi21_linear_bf16's kernel; the tower in FP32 throughout: the BF16 chain's roundings compounded to about 10% over the 27 blocks) */
void lin(const Lin& L, const float* x, int T, const float* res, float* y) {
    const size_t n = (size_t)T * L.out;
    DevBuf t(n);
    sddev::linear_bf16_f32(x, T, L.W, L.in, L.out, t.get());
    sddev::bias_tok(t.get(), T, L.out, L.b);
    if (res) {
        if (res != y) sddev::audio_copy(y, res, n);
        sddev::add(y, t.get(), n);
    } else {
        sddev::audio_copy(y, t.get(), n);
    }
}

/* torch.linspace(0, SIDE - 1, n) in float32 on the CPU: the first half from the start, the second from the end */
std::vector<float> linspace(int n) {
    std::vector<float> v(n);
    if (n == 1) { v[0] = 0.0f; return v; }
    const float start = 0.0f, end = (float)(SIDE - 1), step = (end - start) / (float)(n - 1);
    const int half = n / 2;
    for (int i = 0; i < n; ++i) v[i] = i < half ? start + step * (float)i : end - step * (float)(n - i - 1);
    return v;
}

}  // namespace

bool dev_vision21_encode(const zimage::SafeTensors& st, const float* pixels, int gh, int gw, float* merged, float* ds) {
    if (!sddev::enabled()) { fprintf(stderr, "qwenimage21 vision: no device\n"); return false; }
    if (gh < 2 || gw < 2 || gh % 2 || gw % 2 || gh > 256 || gw > 256) {
        fprintf(stderr, "qwenimage21 vision: a grid of %d x %d patches (even, 2 to 256)\n", gh, gw);
        return false;
    }
    const int T = gh * gw, M = T / 4;
    const std::string p = "model.visual.";
    Lin patch, mfc1, mfc2, dfc1[VISION_DEEPSTACK], dfc2[VISION_DEEPSTACK];
    const uint16_t *pos = nullptr, *mnw = nullptr, *mnb = nullptr, *dnw[VISION_DEEPSTACK], *dnb[VISION_DEEPSTACK];
    if (!bf(st, p + "patch_embed.proj.weight", (size_t)C * IN, patch.W) || !bf(st, p + "patch_embed.proj.bias", C, patch.b) ||
        !bf(st, p + "pos_embed.weight", (size_t)SIDE * SIDE * C, pos) || !bf(st, p + "merger.norm.weight", C, mnw) ||
        !bf(st, p + "merger.norm.bias", C, mnb) || !load_lin(st, p + "merger.linear_fc1", MC, MC, mfc1) ||
        !load_lin(st, p + "merger.linear_fc2", MC, VISION_OUT, mfc2))
        return false;
    patch.in = IN;
    patch.out = C;
    for (int i = 0; i < VISION_DEEPSTACK; ++i) {
        const std::string d = p + "deepstack_merger_list." + std::to_string(i) + ".";
        if (!bf(st, d + "norm.weight", MC, dnw[i]) || !bf(st, d + "norm.bias", MC, dnb[i]) ||
            !load_lin(st, d + "linear_fc1", MC, MC, dfc1[i]) || !load_lin(st, d + "linear_fc2", MC, VISION_OUT, dfc2[i]))
            return false;
    }

    /* the position embedding, as fast_pos_embed_interpolate: the four corners' rows weighted and summed (in that order), raster over
     * the grid, then permuted to merge order */
    DevBuf posd((size_t)T * C);
    {
        const std::vector<float> hi = linspace(gh), wi = linspace(gw);
        std::vector<float> row(C);
        for (int r = 0; r < gh; ++r)
            for (int c = 0; c < gw; ++c) {
                const int hf = (int)hi[r], wf = (int)wi[c];
                const int hc = std::min(hf + 1, SIDE - 1), wc = std::min(wf + 1, SIDE - 1);
                const float dh = hi[r] - (float)hf, dw = wi[c] - (float)wf;
                const int idx[4] = {hf * SIDE + wf, hf * SIDE + wc, hc * SIDE + wf, hc * SIDE + wc};
                const float w[4] = {(1 - dh) * (1 - dw), (1 - dh) * dw, dh * (1 - dw), dh * dw};
                /* merge order: block (r / 2, c / 2), then (r % 2, c % 2) */
                const size_t t = ((size_t)(r / 2) * (gw / 2) + c / 2) * 4 + (r % 2) * 2 + c % 2;
                float* out = posd.get() + t * C;
                for (int k = 0; k < C; ++k) {
                    /* each product rounded, then the sums (no fused multiply-add: separate statements) */
                    const float a0 = bf2f(pos[(size_t)idx[0] * C + k]) * w[0];
                    const float a1 = bf2f(pos[(size_t)idx[1] * C + k]) * w[1];
                    const float a2 = bf2f(pos[(size_t)idx[2] * C + k]) * w[2];
                    const float a3 = bf2f(pos[(size_t)idx[3] * C + k]) * w[3];
                    out[k] = ((a0 + a1) + a2) + a3;
                }
            }
    }
    /* the rotary angles, as rot_pos_emb: the head's 36 rotated pairs (d, d + 36) take the patch row's 18 frequencies, then the
     * column's (theta 10000 over 36 dims); cos and sin [T][72] (each pair's angle twice, as cat(emb, emb)) */
    DevBuf cs((size_t)T * HD), sn((size_t)T * HD);
    for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c) {
            const size_t t = ((size_t)(r / 2) * (gw / 2) + c / 2) * 4 + (r % 2) * 2 + c % 2;
            for (int i = 0; i < HD / 2; ++i) {
                const int k = i % (HD / 4);
                const float inv = (float)(1.0 / std::pow(10000.0, (2.0 * k) / (HD / 2)));
                const float ang = (float)(i < HD / 4 ? r : c) * inv;
                const float co = (float)std::cos((double)ang), si = (float)std::sin((double)ang);
                cs.get()[t * HD + i] = cs.get()[t * HD + i + HD / 2] = co;
                sn.get()[t * HD + i] = sn.get()[t * HD + i + HD / 2] = si;
            }
        }

    /* the patch embedding (the Conv3d of stride = kernel: a linear of the pixel rows), plus the positions */
    DevBuf X((size_t)T * C), px((size_t)T * IN);
    DevBuf qkv((size_t)T * 3 * C), att((size_t)T * C), h((size_t)T * FF);
    std::memcpy(px.get(), pixels, sizeof(float) * (size_t)T * IN);
    lin(patch, px.get(), T, posd.get(), X.get());

    /* a merger: (postshuffle) the LayerNorm over the 4 patches' 4608, else over each patch's 1152; fc1, the exact GELU, fc2 */
    DevBuf mx((size_t)M * MC), mh((size_t)M * MC), mo((size_t)M * VISION_OUT);
    auto merger = [&](bool post, const uint16_t* nw, const uint16_t* nb, const Lin& f1, const Lin& f2, float* host) {
        if (post) sddev::layer_norm_bf16(X.get(), M, MC, nw, nb, EPS, mx.get());
        else sddev::layer_norm_bf16(X.get(), T, C, nw, nb, EPS, mx.get());
        lin(f1, mx.get(), M, nullptr, mh.get());
        sddev::audio_gelu_erf(mh.get(), (size_t)M * MC);
        lin(f2, mh.get(), M, nullptr, mo.get());
        std::memcpy(host, mo.get(), sizeof(float) * (size_t)M * VISION_OUT);
    };

    for (int l = 0; l < DEPTH; ++l) {
        const std::string b = p + "blocks." + std::to_string(l) + ".";
        Lin wqkv, wproj, wfc1, wfc2;
        const uint16_t *n1w = nullptr, *n1b = nullptr, *n2w = nullptr, *n2b = nullptr;
        if (!bf(st, b + "norm1.weight", C, n1w) || !bf(st, b + "norm1.bias", C, n1b) || !bf(st, b + "norm2.weight", C, n2w) ||
            !bf(st, b + "norm2.bias", C, n2b) || !load_lin(st, b + "attn.qkv", C, 3 * C, wqkv) ||
            !load_lin(st, b + "attn.proj", C, C, wproj) || !load_lin(st, b + "mlp.linear_fc1", C, FF, wfc1) ||
            !load_lin(st, b + "mlp.linear_fc2", FF, C, wfc2))
            return false;
        /* attention: LayerNorm, q k v, the rotary positions on q and k, all patches, the projection into the residual */
        sddev::layer_norm_bf16(X.get(), T, C, n1w, n1b, EPS, att.get());
        lin(wqkv, att.get(), T, nullptr, qkv.get());
        sddev::dino_rope(qkv.get(), T, 0, 3 * C, HEADS, HD, cs.get(), sn.get());
        sddev::dino_rope(qkv.get() + C, T, 0, 3 * C, HEADS, HD, cs.get(), sn.get());
        sddev::attention_f32_strided(qkv.get(), 3 * C, T, qkv.get() + C, 3 * C, qkv.get() + 2 * C, 3 * C, T, C, HEADS, att.get());
        lin(wproj, att.get(), T, X.get(), X.get());
        /* the MLP: LayerNorm, fc1, the tanh GELU, fc2 into the residual */
        sddev::layer_norm_bf16(X.get(), T, C, n2w, n2b, EPS, att.get());
        lin(wfc1, att.get(), T, nullptr, h.get());
        sddev::gelu_tanh(h.get(), (size_t)T * FF);
        lin(wfc2, h.get(), T, X.get(), X.get());
        for (int i = 0; i < VISION_DEEPSTACK; ++i)
            if (l == DS_LAYER[i]) merger(true, dnw[i], dnb[i], dfc1[i], dfc2[i], ds + (size_t)i * M * VISION_OUT);
    }
    merger(false, mnw, mnb, mfc1, mfc2, merged);
    return true;
}

}  // namespace qwenimage21
