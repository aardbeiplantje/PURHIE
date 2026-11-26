/* Qwen-Image 2.1's VAE decoder on the device: see qwenimage21_vae.h. Token-major activations [h][w][c]; the 3 x 3 convolutions are
 * the patches (mode 1) of the single frame (their temporal kernel is 1), the 1 x 1 ones linears; the normalisations are the RMS norms
 * over the channels (qv_rms_tok); the residual up blocks add the duplicate-upsample shortcut of their block input. */
#include "qwenimage21_vae.h"

#include "sd_device.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwenimage21 {

namespace {

const float kMean[64] = {0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304};
const float kStd[64] = {3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161};
const size_t kChunkFloats = (size_t)8 << 20;

class DevBuf {
public:
    DevBuf() = default;
    explicit DevBuf(size_t count, bool zero = false) : n_(count) { p_ = static_cast<float*>(sddev::alloc(sizeof(float) * count, zero)); }
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) {
            if (p_) sddev::release(p_, sizeof(float) * n_);
            p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0;
        }
        return *this;
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (p_) sddev::release(p_, sizeof(float) * n_); }
    float* get() const { return p_; }

private:
    float* p_ = nullptr;
    size_t n_ = 0;
};

bool bf(const zimage::SafeTensors& st, const std::string& name, size_t count, const uint16_t*& p) {
    std::vector<size_t> s;
    size_t n = 1;
    if (!st.bf16_in_place(name, p, s)) {
        fprintf(stderr, "qwenimage21 vae: %s missing or not BF16\n", name.c_str());
        return false;
    }
    for (size_t d : s) n *= d;
    if (n != count) {
        fprintf(stderr, "qwenimage21 vae: %s has %zu elements, want %zu\n", name.c_str(), n, count);
        return false;
    }
    return true;
}

/* a convolution: mode 0 (a 1 x 1 or 1 x 1 x 1 kernel: a per-pixel linear) or 1 (a 3 x 3 kernel of the single frame: patches); BF16 */
struct Conv {
    const uint16_t* W = nullptr;
    const uint16_t* b = nullptr;
    int cin = 0, cout = 0, mode = 0;
};

bool load_conv(const zimage::SafeTensors& st, const std::string& prefix, int cin, int cout, int mode, Conv& c) {
    const int k = mode == 0 ? 1 : 9;
    c.cin = cin;
    c.cout = cout;
    c.mode = mode;
    return bf(st, prefix + ".weight", (size_t)cout * cin * k, c.W) && bf(st, prefix + ".bias", (size_t)cout, c.b);
}

void conv(const float* x, int H, int W, const Conv& c, const DevBuf& zeros, float* y) {
    const int HW = H * W;
    if (c.mode == 0) {
        sddev::linear_bf16_f32(x, HW, c.W, c.cin, c.cout, y);
    } else {
        const int K = 9 * c.cin;
        const int chunk = (int)std::max<size_t>(1, kChunkFloats / (size_t)K);
        DevBuf P((size_t)std::min(chunk, HW) * K);
        for (int p0 = 0; p0 < HW; p0 += chunk) {
            const int n = std::min(chunk, HW - p0);
            sddev::patches(x, H, W, c.cin, 1, p0, n, P.get());
            sddev::linear_bf16_f32(P.get(), n, c.W, K, c.cout, y + (size_t)p0 * c.cout);
        }
    }
    sddev::bias_tok(y, HW, c.cout, c.b);
}

/* the residual block: norm, SiLU, 3 x 3, norm, SiLU, 3 x 3, plus the shortcut (a 1 x 1 when the channels change) */
bool resnet(const zimage::SafeTensors& st, const std::string& p, const float* x, int H, int W, int cin, int cout, const DevBuf& zeros,
            DevBuf& out) {
    const int HW = H * W;
    const uint16_t *g1 = nullptr, *g2 = nullptr;
    Conv c1, c2, sc;
    if (!bf(st, p + ".residual.0.gamma", cin, g1) || !bf(st, p + ".residual.3.gamma", cout, g2) ||
        !load_conv(st, p + ".residual.2", cin, cout, 1, c1) || !load_conv(st, p + ".residual.6", cout, cout, 1, c2))
        return false;
    if (cin != cout && !load_conv(st, p + ".shortcut", cin, cout, 0, sc)) return false;
    DevBuf h((size_t)HW * cin);
    sddev::rms_tok(x, HW, cin, g1, h.get());
    sddev::silu(h.get(), (size_t)HW * cin);
    DevBuf a((size_t)HW * cout);
    conv(h.get(), H, W, c1, zeros, a.get());
    sddev::rms_tok(a.get(), HW, cout, g2, a.get());
    sddev::silu(a.get(), (size_t)HW * cout);
    out = DevBuf((size_t)HW * cout);
    conv(a.get(), H, W, c2, zeros, out.get());
    if (cin != cout) {
        DevBuf s((size_t)HW * cout);
        conv(x, H, W, sc, zeros, s.get());
        sddev::add(out.get(), s.get(), (size_t)HW * cout);
    } else {
        sddev::add(out.get(), x, (size_t)HW * cout);
    }
    return true;
}

/* the single-head attention over the pixels: norm, the 1 x 1 qkv, the scaled dot product, proj, residual */
bool attention_block(const zimage::SafeTensors& st, const std::string& p, const float* x, int H, int W, int C, const DevBuf& zeros,
                     DevBuf& out) {
    const int HW = H * W;
    const uint16_t *g = nullptr, *wq = nullptr, *bq = nullptr, *wp = nullptr, *bp = nullptr;
    if (!bf(st, p + ".norm.gamma", C, g) || !bf(st, p + ".to_qkv.weight", (size_t)3 * C * C, wq) ||
        !bf(st, p + ".to_qkv.bias", (size_t)3 * C, bq) || !bf(st, p + ".proj.weight", (size_t)C * C, wp) ||
        !bf(st, p + ".proj.bias", (size_t)C, bp))
        return false;
    DevBuf n((size_t)HW * C);
    sddev::rms_tok(x, HW, C, g, n.get());
    DevBuf q((size_t)HW * C), k((size_t)HW * C), v((size_t)HW * C);
    sddev::linear_bf16_f32(n.get(), HW, wq, C, C, q.get());
    sddev::bias_tok(q.get(), HW, C, bq);
    sddev::linear_bf16_f32(n.get(), HW, wq + (size_t)C * C, C, C, k.get());
    sddev::bias_tok(k.get(), HW, C, bq + C);
    sddev::linear_bf16_f32(n.get(), HW, wq + (size_t)2 * C * C, C, C, v.get());
    sddev::bias_tok(v.get(), HW, C, bq + 2 * C);
    DevBuf o((size_t)HW * C);
    sddev::attention_f32(q.get(), HW, k.get(), v.get(), HW, C, 1, o.get());
    Conv pj;
    pj.W = wp;
    pj.b = bp;
    pj.cin = C;
    pj.cout = C;
    pj.mode = 0;
    out = DevBuf((size_t)HW * C);
    conv(o.get(), H, W, pj, zeros, out.get());
    sddev::add(out.get(), x, (size_t)HW * C);
    return true;
}

}  // namespace

bool dev_vae21_decode(const zimage::SafeTensors& st, const float* latent, int gh, int gw, float* out) {
    if (!sddev::enabled()) { fprintf(stderr, "qwenimage21 vae: no device\n"); return false; }
    if (gh < 1 || gw < 1) { fprintf(stderr, "qwenimage21 vae: bad latent size\n"); return false; }
    const int C0 = 64, H0 = gh, W0 = gw;
    DevBuf zeros(1152, true);

    /* the input: the latent normalised (z * std + mean) and written token-major (the small host step: 64 x gh x gw values) */
    DevBuf z((size_t)H0 * W0 * C0);
    for (int c = 0; c < C0; ++c)
        for (int p = 0; p < H0 * W0; ++p) z.get()[(size_t)p * C0 + c] = latent[(size_t)c * H0 * W0 + p] * kStd[c] + kMean[c];

    /* post_quant_conv (the file's conv2, 1 x 1 x 1), then the decoder's conv_in (decoder.conv1, 3 x 3 of the frame, 64 -> 1152) */
    Conv pq, cin_conv;
    if (!load_conv(st, "conv2", C0, C0, 0, pq) || !load_conv(st, "decoder.conv1", C0, 1152, 1, cin_conv)) return false;
    DevBuf y((size_t)H0 * W0 * C0);
    conv(z.get(), H0, W0, pq, zeros, y.get());
    int H = H0, W = W0;
    DevBuf x((size_t)H * W * 1152);
    conv(y.get(), H, W, cin_conv, zeros, x.get());

    /* the middle: resnet, attention, resnet (gh x gw, 1152 channels) */
    DevBuf r1, a1, r2;
    if (!resnet(st, "decoder.middle.0", x.get(), H, W, 1152, 1152, zeros, r1)) return false;
    if (!attention_block(st, "decoder.middle.1", r1.get(), H, W, 1152, zeros, a1)) return false;
    if (!resnet(st, "decoder.middle.2", a1.get(), H, W, 1152, 1152, zeros, r2)) return false;
    x = std::move(r2);

    /* the up blocks: (in, out, factor_t of the upsample: 2 temporal, 1 spatial, 0 none); each: three resnets, then (nearest 2x, 3 x 3
     * conv) plus the duplicate-upsample shortcut of the block's input */
    const int blk_in[5] = {1152, 1152, 1152, 576, 288}, blk_out[5] = {1152, 1152, 576, 288, 144}, factor_t[5] = {2, 2, 2, 1, 0};
    for (int b = 0; b < 5; ++b) {
        const std::string base = "decoder.upsamples." + std::to_string(b) + ".upsamples.";
        DevBuf xin = std::move(x);  /* the block input (the shortcut's source) */
        DevBuf cur;
        const float* src = xin.get();
        for (int j = 0; j < 3; ++j) {
            DevBuf r;
            const int cin = j == 0 ? blk_in[b] : blk_out[b];
            if (!resnet(st, base + std::to_string(j), src, H, W, cin, blk_out[b], zeros, r)) return false;
            cur = std::move(r);
            src = cur.get();
        }
        if (factor_t[b] == 0) {
            x = std::move(cur);
            continue;
        }
        /* nearest 2x, then the 3 x 3 conv of the upsampler (out -> out), plus the shortcut from the block input */
        const int oc = blk_out[b];
        DevBuf up((size_t)4 * H * W * oc);
        sddev::upsample2_tok(cur.get(), H, W, oc, up.get());
        H *= 2;
        W *= 2;
        Conv uc;
        if (!load_conv(st, base + "3.resample.1", oc, oc, 1, uc)) return false;
        DevBuf u((size_t)H * W * oc);
        conv(up.get(), H, W, uc, zeros, u.get());
        DevBuf sc((size_t)H * W * oc);
        sddev::dupup(xin.get(), H / 2, W / 2, blk_in[b], oc, factor_t[b], sc.get());
        sddev::add(u.get(), sc.get(), (size_t)H * W * oc);
        x = std::move(u);
    }

    /* the head: RMS norm (144), SiLU, the 3 x 3 conv to 4 channels */
    const uint16_t* gamma = nullptr;
    Conv head;
    if (!bf(st, "decoder.head.0.gamma", 144, gamma) || !load_conv(st, "decoder.head.2", 144, 4, 1, head)) return false;
    DevBuf hn((size_t)H * W * 144);
    sddev::rms_tok(x.get(), H * W, 144, gamma, hn.get());
    sddev::silu(hn.get(), (size_t)H * W * 144);
    DevBuf o((size_t)H * W * 4);
    conv(hn.get(), H, W, head, zeros, o.get());

    /* token-major [h][w][4] to channel-major [4][H][W], clamped */
    std::vector<float> tok((size_t)H * W * 4);
    std::memcpy(tok.data(), o.get(), sizeof(float) * tok.size());
    for (int c = 0; c < 4; ++c)
        for (int p = 0; p < H * W; ++p) out[(size_t)c * H * W + p] = std::min(1.0f, std::max(-1.0f, tok[(size_t)p * 4 + c]));
    return H == 16 * H0 && W == 16 * W0;
}

}  // namespace qwenimage21
