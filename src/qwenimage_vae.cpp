/* Qwen-Image's VAE decoder on the device: see qwenimage_vae.h. The file's keys are the Wan layout (decoder.conv1, decoder.middle.N,
 * decoder.upsamples.N, decoder.head.N; conv2 is the post-quantisation conv). The activations are token-major [h][w][c]. */
#include "qwenimage_vae.h"

#include "sd_device.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace qwenimage {

namespace {

/* the latent statistics of vae/config.json (the pipeline's: latents * std + mean before the decode) */
const float kMean[16] = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f, 1.5508f,
                         0.4134f,  -0.0715f, 0.5517f,  -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
const float kStd[16] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                        3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.916f};
const size_t kChunkFloats = (size_t)8 << 20;  /* the patches per GEMM call (32 MB) */

class DevBuf {
public:
    DevBuf() = default;
    explicit DevBuf(size_t count, bool zero = false) : n_(count) { p_ = static_cast<float*>(sddev::alloc(sizeof(float) * count, zero)); }
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) {
            if (p_) sddev::release(p_, sizeof(float) * n_);
            p_ = o.p_;
            n_ = o.n_;
            o.p_ = nullptr;
            o.n_ = 0;
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

/* a BF16 tensor (in place) with its element count checked */
bool bf(const zimage::SafeTensors& st, const std::string& name, size_t count, const uint16_t*& p) {
    std::vector<size_t> s;
    size_t n = 1;
    if (!st.bf16_in_place(name, p, s)) {
        fprintf(stderr, "qwenimage vae: %s missing or not BF16\n", name.c_str());
        return false;
    }
    for (size_t d : s) n *= d;
    if (n != count) {
        fprintf(stderr, "qwenimage vae: %s has %zu elements, want %zu\n", name.c_str(), n, count);
        return false;
    }
    return true;
}

/* the decoder's weights of one convolution: the BF16 weights [cout][cin][kt][kh][kw] (flattened) and the bias */
struct Conv {
    const uint16_t* W = nullptr;
    const uint16_t* b = nullptr;
    int cin = 0, cout = 0, mode = 0;  /* 0: 1 x 1 (or 1 x 1 x 1); 1: 2D 3 x 3; 3: causal 3 x 3 x 3 on one frame */
};

bool load_conv(const zimage::SafeTensors& st, const std::string& prefix, int cin, int cout, int mode, Conv& c) {
    const int k = mode == 0 ? 1 : mode == 1 ? 9 : 27;
    c.cin = cin;
    c.cout = cout;
    c.mode = mode;
    return bf(st, prefix + ".weight", (size_t)cout * cin * k, c.W) && bf(st, prefix + ".bias", (size_t)cout, c.b);
}

/* y [HW][cout] = the convolution of x [H][W][cin] (mode 0: a per-pixel linear; modes 1 and 3: the patches, chunked), plus the bias */
void conv(const float* x, int H, int W, const Conv& c, const DevBuf& zeros, float* y) {
    const int HW = H * W;
    if (c.mode == 0) {
        sddev::linear_bf16_f32(x, HW, c.W, c.cin, c.cout, y);
    } else {
        const int K = (c.mode == 3 ? 27 : 9) * c.cin;
        const int chunk = (int)std::max<size_t>(1, kChunkFloats / (size_t)K);
        DevBuf P((size_t)std::min(chunk, HW) * K);
        for (int p0 = 0; p0 < HW; p0 += chunk) {
            const int n = std::min(chunk, HW - p0);
            sddev::patches(x, H, W, c.cin, c.mode, p0, n, P.get());
            sddev::linear_bf16_f32(P.get(), n, c.W, K, c.cout, y + (size_t)p0 * c.cout);
        }
    }
    sddev::bias_tok(y, HW, c.cout, c.b);
}

/* RMS norm over channels with its BF16 gamma (C values) */
bool gamma(const zimage::SafeTensors& st, const std::string& name, int C, const uint16_t*& g) { return bf(st, name, (size_t)C, g); }

/* one residual block of Wan's layout: norm, SiLU, causal conv, norm, SiLU, causal conv, plus the shortcut */
bool resnet(const zimage::SafeTensors& st, const std::string& p, const float* x, int H, int W, int cin, int cout, const DevBuf& zeros,
            DevBuf& out) {
    const int HW = H * W;
    const uint16_t *g1 = nullptr, *g2 = nullptr;
    Conv c1, c2, sc;
    if (!gamma(st, p + ".residual.0.gamma", cin, g1) || !gamma(st, p + ".residual.3.gamma", cout, g2) ||
        !load_conv(st, p + ".residual.2", cin, cout, 3, c1) || !load_conv(st, p + ".residual.6", cout, cout, 3, c2))
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

/* the single-head attention over the pixels (norm, to_qkv, scaled dot product, proj, residual) */
bool attention_block(const zimage::SafeTensors& st, const std::string& p, const float* x, int H, int W, int C, const DevBuf& zeros,
                     DevBuf& out) {
    const int HW = H * W;
    const uint16_t* g = nullptr;
    const uint16_t *wq = nullptr, *bq = nullptr, *wp = nullptr, *bp = nullptr;
    if (!gamma(st, p + ".norm.gamma", C, g) || !bf(st, p + ".to_qkv.weight", (size_t)3 * C * C, wq) ||
        !bf(st, p + ".to_qkv.bias", (size_t)3 * C, bq) || !bf(st, p + ".proj.weight", (size_t)C * C, wp) ||
        !bf(st, p + ".proj.bias", (size_t)C, bp))
        return false;
    DevBuf n((size_t)HW * C);
    sddev::rms_tok(x, HW, C, g, n.get());
    DevBuf q((size_t)HW * C), k((size_t)HW * C), v((size_t)HW * C);
    /* the qkv rows of to_qkv: q [0, C), k [C, 2C), v [2C, 3C) (contiguous slices of the weight and the bias) */
    sddev::linear_bf16_f32(n.get(), HW, wq, C, C, q.get());
    sddev::bias_tok(q.get(), HW, C, bq);
    sddev::linear_bf16_f32(n.get(), HW, wq + (size_t)C * C, C, C, k.get());
    sddev::bias_tok(k.get(), HW, C, bq + C);
    sddev::linear_bf16_f32(n.get(), HW, wq + (size_t)2 * C * C, C, C, v.get());
    sddev::bias_tok(v.get(), HW, C, bq + 2 * C);
    DevBuf o((size_t)HW * C);
    sddev::attention(q.get(), HW, k.get(), v.get(), HW, C, 1, o.get());
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

bool dev_vae_decode(const zimage::SafeTensors& st, const float* latent, float* out) {
    if (!sddev::enabled()) { fprintf(stderr, "qwenimage vae: no device\n"); return false; }
    const int C0 = 16, H0 = 32, W0 = 32;
    DevBuf zeros(384, true);

    /* the input: the latent (channel-major) normalised and written token-major (the small host step: 16 x 1024 values) */
    DevBuf z((size_t)H0 * W0 * C0);
    for (int c = 0; c < C0; ++c)
        for (int p = 0; p < H0 * W0; ++p) z.get()[(size_t)p * C0 + c] = latent[(size_t)c * H0 * W0 + p] * kStd[c] + kMean[c];

    /* post_quant_conv (the file's conv2: a 1 x 1 x 1 convolution), then the decoder's conv_in (causal 3 x 3 x 3, 16 -> 384) */
    Conv pq, cin;
    if (!load_conv(st, "conv2", C0, C0, 0, pq) || !load_conv(st, "decoder.conv1", C0, 384, 3, cin)) return false;
    DevBuf y((size_t)H0 * W0 * C0);
    conv(z.get(), H0, W0, pq, zeros, y.get());
    DevBuf x((size_t)H0 * W0 * 384);
    conv(y.get(), H0, W0, cin, zeros, x.get());

    /* the middle: resnet, attention, resnet (32 x 32, 384 channels) */
    int H = H0, W = W0;
    DevBuf r1, a1, r2;
    if (!resnet(st, "decoder.middle.0", x.get(), H, W, 384, 384, zeros, r1)) return false;
    if (!attention_block(st, "decoder.middle.1", r1.get(), H, W, 384, zeros, a1)) return false;
    if (!resnet(st, "decoder.middle.2", a1.get(), H, W, 384, 384, zeros, r2)) return false;
    x = std::move(r2);

    /* the up blocks: three resnets each, then a 2x upsample (levels 0 to 2; the first two are temporal, which a single frame skips) */
    const int block_in[4] = {384, 192, 192, 96}, block_c[4] = {384, 384, 192, 96};
    const int ups_in[3] = {384, 384, 192}, ups_out[3] = {192, 192, 96};
    for (int b = 0; b < 4; ++b) {
        for (int j = 0; j < 3; ++j) {
            const std::string p = "decoder.upsamples." + std::to_string(4 * b + j);
            DevBuf r;
            const int cin = j == 0 ? block_in[b] : block_c[b];
            if (!resnet(st, p, x.get(), H, W, cin, block_c[b], zeros, r)) return false;
            x = std::move(r);
        }
        if (b < 3) {
            const std::string p = "decoder.upsamples." + std::to_string(4 * b + 3) + ".resample.1";
            Conv up;
            if (!load_conv(st, p, ups_in[b], ups_out[b], 1, up)) return false;
            DevBuf u((size_t)4 * H * W * ups_in[b]);
            sddev::upsample2_tok(x.get(), H, W, ups_in[b], u.get());
            H *= 2;
            W *= 2;
            DevBuf n((size_t)H * W * ups_out[b]);
            conv(u.get(), H, W, up, zeros, n.get());
            x = std::move(n);
        }
    }

    /* the head: RMS norm (96), SiLU, the causal 3 x 3 x 3 convolution to 3 channels */
    const uint16_t* gh = nullptr;
    Conv head;
    if (!gamma(st, "decoder.head.0.gamma", 96, gh) || !load_conv(st, "decoder.head.2", 96, 3, 3, head)) return false;
    DevBuf hn((size_t)H * W * 96);
    sddev::rms_tok(x.get(), H * W, 96, gh, hn.get());
    sddev::silu(hn.get(), (size_t)H * W * 96);
    DevBuf o((size_t)H * W * 3);
    conv(hn.get(), H, W, head, zeros, o.get());

    /* token-major [h][w][3] to channel-major [3][H][W], clamped (the pipeline's output range) */
    std::vector<float> tok((size_t)H * W * 3);
    std::memcpy(tok.data(), o.get(), sizeof(float) * tok.size());
    for (int c = 0; c < 3; ++c)
        for (int p = 0; p < H * W; ++p) out[(size_t)c * H * W + p] = std::min(1.0f, std::max(-1.0f, tok[(size_t)p * 3 + c]));
    return H == 256 && W == 256;
}

}  // namespace qwenimage
