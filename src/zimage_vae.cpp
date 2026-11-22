/* Z-Image's VAE decoder on the device: see zimage_vae.h. The formulas are the diffusers AutoencoderKL decoder's. */
#include "zimage_vae.h"

#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace zimage {

namespace {

/* a device buffer of floats, released at its end; movable */
class DevBuf {
public:
    explicit DevBuf(size_t count) : n_(count) { p_ = static_cast<float*>(sddev::alloc(sizeof(float) * count, false)); }
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) { reset(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { reset(); }
    float* get() const { return p_; }
    size_t size() const { return n_; }

private:
    void reset() { if (p_) sddev::release(p_, sizeof(float) * n_); p_ = nullptr; n_ = 0; }
    float* p_ = nullptr;
    size_t n_ = 0;
};

const int kGroups = 32;
const float kEps = 1e-6f;

/* the F32 tensor by name (in place), its shape */
bool weight(const SafeTensors& st, const std::string& name, const float*& p, std::vector<size_t>& s) {
    if (!st.f32_in_place(name, p, s)) {
        fprintf(stderr, "zimage vae: %s missing\n", name.c_str());
        return false;
    }
    return true;
}

/* y = conv(x) with the named weight [cout][cin][k][k] and bias, stride 1, pad k / 2; x [cin][H][W] */
bool conv_named(const SafeTensors& st, const std::string& name, const float* x, int cin, int H, int W, DevBuf& y) {
    const float *w = nullptr, *b = nullptr;
    std::vector<size_t> sw, sb;
    if (!weight(st, name + ".weight", w, sw) || !weight(st, name + ".bias", b, sb)) return false;
    if (sw.size() != 4 || (int)sw[1] != cin) {
        fprintf(stderr, "zimage vae: %s has the wrong input channels\n", name.c_str());
        return false;
    }
    const int cout = (int)sw[0], k = (int)sw[2], pad = k / 2;
    y = DevBuf((size_t)cout * H * W);
    sddev::conv2d_f32(x, cin, H, W, w, cout, b, k, pad, 1, y.get(), H, W);
    return true;
}

/* GroupNorm (32 groups, eps 1e-6) with the named affine weights, channel-major x [C][HW] */
bool norm_named(const SafeTensors& st, const std::string& name, const float* x, int C, int HW, DevBuf& y) {
    const float *g = nullptr, *b = nullptr;
    std::vector<size_t> sg, sb;
    if (!weight(st, name + ".weight", g, sg) || !weight(st, name + ".bias", b, sb)) return false;
    if ((int)sg[0] != C) {
        fprintf(stderr, "zimage vae: %s has %zu channels, the input %d\n", name.c_str(), sg[0], C);
        return false;
    }
    y = DevBuf((size_t)C * HW);
    sddev::group_norm_f32(x, C, HW, kGroups, g, b, kEps, y.get());
    return true;
}

/* the ResNet block: h = conv1(silu(norm1(x))); h = conv2(silu(norm2(h))); x' = (nin_shortcut(x) if the channels change) + h */
bool resnet(const SafeTensors& st, const std::string& pre, DevBuf& x, int& C, int H, int W) {
    const int HW = H * W;
    std::vector<size_t> s1;
    const float* g1 = nullptr;
    if (!st.f32_in_place(pre + ".norm1.weight", g1, s1)) return false;
    const int cin = (int)s1[0];
    if (cin != C) { fprintf(stderr, "zimage vae: %s expects %d channels, has %d\n", pre.c_str(), cin, C); return false; }
    DevBuf h1((size_t)0);
    if (!norm_named(st, pre + ".norm1", x.get(), cin, HW, h1)) return false;
    sddev::silu(h1.get(), (size_t)cin * HW);
    DevBuf c1((size_t)0);
    if (!conv_named(st, pre + ".conv1", h1.get(), cin, H, W, c1)) return false;
    const float* g2 = nullptr;
    std::vector<size_t> s2;
    if (!st.f32_in_place(pre + ".norm2.weight", g2, s2)) return false;
    const int cout = (int)s2[0];
    DevBuf h2((size_t)0);
    if (!norm_named(st, pre + ".norm2", c1.get(), cout, HW, h2)) return false;
    sddev::silu(h2.get(), (size_t)cout * HW);
    DevBuf c2((size_t)0);
    if (!conv_named(st, pre + ".conv2", h2.get(), cout, H, W, c2)) return false;
    if (cin != cout) {
        DevBuf sc((size_t)0);
        if (!conv_named(st, pre + ".nin_shortcut", x.get(), cin, H, W, sc)) return false;
        sddev::add(c2.get(), sc.get(), (size_t)cout * HW);
    } else {
        sddev::add(c2.get(), x.get(), (size_t)cout * HW);
    }
    x = std::move(c2);
    C = cout;
    return true;
}

/* the mid attention: x += proj_out(attention(q, k, v)) over the HW tokens (one head of C channels) */
bool mid_attn(const SafeTensors& st, DevBuf& x, int C, int H, int W) {
    const int HW = H * W;
    DevBuf g((size_t)0);
    if (!norm_named(st, "decoder.mid.attn_1.norm", x.get(), C, HW, g)) return false;
    DevBuf q((size_t)0), k((size_t)0), v((size_t)0);
    if (!conv_named(st, "decoder.mid.attn_1.q", g.get(), C, H, W, q) || !conv_named(st, "decoder.mid.attn_1.k", g.get(), C, H, W, k) ||
        !conv_named(st, "decoder.mid.attn_1.v", g.get(), C, H, W, v))
        return false;
    DevBuf qt((size_t)C * HW), kt((size_t)C * HW), vt((size_t)C * HW), ot((size_t)C * HW), o((size_t)C * HW);
    sddev::to_tokens(q.get(), C, HW, qt.get());
    sddev::to_tokens(k.get(), C, HW, kt.get());
    sddev::to_tokens(v.get(), C, HW, vt.get());
    sddev::attention(qt.get(), HW, kt.get(), vt.get(), HW, C, 1, ot.get());
    sddev::to_chw(ot.get(), C, HW, o.get());
    DevBuf p((size_t)0);
    if (!conv_named(st, "decoder.mid.attn_1.proj_out", o.get(), C, H, W, p)) return false;
    sddev::add(x.get(), p.get(), (size_t)C * HW);
    return true;
}

}  // namespace

bool dev_vae_decode(const SafeTensors& st, const float* latent, float* out, int H0, int W0) {
    if (!sddev::enabled()) { fprintf(stderr, "zimage vae: no device\n"); return false; }
    const float scale = 0.3611f, shift = 0.1159f;  /* the Flux autoencoder's scaling and shift (its config) */
    DevBuf x((size_t)16 * H0 * W0);
    for (size_t i = 0; i < x.size(); ++i) x.get()[i] = latent[i] / scale + shift;
    int C = 16, H = H0, W = W0;
    DevBuf c_in((size_t)0);
    if (!conv_named(st, "decoder.conv_in", x.get(), 16, H, W, c_in)) return false;
    x = std::move(c_in);
    C = 512;
    if (!resnet(st, "decoder.mid.block_1", x, C, H, W) || !mid_attn(st, x, C, H, W) || !resnet(st, "decoder.mid.block_2", x, C, H, W))
        return false;
    for (int up = 3; up >= 0; --up) {
        for (int b = 0; b < 3; ++b)
            if (!resnet(st, "decoder.up." + std::to_string(up) + ".block." + std::to_string(b), x, C, H, W)) return false;
        if (up != 0) {
            DevBuf u((size_t)C * 4 * H * W);
            sddev::upsample2(x.get(), C, H, W, u.get());
            H *= 2;
            W *= 2;
            DevBuf c_up((size_t)0);
            if (!conv_named(st, "decoder.up." + std::to_string(up) + ".upsample.conv", u.get(), C, H, W, c_up)) return false;
            x = std::move(c_up);
        }
    }
    DevBuf g((size_t)0);
    if (!norm_named(st, "decoder.norm_out", x.get(), C, H * W, g)) return false;
    sddev::silu(g.get(), (size_t)C * H * W);
    DevBuf o((size_t)0);
    if (!conv_named(st, "decoder.conv_out", g.get(), C, H, W, o)) return false;
    std::memcpy(out, o.get(), sizeof(float) * (size_t)3 * H * W);
    return true;
}

}  // namespace zimage
