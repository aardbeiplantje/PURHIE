#include "audioldm2_vae.h"

#include "sd_device.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace audioldm2 {
namespace {

constexpr float kScaling = 0.4110932946205139f;  /* vae/config.json: scaling_factor */
constexpr float kEps = 1e-6f;                    /* the decoder's GroupNorm epsilon */
constexpr int kGroups = 32;

bool get(const zimage::SafeTensors& st, const std::string& name, const float*& p) {
    std::vector<size_t> s;
    if (!st.f32_in_place(name, p, s)) {
        std::fprintf(stderr, "audioldm2 vae: %s missing or not F32\n", name.c_str());
        return false;
    }
    return true;
}

struct Feat {
    sddev::Floats buf;
    int c = 0, h = 0, w = 0;
};

/* ResnetBlock2D without the time embedding: y = shortcut(x) + conv2(silu(GN(conv1(silu(GN(x))))))  */
bool resnet(const zimage::SafeTensors& st, const std::string& p, const float* x, int cin, int H, int W, int cout, float* y) {
    const int HW = H * W;
    const float *n1w, *n1b, *c1w, *c1b, *n2w, *n2b, *c2w, *c2b;
    if (!get(st, p + "norm1.weight", n1w) || !get(st, p + "norm1.bias", n1b) || !get(st, p + "conv1.weight", c1w) ||
        !get(st, p + "conv1.bias", c1b) || !get(st, p + "norm2.weight", n2w) || !get(st, p + "norm2.bias", n2b) ||
        !get(st, p + "conv2.weight", c2w) || !get(st, p + "conv2.bias", c2b))
        return false;
    sddev::Floats N1((size_t)cin * HW), H1((size_t)cout * HW), N2((size_t)cout * HW), H2((size_t)cout * HW);
    sddev::group_norm_f32(x, cin, HW, kGroups, n1w, n1b, kEps, N1.data());
    sddev::silu(N1.data(), (size_t)cin * HW);
    sddev::conv2d_f32(N1.data(), cin, H, W, c1w, cout, c1b, 3, 1, 1, H1.data(), H, W);
    sddev::group_norm_f32(H1.data(), cout, HW, kGroups, n2w, n2b, kEps, N2.data());
    sddev::silu(N2.data(), (size_t)cout * HW);
    sddev::conv2d_f32(N2.data(), cout, H, W, c2w, cout, c2b, 3, 1, 1, H2.data(), H, W);
    if (cin != cout) {
        const float *sw, *sb;
        if (!get(st, p + "conv_shortcut.weight", sw) || !get(st, p + "conv_shortcut.bias", sb)) return false;
        sddev::Floats S((size_t)cout * HW);
        sddev::conv2d_f32(x, cin, H, W, sw, cout, sb, 1, 0, 1, S.data(), H, W);
        sddev::add(H2.data(), S.data(), (size_t)cout * HW);
    } else {
        sddev::add(H2.data(), x, (size_t)cout * HW);
    }
    sddev::audio_copy(y, H2.data(), (size_t)cout * HW);
    return true;
}

/* the mid-block attention (one head, the residual): y = x + to_out(attn(GN(x))) */
bool mid_attention(const zimage::SafeTensors& st, const std::string& p, const float* x, int C, int H, int W, float* y) {
    const int HW = H * W;
    const float *gw, *gb, *qw, *qb, *kw, *kb, *vw, *vb, *ow, *ob;
    if (!get(st, p + "group_norm.weight", gw) || !get(st, p + "group_norm.bias", gb) || !get(st, p + "to_q.weight", qw) ||
        !get(st, p + "to_q.bias", qb) || !get(st, p + "to_k.weight", kw) || !get(st, p + "to_k.bias", kb) ||
        !get(st, p + "to_v.weight", vw) || !get(st, p + "to_v.bias", vb) || !get(st, p + "to_out.0.weight", ow) ||
        !get(st, p + "to_out.0.bias", ob))
        return false;
    const size_t N = (size_t)HW * C;
    sddev::Floats Nc((size_t)C * HW), Tk(N), Q(N), K(N), V(N), A(N), O(N), Chw((size_t)C * HW);
    sddev::group_norm_f32(x, C, HW, kGroups, gw, gb, kEps, Nc.data());
    sddev::to_tokens(Nc.data(), C, HW, Tk.data());
    sddev::linear_f32(Tk.data(), HW, qw, C, C, qb, Q.data());
    sddev::linear_f32(Tk.data(), HW, kw, C, C, kb, K.data());
    sddev::linear_f32(Tk.data(), HW, vw, C, C, vb, V.data());
    sddev::attention_f32(Q.data(), HW, K.data(), V.data(), HW, C, 1, A.data());
    sddev::linear_f32(A.data(), HW, ow, C, C, ob, O.data());
    sddev::to_chw(O.data(), C, HW, Chw.data());
    sddev::audio_copy(y, x, (size_t)C * HW);
    sddev::add(y, Chw.data(), (size_t)C * HW);
    return true;
}

/* nearest 2x and a 3 x 3 convolution (Upsample2D) */
bool upsample(const zimage::SafeTensors& st, const std::string& p, Feat& f, int cout) {
    const float *w, *b;
    if (!get(st, p + "conv.weight", w) || !get(st, p + "conv.bias", b)) return false;
    sddev::Floats up((size_t)f.c * 4 * f.h * f.w);
    sddev::upsample2(f.buf.data(), f.c, f.h, f.w, up.data());
    Feat nxt;
    nxt.c = cout; nxt.h = 2 * f.h; nxt.w = 2 * f.w;
    nxt.buf = sddev::Floats((size_t)cout * nxt.h * nxt.w);
    sddev::conv2d_f32(up.data(), f.c, nxt.h, nxt.w, w, cout, b, 3, 1, 1, nxt.buf.data(), nxt.h, nxt.w);
    f = std::move(nxt);
    return true;
}

}  // namespace

bool vae_decode(const zimage::SafeTensors& st, const float* z, int h, int w, float* mel) {
    if (!sddev::enabled()) { std::fprintf(stderr, "audioldm2 vae: no device\n"); return false; }
    if (h < 1 || w < 1) return false;
    /* the input on the device, divided by the scaling factor */
    sddev::Floats zin((size_t)8 * h * w);
    for (size_t i = 0; i < zin.size(); ++i) zin.data()[i] = z[i] / kScaling;
    /* post_quant_conv: 1 x 1, 8 -> 8 */
    const float *pqw, *pqb, *cw, *cb;
    if (!get(st, "post_quant_conv.weight", pqw) || !get(st, "post_quant_conv.bias", pqb) || !get(st, "decoder.conv_in.weight", cw) ||
        !get(st, "decoder.conv_in.bias", cb))
        return false;
    sddev::Floats P((size_t)8 * h * w);
    sddev::conv2d_f32(zin.data(), 8, h, w, pqw, 8, pqb, 1, 0, 1, P.data(), h, w);
    Feat cur;
    cur.c = 512; cur.h = h; cur.w = w;
    cur.buf = sddev::Floats((size_t)512 * h * w);
    sddev::conv2d_f32(P.data(), 8, h, w, cw, 512, cb, 3, 1, 1, cur.buf.data(), h, w);
    /* mid block: resnet, attention, resnet (512 channels) */
    {
        Feat y;
        y.c = 512; y.h = h; y.w = w;
        y.buf = sddev::Floats((size_t)512 * h * w);
        if (!resnet(st, "decoder.mid_block.resnets.0.", cur.buf.data(), 512, h, w, 512, y.buf.data())) return false;
        cur = std::move(y);
        Feat a;
        a.c = 512; a.h = h; a.w = w;
        a.buf = sddev::Floats((size_t)512 * h * w);
        if (!mid_attention(st, "decoder.mid_block.attentions.0.", cur.buf.data(), 512, h, w, a.buf.data())) return false;
        cur = std::move(a);
        Feat r;
        r.c = 512; r.h = h; r.w = w;
        r.buf = sddev::Floats((size_t)512 * h * w);
        if (!resnet(st, "decoder.mid_block.resnets.1.", cur.buf.data(), 512, h, w, 512, r.buf.data())) return false;
        cur = std::move(r);
    }
    /* the up blocks: three resnets each (the channels 512, 256, 128), an upsampler in the first two */
    const int uch[3] = {512, 256, 128};
    for (int bi = 0; bi < 3; ++bi) {
        const std::string blk = "decoder.up_blocks." + std::to_string(bi) + ".";
        for (int r = 0; r < 3; ++r) {
            Feat y;
            y.c = uch[bi]; y.h = cur.h; y.w = cur.w;
            y.buf = sddev::Floats((size_t)y.c * y.h * y.w);
            if (!resnet(st, blk + "resnets." + std::to_string(r) + ".", cur.buf.data(), cur.c, cur.h, cur.w, uch[bi], y.buf.data())) return false;
            cur = std::move(y);
        }
        if (bi < 2 && !upsample(st, blk + "upsamplers.0.", cur, uch[bi])) return false;
    }
    /* conv_norm_out, SiLU, conv_out: 128 -> 1 */
    const float *nw, *nb, *ow, *ob;
    if (!get(st, "decoder.conv_norm_out.weight", nw) || !get(st, "decoder.conv_norm_out.bias", nb) || !get(st, "decoder.conv_out.weight", ow) ||
        !get(st, "decoder.conv_out.bias", ob))
        return false;
    const int HW = cur.h * cur.w;
    sddev::Floats N((size_t)cur.c * HW), M((size_t)HW);
    sddev::group_norm_f32(cur.buf.data(), cur.c, HW, kGroups, nw, nb, kEps, N.data());
    sddev::silu(N.data(), (size_t)cur.c * HW);
    sddev::conv2d_f32(N.data(), cur.c, cur.h, cur.w, ow, 1, ob, 3, 1, 1, M.data(), cur.h, cur.w);
    std::memcpy(mel, M.data(), sizeof(float) * (size_t)HW);
    return true;
}

}  // namespace audioldm2
