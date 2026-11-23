#include "audioldm2_unet.h"

#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace audioldm2 {
namespace {

constexpr int kHeads = 8, kTemb = 512, kGroups = 32, kMaxIn = 8 * 640;

using Shape = std::vector<size_t>;

/* the fp32 tensor by name: its pointer (read in place) */
bool get(const zimage::SafeTensors& st, const std::string& name, const float*& p) {
    Shape s;
    if (!st.f32_in_place(name, p, s)) {
        std::fprintf(stderr, "audioldm2 unet: %s missing or not F32\n", name.c_str());
        return false;
    }
    return true;
}

/* a feature map on the device: [C][H][W] */
struct Feat {
    sddev::Floats buf;
    int c = 0, h = 0, w = 0;
};

class Unet {
public:
    explicit Unet(const zimage::SafeTensors& st) : st_(st), zero_(kMaxIn) {}

    bool has_key(const std::string& name) const { return st_.has(name); }

    /* a resnet block (ResnetBlock2D, output_scale_factor 1): y = shortcut(x) + conv2(silu(GN(. + proj(silu(temb))))) */
    bool resnet(const std::string& p, const float* x, int cin, int H, int W, const float* temb_act, int cout, float* y) {
        const int HW = H * W;
        const float *n1w, *n1b, *c1w, *c1b, *tew, *teb, *n2w, *n2b, *c2w, *c2b;
        if (!get(st_, p + "norm1.weight", n1w) || !get(st_, p + "norm1.bias", n1b) || !get(st_, p + "conv1.weight", c1w) ||
            !get(st_, p + "conv1.bias", c1b) || !get(st_, p + "time_emb_proj.weight", tew) || !get(st_, p + "time_emb_proj.bias", teb) ||
            !get(st_, p + "norm2.weight", n2w) || !get(st_, p + "norm2.bias", n2b) || !get(st_, p + "conv2.weight", c2w) ||
            !get(st_, p + "conv2.bias", c2b))
            return false;
        sddev::Floats N1((size_t)cin * HW), H1((size_t)cout * HW), T((size_t)cout), N2((size_t)cout * HW), H2((size_t)cout * HW);
        sddev::group_norm_f32(x, cin, HW, kGroups, n1w, n1b, 1e-5f, N1.data());
        sddev::silu(N1.data(), (size_t)cin * HW);
        sddev::conv2d_f32(N1.data(), cin, H, W, c1w, cout, c1b, 3, 1, 1, H1.data(), H, W);
        sddev::linear_f32(temb_act, 1, tew, kTemb, cout, teb, T.data());
        sddev::audio_add_channel(H1.data(), T.data(), cout, HW);
        sddev::group_norm_f32(H1.data(), cout, HW, kGroups, n2w, n2b, 1e-5f, N2.data());
        sddev::silu(N2.data(), (size_t)cout * HW);
        sddev::conv2d_f32(N2.data(), cout, H, W, c2w, cout, c2b, 3, 1, 1, H2.data(), H, W);
        if (cin != cout) {
            const float *sw, *sb;
            if (!get(st_, p + "conv_shortcut.weight", sw) || !get(st_, p + "conv_shortcut.bias", sb)) return false;
            sddev::Floats S((size_t)cout * HW);
            sddev::conv2d_f32(x, cin, H, W, sw, cout, sb, 1, 0, 1, S.data(), H, W);
            sddev::add(H2.data(), S.data(), (size_t)cout * HW);
        } else {
            sddev::add(H2.data(), x, (size_t)cout * HW);
        }
        sddev::audio_copy(y, H2.data(), (size_t)cout * HW);
        return true;
    }

    /* a Transformer2DModel with one basic block: y = x + proj_out(attn stack(proj_in(GN(x)))). ctx: the cross-attention's context
     * [Tc][cw] (cw its width), or null for self-attention */
    bool transformer(const std::string& p, const float* x, int C, int H, int W, const float* ctx, int Tc, int cw, float* y) {
        const int HW = H * W, ff = 4 * C;
        const float *gw, *gb, *piw, *pib, *poW, *pob;
        if (!get(st_, p + "norm.weight", gw) || !get(st_, p + "norm.bias", gb) || !get(st_, p + "proj_in.weight", piw) ||
            !get(st_, p + "proj_in.bias", pib) || !get(st_, p + "proj_out.weight", poW) || !get(st_, p + "proj_out.bias", pob))
            return false;
        const std::string b = p + "transformer_blocks.0.";
        const float *n1w, *n1b, *n2w, *n2b, *n3w, *n3b, *a1q, *a1k, *a1v, *a1o, *a1ob, *a2q, *a2k, *a2v, *a2o, *a2ob, *fp, *fpb, *fo, *fob;
        if (!get(st_, b + "norm1.weight", n1w) || !get(st_, b + "norm1.bias", n1b) || !get(st_, b + "norm2.weight", n2w) ||
            !get(st_, b + "norm2.bias", n2b) || !get(st_, b + "norm3.weight", n3w) || !get(st_, b + "norm3.bias", n3b) ||
            !get(st_, b + "attn1.to_q.weight", a1q) || !get(st_, b + "attn1.to_k.weight", a1k) || !get(st_, b + "attn1.to_v.weight", a1v) ||
            !get(st_, b + "attn1.to_out.0.weight", a1o) || !get(st_, b + "attn1.to_out.0.bias", a1ob) ||
            !get(st_, b + "attn2.to_q.weight", a2q) || !get(st_, b + "attn2.to_k.weight", a2k) || !get(st_, b + "attn2.to_v.weight", a2v) ||
            !get(st_, b + "attn2.to_out.0.weight", a2o) || !get(st_, b + "attn2.to_out.0.bias", a2ob) ||
            !get(st_, b + "ff.net.0.proj.weight", fp) || !get(st_, b + "ff.net.0.proj.bias", fpb) ||
            !get(st_, b + "ff.net.2.weight", fo) || !get(st_, b + "ff.net.2.bias", fob))
            return false;
        const size_t N = (size_t)HW * C;
        sddev::Floats Nc((size_t)C * HW), Tk(N), X(N), Ln(N), Q(N), K(N), V(N), A(N), O(N), Pj((size_t)HW * 2 * ff), G((size_t)HW * ff),
            Chw((size_t)C * HW), Kc(ctx ? (size_t)Tc * C : 1), Vc(ctx ? (size_t)Tc * C : 1);
        /* GroupNorm, to tokens, proj_in (a 1 x 1 convolution: a linear over the channels) */
        sddev::group_norm_f32(x, C, HW, kGroups, gw, gb, 1e-6f, Nc.data());
        sddev::to_tokens(Nc.data(), C, HW, Tk.data());
        sddev::linear_f32(Tk.data(), HW, piw, C, C, pib, X.data());
        /* self-attention */
        sddev::audio_layer_norm_f32(X.data(), HW, C, n1w, n1b, 1e-5f, Ln.data());
        sddev::linear_f32(Ln.data(), HW, a1q, C, C, zero_.data(), Q.data());
        sddev::linear_f32(Ln.data(), HW, a1k, C, C, zero_.data(), K.data());
        sddev::linear_f32(Ln.data(), HW, a1v, C, C, zero_.data(), V.data());
        sddev::attention_f32(Q.data(), HW, K.data(), V.data(), HW, C, kHeads, A.data());
        sddev::linear_f32(A.data(), HW, a1o, C, C, a1ob, O.data());
        sddev::add(X.data(), O.data(), N);
        /* cross-attention (self-attention when there is no context) */
        sddev::audio_layer_norm_f32(X.data(), HW, C, n2w, n2b, 1e-5f, Ln.data());
        sddev::linear_f32(Ln.data(), HW, a2q, C, C, zero_.data(), Q.data());
        if (ctx) {
            sddev::linear_f32(ctx, Tc, a2k, cw, C, zero_.data(), Kc.data());
            sddev::linear_f32(ctx, Tc, a2v, cw, C, zero_.data(), Vc.data());
            sddev::attention_f32(Q.data(), HW, Kc.data(), Vc.data(), Tc, C, kHeads, A.data());
        } else {
            sddev::linear_f32(Ln.data(), HW, a2k, C, C, zero_.data(), K.data());
            sddev::linear_f32(Ln.data(), HW, a2v, C, C, zero_.data(), V.data());
            sddev::attention_f32(Q.data(), HW, K.data(), V.data(), HW, C, kHeads, A.data());
        }
        sddev::linear_f32(A.data(), HW, a2o, C, C, a2ob, O.data());
        sddev::add(X.data(), O.data(), N);
        /* feed-forward: GEGLU, then the output projection */
        sddev::audio_layer_norm_f32(X.data(), HW, C, n3w, n3b, 1e-5f, Ln.data());
        sddev::linear_f32(Ln.data(), HW, fp, C, 2 * ff, fpb, Pj.data());
        sddev::geglu(Pj.data(), HW, ff, G.data());
        sddev::linear_f32(G.data(), HW, fo, ff, C, fob, O.data());
        sddev::add(X.data(), O.data(), N);
        /* proj_out, back to channels, the residual */
        sddev::linear_f32(X.data(), HW, poW, C, C, pob, O.data());
        sddev::to_chw(O.data(), C, HW, Chw.data());
        sddev::audio_copy(y, x, (size_t)C * HW);
        sddev::add(y, Chw.data(), (size_t)C * HW);
        return true;
    }

    /* the three transformers of one resnet's layer (attentions[first..first+2]): self, the GPT-2 context, the T5 context */
    bool attentions(const std::string& block, int first, Feat& f, int C, const float* ctx1, int T1, const float* ctx2, int T2) {
        for (int idx = 0; idx < 3; ++idx) {
            Feat nxt;
            nxt.buf = sddev::Floats((size_t)C * f.h * f.w);
            nxt.c = C; nxt.h = f.h; nxt.w = f.w;
            const std::string p = block + ".attentions." + std::to_string(first + idx) + ".";
            const float* ctx = idx == 0 ? nullptr : (idx == 1 ? ctx1 : ctx2);
            const int Tc = idx == 0 ? 0 : (idx == 1 ? T1 : T2);
            const int cw = idx == 0 ? C : (idx == 1 ? 768 : 1024);
            if (!transformer(p, f.buf.data(), C, f.h, f.w, ctx, Tc, cw, nxt.buf.data())) return false;
            f = std::move(nxt);
        }
        return true;
    }

    /* stride-2 3 x 3 convolution (Downsample2D, padding 1) */
    bool downsample(const std::string& p, Feat& f, int cout) {
        const float *w, *b;
        if (!get(st_, p + "conv.weight", w) || !get(st_, p + "conv.bias", b)) return false;
        Feat nxt;
        nxt.c = cout; nxt.h = f.h / 2; nxt.w = f.w / 2;
        nxt.buf = sddev::Floats((size_t)cout * nxt.h * nxt.w);
        sddev::conv2d_f32(f.buf.data(), f.c, f.h, f.w, w, cout, b, 3, 1, 2, nxt.buf.data(), nxt.h, nxt.w);
        f = std::move(nxt);
        return true;
    }

    /* nearest 2x, then a 3 x 3 convolution (Upsample2D) */
    bool upsample(const std::string& p, Feat& f, int cout) {
        const float *w, *b;
        if (!get(st_, p + "conv.weight", w) || !get(st_, p + "conv.bias", b)) return false;
        sddev::Floats up((size_t)f.c * 4 * f.h * f.w);
        sddev::upsample2(f.buf.data(), f.c, f.h, f.w, up.data());
        Feat nxt;
        nxt.c = cout; nxt.h = 2 * f.h; nxt.w = 2 * f.w;
        nxt.buf = sddev::Floats((size_t)cout * nxt.h * nxt.w);
        sddev::conv2d_f32(up.data(), f.c, nxt.h, nxt.w, w, cout, b, 3, 1, 1, nxt.buf.data(), nxt.h, nxt.w);
        f = std::move(nxt);
        return true;
    }

    /* the concatenation of two maps along the channels (CHW: the second follows the first in memory) */
    static Feat concat(Feat& a, Feat& b) {
        Feat c;
        c.c = a.c + b.c; c.h = a.h; c.w = a.w;
        c.buf = sddev::Floats((size_t)c.c * c.h * c.w);
        sddev::audio_copy(c.buf.data(), a.buf.data(), (size_t)a.c * a.h * a.w);
        sddev::audio_copy(c.buf.data() + (size_t)a.c * a.h * a.w, b.buf.data(), (size_t)b.c * b.h * b.w);
        return c;
    }

    bool run(const float* x, int H, int W, float t, const float* ctx1, int T1, const float* ctx2, int T2, float* out);

private:
    const zimage::SafeTensors& st_;
    sddev::Floats zero_;
};

/* the time embedding (Timesteps(128, flip_sin_to_cos, shift 0), then Linear, SiLU, Linear), on the host: the sinusoid, the two
 * linears on the device; returns the SiLU of the embedding (the resnets' input) */
bool time_embed(const zimage::SafeTensors& st, float t, sddev::Floats& out_act) {
    const float *w1, *b1, *w2, *b2;
    if (!get(st, "time_embedding.linear_1.weight", w1) || !get(st, "time_embedding.linear_1.bias", b1) ||
        !get(st, "time_embedding.linear_2.weight", w2) || !get(st, "time_embedding.linear_2.bias", b2))
        return false;
    sddev::Floats s(128), h(kTemb), e(kTemb);
    const int half = 64;
    for (int i = 0; i < half; ++i) {
        const double f = std::exp(-std::log(10000.0) * i / half);
        const double a = (double)t * f;
        s.data()[i] = (float)std::cos(a);        /* flip_sin_to_cos: [cos, sin] */
        s.data()[half + i] = (float)std::sin(a);
    }
    sddev::linear_f32(s.data(), 1, w1, 128, kTemb, b1, h.data());
    sddev::silu(h.data(), kTemb);
    sddev::linear_f32(h.data(), 1, w2, kTemb, kTemb, b2, e.data());
    sddev::silu(e.data(), kTemb);
    out_act = std::move(e);
    return true;
}

bool Unet::run(const float* x_host, int H, int W, float t, const float* ctx1_host, int T1, const float* ctx2_host, int T2, float* out_host) {
    /* the kernels read and write device-accessible memory only: the inputs are copied in, the output copied out */
    sddev::Floats x((size_t)8 * H * W), ctx1((size_t)T1 * 768), ctx2((size_t)T2 * 1024), out((size_t)8 * H * W);
    std::memcpy(x.data(), x_host, sizeof(float) * (size_t)8 * H * W);
    std::memcpy(ctx1.data(), ctx1_host, sizeof(float) * (size_t)T1 * 768);
    std::memcpy(ctx2.data(), ctx2_host, sizeof(float) * (size_t)T2 * 1024);
    const float* ctx1p = ctx1.data();
    const float* ctx2p = ctx2.data();
    sddev::Floats temb;
    if (!time_embed(st_, t, temb)) return false;
    /* conv_in: 8 -> 128 */
    const float *cw, *cb;
    if (!get(st_, "conv_in.weight", cw) || !get(st_, "conv_in.bias", cb)) return false;
    Feat cur;
    cur.c = 128; cur.h = H; cur.w = W;
    cur.buf = sddev::Floats((size_t)128 * H * W);
    sddev::conv2d_f32(x.data(), 8, H, W, cw, 128, cb, 3, 1, 1, cur.buf.data(), H, W);
    std::vector<Feat> skips;
    {
        Feat s;  /* the conv_in output is the first skip */
        s.c = 128; s.h = H; s.w = W;
        s.buf = sddev::Floats((size_t)128 * H * W);
        sddev::audio_copy(s.buf.data(), cur.buf.data(), (size_t)128 * H * W);
        skips.push_back(std::move(s));
    }
    const int chans[4] = {128, 256, 384, 640};
    /* down blocks: DownBlock2D (2 resnets, a downsampler), then three CrossAttnDownBlock2D (2 resnets, 3 attentions per resnet,
     * a downsampler but the last) */
    for (int bi = 0; bi < 4; ++bi) {
        const std::string blk = "down_blocks." + std::to_string(bi);
        for (int r = 0; r < 2; ++r) {
            Feat y;
            y.c = chans[bi]; y.h = cur.h; y.w = cur.w;
            y.buf = sddev::Floats((size_t)y.c * y.h * y.w);
            if (!resnet(blk + ".resnets." + std::to_string(r) + ".", cur.buf.data(), cur.c, cur.h, cur.w, temb.data(), chans[bi], y.buf.data()))
                return false;
            cur = std::move(y);
            if (bi > 0 && !attentions(blk, 3 * r, cur, chans[bi], ctx1p, T1, ctx2p, T2)) return false;
            Feat s;
            s.c = cur.c; s.h = cur.h; s.w = cur.w;
            s.buf = sddev::Floats((size_t)cur.c * cur.h * cur.w);
            sddev::audio_copy(s.buf.data(), cur.buf.data(), (size_t)cur.c * cur.h * cur.w);
            skips.push_back(std::move(s));
        }
        if (has_key(blk + ".downsamplers.0.conv.weight")) {
            if (!downsample(blk + ".downsamplers.0.", cur, chans[bi])) return false;
            Feat s;
            s.c = cur.c; s.h = cur.h; s.w = cur.w;
            s.buf = sddev::Floats((size_t)cur.c * cur.h * cur.w);
            sddev::audio_copy(s.buf.data(), cur.buf.data(), (size_t)cur.c * cur.h * cur.w);
            skips.push_back(std::move(s));
        }
    }
    /* mid: resnet, then for each of its layers: three attentions and a resnet */
    {
        Feat y;
        y.c = 640; y.h = cur.h; y.w = cur.w;
        y.buf = sddev::Floats((size_t)y.c * y.h * y.w);
        if (!resnet("mid_block.resnets.0.", cur.buf.data(), cur.c, cur.h, cur.w, temb.data(), 640, y.buf.data())) return false;
        cur = std::move(y);
        if (!attentions("mid_block", 0, cur, 640, ctx1p, T1, ctx2p, T2)) return false;
        Feat z;
        z.c = 640; z.h = cur.h; z.w = cur.w;
        z.buf = sddev::Floats((size_t)z.c * z.h * z.w);
        if (!resnet("mid_block.resnets.1.", cur.buf.data(), cur.c, cur.h, cur.w, temb.data(), 640, z.buf.data())) return false;
        cur = std::move(z);
    }
    /* up blocks: three resnets each (the skip concatenated first), three attentions per resnet, an upsampler but the last */
    const int uch[4] = {640, 384, 256, 128};
    for (int bi = 0; bi < 4; ++bi) {
        const std::string blk = "up_blocks." + std::to_string(bi);
        for (int r = 0; r < 3; ++r) {
            Feat res = std::move(skips.back());
            skips.pop_back();
            Feat in = concat(cur, res);
            Feat y;
            y.c = uch[bi]; y.h = in.h; y.w = in.w;
            y.buf = sddev::Floats((size_t)y.c * y.h * y.w);
            if (!resnet(blk + ".resnets." + std::to_string(r) + ".", in.buf.data(), in.c, in.h, in.w, temb.data(), uch[bi], y.buf.data())) return false;
            cur = std::move(y);
            if (bi < 3 && !attentions(blk, 3 * r, cur, uch[bi], ctx1p, T1, ctx2p, T2)) return false;
        }
        if (has_key(blk + ".upsamplers.0.conv.weight")) {
            if (!upsample(blk + ".upsamplers.0.", cur, uch[bi])) return false;
        }
    }
    /* conv_norm_out, SiLU, conv_out: 128 -> 8 */
    const float *nw, *nb, *ow, *ob;
    if (!get(st_, "conv_norm_out.weight", nw) || !get(st_, "conv_norm_out.bias", nb) || !get(st_, "conv_out.weight", ow) ||
        !get(st_, "conv_out.bias", ob))
        return false;
    sddev::Floats N((size_t)cur.c * H * W);
    sddev::group_norm_f32(cur.buf.data(), cur.c, H * W, kGroups, nw, nb, 1e-5f, N.data());
    sddev::silu(N.data(), (size_t)cur.c * H * W);
    sddev::conv2d_f32(N.data(), cur.c, H, W, ow, 8, ob, 3, 1, 1, out.data(), H, W);
    std::memcpy(out_host, out.data(), sizeof(float) * (size_t)8 * H * W);
    return true;
}

}  // namespace

bool unet_forward(const zimage::SafeTensors& st, const float* x, int H, int W, float t, const float* ctx1, int T1, const float* ctx2, int T2,
                  float* out) {
    if (!sddev::enabled()) { std::fprintf(stderr, "audioldm2 unet: no device\n"); return false; }
    if (H < 8 || W < 8 || H % 8 || W % 8) { std::fprintf(stderr, "audioldm2 unet: H and W multiples of 8\n"); return false; }
    Unet u(st);
    return u.run(x, H, W, t, ctx1, T1, ctx2, T2, out);
}

}  // namespace audioldm2
