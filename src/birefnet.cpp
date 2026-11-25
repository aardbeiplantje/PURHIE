// BiRefNet on the device (birefnet.h). The order of the operations is the Hugging Face model's (birefnet.py of ZhengPeng7/BiRefNet):
// the Swin-L backbone (patch embedding, four stages of shifted-window blocks with the relative position bias, the patch merging, a
// norm on each stage's output) on the image and on its half-size copy, the two concatenated per stage; the context skips (the first
// three stages resized to the last one, in front of it); the squeeze block; the decoder (at 32, 64, 128 and 256: the image's patches
// through two convolutions, the decoder block with the deformable ASPP, the gradient attention's gate, the lateral block of the next
// stage), the image's own features at 1024 and the output convolution. The maps are channel-major; each buffer lives while it is used.
#include "birefnet.h"

#include "sd_device.h"
#include "trellis_image.h"
#include "zimage_dit.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace trellis {
namespace {

// a device buffer (sddev::alloc memory, mapped for the GPU and the host), released when it goes out of scope; movable
class DevBuf {
public:
    DevBuf() = default;
    explicit DevBuf(size_t bytes) : p_(sddev::alloc(bytes ? bytes : 1, false)), bytes_(bytes ? bytes : 1) {}
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), bytes_(o.bytes_) {
        o.p_ = nullptr;
        o.bytes_ = 0;
    }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) {
            reset();
            p_ = o.p_;
            bytes_ = o.bytes_;
            o.p_ = nullptr;
            o.bytes_ = 0;
        }
        return *this;
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { reset(); }
    void reset() {
        if (p_) sddev::release(p_, bytes_);
        p_ = nullptr;
        bytes_ = 0;
    }
    float* f() const { return static_cast<float*>(p_); }

private:
    void* p_ = nullptr;
    size_t bytes_ = 0;
};

DevBuf floats(size_t n) { return DevBuf(sizeof(float) * n); }

float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {  // subnormal: normalise
            int ee = -1;
            uint32_t mm = m;
            do {
                ++ee;
                mm <<= 1;
            } while ((mm & 0x400) == 0);
            bits = s | (uint32_t)(127 - 15 - ee) << 23 | (mm & 0x3ff) << 13;
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | m << 13;
    } else {
        bits = s | (e - 15 + 127) << 23 | m << 13;
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

constexpr int kWin = 12;
constexpr int kEmbed = 192;
const int kDepth[4] = {2, 2, 18, 2};
const int kHeads[4] = {6, 12, 24, 48};
constexpr int kS = 1024;  // the model's input size

struct Lin {  // a linear or convolution's weight and bias, F16 in place (bias null: none)
    const uint16_t* w = nullptr;
    const uint16_t* b = nullptr;
};
struct BN {  // a batch norm as a scale and a shift per channel (inference)
    const float* s = nullptr;
    const float* b = nullptr;
};
struct SwinBlock {
    Lin n1, n2, qkv, proj, fc1, fc2;
    const uint16_t* bias = nullptr;  // the relative position bias table [(2 ws - 1)^2][heads]
};
struct SwinStage {
    std::vector<SwinBlock> blocks;
    Lin out, ds_norm;
    const uint16_t* ds_red = nullptr;
};
struct Deform {
    Lin off, mod;
    const uint16_t* w = nullptr;
    BN bn;
    int k = 1;
};
struct Aspp {
    Deform a1, d[3];
    const uint16_t* gap_w = nullptr;
    BN gap_bn;
    const uint16_t* conv1 = nullptr;
    BN bn1;
};
struct DecBlk {
    Lin in, out;
    BN bn_in, bn_out;
    Aspp att;
};
struct Ipt {
    Lin c1, c2;
};

}  // namespace

struct BiRefNet::Impl {
    explicit Impl(const std::string& path) : st(path) {}
    ~Impl() {
        for (auto& o : owned) sddev::release(o.first, o.second);
    }
    zimage::SafeTensors st;
    std::string err;
    std::vector<std::pair<void*, size_t>> owned;  // the batch norms' vectors

    Lin pe, pe_norm;
    SwinStage stages[4];
    DecBlk squeeze, dec4, dec3, dec2, dec1;
    Ipt ipt5, ipt4, ipt3, ipt2, ipt1;
    Lin lat4, lat3, lat2, gdt[3], gdt_attn[3], out1;
    BN gdt_bn[3];

    const uint16_t* f16(const std::string& name, size_t count) {
        if (!err.empty()) return nullptr;
        const uint16_t* p = nullptr;
        std::vector<size_t> shape;
        if (!st.f16_in_place(name, p, shape)) {
            err = "the BiRefNet checkpoint has no F16 tensor " + name;
            return nullptr;
        }
        size_t n = 1;
        for (size_t s : shape) n *= s;
        if (n != count) {
            err = "the BiRefNet tensor " + name + " has " + std::to_string(n) + " values, not " + std::to_string(count);
            return nullptr;
        }
        return p;
    }
    Lin lin(const std::string& p, size_t wcount, size_t bcount) {
        Lin l;
        l.w = f16(p + ".weight", wcount);
        if (bcount) l.b = f16(p + ".bias", bcount);
        return l;
    }
    BN bn(const std::string& p, int C) {
        BN o;
        const uint16_t* w = f16(p + ".weight", C);
        const uint16_t* b = f16(p + ".bias", C);
        const uint16_t* mean = f16(p + ".running_mean", C);
        const uint16_t* var = f16(p + ".running_var", C);
        if (!err.empty()) return o;
        const size_t bytes = sizeof(float) * 2 * (size_t)C;
        float* v = static_cast<float*>(sddev::alloc(bytes, false));
        owned.push_back({v, bytes});
        for (int c = 0; c < C; ++c) {
            const float s = half_to_float(w[c]) / std::sqrt(half_to_float(var[c]) + 1e-5f);
            v[c] = s;
            v[C + c] = half_to_float(b[c]) - half_to_float(mean[c]) * s;
        }
        o.s = v;
        o.b = v + C;
        return o;
    }
    Deform deform(const std::string& p, int k) {
        Deform d;
        d.k = k;
        const size_t kk = (size_t)k * k;
        d.off = lin(p + ".atrous_conv.offset_conv", 2 * kk * 64 * kk, 2 * kk);
        d.mod = lin(p + ".atrous_conv.modulator_conv", kk * 64 * kk, kk);
        d.w = f16(p + ".atrous_conv.regular_conv.weight", 256 * 64 * kk);
        d.bn = bn(p + ".bn", 256);
        return d;
    }
    DecBlk dec(const std::string& p, int cin, int cout) {
        DecBlk d;
        d.in = lin(p + ".conv_in", (size_t)64 * cin * 9, 64);
        d.bn_in = bn(p + ".bn_in", 64);
        const std::string a = p + ".dec_att";
        d.att.a1 = deform(a + ".aspp1", 1);
        d.att.d[0] = deform(a + ".aspp_deforms.0", 1);
        d.att.d[1] = deform(a + ".aspp_deforms.1", 3);
        d.att.d[2] = deform(a + ".aspp_deforms.2", 7);
        d.att.gap_w = f16(a + ".global_avg_pool.1.weight", 256 * 64);
        d.att.gap_bn = bn(a + ".global_avg_pool.2", 256);
        d.att.conv1 = f16(a + ".conv1.weight", 64 * 1280);
        d.att.bn1 = bn(a + ".bn1", 64);
        d.out = lin(p + ".conv_out", (size_t)cout * 64 * 9, cout);
        d.bn_out = bn(p + ".bn_out", cout);
        return d;
    }
    Ipt ipt(const std::string& p, int cin, int cout) {
        Ipt b;
        b.c1 = lin(p + ".conv1", (size_t)64 * cin * 9, 64);
        b.c2 = lin(p + ".conv_out", (size_t)cout * 64 * 9, cout);
        return b;
    }
};

BiRefNet::BiRefNet(const std::string& safetensors_path) : p_(new Impl(safetensors_path)) {
    Impl& m = *p_;
    if (!m.st.ok()) {
        error_ = m.st.error();
        return;
    }
    if (!m.st.register_device()) {
        error_ = "the BiRefNet checkpoint could not be registered for the device";
        return;
    }
    m.pe = m.lin("bb.patch_embed.proj", (size_t)kEmbed * 3 * 16, kEmbed);
    m.pe_norm = m.lin("bb.patch_embed.norm", kEmbed, kEmbed);
    for (int i = 0; i < 4; ++i) {
        const int C = kEmbed << i, heads = kHeads[i];
        SwinStage& sg = m.stages[i];
        for (int j = 0; j < kDepth[i]; ++j) {
            const std::string p = "bb.layers." + std::to_string(i) + ".blocks." + std::to_string(j) + ".";
            SwinBlock b;
            b.n1 = m.lin(p + "norm1", C, C);
            b.qkv = m.lin(p + "attn.qkv", (size_t)3 * C * C, 3 * C);
            b.bias = m.f16(p + "attn.relative_position_bias_table", (size_t)(2 * kWin - 1) * (2 * kWin - 1) * heads);
            b.proj = m.lin(p + "attn.proj", (size_t)C * C, C);
            b.n2 = m.lin(p + "norm2", C, C);
            b.fc1 = m.lin(p + "mlp.fc1", (size_t)4 * C * C, 4 * C);
            b.fc2 = m.lin(p + "mlp.fc2", (size_t)4 * C * C, C);
            sg.blocks.push_back(b);
        }
        sg.out = m.lin("bb.norm" + std::to_string(i), C, C);
        if (i < 3) {
            const std::string p = "bb.layers." + std::to_string(i) + ".downsample.";
            sg.ds_norm = m.lin(p + "norm", 4 * C, 4 * C);
            sg.ds_red = m.f16(p + "reduction.weight", (size_t)2 * C * 4 * C);
        }
    }
    // the decoder's channels: the two scales concatenated (3072, 1536, 768, 384); the squeeze takes the context skips (5760)
    m.squeeze = m.dec("squeeze_module.0", 5760, 3072);
    m.ipt5 = m.ipt("decoder.ipt_blk5", 3072, 384);
    m.ipt4 = m.ipt("decoder.ipt_blk4", 768, 384);
    m.ipt3 = m.ipt("decoder.ipt_blk3", 192, 192);
    m.ipt2 = m.ipt("decoder.ipt_blk2", 48, 96);
    m.ipt1 = m.ipt("decoder.ipt_blk1", 3, 48);
    m.dec4 = m.dec("decoder.decoder_block4", 3456, 1536);
    m.dec3 = m.dec("decoder.decoder_block3", 1920, 768);
    m.dec2 = m.dec("decoder.decoder_block2", 960, 384);
    m.dec1 = m.dec("decoder.decoder_block1", 480, 192);
    m.lat4 = m.lin("decoder.lateral_block4.conv", (size_t)1536 * 1536, 1536);
    m.lat3 = m.lin("decoder.lateral_block3.conv", (size_t)768 * 768, 768);
    m.lat2 = m.lin("decoder.lateral_block2.conv", (size_t)384 * 384, 384);
    const int gc[3] = {1536, 768, 384};
    for (int i = 0; i < 3; ++i) {
        const std::string k = std::to_string(4 - i);
        m.gdt[i] = m.lin("decoder.gdt_convs_" + k + ".0", (size_t)16 * gc[i] * 9, 16);
        m.gdt_bn[i] = m.bn("decoder.gdt_convs_" + k + ".1", 16);
        m.gdt_attn[i] = m.lin("decoder.gdt_convs_attn_" + k + ".0", 16, 1);
    }
    m.out1 = m.lin("decoder.conv_out1.0", 240, 1);
    if (!m.err.empty()) {
        error_ = m.err;
        return;
    }
    ok_ = true;
}

BiRefNet::~BiRefNet() = default;

namespace {

// the Swin-L backbone on a [3][S][S] map: the four stages' outputs (after their norms) as [C][H][W] into outs
void run_swin(const Lin& pe_proj, const Lin& pe_norm, const SwinStage* stages, const float* x, int S, float* const outs[4]) {
    int H = S / 4, W = S / 4, C = kEmbed;
    DevBuf xb = floats((size_t)H * W * C);
    {
        DevBuf pe = floats((size_t)C * H * W), tok = floats((size_t)H * W * C);
        sddev::conv2d(x, 3, S, S, pe_proj.w, C, pe_proj.b, 4, 0, 4, pe.f(), H, W);
        sddev::to_tokens(pe.f(), C, H * W, tok.f());
        sddev::layer_norm(tok.f(), H * W, C, pe_norm.w, pe_norm.b, 1e-5f, xb.f());
    }
    const float scale = 1.0f / std::sqrt(32.0f);
    for (int i = 0; i < 4; ++i) {
        const SwinStage& sg = stages[i];
        const int T = H * W, heads = kHeads[i];
        const int Hp = (H + kWin - 1) / kWin * kWin, Wp = (W + kWin - 1) / kWin * kWin, TP = Hp * Wp;
        {
            DevBuf n = floats((size_t)T * C), xw = floats((size_t)TP * C), qkv = floats((size_t)TP * 3 * C), att = floats((size_t)TP * C);
            DevBuf hid = floats((size_t)T * 4 * C);
            for (size_t j = 0; j < sg.blocks.size(); ++j) {
                const SwinBlock& b = sg.blocks[j];
                const int shift = (j % 2) ? kWin / 2 : 0;
                // the attention of the (shifted) windows, its output added back to the tokens
                sddev::layer_norm(xb.f(), T, C, b.n1.w, b.n1.b, 1e-5f, n.f());
                sddev::bir_window_gather(n.f(), H, W, C, Hp, Wp, kWin, shift, xw.f());
                sddev::linear_tok(xw.f(), TP, b.qkv.w, C, 3 * C, b.qkv.b, qkv.f());
                sddev::bir_window_attn(qkv.f(), att.f(), Hp, Wp, kWin, shift, heads, C, b.bias, scale);
                sddev::linear_tok(att.f(), TP, b.proj.w, C, C, b.proj.b, xw.f());
                sddev::bir_window_scatter_add(xb.f(), H, W, C, Hp, Wp, kWin, shift, xw.f());
                // the MLP
                sddev::layer_norm(xb.f(), T, C, b.n2.w, b.n2.b, 1e-5f, n.f());
                sddev::linear_tok(n.f(), T, b.fc1.w, C, 4 * C, b.fc1.b, hid.f());
                sddev::audio_gelu_erf(hid.f(), (size_t)T * 4 * C);
                sddev::linear_tok(hid.f(), T, b.fc2.w, 4 * C, C, b.fc2.b, n.f());
                sddev::add(xb.f(), n.f(), (size_t)T * C);
            }
            // the stage's output: its norm, channel-major
            sddev::layer_norm(xb.f(), T, C, sg.out.w, sg.out.b, 1e-5f, n.f());
            sddev::to_chw(n.f(), C, T, outs[i]);
        }
        if (i == 3) break;
        // the patch merging: the 2 x 2 neighbours concatenated, their norm, the reduction to 2C
        DevBuf pm = floats((size_t)(T / 4) * 4 * C), pn = floats((size_t)(T / 4) * 4 * C), nx = floats((size_t)(T / 4) * 2 * C);
        sddev::bir_patch_merge(xb.f(), H, W, C, pm.f());
        sddev::layer_norm(pm.f(), T / 4, 4 * C, sg.ds_norm.w, sg.ds_norm.b, 1e-5f, pn.f());
        sddev::linear_tok(pn.f(), T / 4, sg.ds_red, 4 * C, 2 * C, nullptr, nx.f());
        xb = std::move(nx);
        H /= 2;
        W /= 2;
        C *= 2;
    }
}

// a modulated deformable convolution (64 in, 256 out, k x k, the same size) with its batch norm and ReLU: out [256][H][W]
void deform(const Deform& f, const float* x, int H, int W, float* out) {
    const int k = f.k, kk = k * k, pad = k / 2, ci = 64, co = 256;
    const long HW = (long)H * W, K = (long)ci * kk;
    DevBuf off = floats((size_t)2 * kk * HW), mod = floats((size_t)kk * HW), ytok = floats((size_t)HW * co);
    sddev::conv2d(x, ci, H, W, f.off.w, 2 * kk, f.off.b, k, pad, 1, off.f(), H, W);
    sddev::conv2d(x, ci, H, W, f.mod.w, kk, f.mod.b, k, pad, 1, mod.f(), H, W);
    // the columns in chunks of at most 64 MB: each chunk's GEMM stays well under the 50 ms a launch may take (the 7 x 7 at 256 x 256)
    const long rows = std::max<long>(1, std::min<long>(HW, (16L << 20) / K));
    DevBuf cols = floats((size_t)rows * K);
    for (long r0 = 0; r0 < HW; r0 += rows) {
        const int n = (int)std::min<long>(rows, HW - r0);
        sddev::bir_deform_cols(x, ci, H, W, off.f(), mod.f(), k, pad, r0, n, cols.f());
        sddev::linear_tok(cols.f(), n, f.w, (int)K, co, nullptr, ytok.f() + r0 * co);
    }
    sddev::to_chw(ytok.f(), co, (int)HW, out);
    sddev::bir_bn_act(out, co, HW, f.bn.s, f.bn.b, true);
}

// the deformable ASPP on [64][H][W]: four deformable branches (1, 1, 3, 7) and the global pooling's, concatenated, to 64
void aspp(const Aspp& a, const float* x, int H, int W, float* out) {
    const long HW = (long)H * W;
    DevBuf cat = floats((size_t)1280 * HW);
    deform(a.a1, x, H, W, cat.f());
    for (int i = 0; i < 3; ++i) deform(a.d[i], x, H, W, cat.f() + (size_t)(i + 1) * 256 * HW);
    DevBuf mean = floats(64), v = floats(256);
    sddev::bir_channel_mean(x, 64, HW, mean.f());
    sddev::conv2d(mean.f(), 64, 1, 1, a.gap_w, 256, nullptr, 1, 0, 1, v.f(), 1, 1);
    sddev::bir_bn_act(v.f(), 256, 1, a.gap_bn.s, a.gap_bn.b, true);
    sddev::bir_fill_channels(cat.f() + (size_t)4 * 256 * HW, 256, HW, v.f());
    sddev::conv2d(cat.f(), 1280, H, W, a.conv1, 64, nullptr, 1, 0, 1, out, H, W);
    sddev::bir_bn_act(out, 64, HW, a.bn1.s, a.bn1.b, true);
}

// a decoder block (BasicDecBlk with the deformable ASPP): in [cin][H][W] -> out [cout][H][W]
void dec_block(const DecBlk& d, const float* in, int cin, int H, int W, int cout, float* out) {
    const long HW = (long)H * W;
    DevBuf t = floats((size_t)64 * HW), a = floats((size_t)64 * HW);
    sddev::conv2d(in, cin, H, W, d.in.w, 64, d.in.b, 3, 1, 1, t.f(), H, W);
    sddev::bir_bn_act(t.f(), 64, HW, d.bn_in.s, d.bn_in.b, true);
    aspp(d.att, t.f(), H, W, a.f());
    sddev::conv2d(a.f(), 64, H, W, d.out.w, cout, d.out.b, 3, 1, 1, out, H, W);
    sddev::bir_bn_act(out, cout, HW, d.bn_out.s, d.bn_out.b, false);
}

// the image's patches at a decoder stage (grid x grid patches as channels) through the two convolutions: out [cout][S/grid][S/grid]
void ipt_block(const Ipt& b, const float* img, int grid, int cout, float* out) {
    const int H = kS / grid, ci = 3 * grid * grid;
    const float* src = img;
    DevBuf P;
    if (grid > 1) {
        P = floats((size_t)ci * H * H);
        sddev::bir_patches(img, 3, kS, kS, grid, grid, P.f());
        src = P.f();
    }
    DevBuf t = floats((size_t)64 * H * H);
    sddev::conv2d(src, ci, H, H, b.c1.w, 64, b.c1.b, 3, 1, 1, t.f(), H, H);
    sddev::conv2d(t.f(), 64, H, H, b.c2.w, cout, b.c2.b, 3, 1, 1, out, H, H);
}

}  // namespace

bool BiRefNet::forward(const float* x, float* logits, const StageFn& stage) {
    if (!ok_) return false;
    if (!sddev::enabled()) {
        error_ = "no device";
        return false;
    }
    Impl& m = *p_;
    auto emit = [&](const char* name, const float* d, size_t n) {
        if (stage) stage(name, d, n);
    };
    const size_t n1 = 65536, n2 = 16384, n3 = 4096, n4 = 1024;  // the stages' pixels: 256^2, 128^2, 64^2, 32^2
    // the input into device-visible memory (the caller's may be a plain vector)
    DevBuf xin = floats((size_t)3 * kS * kS);
    std::memcpy(xin.f(), x, sizeof(float) * 3 * kS * kS);

    // 1. the encoder: the backbone on the image and on its half-size copy, concatenated per stage; the context skips in front of x4
    DevBuf X1 = floats(384 * n1), X2 = floats(768 * n2), X3 = floats(1536 * n3), X4c = floats(5760 * n4);
    {
        float* const full[4] = {X1.f(), X2.f(), X3.f(), X4c.f() + 2688 * n4};
        run_swin(m.pe, m.pe_norm, m.stages, xin.f(), kS, full);
        emit("bb1_x1", full[0], 192 * n1);
        emit("bb1_x2", full[1], 384 * n2);
        emit("bb1_x3", full[2], 768 * n3);
        emit("bb1_x4", full[3], 1536 * n4);
        DevBuf xh = floats((size_t)3 * 512 * 512), h1 = floats(192 * n2), h2 = floats(384 * n3), h3 = floats(768 * n4), h4 = floats(1536 * 256);
        sddev::bir_resize_ac(xin.f(), 3, kS, kS, 512, 512, xh.f());
        float* const half[4] = {h1.f(), h2.f(), h3.f(), h4.f()};
        run_swin(m.pe, m.pe_norm, m.stages, xh.f(), 512, half);
        emit("bb2_x1", half[0], 192 * n2);
        emit("bb2_x2", half[1], 384 * n3);
        emit("bb2_x3", half[2], 768 * n4);
        emit("bb2_x4", half[3], 1536 * 256);
        sddev::bir_resize_ac(h1.f(), 192, 128, 128, 256, 256, X1.f() + 192 * n1);
        sddev::bir_resize_ac(h2.f(), 384, 64, 64, 128, 128, X2.f() + 384 * n2);
        sddev::bir_resize_ac(h3.f(), 768, 32, 32, 64, 64, X3.f() + 768 * n3);
        sddev::bir_resize_ac(h4.f(), 1536, 16, 16, 32, 32, X4c.f() + (2688 + 1536) * n4);
    }
    sddev::bir_resize_ac(X1.f(), 384, 256, 256, 32, 32, X4c.f());
    sddev::bir_resize_ac(X2.f(), 768, 128, 128, 32, 32, X4c.f() + 384 * n4);
    sddev::bir_resize_ac(X3.f(), 1536, 64, 64, 32, 32, X4c.f() + 1152 * n4);

    // 2. the squeeze, then the decoder at 32: with the image's patches; the gradient attention's gate
    DevBuf X4in = floats(3456 * n4), p4 = floats(1536 * n4);
    dec_block(m.squeeze, X4c.f(), 5760, 32, 32, 3072, X4in.f());
    X4c.reset();
    emit("squeeze", X4in.f(), 3072 * n4);
    ipt_block(m.ipt5, xin.f(), 32, 384, X4in.f() + 3072 * n4);
    emit("ipt_blk5", X4in.f() + 3072 * n4, 384 * n4);
    dec_block(m.dec4, X4in.f(), 3456, 32, 32, 1536, p4.f());
    X4in.reset();
    emit("decoder_block4", p4.f(), 1536 * n4);
    auto gate = [&](int i, float* p, int C, int H, const char* name) {
        const long HW = (long)H * H;
        DevBuf g = floats((size_t)16 * HW), a = floats((size_t)HW);
        sddev::conv2d(p, C, H, H, m.gdt[i].w, 16, m.gdt[i].b, 3, 1, 1, g.f(), H, H);
        sddev::bir_bn_act(g.f(), 16, HW, m.gdt_bn[i].s, m.gdt_bn[i].b, true);
        sddev::conv2d(g.f(), 16, H, H, m.gdt_attn[i].w, 1, m.gdt_attn[i].b, 1, 0, 1, a.f(), H, H);
        emit(name, a.f(), (size_t)HW);
        sddev::bir_gate(p, C, HW, a.f());
    };
    gate(0, p4.f(), 1536, 32, "gdt_convs_attn_4");

    // 3. at 64: the upsampled stage plus the lateral block of x3, the patches, the decoder block, the gate
    DevBuf X3in = floats(1920 * n3), p3 = floats(768 * n3);
    {
        DevBuf lat = floats(1536 * n3);
        sddev::bir_resize_ac(p4.f(), 1536, 32, 32, 64, 64, X3in.f());
        sddev::conv2d(X3.f(), 1536, 64, 64, m.lat4.w, 1536, m.lat4.b, 1, 0, 1, lat.f(), 64, 64);
        emit("lateral_block4", lat.f(), 1536 * n3);
        sddev::add(X3in.f(), lat.f(), 1536 * n3);
    }
    p4.reset();
    X3.reset();
    ipt_block(m.ipt4, xin.f(), 16, 384, X3in.f() + 1536 * n3);
    emit("ipt_blk4", X3in.f() + 1536 * n3, 384 * n3);
    dec_block(m.dec3, X3in.f(), 1920, 64, 64, 768, p3.f());
    X3in.reset();
    emit("decoder_block3", p3.f(), 768 * n3);
    gate(1, p3.f(), 768, 64, "gdt_convs_attn_3");

    // 4. at 128
    DevBuf X2in = floats(960 * n2), p2 = floats(384 * n2);
    {
        DevBuf lat = floats(768 * n2);
        sddev::bir_resize_ac(p3.f(), 768, 64, 64, 128, 128, X2in.f());
        sddev::conv2d(X2.f(), 768, 128, 128, m.lat3.w, 768, m.lat3.b, 1, 0, 1, lat.f(), 128, 128);
        emit("lateral_block3", lat.f(), 768 * n2);
        sddev::add(X2in.f(), lat.f(), 768 * n2);
    }
    p3.reset();
    X2.reset();
    ipt_block(m.ipt3, xin.f(), 8, 192, X2in.f() + 768 * n2);
    emit("ipt_blk3", X2in.f() + 768 * n2, 192 * n2);
    dec_block(m.dec2, X2in.f(), 960, 128, 128, 384, p2.f());
    X2in.reset();
    emit("decoder_block2", p2.f(), 384 * n2);
    gate(2, p2.f(), 384, 128, "gdt_convs_attn_2");

    // 5. at 256
    DevBuf X1in = floats(480 * n1), p1 = floats(192 * n1);
    {
        DevBuf lat = floats(384 * n1);
        sddev::bir_resize_ac(p2.f(), 384, 128, 128, 256, 256, X1in.f());
        sddev::conv2d(X1.f(), 384, 256, 256, m.lat2.w, 384, m.lat2.b, 1, 0, 1, lat.f(), 256, 256);
        emit("lateral_block2", lat.f(), 384 * n1);
        sddev::add(X1in.f(), lat.f(), 384 * n1);
    }
    p2.reset();
    X1.reset();
    ipt_block(m.ipt2, xin.f(), 4, 96, X1in.f() + 384 * n1);
    emit("ipt_blk2", X1in.f() + 384 * n1, 96 * n1);
    dec_block(m.dec1, X1in.f(), 480, 256, 256, 192, p1.f());
    X1in.reset();
    emit("decoder_block1", p1.f(), 192 * n1);

    // 6. at 1024: the upsampled stage and the image's own features, the output convolution
    const size_t n0 = (size_t)kS * kS;
    DevBuf X0 = floats(240 * n0), lg = floats(n0);
    sddev::bir_resize_ac(p1.f(), 192, 256, 256, kS, kS, X0.f());
    p1.reset();
    ipt_block(m.ipt1, xin.f(), 1, 48, X0.f() + 192 * n0);
    emit("ipt_blk1", X0.f() + 192 * n0, 48 * n0);
    sddev::conv2d(X0.f(), 240, kS, kS, m.out1.w, 1, m.out1.b, 1, 0, 1, lg.f(), kS, kS);
    emit("conv_out1", lg.f(), n0);
    std::memcpy(logits, lg.f(), sizeof(float) * n0);
    return true;
}

bool BiRefNet::mask(const uint8_t* rgb, int w, int h, std::vector<float>& mask) {
    if (!ok_) return false;
    if (!rgb || w <= 0 || h <= 0) {
        error_ = "no image";
        return false;
    }
    // the python wrapper's preprocessing: 1024 x 1024 (PIL's bilinear), [0, 1], the ImageNet mean and std, channel-major
    const std::vector<uint8_t> r = (w == kS && h == kS) ? std::vector<uint8_t>(rgb, rgb + (size_t)w * h * 3)
                                                        : resize_pil8(rgb, w, h, 3, kS, kS, PilFilter::Bilinear);
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    const size_t n0 = (size_t)kS * kS;
    std::vector<float> x(3 * n0), lg(n0);
    for (size_t i = 0; i < n0; ++i)
        for (int c = 0; c < 3; ++c) x[c * n0 + i] = (r[i * 3 + c] / 255.0f - mean[c]) / stdv[c];
    if (!forward(x.data(), lg.data())) return false;
    // the sigmoid in 8 bits (torchvision's ToPILImage), resized back (PIL's bicubic)
    std::vector<uint8_t> m8(n0);
    for (size_t i = 0; i < n0; ++i) m8[i] = (uint8_t)(255.0f / (1.0f + std::exp(-lg[i])));
    const std::vector<uint8_t> mb = (w == kS && h == kS) ? m8 : resize_pil8(m8.data(), kS, kS, 1, w, h, PilFilter::Bicubic);
    mask.resize((size_t)w * h);
    for (size_t i = 0; i < mask.size(); ++i) mask[i] = mb[i] / 255.0f;
    return true;
}

}  // namespace trellis
