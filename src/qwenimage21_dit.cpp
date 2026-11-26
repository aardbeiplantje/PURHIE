/* Qwen-Image 2.1's diffusion transformer on the device: see qwenimage21_dit.h. Formulas as diffusers' QwenImage21Transformer2DModel:
 * the timestep (1000 t, cos then sin) through two bias-free linears; the modulation (shared by all blocks: scale and gate of each
 * sub-layer per token, the text rows from the t = 0 row); LayerNorm (no affine) x (1 + scale), attention with per-head RMS norms
 * and the interleaved RoPE (the text positions 0.., the image frame L with centred height and width), the tanh-gated residuals; the
 * SwiGLU MLP; the output norm (scale only) and the bias-free projection of the image tokens. */
#include "qwenimage21_dit.h"

#include "sd_device.h"


#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwenimage21 {

namespace {


const int D = 4096, HEADS = 32, HEAD = 128, MLP = 12288, LAYERS = 32, TEXT = 4096, OUTC = 64, FREQ = 256;
const float EPS = 1e-6f;

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

/* a ConvRot linear: int8 weights [out][in] in place, the row scales [out] */
struct QLin {
    const int8_t* W = nullptr;
    const float* s = nullptr;
    int in = 0, out = 0;
};

bool load_q(const zimage::SafeTensors& st, const std::string& base, int in, int out, QLin& L) {
    std::vector<size_t> sw, ss;
    if (!st.i8_in_place(base + ".weight", L.W, sw) || sw != std::vector<size_t>{(size_t)out, (size_t)in} ||
        !st.f32_in_place(base + ".weight_scale", L.s, ss) || ss.size() != 2 || ss[0] != (size_t)out) {
        fprintf(stderr, "qwenimage21 dit: %s (int8 ConvRot %dx%d) missing or of the wrong shape\n", base.c_str(), out, in);
        return false;
    }
    L.in = in;
    L.out = out;
    return true;
}

/* a BF16 tensor with the element count checked */
bool bf(const zimage::SafeTensors& st, const std::string& name, size_t count, const uint16_t*& p) {
    std::vector<size_t> s;
    size_t n = 1;
    if (!st.bf16_in_place(name, p, s)) {
        fprintf(stderr, "qwenimage21 dit: %s missing or not BF16\n", name.c_str());
        return false;
    }
    for (size_t d : s) n *= d;
    if (n != count) {
        fprintf(stderr, "qwenimage21 dit: %s has %zu elements, want %zu\n", name.c_str(), n, count);
        return false;
    }
    return true;
}

/* y [T][out] = bf16 weight [out][in] applied to x (no bias) */
void lin_bf(const float* x, int T, const uint16_t* W, int in, int out, const DevBuf& zeros, float* y) {
    (void)zeros;
    sddev::linear_bf16_f32(x, T, W, in, out, y);  /* FP32 activations (linear_bf16 rounds them to BF16) */
}

/* the ConvRot linear on rows: rotate x into xr, then the int8 GEMM over the rows [row0, row0 + out) of the weight */
void lin_q(const QLin& L, const float* x, int T, const float* H, float* xr, int row0, int out, float* y) {
    sddev::linear_convrot(x, T, L.W + (size_t)row0 * L.in, L.s + row0, H, L.in, out, xr, y);
}

}  // namespace

bool dev_dit21_forward(const zimage::SafeTensors& st, const float* packed, int gh, int gw, const float* enc, int L, float t, float* out) {
    if (!sddev::enabled()) { fprintf(stderr, "qwenimage21 dit: no device\n"); return false; }
    if (gh < 1 || gw < 1 || L < 1) { fprintf(stderr, "qwenimage21 dit: bad grid or text length\n"); return false; }
    const int N = gh * gw, J = L + N;
    /* the image rows' attention in launches of about 6M query-key pairs (about 40 ms at 32 heads on the fused kernel: AGENTS.md, GPU
     * use), in whole 64-query blocks */
    const int qrows = (int)std::min<long>(N, std::max<long>(256, ((6L << 20) / J) / 64 * 64));
    if (N % 4 != 0) { fprintf(stderr, "qwenimage21 dit: the grid must hold whole 2 x 2 slots\n"); return false; }

    /* the weights (in place) */
    const uint16_t *wt1 = nullptr, *wt2 = nullptr, *wmod = nullptr, *wnout = nullptr, *wproj = nullptr, *wimg = nullptr, *wtin = nullptr,
                   *wtout = nullptr, *wtnorm = nullptr, *wte1 = nullptr, *wte2 = nullptr;
    if (!bf(st, "time_text_embed.timestep_embedder.linear_1.weight", (size_t)D * FREQ, wte1) ||
        !bf(st, "time_text_embed.timestep_embedder.linear_2.weight", (size_t)D * D, wte2) ||
        !bf(st, "modulation.1.weight", (size_t)16384 * D, wmod) || !bf(st, "norm_out.linear.weight", (size_t)D * D, wnout) ||
        !bf(st, "proj_out.weight", (size_t)OUTC * D, wproj) || !bf(st, "img_in.weight", (size_t)D * OUTC, wimg) ||
        !bf(st, "txt_in.in_layer.weight", (size_t)D * TEXT, wtin) || !bf(st, "txt_in.out_layer.weight", (size_t)D * D, wtout) ||
        !bf(st, "txt_in.text_norm.weight", (size_t)TEXT, wtnorm))
        return false;
    (void)wt1; (void)wt2;

    DevBuf zeros((size_t)16384, true);
    DevBuf Hd((size_t)256 * 256);
    {
        /* the regular Hadamard of 256 (H4 Kronecker power, normalised by 1/16): the rotation of the ConvRot layers */
        static const double H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
        std::vector<double> H = {1.0};
        int n = 1;
        while (n < 256) {
            std::vector<double> K((size_t)n * 4 * n * 4);
            for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
                K[(size_t)(a * 4 + i) * (n * 4) + (b * 4 + j)] = H[(size_t)a * n + b] * H4[i][j];
            H.swap(K);
            n *= 4;
        }
        std::vector<float> Hf(256 * 256);
        for (size_t i = 0; i < Hf.size(); ++i) Hf[i] = (float)(H[i] / 16.0);
        std::memcpy(Hd.get(), Hf.data(), Hf.size() * 4);
    }
    /* the timestep: rows [t, 0] of the sinusoid (cos first), Linear(256, 4096), SiLU, Linear(4096, 4096) */
    std::vector<float> e2(2 * FREQ, 0.0f);
    for (int i = 0; i < FREQ / 2; ++i) {
        const double arg = 1000.0 * t * std::exp(-std::log(10000.0) * i / (FREQ / 2));
        e2[i] = (float)std::cos(arg);
        e2[FREQ / 2 + i] = (float)std::sin(arg);
        e2[FREQ + i] = 1.0f;  /* t = 0: cos(0) = 1, sin(0) = 0 */
    }
    DevBuf e(2 * FREQ), h1(2 * D), temb(2 * D), sil(2 * D);
    std::memcpy(e.get(), e2.data(), sizeof(float) * 2 * FREQ);
    lin_bf(e.get(), 2, wte1, FREQ, D, zeros, h1.get());
    sddev::silu(h1.get(), 2 * D);
    lin_bf(h1.get(), 2, wte2, D, D, zeros, temb.get());
    std::memcpy(sil.get(), temb.get(), sizeof(float) * 2 * D);
    sddev::silu(sil.get(), 2 * D);
    /* the shared modulation (four chunks of 4096: scale1, gate1, scale2, gate2) and the output norm's scale, per row [2][4096] */
    DevBuf mods[4];
    for (int k = 0; k < 4; ++k) {
        mods[k] = DevBuf(2 * D);
        sddev::linear_bf16(sil.get(), 2, wmod + (size_t)k * D * D, D, D, zeros.get(), mods[k].get());
    }
    DevBuf mout(2 * D);
    lin_bf(sil.get(), 2, wnout, D, D, zeros, mout.get());
    /* the token rows: the text (row 1: t = 0) and the image (row 0: the sampled step) */
    std::vector<float> sel(J);
    for (int j = 0; j < J; ++j) sel[j] = j < L ? 1.0f : 0.0f;
    DevBuf self(J);
    std::memcpy(self.get(), sel.data(), sizeof(float) * J);
    DevBuf sc1((size_t)J * D), gt1((size_t)J * D), sc2((size_t)J * D), gt2((size_t)J * D), scout((size_t)J * D);
    sddev::gather_mod(mods[0].get(), self.get(), J, D, sc1.get());
    sddev::gather_mod(mods[1].get(), self.get(), J, D, gt1.get());
    sddev::gather_mod(mods[2].get(), self.get(), J, D, sc2.get());
    sddev::gather_mod(mods[3].get(), self.get(), J, D, gt2.get());
    sddev::gather_mod(mout.get(), self.get(), J, D, scout.get());
    /* the tokens: the text (the RMS norm, then the projection), the image (the packed latent) */
    DevBuf pk((size_t)N * OUTC), ec((size_t)L * TEXT);
    std::memcpy(pk.get(), packed, sizeof(float) * (size_t)N * OUTC);
    std::memcpy(ec.get(), enc, sizeof(float) * (size_t)L * TEXT);
    DevBuf X((size_t)J * D);
    {
        DevBuf tn((size_t)L * TEXT), th((size_t)L * D);
        sddev::zc_rms(ec.get(), L, TEXT, wtnorm, EPS, tn.get());
        lin_bf(tn.get(), L, wtin, TEXT, D, zeros, th.get());
        sddev::gelu_tanh(th.get(), (size_t)L * D);
        lin_bf(th.get(), L, wtout, D, D, zeros, X.get());
        lin_bf(pk.get(), N, wimg, OUTC, D, zeros, X.get() + (size_t)L * D);
    }
    /* the RoPE of the joint sequence: text j at position j on all three axes; the image at frame L, centred (h - gh/2, w - gw/2) */
    std::vector<float> fr((size_t)J * 64 * 2, 0.0f);
    auto put = [&](int tok, int pair, double angle) {
        fr[((size_t)tok * 64 + pair) * 2] = (float)std::cos(angle);
        fr[((size_t)tok * 64 + pair) * 2 + 1] = (float)std::sin(angle);
    };
    auto inv = [](int dim, int k) { return std::pow(10000.0, -2.0 * k / dim); };
    for (int j = 0; j < L; ++j) {
        for (int k = 0; k < 8; ++k) put(j, k, j * inv(16, k));
        for (int k = 0; k < 28; ++k) put(j, 8 + k, j * inv(56, k));
        for (int k = 0; k < 28; ++k) put(j, 36 + k, j * inv(56, k));
    }
    const int hh = gh - gh / 2, ww = gw - gw / 2;
    for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c) {
            const int tok = L + r * gw + c;
            for (int k = 0; k < 8; ++k) put(tok, k, (double)L * inv(16, k));
            for (int k = 0; k < 28; ++k) put(tok, 8 + k, (r - hh) * inv(56, k));
            for (int k = 0; k < 28; ++k) put(tok, 36 + k, (c - ww) * inv(56, k));
        }
    DevBuf frd((size_t)J * 64 * 2);
    std::memcpy(frd.get(), fr.data(), sizeof(float) * fr.size());
    /* the blocks */
    DevBuf xn((size_t)J * D), q((size_t)J * D), k((size_t)J * D), v((size_t)J * D), att((size_t)J * D), o((size_t)J * D);
    DevBuf xr((size_t)J * MLP), gate_up_g((size_t)J * MLP), gate_up_p((size_t)J * MLP), mlp_g((size_t)J * MLP), mlp_o((size_t)J * D);
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "transformer_blocks." + std::to_string(l) + ".";
        QLin wq, wk, wv, wo, wgu, wdn;
        const uint16_t *nq = nullptr, *nk = nullptr;
        if (!load_q(st, p + "attn.to_q", D, D, wq) || !load_q(st, p + "attn.to_k", D, D, wk) || !load_q(st, p + "attn.to_v", D, D, wv) ||
            !load_q(st, p + "attn.to_out.0", D, D, wo) || !load_q(st, p + "img_mlp.gate_up", D, 2 * MLP, wgu) ||
            !load_q(st, p + "img_mlp.out", MLP, D, wdn) || !bf(st, p + "attn.norm_q.weight", HEAD, nq) ||
            !bf(st, p + "attn.norm_k.weight", HEAD, nk))
            return false;
        /* attention: LayerNorm, (1 + scale1), the projections, the per-head RMS norms, the RoPE, the masked attention, the output */
        sddev::layer_norm_plain(X.get(), J, D, EPS, xn.get());
        sddev::mul_1p(xn.get(), sc1.get(), (size_t)J * D);
        lin_q(wq, xn.get(), J, Hd.get(), xr.get(), 0, D, q.get());
        lin_q(wk, xn.get(), J, Hd.get(), xr.get(), 0, D, k.get());
        lin_q(wv, xn.get(), J, Hd.get(), xr.get(), 0, D, v.get());
        sddev::rms_norm(q.get(), J * HEADS, HEAD, nq, EPS, q.get());
        sddev::rms_norm(k.get(), J * HEADS, HEAD, nk, EPS, k.get());
        sddev::rope(q.get(), J, HEADS, frd.get());
        sddev::rope(k.get(), J, HEADS, frd.get());
        /* the text rows: causal among the text; the image rows: every token (the FP32 attention, no mask) */
        sddev::attn21_text(q.get(), k.get(), v.get(), J, L, HEADS, att.get());
        for (int r0 = 0; r0 < N; r0 += qrows)
            sddev::attention_f32(q.get() + (size_t)(L + r0) * D, std::min(qrows, N - r0), k.get(), v.get(), J, D, HEADS,
                                 att.get() + (size_t)(L + r0) * D);
        lin_q(wo, att.get(), J, Hd.get(), xr.get(), 0, D, o.get());
        sddev::gated_add(X.get(), gt1.get(), o.get(), (size_t)J * D);
        /* the SwiGLU MLP: gate_layer rows [0, 12288) and proj rows [12288, 24576) of the fused weight, then out(silu(gate) * proj) */
        sddev::layer_norm_plain(X.get(), J, D, EPS, xn.get());
        sddev::mul_1p(xn.get(), sc2.get(), (size_t)J * D);
        lin_q(wgu, xn.get(), J, Hd.get(), xr.get(), 0, MLP, gate_up_g.get());
        lin_q(wgu, xn.get(), J, Hd.get(), xr.get(), MLP, MLP, gate_up_p.get());
        sddev::swiglu(gate_up_g.get(), gate_up_p.get(), mlp_g.get(), (size_t)J * MLP);
        lin_q(wdn, mlp_g.get(), J, Hd.get(), xr.get(), 0, D, mlp_o.get());
        sddev::gated_add(X.get(), gt2.get(), mlp_o.get(), (size_t)J * D);
    }
    /* the output: LayerNorm, (1 + scale) from the output norm's row, the projection of the image tokens */
    sddev::layer_norm_plain(X.get(), J, D, EPS, xn.get());
    sddev::mul_1p(xn.get(), scout.get(), (size_t)J * D);
    DevBuf res((size_t)N * OUTC);
    lin_bf(xn.get() + (size_t)L * D, N, wproj, D, OUTC, zeros, res.get());
    std::memcpy(out, res.get(), sizeof(float) * (size_t)N * OUTC);
    return true;
}

}  // namespace qwenimage21
