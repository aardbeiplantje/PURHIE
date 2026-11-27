/* Qwen-Image 2.1's diffusion transformer on the device: see qwenimage21_dit.h. Formulas as diffusers' QwenImage21Transformer2DModel:
 * the timestep (1000 t, cos then sin) through two bias-free linears; the modulation (shared by all blocks: scale and gate of each
 * sub-layer per token; the text and condition rows from the t = 0 row); LayerNorm (no affine) x (1 + scale), attention with per-head
 * RMS norms and the interleaved RoPE (QwenImage21Rope: the text advances one position on all three axes, an image block takes the
 * frame position reached and a height / width grid centred on zero, then advances max(h, w)), the tanh-gated residuals; the SwiGLU
 * MLP; the output norm (scale only) and the bias-free projection of the target's tokens. The attention is block-causal
 * (build_qwenimage21_block_causal_mask): a text row sees the rows up to itself, an image block's rows see everything up to the block's
 * end. The prefix (text and condition images) runs once and keeps every block's keys and values; a step runs the target's rows, whose
 * queries see the whole prefix and the target (the pipeline's "cached" mode). */
#include "qwenimage21_dit.h"

#include "sd_device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace qwenimage21 {

namespace {

const int D = 4096, HEADS = 32, HEAD = 128, MLP = 12288, LAYERS = 32, TEXT = 4096, OUTC = 64, FREQ = 256;
const float EPS = 1e-6f;

/* a device buffer of n elements (FP32 from sddev::alloc; BF16 from device memory, sddev::dalloc: never read by the host) */
template <typename T>
class Buf {
public:
    Buf() = default;
    explicit Buf(size_t count, bool zero = false) : n_(count) {
        p_ = static_cast<T*>(dev() ? sddev::dalloc(sizeof(T) * count) : sddev::alloc(sizeof(T) * count, zero));
    }
    Buf(Buf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) {
            free_();
            p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0;
        }
        return *this;
    }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    ~Buf() { free_(); }
    T* get() const { return p_; }
    size_t size() const { return n_; }

private:
    static constexpr bool dev() { return sizeof(T) == 2; }
    void free_() {
        if (!p_) return;
        if (dev()) sddev::drelease(p_, sizeof(T) * n_);
        else sddev::release(p_, sizeof(T) * n_);
        p_ = nullptr;
    }
    T* p_ = nullptr;
    size_t n_ = 0;
};
using DevBuf = Buf<float>;
using Buf16 = Buf<uint16_t>;

/* a buffer of at least n elements: reallocated only when it grows */
template <typename T>
void need(Buf<T>& b, size_t n) {
    if (b.size() < n) b = Buf<T>(n);
}

/* a block's linear [out][in] in place: the ComfyUI int8 file's ConvRot weights with their row scales [out], or the BF16 file's
 * weights (the same names; hipBLASLt's BF16 products, about ten times the int8 kernel's speed) */
struct QLin {
    const int8_t* W = nullptr;
    const float* s = nullptr;
    const uint16_t* Wb = nullptr;
    int in = 0, out = 0;
};

bool load_q(const zimage::SafeTensors& st, const std::string& base, int in, int out, QLin& L) {
    std::vector<size_t> sw, ss;
    const std::vector<size_t> want{(size_t)out, (size_t)in};
    bool ok = false;
    if (st.i8_in_place(base + ".weight", L.W, sw))
        ok = sw == want && st.f32_in_place(base + ".weight_scale", L.s, ss) && ss.size() == 2 && ss[0] == (size_t)out;
    else if (st.bf16_in_place(base + ".weight", L.Wb, sw))
        ok = sw == want;
    if (!ok) {
        fprintf(stderr, "qwenimage21 dit: %s (int8 ConvRot or BF16, %dx%d) missing or of the wrong shape\n", base.c_str(), out, in);
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

/* y [T][out] = bf16 weight [out][in] applied to x (no bias; FP32 activations) */
void lin_bf(const float* x, int T, const uint16_t* W, int in, int out, float* y) { sddev::linear_bf16_f32(x, T, W, in, out, y); }

/* the block's linear over the weight's rows [row0, row0 + out): the BF16 product, or the ConvRot one (rotate x into xr, then the int8
 * GEMM); the tokens in pieces, so each launch stays short (AGENTS.md: the int8 kernel at about 2 TFLOPS gets about 40 ms of work,
 * hipBLASLt at most 8192 tokens, about 40 ms), split evenly (no small remainder: it would read the whole weight again) */
void lin_q(const QLin& L, const float* x, int T, const float* H, float* xr, int row0, int out, float* y) {
    const int most = L.Wb ? 8192 : std::max(64, (int)(4e10 / ((double)out * L.in)) / 64 * 64);
    const int pieces = (T + most - 1) / most;
    const int rows = (T + pieces - 1) / pieces;
    for (int r0 = 0; r0 < T; r0 += rows) {
        const int n = std::min(rows, T - r0);
        if (L.Wb)
            sddev::linear_bf16_via16(x + (size_t)r0 * L.in, n, L.Wb + (size_t)row0 * L.in, L.in, out, nullptr, y + (size_t)r0 * out);
        else
            sddev::linear_convrot(x + (size_t)r0 * L.in, n, L.W + (size_t)row0 * L.in, L.s + row0, H, L.in, out, xr,
                                  y + (size_t)r0 * out);
    }
}

/* the BF16 chain's product over the weight's rows [row0, row0 + out): BF16 in and out (hipBLASLt's BF16 output; the FP32 product
 * rounded where it does not take the shape), the tokens (a multiple of 64) split evenly into pieces of at most 8192, in 64s */
void lin16(const QLin& L, const uint16_t* x, int T, int row0, int out, uint16_t* y) {
    const int pieces = (T + 8191) / 8192;
    const int rows = ((T + pieces - 1) / pieces + 63) / 64 * 64;
    for (int r0 = 0; r0 < T; r0 += rows) {
        const int n = std::min(rows, T - r0);
        sddev::linear_bf16_xb16(x + (size_t)r0 * L.in, n, L.Wb + (size_t)row0 * L.in, L.in, out, nullptr, y + (size_t)r0 * out);
    }
}

/* a segment of the joint prefix: [a, a + n) text rows (causal) or one image block (bidirectional, sees everything before it) */
struct Seg {
    int a = 0, n = 0;
    bool image = false;
};

/* RoPE angles of a token: frame f, height h, width w -> [64][2] (cos, sin): 8 pairs of the frame (16 dims), 28 of the height and 28 of
 * the width (56 each), theta 10000 */
void put_rope(float* fr, int tok, double f, double h, double w) {
    auto inv = [](int dim, int k) { return std::pow(10000.0, -2.0 * k / dim); };
    auto put = [&](int pair, double angle) {
        fr[((size_t)tok * 64 + pair) * 2] = (float)std::cos(angle);
        fr[((size_t)tok * 64 + pair) * 2 + 1] = (float)std::sin(angle);
    };
    for (int k = 0; k < 8; ++k) put(k, f * inv(16, k));
    for (int k = 0; k < 28; ++k) put(8 + k, h * inv(56, k));
    for (int k = 0; k < 28; ++k) put(36 + k, w * inv(56, k));
}

/* an image block of gh x gw at frame f, from token tok0: the centred grid (h - (gh - gh / 2), w - (gw - gw / 2)) */
void put_block(float* fr, int tok0, int f, int gh, int gw) {
    const int hh = gh - gh / 2, ww = gw - gw / 2;
    for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c) put_rope(fr, tok0 + r * gw + c, f, r - hh, c - ww);
}

}  // namespace

struct Dit21::Impl {
    const zimage::SafeTensors& st;
    bool ok = false, bf16 = false, have_prefix = false;
    const uint16_t *wte1 = nullptr, *wte2 = nullptr, *wmod = nullptr, *wnout = nullptr, *wproj = nullptr, *wimg = nullptr,
                   *wtin = nullptr, *wtout = nullptr, *wtnorm = nullptr;
    struct Layer {
        QLin q, k, v, o, gu, dn;
        const uint16_t *nq = nullptr, *nk = nullptr;
    };
    std::vector<Layer> layers;
    DevBuf Hd;  /* the regular Hadamard of 256 (the ConvRot layers' rotation) */
    /* the prefix: its rows, the target's grid and RoPE, every block's keys and values of the prefix (BF16 [Pp][D] with the BF16 file,
     * FP32 [P][D] with the int8 one) */
    int P = 0, gh = 0, gw = 0, N = 0;
    std::vector<Buf16> K16, V16;
    std::vector<DevBuf> K32, V32;
    DevBuf frt;
    /* a step's buffers (grown as needed) */
    DevBuf X, xn, q, att, o, xr, g1, g2, mo, sel, sc1, gt1, sc2, gt2, scout, kst, vst, pk;
    Buf16 xb, qb, ab, ob, gb, pb, kb, vb;

    explicit Impl(const zimage::SafeTensors& s) : st(s) {}

    /* the modulation of one timestep: the four chunks of 4096 (scale1, gate1, scale2, gate2) and the output norm's scale, [5][D] */
    void modulation(float t, DevBuf& m) {
        std::vector<float> e1(FREQ);
        for (int i = 0; i < FREQ / 2; ++i) {
            const double arg = 1000.0 * t * std::exp(-std::log(10000.0) * i / (FREQ / 2));
            e1[i] = (float)std::cos(arg);
            e1[FREQ / 2 + i] = (float)std::sin(arg);
        }
        DevBuf e(FREQ), h1(D), temb(D);
        std::memcpy(e.get(), e1.data(), sizeof(float) * FREQ);
        lin_bf(e.get(), 1, wte1, FREQ, D, h1.get());
        sddev::silu(h1.get(), D);
        lin_bf(h1.get(), 1, wte2, D, D, temb.get());
        sddev::silu(temb.get(), D);
        m = DevBuf((size_t)5 * D);
        DevBuf zeros(D, true);
        for (int k = 0; k < 4; ++k) sddev::linear_bf16(temb.get(), 1, wmod + (size_t)k * D * D, D, D, zeros.get(), m.get() + (size_t)k * D);
        lin_bf(temb.get(), 1, wnout, D, D, m.get() + (size_t)4 * D);
    }

    /* the per-token rows of the modulation m (one timestep) for T tokens */
    void gather(const DevBuf& m, int T, bool with_out) {
        need(sel, (size_t)T);
        std::memset(sel.get(), 0, sizeof(float) * T);
        const size_t n = (size_t)T * D;
        need(sc1, n), need(gt1, n), need(sc2, n), need(gt2, n);
        sddev::gather_mod(m.get(), sel.get(), T, D, sc1.get());
        sddev::gather_mod(m.get() + (size_t)D, sel.get(), T, D, gt1.get());
        sddev::gather_mod(m.get() + (size_t)2 * D, sel.get(), T, D, sc2.get());
        sddev::gather_mod(m.get() + (size_t)3 * D, sel.get(), T, D, gt2.get());
        if (with_out) {
            need(scout, n);
            sddev::gather_mod(m.get() + (size_t)4 * D, sel.get(), T, D, scout.get());
        }
    }

    /* the query rows of the image attention per launch: the BF16 flash kernel about 32M query-key pairs (about 30 ms at 32 heads), the
     * FP32 fused kernel about 6M (about 40 ms) */
    int qrows(int rows, int Tk) const {
        return bf16 ? (int)std::min<long>(rows, std::max<long>(128, ((32L << 20) / Tk) / 128 * 128))
                    : (int)std::min<long>(rows, std::max<long>(256, ((6L << 20) / Tk) / 64 * 64));
    }

    /* one block over T rows of X (FP32 [T][D]) with the gathered modulation; q, k, v of the rows (k and v written to kd / vd, or kf /
     * vf, at those rows: the caller's key buffers), then attn(...) fills att from them, then the output and the MLP into X */
    void block(const Layer& Ly, int T, const float* rope, uint16_t* kd, uint16_t* vd, float* kf, float* vf, bool mlp,
               const std::function<void()>& attn) {
        const int Tp = (T + 63) / 64 * 64;
        const size_t td = (size_t)T * D;
        if (bf16) {
            /* the BF16 chain, as Z-Image's: the residual X stays FP32; the normed input, the products' outputs and the MLP's values are
             * BF16 (each rounded once). The products run on Tp rows (a multiple of 64); the rows past T are never read. */
            need(xb, (size_t)Tp * D), need(qb, (size_t)Tp * D), need(ab, (size_t)Tp * D), need(ob, (size_t)Tp * D);
            sddev::qi21_ln_mod_bf16(X.get(), T, D, EPS, sc1.get(), xb.get());
            lin16(Ly.q, xb.get(), Tp, 0, D, qb.get());
            lin16(Ly.k, xb.get(), Tp, 0, D, kd);
            lin16(Ly.v, xb.get(), Tp, 0, D, vd);
            sddev::head_norm_rope_bf16(qb.get(), T, HEADS, Ly.nq, EPS, rope, qb.get());
            sddev::head_norm_rope_bf16(kd, T, HEADS, Ly.nk, EPS, rope, kd);
            if (!mlp && !attn) return;
            need(att, td);
            attn();
            sddev::to_bf16(att.get(), ab.get(), td);
            lin16(Ly.o, ab.get(), Tp, 0, D, ob.get());
            sddev::qi21_gated_add_bf16(X.get(), gt1.get(), ob.get(), td);
            if (!mlp) return;
            need(gb, (size_t)Tp * MLP), need(pb, (size_t)Tp * MLP);
            sddev::qi21_ln_mod_bf16(X.get(), T, D, EPS, sc2.get(), xb.get());
            lin16(Ly.gu, xb.get(), Tp, 0, MLP, gb.get());
            lin16(Ly.gu, xb.get(), Tp, MLP, MLP, pb.get());
            sddev::swiglu_bf16_in(gb.get(), pb.get(), gb.get(), (size_t)T * MLP);
            lin16(Ly.dn, gb.get(), Tp, 0, D, ob.get());
            sddev::qi21_gated_add_bf16(X.get(), gt2.get(), ob.get(), td);
            return;
        }
        /* the FP32 chain (the int8 file): LayerNorm, (1 + scale1), the projections, the per-head RMS norms, the RoPE */
        need(xn, td), need(q, td), need(xr, (size_t)T * MLP);
        sddev::layer_norm_plain(X.get(), T, D, EPS, xn.get());
        sddev::mul_1p(xn.get(), sc1.get(), td);
        lin_q(Ly.q, xn.get(), T, Hd.get(), xr.get(), 0, D, q.get());
        lin_q(Ly.k, xn.get(), T, Hd.get(), xr.get(), 0, D, kf);
        lin_q(Ly.v, xn.get(), T, Hd.get(), xr.get(), 0, D, vf);
        sddev::rms_norm(q.get(), T * HEADS, HEAD, Ly.nq, EPS, q.get());
        sddev::rms_norm(kf, T * HEADS, HEAD, Ly.nk, EPS, kf);
        sddev::rope(q.get(), T, HEADS, rope);
        sddev::rope(kf, T, HEADS, rope);
        if (!mlp && !attn) return;
        need(att, td), need(o, td);
        attn();
        lin_q(Ly.o, att.get(), T, Hd.get(), xr.get(), 0, D, o.get());
        sddev::gated_add(X.get(), gt1.get(), o.get(), td);
        if (!mlp) return;
        need(g1, (size_t)T * MLP), need(g2, (size_t)T * MLP), need(mo, td);
        sddev::layer_norm_plain(X.get(), T, D, EPS, xn.get());
        sddev::mul_1p(xn.get(), sc2.get(), td);
        lin_q(Ly.gu, xn.get(), T, Hd.get(), xr.get(), 0, MLP, g1.get());
        lin_q(Ly.gu, xn.get(), T, Hd.get(), xr.get(), MLP, MLP, g2.get());
        sddev::swiglu(g1.get(), g2.get(), g1.get(), (size_t)T * MLP);
        lin_q(Ly.dn, g1.get(), T, Hd.get(), xr.get(), 0, D, mo.get());
        sddev::gated_add(X.get(), gt2.get(), mo.get(), td);
    }
};

Dit21::Dit21(const zimage::SafeTensors& st) : p_(new Impl(st)) {
    Impl& p = *p_;
    if (!sddev::enabled()) { fprintf(stderr, "qwenimage21 dit: no device\n"); return; }
    {
        const uint16_t* w = nullptr;
        std::vector<size_t> sh;
        p.bf16 = st.bf16_in_place("transformer_blocks.0.attn.to_q.weight", w, sh);
    }
    if (!bf(st, "time_text_embed.timestep_embedder.linear_1.weight", (size_t)D * FREQ, p.wte1) ||
        !bf(st, "time_text_embed.timestep_embedder.linear_2.weight", (size_t)D * D, p.wte2) ||
        !bf(st, "modulation.1.weight", (size_t)16384 * D, p.wmod) || !bf(st, "norm_out.linear.weight", (size_t)D * D, p.wnout) ||
        !bf(st, "proj_out.weight", (size_t)OUTC * D, p.wproj) || !bf(st, "img_in.weight", (size_t)D * OUTC, p.wimg) ||
        !bf(st, "txt_in.in_layer.weight", (size_t)D * TEXT, p.wtin) || !bf(st, "txt_in.out_layer.weight", (size_t)D * D, p.wtout) ||
        !bf(st, "txt_in.text_norm.weight", (size_t)TEXT, p.wtnorm))
        return;
    p.layers.resize(LAYERS);
    for (int l = 0; l < LAYERS; ++l) {
        const std::string b = "transformer_blocks." + std::to_string(l) + ".";
        Impl::Layer& Ly = p.layers[l];
        if (!load_q(st, b + "attn.to_q", D, D, Ly.q) || !load_q(st, b + "attn.to_k", D, D, Ly.k) ||
            !load_q(st, b + "attn.to_v", D, D, Ly.v) || !load_q(st, b + "attn.to_out.0", D, D, Ly.o) ||
            !load_q(st, b + "img_mlp.gate_up", D, 2 * MLP, Ly.gu) || !load_q(st, b + "img_mlp.out", MLP, D, Ly.dn) ||
            !bf(st, b + "attn.norm_q.weight", HEAD, Ly.nq) || !bf(st, b + "attn.norm_k.weight", HEAD, Ly.nk))
            return;
        if (p.bf16 != (Ly.q.Wb != nullptr)) {
            fprintf(stderr, "qwenimage21 dit: the file mixes int8 and BF16 blocks\n");
            return;
        }
    }
    if (!p.bf16) {
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
        p.Hd = DevBuf((size_t)256 * 256);
        for (size_t i = 0; i < H.size(); ++i) p.Hd.get()[i] = (float)(H[i] / 16.0);
    }
    p.ok = true;
}

Dit21::~Dit21() { delete p_; }

bool Dit21::ok() const { return p_->ok; }

bool Dit21::prefix(const Condition& c, int gh, int gw) {
    Impl& p = *p_;
    p.have_prefix = false;
    if (!p.ok) return false;
    if (!c.enc || c.L < 1 || gh < 1 || gw < 1 || (gh * gw) % 4 != 0) {
        fprintf(stderr, "qwenimage21 dit: bad text or grid (the target must hold whole 2 x 2 slots)\n");
        return false;
    }
    /* the segments: text runs, and each run of pads one condition image of 4 x its pads tokens (the grids in order) */
    std::vector<Seg> segs;
    std::vector<int> src;  /* per segment: the first encoder row (text) or the condition image's index (image) */
    size_t img = 0;
    int P = 0;
    for (int j = 0; j < c.L;) {
        const bool pad = c.pad && c.pad[j];
        int e = j;
        while (e < c.L && (c.pad && c.pad[e]) == pad) ++e;
        Seg s;
        s.a = P;
        s.image = pad;
        if (pad) {
            if (img >= c.grids.size() || c.grids[img].first * c.grids[img].second != 4 * (e - j) || !c.cond) {
                fprintf(stderr, "qwenimage21 dit: the image pads (%d) do not match condition image %zu\n", e - j, img);
                return false;
            }
            s.n = 4 * (e - j);
            src.push_back((int)img++);
        } else {
            s.n = e - j;
            src.push_back(j);
        }
        segs.push_back(s);
        P += s.n;
        j = e;
    }
    if (img != c.grids.size()) {
        fprintf(stderr, "qwenimage21 dit: %zu condition images for %zu runs of pads\n", c.grids.size(), img);
        return false;
    }
    const int Pp = (P + 63) / 64 * 64;
    /* the RoPE of the prefix and the target (QwenImage21Rope) */
    std::vector<float> frp((size_t)P * 128), ft((size_t)gh * gw * 128);
    int pos = 0;
    for (size_t s = 0; s < segs.size(); ++s) {
        if (segs[s].image) {
            const auto& g = c.grids[src[s]];
            put_block(frp.data(), segs[s].a, pos, g.first, g.second);
            pos += std::max(g.first, g.second);
        } else {
            for (int i = 0; i < segs[s].n; ++i) put_rope(frp.data(), segs[s].a + i, pos + i, pos + i, pos + i);
            pos += segs[s].n;
        }
    }
    put_block(ft.data(), 0, pos, gh, gw);
    DevBuf frpd((size_t)P * 128);
    std::memcpy(frpd.get(), frp.data(), sizeof(float) * frp.size());
    p.frt = DevBuf(ft.size());
    std::memcpy(p.frt.get(), ft.data(), sizeof(float) * ft.size());

    /* the prefix's tokens: the text (the RMS norm, then the projection) and the condition images (their packed latents) */
    need(p.X, (size_t)P * D);
    {
        std::vector<size_t> coff(c.grids.size() + 1, 0);
        for (size_t i = 0; i < c.grids.size(); ++i) coff[i + 1] = coff[i] + (size_t)c.grids[i].first * c.grids[i].second;
        for (size_t s = 0; s < segs.size(); ++s) {
            const int n = segs[s].n;
            float* dst = p.X.get() + (size_t)segs[s].a * D;
            if (segs[s].image) {
                DevBuf pk((size_t)n * OUTC);
                std::memcpy(pk.get(), c.cond + coff[src[s]] * OUTC, sizeof(float) * (size_t)n * OUTC);
                lin_bf(pk.get(), n, p.wimg, OUTC, D, dst);
            } else {
                DevBuf ec((size_t)n * TEXT), tn((size_t)n * TEXT), th((size_t)n * D);
                std::memcpy(ec.get(), c.enc + (size_t)src[s] * TEXT, sizeof(float) * (size_t)n * TEXT);
                sddev::zc_rms(ec.get(), n, TEXT, p.wtnorm, EPS, tn.get());
                lin_bf(tn.get(), n, p.wtin, TEXT, D, th.get());
                sddev::gelu_tanh(th.get(), (size_t)n * D);
                lin_bf(th.get(), n, p.wtout, D, D, dst);
            }
        }
    }
    /* the t = 0 modulation for every prefix row */
    DevBuf m0;
    p.modulation(0.0f, m0);
    p.gather(m0, P, false);

    /* the blocks: every block's keys and values kept; the last block's attention output and MLP are never read */
    p.K16.clear(), p.V16.clear(), p.K32.clear(), p.V32.clear();
    DevBuf q32, k32, v32;
    for (int l = 0; l < LAYERS; ++l) {
        const Impl::Layer& Ly = p.layers[l];
        uint16_t *kd = nullptr, *vd = nullptr;
        float *kf = nullptr, *vf = nullptr;
        if (p.bf16) {
            p.K16.emplace_back((size_t)Pp * D);
            p.V16.emplace_back((size_t)Pp * D);
            kd = p.K16.back().get();
            vd = p.V16.back().get();
        } else {
            p.K32.emplace_back((size_t)P * D);
            p.V32.emplace_back((size_t)P * D);
            kf = p.K32.back().get();
            vf = p.V32.back().get();
        }
        const bool last = l == LAYERS - 1;
        auto attn = [&]() {
            for (const Seg& s : segs) {
                const int end = s.a + s.n;
                if (!s.image) {
                    /* text rows: causal, on the FP32 kernel (their keys and values widened up to the segment's end) */
                    if (p.bf16) {
                        need(q32, (size_t)end * D), need(k32, (size_t)end * D), need(v32, (size_t)end * D);
                        sddev::from_bf16(p.qb.get(), q32.get(), (size_t)end * D);
                        sddev::from_bf16(kd, k32.get(), (size_t)end * D);
                        sddev::from_bf16(vd, v32.get(), (size_t)end * D);
                        sddev::attn21_causal(q32.get(), k32.get(), v32.get(), s.a, s.n, HEADS, p.att.get());
                    } else {
                        sddev::attn21_causal(p.q.get(), kf, vf, s.a, s.n, HEADS, p.att.get());
                    }
                    continue;
                }
                /* an image block: every row sees the keys up to the block's end */
                const int qr = p.qrows(s.n, end);
                for (int r0 = 0; r0 < s.n; r0 += qr) {
                    const int rows = std::min(qr, s.n - r0);
                    const size_t off = (size_t)(s.a + r0) * D;
                    if (p.bf16) sddev::attention_bf16(p.qb.get() + off, rows, kd, vd, end, D, HEADS, p.att.get() + off);
                    else sddev::attention_f32(p.q.get() + off, rows, kf, vf, end, D, HEADS, p.att.get() + off);
                }
            }
        };
        p.block(Ly, P, frpd.get(), kd, vd, kf, vf, !last, last ? std::function<void()>() : std::function<void()>(attn));
    }
    p.P = P;
    p.gh = gh;
    p.gw = gw;
    p.N = gh * gw;
    p.have_prefix = true;
    return true;
}

bool Dit21::step(const float* packed, float t, float* out) {
    Impl& p = *p_;
    if (!p.ok || !p.have_prefix) return false;
    const int N = p.N, Np = (N + 63) / 64 * 64, P = p.P, Tk = P + N;
    const size_t nd = (size_t)N * D;
    DevBuf m;
    p.modulation(t, m);
    p.gather(m, N, true);
    /* the target's tokens */
    need(p.X, nd);
    need(p.pk, (size_t)N * OUTC);
    std::memcpy(p.pk.get(), packed, sizeof(float) * (size_t)N * OUTC);
    lin_bf(p.pk.get(), N, p.wimg, OUTC, D, p.X.get());
    /* the keys and values of a step: the cached prefix's rows, then the target's (written by its products) */
    if (p.bf16) need(p.kb, (size_t)(P + Np) * D), need(p.vb, (size_t)(P + Np) * D);
    else need(p.kst, (size_t)Tk * D), need(p.vst, (size_t)Tk * D);
    for (int l = 0; l < LAYERS; ++l) {
        const Impl::Layer& Ly = p.layers[l];
        uint16_t *kd = nullptr, *vd = nullptr;
        float *kf = nullptr, *vf = nullptr;
        if (p.bf16) {
            /* the prefix's rows copied in (as 32-bit words: P D BF16 values, P D even) */
            sddev::audio_copy(reinterpret_cast<float*>(p.kb.get()), reinterpret_cast<const float*>(p.K16[l].get()), (size_t)P * D / 2);
            sddev::audio_copy(reinterpret_cast<float*>(p.vb.get()), reinterpret_cast<const float*>(p.V16[l].get()), (size_t)P * D / 2);
            kd = p.kb.get() + (size_t)P * D;
            vd = p.vb.get() + (size_t)P * D;
        } else {
            sddev::audio_copy(p.kst.get(), p.K32[l].get(), (size_t)P * D);
            sddev::audio_copy(p.vst.get(), p.V32[l].get(), (size_t)P * D);
            kf = p.kst.get() + (size_t)P * D;
            vf = p.vst.get() + (size_t)P * D;
        }
        /* the target's rows see the whole prefix and the target */
        auto attn = [&]() {
            const int qr = p.qrows(N, Tk);
            for (int r0 = 0; r0 < N; r0 += qr) {
                const int rows = std::min(qr, N - r0);
                const size_t off = (size_t)r0 * D;
                if (p.bf16) sddev::attention_bf16(p.qb.get() + off, rows, p.kb.get(), p.vb.get(), Tk, D, HEADS, p.att.get() + off);
                else sddev::attention_f32(p.q.get() + off, rows, p.kst.get(), p.vst.get(), Tk, D, HEADS, p.att.get() + off);
            }
        };
        p.block(Ly, N, p.frt.get(), kd, vd, kf, vf, true, attn);
    }
    /* the output: LayerNorm, (1 + scale) from the output norm's row, the projection */
    need(p.xn, nd);
    sddev::layer_norm_plain(p.X.get(), N, D, EPS, p.xn.get());
    sddev::mul_1p(p.xn.get(), p.scout.get(), nd);
    DevBuf res((size_t)N * OUTC);
    lin_bf(p.xn.get(), N, p.wproj, D, OUTC, res.get());
    std::memcpy(out, res.get(), sizeof(float) * (size_t)N * OUTC);
    return true;
}

bool dev_dit21_forward(const zimage::SafeTensors& st, const float* packed, int gh, int gw, const float* enc, int L, float t, float* out) {
    Dit21 d(st);
    Condition c;
    c.enc = enc;
    c.L = L;
    return d.ok() && d.prefix(c, gh, gw) && d.step(packed, t, out);
}

}  // namespace qwenimage21
