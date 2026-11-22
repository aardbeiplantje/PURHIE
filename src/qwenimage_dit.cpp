/* Qwen-Image's diffusion transformer on the device: see qwenimage_dit.h. The block is the reference's QwenImageTransformerBlock:
 * the modulations (SiLU of the timestep embedding, then a linear: six chunks per stream, shift, scale, gate for the attention and
 * for the MLP), LayerNorm (no affine) and the adaLN affine, the joint attention (text rows first, the image rows after; per-head RMS
 * norms; interleaved complex RoPE), the gated residuals, and the GELU (tanh) MLPs. */
#include "qwenimage_dit.h"

#include "sd_device.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwenimage {

namespace {

const int D = 3072, HEADS = 24, HEAD = 128, FF = 12288, LAYERS = 60, TEXT = 3584, OUTC = 64, FREQ = 256;
const float EPS = 1e-6f;

class DevBuf {
public:
    explicit DevBuf(size_t count, bool zero = false) : n_(count) { p_ = static_cast<float*>(sddev::alloc(sizeof(float) * count, zero)); }
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { if (p_) sddev::release(p_, sizeof(float) * n_); }
    float* get() const { return p_; }

private:
    float* p_ = nullptr;
    size_t n_ = 0;
};

/* one linear: E4M3 weights [out][in] (in place), the per-tensor scale when the file has one (else 1), the E4M3 bias */
struct Lin {
    const uint8_t* W = nullptr;
    float scale = 1.0f;
    const uint8_t* bias = nullptr;
    int in = 0, out = 0;
};

bool load_lin(const zimage::SafeTensors& st, const std::string& prefix, int in, int out, Lin& L) {
    std::vector<size_t> s;
    if (!st.fp8_in_place(prefix + ".weight", L.W, s) || s != std::vector<size_t>{(size_t)out, (size_t)in}) {
        fprintf(stderr, "qwenimage dit: %s.weight missing or of the wrong shape\n", prefix.c_str());
        return false;
    }
    const float* sc = nullptr;
    std::vector<size_t> ss;
    if (st.f32_in_place(prefix + ".scale_weight", sc, ss) && ss.empty()) L.scale = sc[0];
    if (!st.fp8_in_place(prefix + ".bias", L.bias, s) || s != std::vector<size_t>{(size_t)out}) {
        fprintf(stderr, "qwenimage dit: %s.bias missing\n", prefix.c_str());
        return false;
    }
    L.in = in;
    L.out = out;
    return true;
}

void apply(const Lin& L, const float* x, int T, float* y) {
    sddev::linear_f8(x, T, L.W, L.scale, L.in, L.out, L.bias, 1, y);
}

bool load_f8vec(const zimage::SafeTensors& st, const std::string& name, size_t n, const uint8_t*& p) {
    std::vector<size_t> s;
    if (!st.fp8_in_place(name, p, s) || s != std::vector<size_t>{n}) {
        fprintf(stderr, "qwenimage dit: %s missing or of the wrong shape\n", name.c_str());
        return false;
    }
    return true;
}

/* the RoPE table of the joint sequence: fr[(L + N)][64][2] = (cos, sin) per complex pair. The pairs are the frame axis (8, at
 * position 0), the height axis (28) and the width axis (28; inverse frequencies 10000^(-2k/16) and 10000^(-2k/56)). Image tokens
 * (row-major over gh x gw) take centred positions (row - (gh - gh/2)); the text tokens take the positions max(gh/2, gw/2) + j. */
void rope_table(int L, int gh, int gw, std::vector<float>& fr) {
    const int N = gh * gw;
    fr.assign((size_t)(L + N) * 64 * 2, 0.0f);
    auto inv = [](int dim, int k) { return std::pow(10000.0, -2.0 * k / dim); };
    auto put = [&](size_t tok, int pair, double angle) {
        fr[(tok * 64 + pair) * 2] = (float)std::cos(angle);
        fr[(tok * 64 + pair) * 2 + 1] = (float)std::sin(angle);
    };
    const int mx = std::max(gh / 2, gw / 2);
    for (int j = 0; j < L; ++j) {
        const double p = mx + j;
        for (int k = 0; k < 8; ++k) put(j, k, p * inv(16, k));
        for (int k = 0; k < 28; ++k) put(j, 8 + k, p * inv(56, k));
        for (int k = 0; k < 28; ++k) put(j, 36 + k, p * inv(56, k));
    }
    const int hh = gh - gh / 2, ww = gw - gw / 2;
    for (int r = 0; r < gh; ++r)
        for (int c = 0; c < gw; ++c) {
            const size_t tok = (size_t)L + (size_t)r * gw + c;
            for (int k = 0; k < 8; ++k) put(tok, k, 0.0);
            for (int k = 0; k < 28; ++k) put(tok, 8 + k, (r - hh) * inv(56, k));
            for (int k = 0; k < 28; ++k) put(tok, 36 + k, (c - ww) * inv(56, k));
        }
}

}  // namespace

bool dev_dit_forward(const zimage::SafeTensors& st, const float* packed, int gh, int gw, const float* enc, int L, float t, float* out) {
    if (!sddev::enabled()) { fprintf(stderr, "qwenimage dit: no device\n"); return false; }
    if (gh < 1 || gw < 1 || L < 1) { fprintf(stderr, "qwenimage dit: bad grid or text length\n"); return false; }
    const int N = gh * gw, J = L + N;

    /* the timestep: the sinusoid of 1000 t (cos block first), then Linear(256, 3072), SiLU, Linear(3072, 3072) */
    std::vector<float> e256(FREQ);
    for (int i = 0; i < FREQ / 2; ++i) {
        const double arg = 1000.0 * t * std::exp(-std::log(10000.0) * i / (FREQ / 2));
        e256[i] = (float)std::cos(arg);
        e256[FREQ / 2 + i] = (float)std::sin(arg);
    }
    Lin t1, t2, imgin, txtin, projout, nout;
    if (!load_lin(st, "time_text_embed.timestep_embedder.linear_1", FREQ, D, t1) ||
        !load_lin(st, "time_text_embed.timestep_embedder.linear_2", D, D, t2) ||
        !load_lin(st, "img_in", OUTC, D, imgin) || !load_lin(st, "txt_in", TEXT, D, txtin) ||
        !load_lin(st, "proj_out", D, OUTC, projout) || !load_lin(st, "norm_out.linear", D, 2 * D, nout))
        return false;
    const uint8_t *wtxtnorm = nullptr;
    if (!load_f8vec(st, "txt_norm.weight", TEXT, wtxtnorm)) return false;

    DevBuf e(FREQ);
    std::memcpy(e.get(), e256.data(), sizeof(float) * FREQ);
    DevBuf h1(D), temb(D), siluT(D);
    apply(t1, e.get(), 1, h1.get());
    sddev::silu(h1.get(), D);
    apply(t2, h1.get(), 1, temb.get());
    std::memcpy(siluT.get(), temb.get(), sizeof(float) * D);
    sddev::silu(siluT.get(), D);

    /* the streams: the image tokens from the packed latent, the text from the encoder (RMS norm, then the input linear). The
     * inputs are copied into shared memory first: the device reads the activations there, not the caller's memory. */
    DevBuf pk((size_t)N * 64), ec((size_t)L * TEXT);
    std::memcpy(pk.get(), packed, sizeof(float) * (size_t)N * 64);
    std::memcpy(ec.get(), enc, sizeof(float) * (size_t)L * TEXT);
    DevBuf img((size_t)N * D), txt((size_t)L * D);
    apply(imgin, pk.get(), N, img.get());
    DevBuf tn((size_t)L * TEXT);
    sddev::rms_norm_f8(ec.get(), L, TEXT, wtxtnorm, EPS, tn.get());
    apply(txtin, tn.get(), L, txt.get());

    std::vector<float> fr;
    rope_table(L, gh, gw, fr);
    DevBuf frd((size_t)J * 64 * 2);
    std::memcpy(frd.get(), fr.data(), sizeof(float) * fr.size());

    /* the blocks */
    DevBuf Q((size_t)J * D), K((size_t)J * D), V((size_t)J * D), A((size_t)J * D), LN((size_t)J * D);
    DevBuf Oi((size_t)N * D), Ot((size_t)L * D), P((size_t)N * FF), Pt((size_t)L * FF), M((size_t)N * D), Mt((size_t)L * D);
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "transformer_blocks." + std::to_string(l) + ".";
        Lin imod, tmod, aq, ak, av, tq, tk, tv, ao, tao, imlp1, imlp2, tmlp1, tmlp2;
        const uint8_t *nq, *nk, *naq, *nak;
        if (!load_lin(st, p + "img_mod.1", D, 6 * D, imod) || !load_lin(st, p + "txt_mod.1", D, 6 * D, tmod) ||
            !load_lin(st, p + "attn.to_q", D, D, aq) || !load_lin(st, p + "attn.to_k", D, D, ak) ||
            !load_lin(st, p + "attn.to_v", D, D, av) || !load_lin(st, p + "attn.add_q_proj", D, D, tq) ||
            !load_lin(st, p + "attn.add_k_proj", D, D, tk) || !load_lin(st, p + "attn.add_v_proj", D, D, tv) ||
            !load_lin(st, p + "attn.to_out.0", D, D, ao) || !load_lin(st, p + "attn.to_add_out", D, D, tao) ||
            !load_lin(st, p + "img_mlp.net.0.proj", D, FF, imlp1) || !load_lin(st, p + "img_mlp.net.2", FF, D, imlp2) ||
            !load_lin(st, p + "txt_mlp.net.0.proj", D, FF, tmlp1) || !load_lin(st, p + "txt_mlp.net.2", FF, D, tmlp2) ||
            !load_f8vec(st, p + "attn.norm_q.weight", HEAD, nq) || !load_f8vec(st, p + "attn.norm_k.weight", HEAD, nk) ||
            !load_f8vec(st, p + "attn.norm_added_q.weight", HEAD, naq) || !load_f8vec(st, p + "attn.norm_added_k.weight", HEAD, nak))
            return false;
        /* modulations: six chunks of 3072 per stream: (shift, scale, gate) for the attention, then for the MLP */
        DevBuf mi(6 * D), mt(6 * D);
        apply(imod, siluT.get(), 1, mi.get());
        apply(tmod, siluT.get(), 1, mt.get());
        const float *ish1 = mi.get(), *isc1 = mi.get() + D, *ig1 = mi.get() + 2 * D;
        const float *ish2 = mi.get() + 3 * D, *isc2 = mi.get() + 4 * D, *ig2 = mi.get() + 5 * D;
        const float *tsh1 = mt.get(), *tsc1 = mt.get() + D, *tg1 = mt.get() + 2 * D;
        const float *tsh2 = mt.get() + 3 * D, *tsc2 = mt.get() + 4 * D, *tg2 = mt.get() + 5 * D;

        /* attention: the normed, modulated streams projected into the joint buffers (text rows first) */
        sddev::layer_norm_plain(img.get(), N, D, EPS, LN.get() + (size_t)L * D);
        sddev::affine(LN.get() + (size_t)L * D, N, D, isc1, ish1);
        sddev::layer_norm_plain(txt.get(), L, D, EPS, LN.get());
        sddev::affine(LN.get(), L, D, tsc1, tsh1);
        apply(tq, LN.get(), L, Q.get());
        apply(aq, LN.get() + (size_t)L * D, N, Q.get() + (size_t)L * D);
        apply(tk, LN.get(), L, K.get());
        apply(ak, LN.get() + (size_t)L * D, N, K.get() + (size_t)L * D);
        apply(tv, LN.get(), L, V.get());
        apply(av, LN.get() + (size_t)L * D, N, V.get() + (size_t)L * D);
        /* the per-head RMS norms (the text's and the image's weights), then the RoPE over the joint rows */
        sddev::rms_norm_f8(Q.get(), L * HEADS, HEAD, naq, EPS, Q.get());
        sddev::rms_norm_f8(Q.get() + (size_t)L * D, N * HEADS, HEAD, nq, EPS, Q.get() + (size_t)L * D);
        sddev::rms_norm_f8(K.get(), L * HEADS, HEAD, nak, EPS, K.get());
        sddev::rms_norm_f8(K.get() + (size_t)L * D, N * HEADS, HEAD, nk, EPS, K.get() + (size_t)L * D);
        sddev::rope(Q.get(), J, HEADS, frd.get());
        sddev::rope(K.get(), J, HEADS, frd.get());
        sddev::attention(Q.get(), J, K.get(), V.get(), J, D, HEADS, A.get());
        apply(ao, A.get() + (size_t)L * D, N, Oi.get());
        apply(tao, A.get(), L, Ot.get());
        sddev::gate_add(img.get(), N, D, ig1, Oi.get());
        sddev::gate_add(txt.get(), L, D, tg1, Ot.get());

        /* MLPs: norm, modulate, GELU (tanh) feed-forward, gated residual */
        sddev::layer_norm_plain(img.get(), N, D, EPS, LN.get() + (size_t)L * D);
        sddev::affine(LN.get() + (size_t)L * D, N, D, isc2, ish2);
        apply(imlp1, LN.get() + (size_t)L * D, N, P.get());
        sddev::gelu_tanh(P.get(), (size_t)N * FF);
        apply(imlp2, P.get(), N, M.get());
        sddev::gate_add(img.get(), N, D, ig2, M.get());
        sddev::layer_norm_plain(txt.get(), L, D, EPS, LN.get());
        sddev::affine(LN.get(), L, D, tsc2, tsh2);
        apply(tmlp1, LN.get(), L, Pt.get());
        sddev::gelu_tanh(Pt.get(), (size_t)L * FF);
        apply(tmlp2, Pt.get(), L, Mt.get());
        sddev::gate_add(txt.get(), L, D, tg2, Mt.get());
    }

    /* the output: AdaLayerNormContinuous (scale first, then shift, of the SiLU'd timestep), then proj_out */
    DevBuf em(2 * D);
    apply(nout, siluT.get(), 1, em.get());
    sddev::layer_norm_plain(img.get(), N, D, EPS, LN.get());
    sddev::affine(LN.get(), N, D, em.get(), em.get() + D);
    DevBuf res((size_t)N * OUTC);
    apply(projout, LN.get(), N, res.get());
    std::memcpy(out, res.get(), sizeof(float) * (size_t)N * OUTC);
    return true;
}

}  // namespace qwenimage
