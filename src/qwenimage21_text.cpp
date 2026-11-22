/* Qwen-Image 2.1's text encoder on the device: see qwenimage21_text.h. The embedding rows are dequantised and rotated on the host (a
 * lookup of T rows); everything else is on the device: the ConvRot linears, the RMS norms (BF16 weights), rotate-half RoPE, causal
 * grouped-query attention (T <= 64), SwiGLU. */
#include "qwenimage21_text.h"

#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwenimage21 {

namespace {

const int HID = 4096, HEADS = 32, KV = 8, HD = 128, MLP = 12288, LAYERS = 36, VOCAB = 151936;
const float EPS = 1e-6f;
const double THETA = 5000000.0;

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

struct QLin {
    const int8_t* W = nullptr;
    const float* s = nullptr;
    int in = 0, out = 0;
};

bool load_q(const zimage::SafeTensors& st, const std::string& base, int in, int out, QLin& L) {
    std::vector<size_t> sw, ss;
    if (!st.i8_in_place(base + ".weight", L.W, sw) || sw != std::vector<size_t>{(size_t)out, (size_t)in} ||
        !st.f32_in_place(base + ".weight_scale", L.s, ss) || ss.size() != 2 || ss[0] != (size_t)out) {
        fprintf(stderr, "qwenimage21 text: %s (int8 %dx%d) missing or of the wrong shape\n", base.c_str(), out, in);
        return false;
    }
    L.in = in;
    L.out = out;
    return true;
}

bool bf(const zimage::SafeTensors& st, const std::string& name, size_t count, const uint16_t*& p) {
    std::vector<size_t> s;
    size_t n = 1;
    if (!st.bf16_in_place(name, p, s)) {
        fprintf(stderr, "qwenimage21 text: %s missing or not BF16\n", name.c_str());
        return false;
    }
    for (size_t d : s) n *= d;
    if (n != count) {
        fprintf(stderr, "qwenimage21 text: %s has %zu elements, want %zu\n", name.c_str(), n, count);
        return false;
    }
    return true;
}

}  // namespace

bool dev_text21_encode(const zimage::SafeTensors& st, const int* ids, int T, float* out) {
    if (!sddev::enabled()) { fprintf(stderr, "qwenimage21 text: no device\n"); return false; }
    if (T < 1 || T > 64) { fprintf(stderr, "qwenimage21 text: %d tokens (1 to 64)\n", T); return false; }

    /* the Hadamard of 256 (the same as the DiT's ConvRot layers) */
    static const double H4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
    std::vector<double> Hd = {1.0};
    int n = 1;
    while (n < 256) {
        std::vector<double> K((size_t)n * 4 * n * 4);
        for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
            K[(size_t)(a * 4 + i) * (n * 4) + (b * 4 + j)] = Hd[(size_t)a * n + b] * H4[i][j];
        Hd.swap(K);
        n *= 4;
    }
    std::vector<float> Hf(256 * 256);
    for (size_t i = 0; i < Hf.size(); ++i) Hf[i] = (float)(Hd[i] / 16.0);
    DevBuf H((size_t)256 * 256);
    std::memcpy(H.get(), Hf.data(), Hf.size() * 4);

    /* the embedding rows: W[r] = (q[r] s[r]) H (the rotation along the hidden dimension), gathered for the T tokens on the host */
    const int8_t* Eq = nullptr;
    const float* Es = nullptr;
    std::vector<size_t> se, ses;
    if (!st.i8_in_place("model.embed_tokens.weight", Eq, se) || se != std::vector<size_t>{(size_t)VOCAB, (size_t)HID} ||
        !st.f32_in_place("model.embed_tokens.weight_scale", Es, ses)) {
        fprintf(stderr, "qwenimage21 text: the embedding table missing\n");
        return false;
    }
    DevBuf X((size_t)T * HID);
    {
        std::vector<float> rows((size_t)T * HID);
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= VOCAB) { fprintf(stderr, "qwenimage21 text: token %d out of range\n", ids[t]); return false; }
            const int8_t* q = Eq + (size_t)ids[t] * HID;
            const float s = Es[ids[t]];
            for (int b = 0; b < HID / 256; ++b)
                for (int c = 0; c < 256; ++c) {
                    double acc = 0;
                    for (int j = 0; j < 256; ++j) acc += (double)q[b * 256 + j] * s * Hd[(size_t)j * 256 + c] / 16.0;  /* the normalised H */
                    rows[(size_t)t * HID + b * 256 + c] = (float)acc;
                }
        }
        std::memcpy(X.get(), rows.data(), rows.size() * 4);
    }

    /* RoPE: rotate-half at theta 5e6, the cos and sin per position [T][64] */
    DevBuf cs((size_t)T * 64), sn((size_t)T * 64);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < 64; ++i) {
            const double f = std::pow(THETA, -2.0 * i / HD);
            cs.get()[t * 64 + i] = (float)std::cos(t * f);
            sn.get()[t * 64 + i] = (float)std::sin(t * f);
        }

    DevBuf xr((size_t)T * MLP), h((size_t)T * HID), q((size_t)T * HID), k((size_t)T * KV * HD), v((size_t)T * KV * HD),
        att((size_t)T * HID), o((size_t)T * HID), g((size_t)T * MLP), u((size_t)T * MLP), d((size_t)T * HID);
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        QLin wq, wk, wv, wo, wg, wu, wd;
        const uint16_t *ln1 = nullptr, *ln2 = nullptr, *qn = nullptr, *kn = nullptr;
        if (!bf(st, p + "input_layernorm.weight", HID, ln1) || !bf(st, p + "post_attention_layernorm.weight", HID, ln2) ||
            !bf(st, p + "self_attn.q_norm.weight", HD, qn) || !bf(st, p + "self_attn.k_norm.weight", HD, kn) ||
            !load_q(st, p + "self_attn.q_proj", HID, HEADS * HD, wq) || !load_q(st, p + "self_attn.k_proj", HID, KV * HD, wk) ||
            !load_q(st, p + "self_attn.v_proj", HID, KV * HD, wv) || !load_q(st, p + "self_attn.o_proj", HEADS * HD, HID, wo) ||
            !load_q(st, p + "mlp.gate_proj", HID, MLP, wg) || !load_q(st, p + "mlp.up_proj", HID, MLP, wu) ||
            !load_q(st, p + "mlp.down_proj", MLP, HID, wd))
            return false;
        /* attention */
        sddev::rms_norm(X.get(), T, HID, ln1, EPS, h.get());
        sddev::linear_convrot(h.get(), T, wq.W, wq.s, H.get(), HID, HEADS * HD, xr.get(), q.get());
        sddev::linear_convrot(h.get(), T, wk.W, wk.s, H.get(), HID, KV * HD, xr.get(), k.get());
        sddev::linear_convrot(h.get(), T, wv.W, wv.s, H.get(), HID, KV * HD, xr.get(), v.get());
        sddev::rms_norm(q.get(), T * HEADS, HD, qn, EPS, q.get());
        sddev::rms_norm(k.get(), T * KV, HD, kn, EPS, k.get());
        sddev::rope_half(q.get(), T, HEADS, cs.get(), sn.get());
        sddev::rope_half(k.get(), T, KV, cs.get(), sn.get());
        sddev::causal_attn(q.get(), k.get(), v.get(), T, HEADS, KV, att.get());
        sddev::linear_convrot(att.get(), T, wo.W, wo.s, H.get(), HEADS * HD, HID, xr.get(), o.get());
        sddev::add(X.get(), o.get(), (size_t)T * HID);
        /* SwiGLU */
        sddev::rms_norm(X.get(), T, HID, ln2, EPS, h.get());
        sddev::linear_convrot(h.get(), T, wg.W, wg.s, H.get(), HID, MLP, xr.get(), g.get());
        sddev::linear_convrot(h.get(), T, wu.W, wu.s, H.get(), HID, MLP, xr.get(), u.get());
        sddev::swiglu(g.get(), u.get(), g.get(), (size_t)T * MLP);
        sddev::linear_convrot(g.get(), T, wd.W, wd.s, H.get(), MLP, HID, xr.get(), d.get());
        sddev::add(X.get(), d.get(), (size_t)T * HID);
    }
    std::memcpy(out, X.get(), sizeof(float) * (size_t)T * HID);
    return true;
}

}  // namespace qwenimage21
