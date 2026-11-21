/* Z-Image's text encoder on the device: see zimage_text.h. The formulas are the Qwen3 decoder layer's (RMS norms with eps 1e-6,
 * per-head QK norms, rotate-half RoPE at theta 1e6, grouped-query causal attention, SwiGLU). */
#include "zimage_text.h"

#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace zimage {

namespace {

class DevBuf {
public:
    explicit DevBuf(size_t count, bool zero = false) : n_(count) { p_ = static_cast<float*>(sddev::alloc(sizeof(float) * count, zero)); }
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) { reset(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { reset(); }
    float* get() const { return p_; }

private:
    void reset() { if (p_) sddev::release(p_, sizeof(float) * n_); p_ = nullptr; n_ = 0; }
    float* p_ = nullptr;
    size_t n_ = 0;
};

const int D_MODEL = 2560, HEADS = 32, KV_HEADS = 8, HEAD = 128, MLP = 9728, LAYERS = 35;
const float EPS = 1e-6f;

bool bf16(const SafeTensors& st, const std::string& name, std::vector<size_t> shape, const uint16_t*& p) {
    std::vector<size_t> s;
    if (!st.bf16_in_place(name, p, s) || s != shape) {
        fprintf(stderr, "zimage text: %s missing or of the wrong shape\n", name.c_str());
        return false;
    }
    return true;
}

/* y = x W^T (no bias), x [T][in] */
void lin(const float* x, int T, const uint16_t* W, int in, int out, const float* zeros, float* y) {
    sddev::linear_bf16(x, T, W, in, out, zeros, y);
}

}  // namespace

bool dev_text_encode(const SafeTensors& st, const int* ids, int T, float* out) {
    if (!sddev::enabled()) { fprintf(stderr, "zimage text: no device\n"); return false; }
    if (T < 1 || T > 64) { fprintf(stderr, "zimage text: %d tokens (1 to 64)\n", T); return false; }
    const uint16_t* emb = nullptr;
    if (!bf16(st, "model.embed_tokens.weight", {151936, D_MODEL}, emb)) return false;
    /* the embedding rows (read in place from the mapping: the rows are the activations' input) */
    DevBuf X((size_t)T * D_MODEL);
    for (int t = 0; t < T; ++t) {
        if (ids[t] < 0 || ids[t] >= 151936) { fprintf(stderr, "zimage text: token %d out of range\n", ids[t]); return false; }
        const uint16_t* row = emb + (size_t)ids[t] * D_MODEL;
        for (int c = 0; c < D_MODEL; ++c) {
            const uint32_t bits = (uint32_t)row[c] << 16;
            float f;
            std::memcpy(&f, &bits, 4);
            X.get()[(size_t)t * D_MODEL + c] = f;
        }
    }
    /* RoPE: angle = position * theta^(-2i/128), i in 0..63 (rotate-half pairs) */
    DevBuf cs((size_t)T * 64), sn((size_t)T * 64);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < 64; ++i) {
            const double f = std::pow(1e6, -2.0 * i / 128.0);
            cs.get()[t * 64 + i] = (float)std::cos(t * f);
            sn.get()[t * 64 + i] = (float)std::sin(t * f);
        }
    DevBuf zeros((size_t)MLP, true);
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const uint16_t *wln = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wqn = nullptr, *wkn = nullptr, *wo = nullptr,
                       *wpln = nullptr, *wg = nullptr, *wu = nullptr, *wd = nullptr;
        if (!bf16(st, p + "input_layernorm.weight", {D_MODEL}, wln) ||
            !bf16(st, p + "self_attn.q_proj.weight", {HEADS * HEAD, D_MODEL}, wq) ||
            !bf16(st, p + "self_attn.k_proj.weight", {KV_HEADS * HEAD, D_MODEL}, wk) ||
            !bf16(st, p + "self_attn.v_proj.weight", {KV_HEADS * HEAD, D_MODEL}, wv) ||
            !bf16(st, p + "self_attn.q_norm.weight", {HEAD}, wqn) || !bf16(st, p + "self_attn.k_norm.weight", {HEAD}, wkn) ||
            !bf16(st, p + "self_attn.o_proj.weight", {D_MODEL, HEADS * HEAD}, wo) ||
            !bf16(st, p + "post_attention_layernorm.weight", {D_MODEL}, wpln) ||
            !bf16(st, p + "mlp.gate_proj.weight", {MLP, D_MODEL}, wg) || !bf16(st, p + "mlp.up_proj.weight", {MLP, D_MODEL}, wu) ||
            !bf16(st, p + "mlp.down_proj.weight", {D_MODEL, MLP}, wd))
            return false;
        /* attention */
        DevBuf h((size_t)T * D_MODEL);
        sddev::rms_norm(X.get(), T, D_MODEL, wln, EPS, h.get());
        DevBuf q((size_t)T * HEADS * HEAD), k((size_t)T * KV_HEADS * HEAD), v((size_t)T * KV_HEADS * HEAD);
        lin(h.get(), T, wq, D_MODEL, HEADS * HEAD, zeros.get(), q.get());
        lin(h.get(), T, wk, D_MODEL, KV_HEADS * HEAD, zeros.get(), k.get());
        lin(h.get(), T, wv, D_MODEL, KV_HEADS * HEAD, zeros.get(), v.get());
        sddev::rms_norm(q.get(), T * HEADS, HEAD, wqn, EPS, q.get());
        sddev::rms_norm(k.get(), T * KV_HEADS, HEAD, wkn, EPS, k.get());
        sddev::rope_half(q.get(), T, HEADS, cs.get(), sn.get());
        sddev::rope_half(k.get(), T, KV_HEADS, cs.get(), sn.get());
        DevBuf att((size_t)T * HEADS * HEAD);
        sddev::causal_attn(q.get(), k.get(), v.get(), T, HEADS, KV_HEADS, att.get());
        DevBuf o((size_t)T * D_MODEL);
        lin(att.get(), T, wo, HEADS * HEAD, D_MODEL, zeros.get(), o.get());
        sddev::add(X.get(), o.get(), (size_t)T * D_MODEL);
        /* SwiGLU MLP */
        DevBuf h2((size_t)T * D_MODEL);
        sddev::rms_norm(X.get(), T, D_MODEL, wpln, EPS, h2.get());
        DevBuf g((size_t)T * MLP), u((size_t)T * MLP);
        lin(h2.get(), T, wg, D_MODEL, MLP, zeros.get(), g.get());
        lin(h2.get(), T, wu, D_MODEL, MLP, zeros.get(), u.get());
        sddev::swiglu(g.get(), u.get(), g.get(), (size_t)T * MLP);
        DevBuf dn((size_t)T * D_MODEL);
        lin(g.get(), T, wd, MLP, D_MODEL, zeros.get(), dn.get());
        sddev::add(X.get(), dn.get(), (size_t)T * D_MODEL);
    }
    std::memcpy(out, X.get(), sizeof(float) * (size_t)T * D_MODEL);
    return true;
}

}  // namespace zimage
