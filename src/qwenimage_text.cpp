/* Qwen-Image's text encoder on the device: see qwenimage_text.h. The formulas are Qwen2's decoder layer (RMS norms with eps 1e-6,
 * q, k and v with biases, no QK norms, rotate-half RoPE at theta 1e6 (text positions: the three M-RoPE sections agree), grouped-query
 * causal attention with 28 query and 4 key/value heads, SwiGLU). */
#include "qwenimage_text.h"

#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace qwenimage {

namespace {

const int D = 3584, HEADS = 28, KV_HEADS = 4, HEAD = 128, MLP = 18944, LAYERS = 28, VOCAB = 152064;
const float EPS = 1e-6f;
const double THETA = 1e6;

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

/* one FP8 linear: the E4M3 weights (in place), the scale, the bias (BF16 in place, or none) */
struct F8Linear {
    const uint8_t* W = nullptr;
    float scale = 1.0f;
    const void* bias = nullptr;
    int bias_kind = 0;
    int in = 0, out = 0;
};

bool load_f8(const zimage::SafeTensors& st, const std::string& prefix, int in, int out, bool has_bias, F8Linear& L) {
    std::vector<size_t> s;
    if (!st.fp8_in_place(prefix + ".weight", L.W, s) || s != std::vector<size_t>{(size_t)out, (size_t)in}) {
        fprintf(stderr, "qwenimage text: %s.weight missing or of the wrong shape\n", prefix.c_str());
        return false;
    }
    const float* sc = nullptr;
    std::vector<size_t> ss;
    if (!st.f32_in_place(prefix + ".scale_weight", sc, ss) || !ss.empty()) {
        fprintf(stderr, "qwenimage text: %s.scale_weight missing (a per-tensor scalar)\n", prefix.c_str());
        return false;
    }
    L.scale = sc[0];
    L.in = in;
    L.out = out;
    if (has_bias) {
        const uint16_t* b = nullptr;
        if (!st.bf16_in_place(prefix + ".bias", b, s) || s != std::vector<size_t>{(size_t)out}) {
            fprintf(stderr, "qwenimage text: %s.bias missing\n", prefix.c_str());
            return false;
        }
        L.bias = b;
        L.bias_kind = 2;
    }
    return true;
}

void lin(const float* x, int T, const F8Linear& L, float* y) {
    sddev::linear_f8(x, T, L.W, L.scale, L.in, L.out, L.bias, L.bias_kind, y);
}

bool bf16(const zimage::SafeTensors& st, const std::string& name, std::vector<size_t> shape, const uint16_t*& p) {
    std::vector<size_t> s;
    if (!st.bf16_in_place(name, p, s) || s != shape) {
        fprintf(stderr, "qwenimage text: %s missing or of the wrong shape\n", name.c_str());
        return false;
    }
    return true;
}

}  // namespace

bool dev_text_encode(const zimage::SafeTensors& st, const int* ids, int T, float* out) {
    if (!sddev::enabled()) { fprintf(stderr, "qwenimage text: no device\n"); return false; }
    if (T < 1 || T > 64) { fprintf(stderr, "qwenimage text: %d tokens (1 to 64)\n", T); return false; }
    const uint16_t* emb = nullptr;
    if (!bf16(st, "model.embed_tokens.weight", {(size_t)VOCAB, (size_t)D}, emb)) return false;
    DevBuf X((size_t)T * D);
    for (int t = 0; t < T; ++t) {
        if (ids[t] < 0 || ids[t] >= VOCAB) { fprintf(stderr, "qwenimage text: token %d out of range\n", ids[t]); return false; }
        const uint16_t* row = emb + (size_t)ids[t] * D;
        for (int c = 0; c < D; ++c) {
            const uint32_t bits = (uint32_t)row[c] << 16;
            float f;
            std::memcpy(&f, &bits, 4);
            X.get()[(size_t)t * D + c] = f;
        }
    }
    /* RoPE: angle = position * theta^(-2i/128), i in 0..63 (rotate-half pairs) */
    DevBuf cs((size_t)T * 64), sn((size_t)T * 64);
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < 64; ++i) {
            const double f = std::pow(THETA, -2.0 * i / 128.0);
            cs.get()[t * 64 + i] = (float)std::cos(t * f);
            sn.get()[t * 64 + i] = (float)std::sin(t * f);
        }
    for (int l = 0; l < LAYERS; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const uint16_t *wln = nullptr, *wpln = nullptr;
        F8Linear q, k, v, o, g, u, d;
        if (!bf16(st, p + "input_layernorm.weight", {(size_t)D}, wln) ||
            !bf16(st, p + "post_attention_layernorm.weight", {(size_t)D}, wpln) ||
            !load_f8(st, p + "self_attn.q_proj", D, HEADS * HEAD, true, q) ||
            !load_f8(st, p + "self_attn.k_proj", D, KV_HEADS * HEAD, true, k) ||
            !load_f8(st, p + "self_attn.v_proj", D, KV_HEADS * HEAD, true, v) ||
            !load_f8(st, p + "self_attn.o_proj", HEADS * HEAD, D, false, o) ||
            !load_f8(st, p + "mlp.gate_proj", D, MLP, false, g) || !load_f8(st, p + "mlp.up_proj", D, MLP, false, u) ||
            !load_f8(st, p + "mlp.down_proj", MLP, D, false, d))
            return false;
        /* attention */
        DevBuf h((size_t)T * D);
        sddev::rms_norm(X.get(), T, D, wln, EPS, h.get());
        DevBuf qb((size_t)T * HEADS * HEAD), kb((size_t)T * KV_HEADS * HEAD), vb((size_t)T * KV_HEADS * HEAD);
        lin(h.get(), T, q, qb.get());
        lin(h.get(), T, k, kb.get());
        lin(h.get(), T, v, vb.get());
        sddev::rope_half(qb.get(), T, HEADS, cs.get(), sn.get());
        sddev::rope_half(kb.get(), T, KV_HEADS, cs.get(), sn.get());
        DevBuf att((size_t)T * HEADS * HEAD);
        sddev::causal_attn(qb.get(), kb.get(), vb.get(), T, HEADS, KV_HEADS, att.get());
        DevBuf ob((size_t)T * D);
        lin(att.get(), T, o, ob.get());
        sddev::add(X.get(), ob.get(), (size_t)T * D);
        /* SwiGLU MLP */
        DevBuf h2((size_t)T * D);
        sddev::rms_norm(X.get(), T, D, wpln, EPS, h2.get());
        DevBuf gb((size_t)T * MLP), ub((size_t)T * MLP);
        lin(h2.get(), T, g, gb.get());
        lin(h2.get(), T, u, ub.get());
        sddev::swiglu(gb.get(), ub.get(), gb.get(), (size_t)T * MLP);
        DevBuf db((size_t)T * D);
        lin(gb.get(), T, d, db.get());
        sddev::add(X.get(), db.get(), (size_t)T * D);
    }
    const uint16_t* wnorm = nullptr;
    if (!bf16(st, "model.norm.weight", {(size_t)D}, wnorm)) return false;
    DevBuf fin((size_t)T * D);
    sddev::rms_norm(X.get(), T, D, wnorm, EPS, fin.get());
    std::memcpy(out, fin.get(), sizeof(float) * (size_t)T * D);
    return true;
}

}  // namespace qwenimage
