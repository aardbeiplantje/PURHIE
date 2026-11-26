// The TRELLIS.2 sparse-structure flow on the device (trellis_ss.h). The stage order and the arithmetic are those of
// trellis2/models/sparse_structure_flow.py: the python model is the reference, and tests/trellis_ss_verify.cpp checks
// each stage against its dump.
#include "trellis_ss.h"

#include "sd_device.h"
#include "zimage_dit.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace trellis {
namespace {

constexpr int kN = 4096;      // tokens: 16^3
constexpr int kC = 1536;      // model channels
constexpr int kH = 12;        // heads
constexpr int kD = 128;       // head dim
constexpr int kF = 8192;      // MLP hidden (mlp_ratio 5.3334)
constexpr int kBlocks = 30;
constexpr int kCond = 1024;   // image feature channels
constexpr int kIn = 8;        // latent channels
constexpr int kFreq = 256;    // timestep embedding size
// queries per attention launch: about kAttnPairs query-key pairs (about 40 ms per launch on the fused kernel at 12 heads), so each
// launch stays short (AGENTS.md, GPU use) and still has enough workgroups to fill the GPU; at least 256, in whole 64-query blocks
constexpr size_t kAttnPairs = (size_t)1 << 24;
int attn_rows(int tq, int tk) {
    size_t r = kAttnPairs / (size_t)(tk > 0 ? tk : 1);
    r -= r % 64;
    if (r < 256) r = 256;
    return (int)std::min<size_t>(r, (size_t)(tq > 0 ? tq : 1));
}

float bf16_to_f32(uint16_t v) {
    const uint32_t u = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

struct SsFlow::Impl {
    explicit Impl(const std::string& path) : st(path) {}
    ~Impl() {
        for (auto& o : owned) sddev::release(o.first, o.second);
    }

    zimage::SafeTensors st;
    std::vector<std::pair<void*, size_t>> owned;  // the fp32 vectors made at open, released at destruction

    // weights, in place: bf16, [out][in] row-major
    const uint16_t *w_in = nullptr, *w_t0 = nullptr, *w_t2 = nullptr, *w_ada = nullptr, *w_out = nullptr;
    // small vectors, fp32
    float *b_in = nullptr, *b_t0 = nullptr, *b_t2 = nullptr, *b_ada = nullptr, *b_out = nullptr;
    struct Block {
        const uint16_t *wqkv, *wso, *wcq, *wkv, *wco, *wm0, *wm2;  // wkv: K rows, then V rows
        float *n2w, *n2b;  // the cross-attention norm is affine in this checkpoint (the other two norms are not)
        float *mod, *bqkv, *bso, *gq, *gk, *bcq, *bkv, *gcq, *gck, *bco, *bm0, *bm2;
    };
    std::vector<Block> blk;

    // the checkpoint's bf16 tensor, read in place; false when it is missing or has another size
    bool weight(const std::string& name, const uint16_t*& p, size_t n) {
        std::vector<size_t> shape;
        if (!st.bf16_in_place(name, p, shape)) return false;
        size_t total = 1;
        for (size_t s : shape) total *= s;
        return total == n;
    }
    // a small tensor as fp32, in device-visible memory (converted once)
    bool vec(const std::string& name, float*& out, size_t n) {
        const uint16_t* p = nullptr;
        if (!weight(name, p, n)) return false;
        out = static_cast<float*>(sddev::alloc(sizeof(float) * n, false));
        owned.push_back({out, sizeof(float) * n});
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
        return true;
    }
};

SsFlow::SsFlow(const std::string& safetensors_path) : p_(new Impl(safetensors_path)) {
    Impl& m = *p_;
    if (!m.st.ok()) {
        error_ = m.st.error();
        return;
    }
    if (!m.st.register_device()) {
        error_ = "the checkpoint could not be registered for the device";
        return;
    }
    const char* what = nullptr;
    auto need = [&](bool good, const char* name) {
        if (!good && !what) what = name;
        return good;
    };
    need(m.weight("input_layer.weight", m.w_in, (size_t)kC * kIn), "input_layer.weight");
    need(m.vec("input_layer.bias", m.b_in, kC), "input_layer.bias");
    need(m.weight("t_embedder.mlp.0.weight", m.w_t0, (size_t)kC * kFreq), "t_embedder.mlp.0.weight");
    need(m.vec("t_embedder.mlp.0.bias", m.b_t0, kC), "t_embedder.mlp.0.bias");
    need(m.weight("t_embedder.mlp.2.weight", m.w_t2, (size_t)kC * kC), "t_embedder.mlp.2.weight");
    need(m.vec("t_embedder.mlp.2.bias", m.b_t2, kC), "t_embedder.mlp.2.bias");
    need(m.weight("adaLN_modulation.1.weight", m.w_ada, (size_t)6 * kC * kC), "adaLN_modulation.1.weight");
    need(m.vec("adaLN_modulation.1.bias", m.b_ada, 6 * kC), "adaLN_modulation.1.bias");
    need(m.weight("out_layer.weight", m.w_out, (size_t)kIn * kC), "out_layer.weight");
    need(m.vec("out_layer.bias", m.b_out, kIn), "out_layer.bias");
    m.blk.resize(kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        const std::string p = "blocks." + std::to_string(b) + ".";
        Impl::Block& k = m.blk[b];
        need(m.vec(p + "modulation", k.mod, 6 * kC), (p + "modulation").c_str());
        need(m.vec(p + "norm2.weight", k.n2w, kC), (p + "norm2.weight").c_str());
        need(m.vec(p + "norm2.bias", k.n2b, kC), (p + "norm2.bias").c_str());
        need(m.weight(p + "self_attn.to_qkv.weight", k.wqkv, (size_t)3 * kC * kC), (p + "self_attn.to_qkv.weight").c_str());
        need(m.vec(p + "self_attn.to_qkv.bias", k.bqkv, 3 * kC), (p + "self_attn.to_qkv.bias").c_str());
        need(m.vec(p + "self_attn.q_rms_norm.gamma", k.gq, (size_t)kH * kD), (p + "self_attn.q_rms_norm.gamma").c_str());
        need(m.vec(p + "self_attn.k_rms_norm.gamma", k.gk, (size_t)kH * kD), (p + "self_attn.k_rms_norm.gamma").c_str());
        need(m.weight(p + "self_attn.to_out.weight", k.wso, (size_t)kC * kC), (p + "self_attn.to_out.weight").c_str());
        need(m.vec(p + "self_attn.to_out.bias", k.bso, kC), (p + "self_attn.to_out.bias").c_str());
        need(m.weight(p + "cross_attn.to_q.weight", k.wcq, (size_t)kC * kC), (p + "cross_attn.to_q.weight").c_str());
        need(m.vec(p + "cross_attn.to_q.bias", k.bcq, kC), (p + "cross_attn.to_q.bias").c_str());
        need(m.weight(p + "cross_attn.to_kv.weight", k.wkv, (size_t)2 * kC * kCond), (p + "cross_attn.to_kv.weight").c_str());
        need(m.vec(p + "cross_attn.to_kv.bias", k.bkv, 2 * kC), (p + "cross_attn.to_kv.bias").c_str());
        need(m.vec(p + "cross_attn.q_rms_norm.gamma", k.gcq, (size_t)kH * kD), (p + "cross_attn.q_rms_norm.gamma").c_str());
        need(m.vec(p + "cross_attn.k_rms_norm.gamma", k.gck, (size_t)kH * kD), (p + "cross_attn.k_rms_norm.gamma").c_str());
        need(m.weight(p + "cross_attn.to_out.weight", k.wco, (size_t)kC * kC), (p + "cross_attn.to_out.weight").c_str());
        need(m.vec(p + "cross_attn.to_out.bias", k.bco, kC), (p + "cross_attn.to_out.bias").c_str());
        need(m.weight(p + "mlp.mlp.0.weight", k.wm0, (size_t)kF * kC), (p + "mlp.mlp.0.weight").c_str());
        need(m.vec(p + "mlp.mlp.0.bias", k.bm0, kF), (p + "mlp.mlp.0.bias").c_str());
        need(m.weight(p + "mlp.mlp.2.weight", k.wm2, (size_t)kC * kF), (p + "mlp.mlp.2.weight").c_str());
        need(m.vec(p + "mlp.mlp.2.bias", k.bm2, kC), (p + "mlp.mlp.2.bias").c_str());
        if (what) break;
    }
    if (what) {
        error_ = std::string("the checkpoint has no tensor ") + what + " of the expected size";
        return;
    }
    ok_ = true;
}

SsFlow::~SsFlow() = default;

bool SsFlow::forward(const float* x, float t, const float* cond, int Lc, const float* rope, float* out,
                     const std::function<void(const char*, const float*, size_t)>& stage) {
    if (!ok_) return false;
    if (!sddev::enabled()) {
        error_ = "no device";
        return false;
    }
    Impl& m = *p_;

    // per-forward device buffers, released on every return
    std::vector<std::pair<void*, size_t>> bufs;
    struct Guard {
        std::vector<std::pair<void*, size_t>>& b;
        ~Guard() {
            for (auto& o : b) sddev::release(o.first, o.second);
        }
    } guard{bufs};
    auto S = [&](size_t n) {
        void* p = sddev::alloc(sizeof(float) * n, false);
        bufs.push_back({p, sizeof(float) * n});
        return static_cast<float*>(p);
    };
    auto emit = [&](const char* name, const float* d, size_t n) {
        if (stage) {

            stage(name, d, n);
        }
    };

    // the timestep embedding: the sinusoid on the host (256 values), then the two linears and the SiLU between them
    float* freq = S(kFreq);
    for (int i = 0; i < kFreq / 2; ++i) {
        const double f = std::exp(-std::log(10000.0) * i / (kFreq / 2));
        const double a = (double)t * f;
        freq[i] = (float)std::cos(a);
        freq[kFreq / 2 + i] = (float)std::sin(a);
    }
    float* e1 = S(kC);
    sddev::linear_bf16(freq, 1, m.w_t0, kFreq, kC, m.b_t0, e1);
    sddev::silu(e1, kC);
    float* temb = S(kC);
    sddev::linear_bf16(e1, 1, m.w_t2, kC, kC, m.b_t2, temb);
    emit("t_embedder", temb, kC);

    // the shared adaLN modulation: SiLU, then a linear to 6 x C
    float* sa = S(kC);
    sddev::audio_copy(sa, temb, kC);
    sddev::silu(sa, kC);
    float* mod = S((size_t)6 * kC);
    sddev::linear_bf16(sa, 1, m.w_ada, kC, 6 * kC, m.b_ada, mod);
    emit("mod_adaLN", mod, (size_t)6 * kC);

    // the input: x [8][4096] (channel-major) to tokens [4096][8], then the input layer
    float* tok = S((size_t)kN * kIn);
    for (int c = 0; c < kIn; ++c)
        for (int n = 0; n < kN; ++n) tok[(size_t)n * kIn + c] = x[(size_t)c * kN + n];
    float* h = S((size_t)kN * kC);
    sddev::linear_bf16(tok, kN, m.w_in, kIn, kC, m.b_in, h);
    emit("h0_input_layer", h, (size_t)kN * kC);

    // the RoPE table and the conditioning, copied in once
    float* ph = S((size_t)kN * (kD / 2) * 2);
    std::memcpy(ph, rope, sizeof(float) * (size_t)kN * (kD / 2) * 2);
    float* cd = S((size_t)Lc * kCond);
    std::memcpy(cd, cond, sizeof(float) * (size_t)Lc * kCond);

    // block buffers
    float* A = S((size_t)kN * kC);
    float* QKV = S((size_t)kN * 3 * kC);
    float* O = S((size_t)kN * kC);
    float* P = S((size_t)kN * kC);
    float* Bq = S((size_t)kN * kC);
    float* Qc = S((size_t)kN * kC);
    float* Kc = S((size_t)Lc * kC);
    float* Vc = S((size_t)Lc * kC);
    float* Oc = S((size_t)kN * kC);
    float* Pc = S((size_t)kN * kC);
    float* F1 = S((size_t)kN * kF);
    float* P3 = S((size_t)kN * kC);
    float* ones = S(kC);
    for (int c = 0; c < kC; ++c) ones[c] = 1.0f;
    // the six per-channel vectors of a block (shift, 1 + scale, gate, for the attention and the MLP)
    float* vs[6];
    for (int j = 0; j < 6; ++j) vs[j] = S(kC);
    std::vector<float> ms(6 * kC);

    for (int b = 0; b < kBlocks; ++b) {
        const Impl::Block& k = m.blk[b];
        const bool dbg = (b == 0);  // block 0 taps for the stage check (emitted only when a stage callback is set)
        // modulation = the block's own vector + the timestep's, split in six (the python chunk order)

        for (int j = 0; j < 6 * kC; ++j) ms[j] = k.mod[j] + mod[j];
        for (int j = 0; j < kC; ++j) {
            vs[0][j] = ms[j];                          // shift_msa
            vs[1][j] = 1.0f + ms[kC + j];              // 1 + scale_msa
            vs[2][j] = ms[2 * kC + j];                 // gate_msa
            vs[3][j] = ms[3 * kC + j];                 // shift_mlp
            vs[4][j] = 1.0f + ms[4 * kC + j];          // 1 + scale_mlp
            vs[5][j] = ms[5 * kC + j];                 // gate_mlp
        }

        // self-attention: norm, modulate, qkv, q/k norm and RoPE, attention in query chunks, output, gated residual
        sddev::layer_norm_plain(h, kN, kC, 1e-6f, A);
        sddev::ss_affine(A, kN, kC, vs[1], vs[0]);
        if (dbg) emit("dbg_ln1", A, (size_t)kN * kC);
        sddev::linear_bf16(A, kN, k.wqkv, kC, 3 * kC, k.bqkv, QKV);
        sddev::ss_head_norm_rope(QKV, kN, 3 * kC, kH, kD, k.gq, ph);
        sddev::ss_head_norm_rope(QKV + kC, kN, 3 * kC, kH, kD, k.gk, ph);
        if (dbg) emit("dbg_qk", QKV, (size_t)kN * 3 * kC);
        const int qrows = attn_rows(kN, kN);
        for (int c0 = 0; c0 < kN; c0 += qrows)
            sddev::attention_f32_strided(QKV + (size_t)c0 * 3 * kC, 3 * kC, std::min(qrows, kN - c0), QKV + kC, 3 * kC,
                                         QKV + 2 * kC, 3 * kC, kN, kC, kH, O + (size_t)c0 * kC);
        if (dbg) emit("dbg_attn", O, (size_t)kN * kC);
        sddev::linear_bf16(O, kN, k.wso, kC, kC, k.bso, P);
        sddev::ss_gate_add(h, kN, kC, vs[2], P);
        if (dbg) emit("dbg_self", h, (size_t)kN * kC);

        // cross-attention to the image features (no RoPE; keys and values from the two halves of to_kv)
        sddev::layer_norm_plain(h, kN, kC, 1e-6f, Bq);
        sddev::ss_affine(Bq, kN, kC, k.n2w, k.n2b);  // the affine norm: x * w + b
        if (dbg) emit("dbg_cq_ln", Bq, (size_t)kN * kC);
        sddev::linear_bf16(Bq, kN, k.wcq, kC, kC, k.bcq, Qc);
        if (dbg) emit("dbg_cq_lin", Qc, (size_t)kN * kC);
        sddev::linear_bf16(cd, Lc, k.wkv, kCond, kC, k.bkv, Kc);
        sddev::linear_bf16(cd, Lc, k.wkv + (size_t)kC * kCond, kCond, kC, k.bkv + kC, Vc);
        sddev::ss_head_norm_rope(Qc, kN, kC, kH, kD, k.gcq, nullptr);
        sddev::ss_head_norm_rope(Kc, Lc, kC, kH, kD, k.gck, nullptr);
        if (dbg) {
            emit("dbg_cq", Qc, (size_t)kN * kC);
            emit("dbg_ck", Kc, (size_t)Lc * kC);
            emit("dbg_cv", Vc, (size_t)Lc * kC);
        }
        const int crows = attn_rows(kN, Lc);
        for (int c0 = 0; c0 < kN; c0 += crows)
            sddev::attention_f32_strided(Qc + (size_t)c0 * kC, kC, std::min(crows, kN - c0), Kc, kC, Vc, kC, Lc, kC, kH,
                                         Oc + (size_t)c0 * kC);
        if (dbg) emit("dbg_cross_attn", Oc, (size_t)kN * kC);
        sddev::linear_bf16(Oc, kN, k.wco, kC, kC, k.bco, Pc);
        sddev::ss_gate_add(h, kN, kC, ones, Pc);
        if (dbg) emit("dbg_cross", h, (size_t)kN * kC);

        // MLP: norm, modulate, GELU (tanh), gated residual
        sddev::layer_norm_plain(h, kN, kC, 1e-6f, A);
        sddev::ss_affine(A, kN, kC, vs[4], vs[3]);
        sddev::linear_bf16(A, kN, k.wm0, kC, kF, k.bm0, F1);
        sddev::gelu_tanh(F1, (size_t)kN * kF);
        sddev::linear_bf16(F1, kN, k.wm2, kF, kC, k.bm2, P3);
        sddev::ss_gate_add(h, kN, kC, vs[5], P3);

        if (b == 0 || b == 1 || b == 14 || b == 28 || b == 29) {
            char name[16];
            std::snprintf(name, sizeof(name), "block%02d", b);
            emit(name, h, (size_t)kN * kC);
        }
    }

    // the output: LayerNorm (no affine), the output layer to 8 channels, then [4096][8] to [8][4096]
    float* Z = S((size_t)kN * kC);
    sddev::layer_norm_plain(h, kN, kC, 1e-5f, Z);
    float* Y = S((size_t)kN * kIn);
    sddev::linear_bf16(Z, kN, m.w_out, kC, kIn, m.b_out, Y);
    emit("out_pre_layer", Y, (size_t)kN * kIn);

    for (int c = 0; c < kIn; ++c)
        for (int n = 0; n < kN; ++n) out[(size_t)c * kN + n] = Y[(size_t)n * kIn + c];
    return true;
}

}  // namespace trellis
