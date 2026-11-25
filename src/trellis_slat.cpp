// The TRELLIS.2 shape-SLAT flow on the device (trellis_slat.h). The stage order and arithmetic are those of
// trellis2/models/structured_latent_flow.py (SLatFlowModel) with the sparse transformer blocks over one sequence: the same
// blocks as the sparse-structure flow (trellis_ss.cpp), with the token count and the channels of the latent (32), and the
// RoPE of the voxel coordinates (SparseRotaryPositionEmbedder: three axes, 21 frequencies each, one identity pair).
#include "trellis_slat.h"

#include "sd_device.h"
#include "zimage_dit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace trellis {
namespace {

constexpr int kC = 1536;      // model channels
constexpr int kH = 12;        // heads
constexpr int kD = 128;       // head dim
constexpr int kF = 8192;      // MLP hidden (mlp_ratio 5.3334)
constexpr int kBlocks = 30;
constexpr int kCond = 1024;   // image feature channels
constexpr int kLat = 32;      // latent channels per voxel (in and out)
constexpr int kFreq = 256;    // timestep embedding size
constexpr int kQChunk = 256;  // queries per attention launch (AGENTS.md: keep each launch short)
constexpr int kAxisFreq = 21; // head_dim / 2 / 3 frequencies per axis (SparseRotaryPositionEmbedder)

float bf16_to_f32(uint16_t v) {
    const uint32_t u = (uint32_t)v << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

}  // namespace

struct SlatFlow::Impl {
    explicit Impl(const std::string& path) : st(path) {}
    ~Impl() {
        for (auto& o : owned) sddev::release(o.first, o.second);
    }

    zimage::SafeTensors st;
    std::vector<std::pair<void*, size_t>> owned;  // the fp32 vectors made at open, released at destruction

    int in = kLat;  // the input features per voxel
    const uint16_t *w_in = nullptr, *w_t0 = nullptr, *w_t2 = nullptr, *w_ada = nullptr, *w_out = nullptr;
    float *b_in = nullptr, *b_t0 = nullptr, *b_t2 = nullptr, *b_ada = nullptr, *b_out = nullptr;
    struct Block {
        const uint16_t *wqkv, *wso, *wcq, *wkv, *wco, *wm0, *wm2;  // wkv: K rows, then V rows
        float *mod, *bqkv, *bso, *gq, *gk, *bcq, *bkv, *gcq, *gck, *bco, *bm0, *bm2;
        float *n2w, *n2b;  // the cross-attention norm is affine in this checkpoint
    };
    std::vector<Block> blk;

    bool weight(const std::string& name, const uint16_t*& p, size_t n) {
        std::vector<size_t> shape;
        if (!st.bf16_in_place(name, p, shape)) return false;
        size_t total = 1;
        for (size_t s : shape) total *= s;
        return total == n;
    }
    bool vec(const std::string& name, float*& out, size_t n) {
        const uint16_t* p = nullptr;
        if (!weight(name, p, n)) return false;
        out = static_cast<float*>(sddev::alloc(sizeof(float) * n, false));
        owned.push_back({out, sizeof(float) * n});
        for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
        return true;
    }
};

SlatFlow::SlatFlow(const std::string& safetensors_path, int in_channels) : p_(new Impl(safetensors_path)) {
    Impl& m = *p_;
    m.in = in_channels;
    if (!m.st.ok()) {
        error_ = m.st.error();
        return;
    }
    if (!m.st.register_device()) {
        error_ = "the checkpoint could not be registered for the device";
        return;
    }
    const char* what = nullptr;
    auto need = [&](bool good, const std::string& name) {
        if (!good && !what) what = strdup(name.c_str());
        return good;
    };
    need(m.weight("input_layer.weight", m.w_in, (size_t)kC * m.in), "input_layer.weight");
    need(m.vec("input_layer.bias", m.b_in, kC), "input_layer.bias");
    need(m.weight("t_embedder.mlp.0.weight", m.w_t0, (size_t)kC * kFreq), "t_embedder.mlp.0.weight");
    need(m.vec("t_embedder.mlp.0.bias", m.b_t0, kC), "t_embedder.mlp.0.bias");
    need(m.weight("t_embedder.mlp.2.weight", m.w_t2, (size_t)kC * kC), "t_embedder.mlp.2.weight");
    need(m.vec("t_embedder.mlp.2.bias", m.b_t2, kC), "t_embedder.mlp.2.bias");
    need(m.weight("adaLN_modulation.1.weight", m.w_ada, (size_t)6 * kC * kC), "adaLN_modulation.1.weight");
    need(m.vec("adaLN_modulation.1.bias", m.b_ada, 6 * kC), "adaLN_modulation.1.bias");
    need(m.weight("out_layer.weight", m.w_out, (size_t)kLat * kC), "out_layer.weight");
    need(m.vec("out_layer.bias", m.b_out, kLat), "out_layer.bias");
    m.blk.resize(kBlocks);
    for (int b = 0; b < kBlocks; ++b) {
        const std::string p = "blocks." + std::to_string(b) + ".";
        Impl::Block& k = m.blk[b];
        need(m.vec(p + "modulation", k.mod, 6 * kC), p + "modulation");
        need(m.weight(p + "self_attn.to_qkv.weight", k.wqkv, (size_t)3 * kC * kC), p + "self_attn.to_qkv.weight");
        need(m.vec(p + "self_attn.to_qkv.bias", k.bqkv, 3 * kC), p + "self_attn.to_qkv.bias");
        need(m.vec(p + "self_attn.q_rms_norm.gamma", k.gq, (size_t)kH * kD), p + "self_attn.q_rms_norm.gamma");
        need(m.vec(p + "self_attn.k_rms_norm.gamma", k.gk, (size_t)kH * kD), p + "self_attn.k_rms_norm.gamma");
        need(m.weight(p + "self_attn.to_out.weight", k.wso, (size_t)kC * kC), p + "self_attn.to_out.weight");
        need(m.vec(p + "self_attn.to_out.bias", k.bso, kC), p + "self_attn.to_out.bias");
        need(m.weight(p + "cross_attn.to_q.weight", k.wcq, (size_t)kC * kC), p + "cross_attn.to_q.weight");
        need(m.vec(p + "cross_attn.to_q.bias", k.bcq, kC), p + "cross_attn.to_q.bias");
        need(m.weight(p + "cross_attn.to_kv.weight", k.wkv, (size_t)2 * kC * kCond), p + "cross_attn.to_kv.weight");
        need(m.vec(p + "cross_attn.to_kv.bias", k.bkv, 2 * kC), p + "cross_attn.to_kv.bias");
        need(m.vec(p + "cross_attn.q_rms_norm.gamma", k.gcq, (size_t)kH * kD), p + "cross_attn.q_rms_norm.gamma");
        need(m.vec(p + "cross_attn.k_rms_norm.gamma", k.gck, (size_t)kH * kD), p + "cross_attn.k_rms_norm.gamma");
        need(m.weight(p + "cross_attn.to_out.weight", k.wco, (size_t)kC * kC), p + "cross_attn.to_out.weight");
        need(m.vec(p + "cross_attn.to_out.bias", k.bco, kC), p + "cross_attn.to_out.bias");
        need(m.vec(p + "norm2.weight", k.n2w, kC), p + "norm2.weight");
        need(m.vec(p + "norm2.bias", k.n2b, kC), p + "norm2.bias");
        need(m.weight(p + "mlp.mlp.0.weight", k.wm0, (size_t)kF * kC), p + "mlp.mlp.0.weight");
        need(m.vec(p + "mlp.mlp.0.bias", k.bm0, kF), p + "mlp.mlp.0.bias");
        need(m.weight(p + "mlp.mlp.2.weight", k.wm2, (size_t)kC * kF), p + "mlp.mlp.2.weight");
        need(m.vec(p + "mlp.mlp.2.bias", k.bm2, kC), p + "mlp.mlp.2.bias");
        if (what) break;
    }
    if (what) {
        error_ = std::string("the checkpoint has no tensor ") + what + " of the expected size";
        return;
    }
    ok_ = true;
}

SlatFlow::~SlatFlow() = default;

bool SlatFlow::forward(const int32_t* coords, int n, const float* feats, float t, const float* cond, int Lc, float* out,
                       const std::function<void(const char*, const float*, size_t)>& stage) {
    if (!ok_) return false;
    if (!sddev::enabled()) {
        error_ = "no device";
        return false;
    }
    if (n <= 0) {
        error_ = "no voxels";
        return false;
    }
    Impl& m = *p_;

    std::vector<std::pair<void*, size_t>> bufs;
    struct Guard {
        std::vector<std::pair<void*, size_t>>& b;
        ~Guard() {
            for (auto& o : b) sddev::release(o.first, o.second);
        }
    } guard{bufs};
    auto S = [&](size_t cnt) {
        void* p = sddev::alloc(sizeof(float) * cnt, false);
        bufs.push_back({p, sizeof(float) * cnt});
        return static_cast<float*>(p);
    };
    auto emit = [&](const char* name, const float* d, size_t cnt) {
        if (stage) stage(name, d, cnt);
    };
    const size_t N = (size_t)n;

    // the timestep embedding: the sinusoid on the host, then the linears (as in the sparse-structure flow)
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
    emit("slat_temb", temb, kC);
    float* sa = S(kC);
    sddev::audio_copy(sa, temb, kC);
    sddev::silu(sa, kC);
    float* mod = S((size_t)6 * kC);
    sddev::linear_bf16(sa, 1, m.w_ada, kC, 6 * kC, m.b_ada, mod);
    emit("slat_mod", mod, (size_t)6 * kC);

    // the tokens: the latent per voxel, then the input layer (32 -> 1536)
    float* tok = S(N * m.in);
    std::memcpy(tok, feats, sizeof(float) * N * m.in);
    float* h = S(N * kC);
    sddev::linear_bf16(tok, n, m.w_in, m.in, kC, m.b_in, h);
    emit("slat_h0", h, N * kC);

    // the RoPE table of the voxel coordinates (host, small): 63 pairs over the three axes (21 frequencies each) and one identity
    std::vector<float> freqs(kAxisFreq);
    for (int f = 0; f < kAxisFreq; ++f) freqs[f] = (float)(1.0 / std::pow(10000.0, (double)f / kAxisFreq));
    float* ph = S(N * (kD / 2) * 2);
    for (size_t v = 0; v < N; ++v) {
        const double c[3] = {(double)coords[v * 4 + 1], (double)coords[v * 4 + 2], (double)coords[v * 4 + 3]};
        float* row = ph + v * (kD / 2) * 2;
        for (int j = 0; j < 3 * kAxisFreq; ++j) {
            const int a = j / kAxisFreq, f = j % kAxisFreq;
            const double ang = c[a] * (double)freqs[f];
            row[2 * j] = (float)std::cos(ang);
            row[2 * j + 1] = (float)std::sin(ang);
        }
        row[2 * (kD / 2 - 1)] = 1.0f;  // the padding pair: the identity
        row[2 * (kD / 2 - 1) + 1] = 0.0f;
    }
    float* cd = S((size_t)Lc * kCond);
    std::memcpy(cd, cond, sizeof(float) * (size_t)Lc * kCond);

    // block buffers
    float* A = S(N * kC);
    float* QKV = S(N * 3 * kC);
    float* O = S(N * kC);
    float* P = S(N * kC);
    float* Bq = S(N * kC);
    float* Qc = S(N * kC);
    float* Kc = S((size_t)Lc * kC);
    float* Vc = S((size_t)Lc * kC);
    float* Oc = S(N * kC);
    float* Pc = S(N * kC);
    float* F1 = S(N * kF);
    float* P3 = S(N * kC);
    float* ones = S(kC);
    for (int c = 0; c < kC; ++c) ones[c] = 1.0f;
    float* vs[6];
    for (int j = 0; j < 6; ++j) vs[j] = S(kC);
    std::vector<float> ms(6 * kC);

    const auto fwd_t0 = std::chrono::steady_clock::now();
    for (int b = 0; b < kBlocks; ++b) {
        const Impl::Block& k = m.blk[b];
        const auto blk_t0 = std::chrono::steady_clock::now();
        double att_s = 0;
        for (int j = 0; j < 6 * kC; ++j) ms[j] = k.mod[j] + mod[j];
        for (int j = 0; j < kC; ++j) {
            vs[0][j] = ms[j];
            vs[1][j] = 1.0f + ms[kC + j];
            vs[2][j] = ms[2 * kC + j];
            vs[3][j] = ms[3 * kC + j];
            vs[4][j] = 1.0f + ms[4 * kC + j];
            vs[5][j] = ms[5 * kC + j];
        }
        // self-attention over the voxels
        sddev::layer_norm_plain(h, n, kC, 1e-6f, A);
        sddev::ss_affine(A, n, kC, vs[1], vs[0]);
        sddev::linear_bf16(A, n, k.wqkv, kC, 3 * kC, k.bqkv, QKV);
        sddev::ss_head_norm_rope(QKV, n, 3 * kC, kH, kD, k.gq, ph);
        sddev::ss_head_norm_rope(QKV + kC, n, 3 * kC, kH, kD, k.gk, ph);
        const auto a0 = std::chrono::steady_clock::now();
        for (int c0 = 0; c0 < n; c0 += kQChunk) {
            const int qn = std::min(kQChunk, n - c0);
            sddev::attention_f32_strided(QKV + (size_t)c0 * 3 * kC, 3 * kC, qn, QKV + kC, 3 * kC, QKV + 2 * kC, 3 * kC, n, kC, kH,
                                         O + (size_t)c0 * kC);
        }
        att_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();
        sddev::linear_bf16(O, n, k.wso, kC, kC, k.bso, P);
        sddev::ss_gate_add(h, n, kC, vs[2], P);
        // cross-attention to the image features
        sddev::layer_norm_plain(h, n, kC, 1e-6f, Bq);
        sddev::ss_affine(Bq, n, kC, k.n2w, k.n2b);
        sddev::linear_bf16(Bq, n, k.wcq, kC, kC, k.bcq, Qc);
        sddev::linear_bf16(cd, Lc, k.wkv, kCond, kC, k.bkv, Kc);
        sddev::linear_bf16(cd, Lc, k.wkv + (size_t)kC * kCond, kCond, kC, k.bkv + kC, Vc);
        sddev::ss_head_norm_rope(Qc, n, kC, kH, kD, k.gcq, nullptr);
        sddev::ss_head_norm_rope(Kc, Lc, kC, kH, kD, k.gck, nullptr);
        const auto a1 = std::chrono::steady_clock::now();
        for (int c0 = 0; c0 < n; c0 += kQChunk) {
            const int qn = std::min(kQChunk, n - c0);
            sddev::attention_f32_strided(Qc + (size_t)c0 * kC, kC, qn, Kc, kC, Vc, kC, Lc, kC, kH, Oc + (size_t)c0 * kC);
        }
        att_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a1).count();
        sddev::linear_bf16(Oc, n, k.wco, kC, kC, k.bco, Pc);
        sddev::ss_gate_add(h, n, kC, ones, Pc);
        // MLP
        sddev::layer_norm_plain(h, n, kC, 1e-6f, A);
        sddev::ss_affine(A, n, kC, vs[4], vs[3]);
        sddev::linear_bf16(A, n, k.wm0, kC, kF, k.bm0, F1);
        sddev::gelu_tanh(F1, N * kF);
        sddev::linear_bf16(F1, n, k.wm2, kF, kC, k.bm2, P3);
        sddev::ss_gate_add(h, n, kC, vs[5], P3);
        if (progress_) {
            std::fprintf(stderr, "      block %2d/%d: %.1f s (attention %.1f s), %.1f s into the flow\n", b + 1, kBlocks,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - blk_t0).count(), att_s,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - fwd_t0).count());
        }
        if (b == 0 || b == 1 || b == 14 || b == 28 || b == 29) {
            char name[24];
            std::snprintf(name, sizeof(name), "slat_block%02d", b);
            emit(name, h, N * kC);
        }
    }

    // the output: LayerNorm (no affine), the output layer to 32 channels (token-major, as the input)
    float* Z = S(N * kC);
    sddev::layer_norm_plain(h, n, kC, 1e-5f, Z);
    float* Y = S(N * kLat);
    sddev::linear_bf16(Z, n, m.w_out, kC, kLat, m.b_out, Y);
    emit("slat_out", Y, N * kLat);
    std::memcpy(out, Y, sizeof(float) * N * kLat);
    return true;
}

}  // namespace trellis
