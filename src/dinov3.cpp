// The DINOv3 ViT-L/16 image encoder on the device (dinov3.h). The order of the operations is transformers' DINOv3ViTModel (embeddings,
// rotary embedding, the layers) with TRELLIS's output (a layer norm without affine). The rotary angles follow get_patches_center_coordinates
// (the patch centres at (i + 0.5) / n mapped to [-1, 1]) and inv_freq = 1 / rope_theta ** (k / 16); each patch's angles are the y angles
// then the x angles, tiled over the head dimension (rotate_half on the two halves of the 64 dims).
#include "dinov3.h"

#include "sd_device.h"
#include "zimage_dit.h"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace trellis {
namespace {

constexpr int kC = 1024;   // hidden size
constexpr int kH = 16;     // heads
constexpr int kD = 64;     // head dim
constexpr int kFF = 4096;  // MLP width
constexpr int kLayers = 24;
constexpr int kPrefix = 5; // CLS and 4 registers
constexpr float kEps = 1e-5f;

}  // namespace

struct DinoV3::Impl {
    explicit Impl(const std::string& path) : st(path) {}
    zimage::SafeTensors st;

    struct Block {
        const float *n1w, *n1b, *wq, *bq, *wk, *wv, *bv, *wo, *bo, *ls1, *n2w, *n2b, *wup, *bup, *wdown, *bdown, *ls2;
    };
    std::vector<Block> blocks;
    const float *cls = nullptr, *reg = nullptr, *pe_w = nullptr, *pe_b = nullptr;
    std::vector<double> periods;  // rope_theta ** (k / 16), 16 of them

    bool get(const std::string& name, const float*& p, size_t count) {
        std::vector<size_t> shape;
        if (!st.f32_in_place(name, p, shape)) return false;
        size_t n = 1;
        for (size_t s : shape) n *= s;
        return n == count;
    }
};

DinoV3::DinoV3(const std::string& safetensors_path) : p_(new Impl(safetensors_path)) {
    Impl& m = *p_;
    if (!m.st.ok()) {
        error_ = m.st.error();
        return;
    }
    if (!m.st.register_device()) {
        error_ = "the image encoder checkpoint could not be registered for the device";
        return;
    }
    bool good = m.get("embeddings.cls_token", m.cls, kC) && m.get("embeddings.register_tokens", m.reg, 4 * kC) &&
                m.get("embeddings.patch_embeddings.weight", m.pe_w, (size_t)kC * 3 * 16 * 16) &&
                m.get("embeddings.patch_embeddings.bias", m.pe_b, kC);
    m.blocks.resize(kLayers);
    for (int l = 0; good && l < kLayers; ++l) {
        const std::string p = "layer." + std::to_string(l) + ".";
        Impl::Block& b = m.blocks[l];
        good = m.get(p + "norm1.weight", b.n1w, kC) && m.get(p + "norm1.bias", b.n1b, kC) &&
               m.get(p + "attention.q_proj.weight", b.wq, (size_t)kC * kC) && m.get(p + "attention.q_proj.bias", b.bq, kC) &&
               m.get(p + "attention.k_proj.weight", b.wk, (size_t)kC * kC) &&
               m.get(p + "attention.v_proj.weight", b.wv, (size_t)kC * kC) && m.get(p + "attention.v_proj.bias", b.bv, kC) &&
               m.get(p + "attention.o_proj.weight", b.wo, (size_t)kC * kC) && m.get(p + "attention.o_proj.bias", b.bo, kC) &&
               m.get(p + "layer_scale1.lambda1", b.ls1, kC) && m.get(p + "norm2.weight", b.n2w, kC) &&
               m.get(p + "norm2.bias", b.n2b, kC) && m.get(p + "mlp.up_proj.weight", b.wup, (size_t)kFF * kC) &&
               m.get(p + "mlp.up_proj.bias", b.bup, kFF) && m.get(p + "mlp.down_proj.weight", b.wdown, (size_t)kC * kFF) &&
               m.get(p + "mlp.down_proj.bias", b.bdown, kC) && m.get(p + "layer_scale2.lambda1", b.ls2, kC);
    }
    if (!good) {
        error_ = "the image encoder checkpoint has a tensor missing or of another size";
        return;
    }
    // the rotary periods (the file's copy of rope_theta ** (k / 16), 16 values): the angles use their inverses
    std::vector<size_t> shape;
    const float* per = nullptr;
    if (!m.st.f32_in_place("naf.image_encoder.rope.periods", per, shape) || shape.size() != 1 || shape[0] != 16) {
        error_ = "the image encoder checkpoint has no rope periods (naf.image_encoder.rope.periods)";
        return;
    }
    m.periods.resize(16);
    for (int k = 0; k < 16; ++k) m.periods[k] = per[k];
    ok_ = true;
}

DinoV3::~DinoV3() = default;

bool DinoV3::forward(const float* x, int S, float* out) {
    if (!ok_) return false;
    if (S <= 0 || S % 16 != 0) {
        error_ = "the image size must be a positive multiple of 16";
        return false;
    }
    if (!sddev::enabled()) {
        error_ = "no device";
        return false;
    }
    Impl& m = *p_;
    const int g = S / 16, P = g * g, T = kPrefix + P;
    std::vector<std::pair<void*, size_t>> bufs;
    struct Guard {
        std::vector<std::pair<void*, size_t>>& b;
        ~Guard() {
            for (auto& o : b) sddev::release(o.first, o.second);
        }
    } guard{bufs};
    auto A = [&](size_t count, bool zero) {  // count floats
        const size_t bytes = sizeof(float) * count;
        void* p = sddev::alloc(bytes, zero);
        bufs.push_back({p, bytes});
        return static_cast<float*>(p);
    };

    // 1. the patches as rows of 3 * 16 * 16 values (channel, row, column: the convolution's order), then the patch embedding
    float* rows = A((size_t)P * 768, false);
    for (int py = 0; py < g; ++py)
        for (int px = 0; px < g; ++px) {
            float* r = rows + (size_t)(py * g + px) * 768;
            for (int c = 0; c < 3; ++c)
                for (int kh = 0; kh < 16; ++kh)
                    for (int kw = 0; kw < 16; ++kw) r[c * 256 + kh * 16 + kw] = x[((size_t)c * S + py * 16 + kh) * S + px * 16 + kw];
        }
    float* pe = A((size_t)P * kC, false);
    sddev::linear_f32(rows, P, m.pe_w, 768, kC, m.pe_b, pe);

    // 2. the tokens: CLS, the registers, the patches
    float* X = A((size_t)T * kC, false);
    std::memcpy(X, m.cls, sizeof(float) * kC);
    std::memcpy(X + kC, m.reg, sizeof(float) * 4 * kC);
    std::memcpy(X + (size_t)kPrefix * kC, pe, sizeof(float) * (size_t)P * kC);

    // 3. the rotary cosines and sines of the patches: the 16 y angles then the 16 x angles, tiled twice over the 64 dims
    float* cs = A((size_t)P * kD, false);
    float* sn = A((size_t)P * kD, false);
    for (int p = 0; p < P; ++p) {
        const double yc = ((p / g) + 0.5) / g * 2.0 - 1.0, xc = ((p % g) + 0.5) / g * 2.0 - 1.0;
        double a[32];
        for (int f = 0; f < 16; ++f) {
            const double inv = 1.0 / m.periods[f];
            a[f] = 2.0 * M_PI * yc * inv;
            a[16 + f] = 2.0 * M_PI * xc * inv;
        }
        for (int d = 0; d < kD; ++d) {
            cs[(size_t)p * kD + d] = (float)std::cos(a[d % 32]);
            sn[(size_t)p * kD + d] = (float)std::sin(a[d % 32]);
        }
    }

    // 4. the blocks
    float* zeros = A(kC, true);  // the shift of the layer scales (none)
    for (int l = 0; l < kLayers; ++l) {
        const Impl::Block& b = m.blocks[l];
        float* n1 = A((size_t)T * kC, false);
        sddev::audio_layer_norm_f32(X, T, kC, b.n1w, b.n1b, kEps, n1);
        float* q = A((size_t)T * kC, false);
        float* k = A((size_t)T * kC, false);
        float* v = A((size_t)T * kC, false);
        sddev::linear_f32(n1, T, b.wq, kC, kC, b.bq, q);
        sddev::linear_f32(n1, T, b.wk, kC, kC, nullptr, k);  // the key projection has no bias
        sddev::linear_f32(n1, T, b.wv, kC, kC, b.bv, v);
        sddev::dino_rope(q, T, kPrefix, kC, kH, kD, cs, sn);
        sddev::dino_rope(k, T, kPrefix, kC, kH, kD, cs, sn);
        float* att = A((size_t)T * kC, false);
        sddev::attention_f32_strided(q, kC, T, k, kC, v, kC, T, kC, kH, att);
        float* o = A((size_t)T * kC, false);
        sddev::linear_f32(att, T, b.wo, kC, kC, b.bo, o);
        sddev::ss_affine(o, T, kC, b.ls1, zeros);  // the layer scale
        sddev::add(X, o, (size_t)T * kC);          // the residual
        float* n2 = A((size_t)T * kC, false);
        sddev::audio_layer_norm_f32(X, T, kC, b.n2w, b.n2b, kEps, n2);
        float* u = A((size_t)T * kFF, false);
        sddev::linear_f32(n2, T, b.wup, kC, kFF, b.bup, u);
        sddev::audio_gelu_erf(u, (size_t)T * kFF);
        float* dn = A((size_t)T * kC, false);
        sddev::linear_f32(u, T, b.wdown, kFF, kC, b.bdown, dn);
        sddev::ss_affine(dn, T, kC, b.ls2, zeros);
        sddev::add(X, dn, (size_t)T * kC);
    }

    // 5. the output: a layer norm without affine (into a device-visible buffer: the caller's vector is not mapped for the GPU)
    float* Y = A((size_t)T * kC, false);
    sddev::layer_norm_plain(X, T, kC, kEps, Y);
    std::memcpy(out, Y, sizeof(float) * (size_t)T * kC);
    return true;
}

}  // namespace trellis
