// The TRELLIS.2 sparse-structure decoder on the device (trellis_ss_dec.h). The stage order follows
// trellis2/models/sparse_structure_vae.py (SparseStructureDecoder): the input conv, two middle residual blocks, two blocks of
// residual blocks at 512 channels, a conv upsampling (512 -> 128, 16^3 -> 32^3), two residual blocks at 128, a conv
// upsampling (128 -> 32, 32^3 -> 64^3), two residual blocks at 32, then the output norm, SiLU and conv to one channel.
// Activations are token-major [V][C] with V = n^3 (the python layout [C][n][n][n] transposed).
#include "trellis_ss_dec.h"

#include "sd_device.h"
#include "zimage_dit.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace trellis {
namespace {

constexpr float kLnEps = 1e-5f;  // nn.LayerNorm's default, as the decoder's norm_layer builds it
constexpr int kChunkV = 256;     // voxels per fp16 GEMM launch
constexpr int kChunkO = 128;     // output channels per fp16 GEMM launch

// one residual block: the fp32 norms, the fp16 convolutions (in place) and their fp32 biases
struct DecRes {
    const float *n1w, *n1b, *n2w, *n2b;
    const uint16_t *c1w, *c2w;
    float *c1b, *c2b;
    int C;
};

float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal
            uint32_t m = man, e = 127 - 15 + 1;
            while (!(m & 0x400u)) {
                m <<= 1;
                --e;
            }
            bits = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // namespace

struct SsDecoder::Impl {
    explicit Impl(const std::string& path) : st(path) {}
    ~Impl() {
        for (auto& o : owned) sddev::release(o.first, o.second);
    }

    zimage::SafeTensors st;
    std::vector<std::pair<void*, size_t>> owned;  // fp32 bias vectors made at open

    using Res = DecRes;
    struct Up {
        const uint16_t* w;  // fp16 [8 * cout][cin][3][3][3]
        float* b;           // [8 * cout]
        int cin, cout;
    };
    const float *in_w, *in_b;  // input conv: fp32 [512][8][3][3][3], [512]
    const float *out_nw, *out_nb, *out_w, *out_b;  // output: norm (fp32), conv (fp32 [1][32][3][3][3]), [1]
    std::vector<Res> mid, blk;  // mid: 2 blocks at 512; blk: 512 x2, 128 x2, 32 x2
    Up up1, up2;                // 512 -> 128 and 128 -> 32

    bool f32(const std::string& name, const float*& p, size_t n) {
        std::vector<size_t> shape;
        if (!st.f32_in_place(name, p, shape)) return false;
        size_t total = 1;
        for (size_t s : shape) total *= s;
        return total == n;
    }
    bool f16(const std::string& name, const uint16_t*& p, size_t n) {
        std::vector<size_t> shape;
        if (!st.f16_in_place(name, p, shape)) return false;
        size_t total = 1;
        for (size_t s : shape) total *= s;
        return total == n;
    }
    // an fp16 bias as fp32, in device-visible memory (small: converted once)
    bool bias16(const std::string& name, float*& out, size_t n) {
        const uint16_t* p = nullptr;
        if (!f16(name, p, n)) return false;
        out = static_cast<float*>(sddev::alloc(sizeof(float) * n, false));
        owned.push_back({out, sizeof(float) * n});
        for (size_t i = 0; i < n; ++i) out[i] = fp16_to_f32(p[i]);
        return true;
    }
    bool res(const std::string& p, int C, Res& r) {
        r.C = C;
        return f32(p + "norm1.weight", r.n1w, C) && f32(p + "norm1.bias", r.n1b, C) &&
               f32(p + "norm2.weight", r.n2w, C) && f32(p + "norm2.bias", r.n2b, C) &&
               f16(p + "conv1.weight", r.c1w, (size_t)C * C * 27) && bias16(p + "conv1.bias", r.c1b, C) &&
               f16(p + "conv2.weight", r.c2w, (size_t)C * C * 27) && bias16(p + "conv2.bias", r.c2b, C);
    }
};

SsDecoder::SsDecoder(const std::string& safetensors_path) : p_(new Impl(safetensors_path)) {
    Impl& m = *p_;
    if (!m.st.ok()) {
        error_ = m.st.error();
        return;
    }
    if (!m.st.register_device()) {
        error_ = "the checkpoint could not be registered for the device";
        return;
    }
    bool good = m.f32("input_layer.weight", m.in_w, (size_t)512 * 8 * 27) && m.f32("input_layer.bias", m.in_b, 512);
    good = good && m.f32("out_layer.0.weight", m.out_nw, 32) && m.f32("out_layer.0.bias", m.out_nb, 32);
    good = good && m.f32("out_layer.2.weight", m.out_w, 32 * 27) && m.f32("out_layer.2.bias", m.out_b, 1);
    good = good && m.f16("blocks.2.conv.weight", m.up1.w, (size_t)1024 * 512 * 27) && m.bias16("blocks.2.conv.bias", m.up1.b, 1024);
    good = good && m.f16("blocks.5.conv.weight", m.up2.w, (size_t)256 * 128 * 27) && m.bias16("blocks.5.conv.bias", m.up2.b, 256);
    m.up1.cin = 512;
    m.up1.cout = 128;
    m.up2.cin = 128;
    m.up2.cout = 32;
    for (int i = 0; good && i < 2; ++i) {
        Impl::Res r{};
        good = m.res("middle_block." + std::to_string(i) + ".", 512, r);
        if (good) m.mid.push_back(r);
    }
    const int chans[8] = {512, 512, 128, 128, 128, 32, 32, 32};  // blocks 0,1 / 3,4 / 6,7 (the upsampling blocks are 2 and 5)
    const int idx[6] = {0, 1, 3, 4, 6, 7};
    for (int i = 0; good && i < 6; ++i) {
        Impl::Res r{};
        good = m.res("blocks." + std::to_string(idx[i]) + ".", chans[idx[i]], r);
        if (good) m.blk.push_back(r);
    }
    if (!good) {
        error_ = "the decoder checkpoint has a tensor missing or of another size";
        return;
    }
    ok_ = true;
}

SsDecoder::~SsDecoder() = default;

namespace {

// y[V][cout] = conv3x3x3(x[V][cin]) with fp16 weights; P is scratch [V][cin * 27]
void conv_f16(const float* x, int cin, int n, const uint16_t* w, const float* b, int cout, float* y, float* P) {
    const long V = (long)n * n * n, K = (long)cin * 27;
    sddev::ss_im2col3(x, cin, n, P);
    for (long v0 = 0; v0 < V; v0 += kChunkV)
        for (int o0 = 0; o0 < cout; o0 += kChunkO)
            sddev::ss_gemm_f16w(P, K, w, b, cout, V, y, (int)v0, (int)std::min<long>(kChunkV, V - v0), o0,
                                std::min(kChunkO, cout - o0));
}

// the same with fp32 weights ([cout][cin * 27], the linear's layout)
void conv_f32(const float* x, int cin, int n, const float* w, const float* b, int cout, float* y, float* P) {
    const int V = n * n * n, K = cin * 27;
    sddev::ss_im2col3(x, cin, n, P);
    sddev::linear_f32(P, V, w, K, cout, b, y);
}

// the residual block: h += conv2(silu(norm2(conv1(silu(norm1(h))))))  (the skip is the identity: the channels match)
void res_block(float* h, const DecRes& r, int n, float* A, float* B, float* P) {
    const int V = n * n * n, C = r.C;
    sddev::layer_norm_plain(h, V, C, kLnEps, A);
    sddev::ss_affine(A, V, C, r.n1w, r.n1b);
    sddev::silu(A, (size_t)V * C);
    conv_f16(A, C, n, r.c1w, r.c1b, C, B, P);
    sddev::layer_norm_plain(B, V, C, kLnEps, A);
    sddev::ss_affine(A, V, C, r.n2w, r.n2b);
    sddev::silu(A, (size_t)V * C);
    conv_f16(A, C, n, r.c2w, r.c2b, C, B, P);
    sddev::add(h, B, (size_t)V * C);
}

}  // namespace

bool SsDecoder::forward(const float* z, float* out, const std::function<void(const char*, const float*, size_t)>& stage) {
    if (!ok_) return false;
    if (!sddev::enabled()) {
        error_ = "no device";
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
    auto S = [&](size_t n) {
        void* p = sddev::alloc(sizeof(float) * n, false);
        bufs.push_back({p, sizeof(float) * n});
        return static_cast<float*>(p);
    };
    auto emit = [&](const char* name, const float* d, size_t n) {
        if (stage) stage(name, d, n);
    };

    const int n16 = 16, n32 = 32, n64 = 64;
    const size_t V16 = (size_t)n16 * n16 * n16, V32 = (size_t)n32 * n32 * n32, V64 = (size_t)n64 * n64 * n64;

    // the input: z [8][4096] (channel-major) to tokens [4096][8]; the input conv (fp32) to 512 channels
    float* zt = S(V16 * 8);
    for (int c = 0; c < 8; ++c)
        for (size_t v = 0; v < V16; ++v) zt[v * 8 + c] = z[(size_t)c * V16 + v];
    float* P = S(std::max({V16 * 512 * 27, V32 * 128 * 27, V64 * 32 * 27}));
    float* h = S(V16 * 512);
    conv_f32(zt, 8, n16, m.in_w, m.in_b, 512, h, P);
    emit("dec_in", h, V16 * 512);

    // the middle: two residual blocks at 512 channels, 16^3
    float* A = S(V16 * 512);
    float* B = S(V16 * 512);
    for (const auto& r : m.mid) res_block(h, r, n16, A, B, P);
    emit("dec_mid", h, V16 * 512);

    // blocks 0, 1: residual blocks at 512, 16^3; block 2: the upsampling 512 -> 128 (conv to 8 x 128, then the shuffle)
    res_block(h, m.blk[0], n16, A, B, P);
    res_block(h, m.blk[1], n16, A, B, P);
    float* Y = S(V16 * 8 * 128);
    conv_f16(h, 512, n16, m.up1.w, m.up1.b, 8 * 128, Y, P);
    float* h2 = S(V32 * 128);
    sddev::ss_pixel_shuffle3(Y, 128, n16, h2);
    emit("dec_b2", h2, V32 * 128);

    // blocks 3, 4: residual blocks at 128, 32^3; block 5: the upsampling 128 -> 32 (conv to 8 x 32, then the shuffle)
    float* A2 = S(V32 * 128);
    float* B2 = S(V32 * 128);
    res_block(h2, m.blk[2], n32, A2, B2, P);
    res_block(h2, m.blk[3], n32, A2, B2, P);
    float* Y2 = S(V32 * 8 * 32);
    conv_f16(h2, 128, n32, m.up2.w, m.up2.b, 8 * 32, Y2, P);
    float* h3 = S(V64 * 32);
    sddev::ss_pixel_shuffle3(Y2, 32, n32, h3);
    emit("dec_b5", h3, V64 * 32);

    // blocks 6, 7: residual blocks at 32, 64^3; the output: norm (affine), SiLU, conv (fp32) to one channel
    float* A3 = S(V64 * 32);
    float* B3 = S(V64 * 32);
    res_block(h3, m.blk[4], n64, A3, B3, P);
    res_block(h3, m.blk[5], n64, A3, B3, P);
    sddev::layer_norm_plain(h3, (int)V64, 32, kLnEps, A3);
    sddev::ss_affine(A3, (int)V64, 32, m.out_nw, m.out_nb);
    sddev::silu(A3, V64 * 32);
    float* o = S(V64);
    conv_f32(A3, 32, n64, m.out_w, m.out_b, 1, o, P);
    emit("dec_out", o, V64);
    for (size_t v = 0; v < V64; ++v) out[v] = o[v];
    return true;
}

}  // namespace trellis
