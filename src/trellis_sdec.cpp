// The TRELLIS.2 shape decoder's sparse UNet on the device (trellis_sdec.h). The order is that of
// trellis2/models/sc_vaes/sparse_unet_vae.py (SparseUnetVaeDecoder.forward): from_latent; for each level, its ConvNeXt
// blocks and, except at the last level, the channel-to-spatial block (which predicts the subdivision); then the layer norm
// (no affine) and the output layer. The sparse convolutions are the engine's (sd_gpu.hip), with the neighbour map of each
// voxel set built once.
#include "trellis_sdec.h"

#include "sd_device.h"
#include "zimage_dit.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace trellis {
namespace {

const int kChans[5] = {1024, 512, 256, 128, 64};
const int kBlocks[5] = {4, 16, 8, 4, 0};
constexpr int kLatent = 32;
constexpr int kOut = 7;
constexpr int kOChunk = 32;  // output channels per sparse-conv launch
constexpr int kVChunk = 256; // voxels per dense-linear launch
constexpr int kLChunk = 128; // output channels per dense-linear launch

float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
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

struct ShapeDecoder::Impl {
    explicit Impl(const std::string& path) : st(path) {}
    ~Impl() {
        for (auto& o : owned) sddev::release(o.first, o.second);
    }

    zimage::SafeTensors st;
    std::vector<std::pair<void*, size_t>> owned;

    // fp16 weights, in place
    struct ConvNeXt {
        const uint16_t *cw, *m0w, *m2w;
        float *cb, *nw, *nb, *m0b, *m2b;
    };
    struct C2S {
        const uint16_t *c1w, *sw, *c2w;
        float *c1b, *sb, *c2b, *n1w, *n1b;
        int in, out;
    };
    std::vector<std::vector<ConvNeXt>> levels;  // [5][num_blocks]
    std::vector<C2S> up;                        // [4]
    const uint16_t *lw = nullptr, *ow = nullptr;
    float *lb = nullptr, *ob = nullptr;

    bool f16(const std::string& name, const uint16_t*& p, size_t n) {
        std::vector<size_t> shape;
        if (!st.f16_in_place(name, p, shape)) return false;
        size_t total = 1;
        for (size_t s : shape) total *= s;
        return total == n;
    }
    // a small tensor as fp32 (the checkpoint's fp16 or fp32 vectors), device-visible, made once
    bool vec(const std::string& name, float*& out, size_t n) {
        std::vector<size_t> shape;
        const uint16_t* p16 = nullptr;
        const float* p32 = nullptr;
        size_t total = 0;
        if (st.f16_in_place(name, p16, shape)) {
            total = 1;
            for (size_t s : shape) total *= s;
            if (total != n) return false;
            out = static_cast<float*>(sddev::alloc(sizeof(float) * n, false));
            owned.push_back({out, sizeof(float) * n});
            for (size_t i = 0; i < n; ++i) out[i] = fp16_to_f32(p16[i]);
            return true;
        }
        if (st.f32_in_place(name, p32, shape)) {
            total = 1;
            for (size_t s : shape) total *= s;
            if (total != n) return false;
            out = const_cast<float*>(p32);  // read in place
            return true;
        }
        return false;
    }
};

ShapeDecoder::ShapeDecoder(const std::string& safetensors_path) : p_(new Impl(safetensors_path)) {
    Impl& m = *p_;
    if (!m.st.ok()) {
        error_ = m.st.error();
        return;
    }
    if (!m.st.register_device()) {
        error_ = "the checkpoint could not be registered for the device";
        return;
    }
    bool good = m.f16("from_latent.weight", m.lw, (size_t)kChans[0] * kLatent) && m.vec("from_latent.bias", m.lb, kChans[0]);
    good = good && m.f16("output_layer.weight", m.ow, (size_t)kOut * kChans[4]) && m.vec("output_layer.bias", m.ob, kOut);
    m.levels.resize(5);
    m.up.resize(4);
    for (int i = 0; good && i < 5; ++i) {
        const int C = kChans[i];
        for (int j = 0; good && j < kBlocks[i]; ++j) {
            const std::string p = "blocks." + std::to_string(i) + "." + std::to_string(j) + ".";
            Impl::ConvNeXt cn{};
            good = m.f16(p + "conv.weight", cn.cw, (size_t)C * 27 * C) && m.vec(p + "conv.bias", cn.cb, C) &&
                   m.vec(p + "norm.weight", cn.nw, C) && m.vec(p + "norm.bias", cn.nb, C) &&
                   m.f16(p + "mlp.0.weight", cn.m0w, (size_t)4 * C * C) && m.vec(p + "mlp.0.bias", cn.m0b, 4 * C) &&
                   m.f16(p + "mlp.2.weight", cn.m2w, (size_t)C * 4 * C) && m.vec(p + "mlp.2.bias", cn.m2b, C);
            if (good) m.levels[i].push_back(cn);
        }
        if (good && i < 4) {  // the channel-to-spatial block after the level's ConvNeXt blocks
            const int O = kChans[i + 1];
            const std::string p = "blocks." + std::to_string(i) + "." + std::to_string(kBlocks[i]) + ".";
            Impl::C2S& u = m.up[i];
            u.in = C;
            u.out = O;
            good = m.f16(p + "conv1.weight", u.c1w, (size_t)8 * O * 27 * C) && m.vec(p + "conv1.bias", u.c1b, 8 * O) &&
                   m.vec(p + "norm1.weight", u.n1w, C) && m.vec(p + "norm1.bias", u.n1b, C) &&
                   m.f16(p + "to_subdiv.weight", u.sw, (size_t)8 * C) && m.vec(p + "to_subdiv.bias", u.sb, 8) &&
                   m.f16(p + "conv2.weight", u.c2w, (size_t)O * 27 * O) && m.vec(p + "conv2.bias", u.c2b, O);
        }
    }
    if (!good) {
        error_ = "the shape decoder checkpoint has a tensor missing or of another size";
        return;
    }
    ok_ = true;
}

ShapeDecoder::~ShapeDecoder() = default;

namespace {

// a voxel set's neighbour map, built on the device: the table (keys, values) and nb[n][27]
struct VoxelMap {
    int n = 0;
    unsigned int tsize = 0;
    unsigned long long* keys = nullptr;
    int* vals = nullptr;
    int* nb = nullptr;
};

}  // namespace

bool ShapeDecoder::forward(const int32_t* coords, int n, const float* feats, float* out_feats, int32_t* out_coords, int* out_m,
                           int out_capacity, const StageFn& stage) {
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
    auto A = [&](size_t bytes) {
        void* p = sddev::alloc(bytes, false);
        bufs.push_back({p, bytes});
        return p;
    };
    auto F = [&](size_t cnt) { return static_cast<float*>(A(sizeof(float) * cnt)); };
    auto I = [&](size_t cnt) { return static_cast<int*>(A(sizeof(int) * cnt)); };

    // a dense linear over token rows: y[v][o] = b[o] + sum_k x[v][k] W[o][k]  (fp16 W in place)
    auto linear16 = [&](const float* x, int V, int K, const uint16_t* W, const float* b, int O, float* y) {
        for (int v0 = 0; v0 < V; v0 += kVChunk)
            for (int o0 = 0; o0 < O; o0 += kLChunk)
                sddev::ss_gemm_f16w(x, K, W, b, O, V, y, v0, std::min(kVChunk, V - v0), o0, std::min(kLChunk, O - o0));
    };
    // the neighbour map of a voxel set
    auto build_map = [&](const int* vc, int nv) {
        VoxelMap mp;
        mp.n = nv;
        mp.tsize = 1;
        while (mp.tsize < 2u * (unsigned)nv) mp.tsize <<= 1;
        mp.keys = static_cast<unsigned long long*>(A(sizeof(unsigned long long) * mp.tsize));
        mp.vals = I(mp.tsize);
        mp.nb = I((size_t)nv * 27);
        sddev::sp_hash_build(vc, nv, mp.keys, mp.vals, mp.tsize);
        sddev::sp_neighbor_map(vc, nv, mp.keys, mp.vals, mp.tsize, mp.nb);
        return mp;
    };
    // the sparse 3x3x3 convolution of x [N][cin] on a voxel set: y [N][cout]
    auto conv = [&](const float* x, int cin, const VoxelMap& mp, const uint16_t* W, const float* b, int cout, float* y) {
        for (int o0 = 0; o0 < cout; o0 += kOChunk)
            sddev::sp_subm_conv_f16w(x, mp.n, cin, mp.nb, W, b, cout, y, o0, std::min(kOChunk, cout - o0), 1);
    };
    // the layer norm with the affine gains (fp32), in place from x into y
    auto ln_affine = [&](const float* x, int V, int C, const float* w, const float* bb, float eps, float* y) {
        sddev::layer_norm_plain(x, V, C, eps, y);
        sddev::ss_affine(y, V, C, w, bb);
    };

    // the tokens: from_latent
    int* ccoords = I((size_t)n * 4);  // the current voxel set's coordinates
    std::memcpy(ccoords, coords, sizeof(int) * (size_t)n * 4);
    float* fin = F((size_t)n * kLatent);
    std::memcpy(fin, feats, sizeof(float) * (size_t)n * kLatent);
    int N = n;
    float* h = F((size_t)N * kChans[0]);
    linear16(fin, N, kLatent, m.lw, m.lb, kChans[0], h);
    int C = kChans[0];

    for (int i = 0; i < 5; ++i) {
        // the level's ConvNeXt blocks: h = mlp(LN(conv(h))) + h
        if (!m.levels[i].empty()) {
            VoxelMap mp = build_map(ccoords, N);
            for (const auto& cn : m.levels[i]) {
                float* t1 = F((size_t)N * C);
                float* t2 = F((size_t)N * 4 * C);
                float* t3 = F((size_t)N * C);
                float* t4 = F((size_t)N * C);
                conv(h, C, mp, cn.cw, cn.cb, C, t1);
                ln_affine(t1, N, C, cn.nw, cn.nb, 1e-6f, t4);
                linear16(t4, N, C, cn.m0w, cn.m0b, 4 * C, t2);
                sddev::silu(t2, (size_t)N * 4 * C);
                linear16(t2, N, 4 * C, cn.m2w, cn.m2b, C, t3);
                sddev::add(t3, h, (size_t)N * C);  // t3 += h
                h = t3;
            }
        }
        if (i == 4) break;  // the last level has no upsampling
        // the channel-to-spatial block (ResBlockC2S with the predicted subdivision)
        const Impl::C2S& u = m.up[i];
        const int O = u.out;
        float* logits = F((size_t)N * 8);
        linear16(h, N, C, u.sw, u.sb, 8, logits);
        // the residual branch: LN (affine), SiLU, conv1 (8 children per voxel), then the upsampling
        VoxelMap mp = build_map(ccoords, N);
        float* a1 = F((size_t)N * C);
        ln_affine(h, N, C, u.n1w, u.n1b, 1e-6f, a1);
        sddev::silu(a1, (size_t)N * C);
        float* c1 = F((size_t)N * 8 * O);
        conv(a1, C, mp, u.c1w, u.c1b, 8 * O, c1);
        // the subdivision: the children where the logit is positive, their voxels
        unsigned char* bits = static_cast<unsigned char*>(A((size_t)N));
        int* cnt = I((size_t)N);
        int* off = I((size_t)N + 1);
        sddev::sp_c2s_mask(logits, N, bits, cnt);
        sddev::sp_exclusive_scan(cnt, N, off);
        const int M = off[N];
        if (M <= 0) {
            error_ = "the subdivision produced no voxels";
            return false;
        }
        int* newc = I((size_t)M * 4);
        int* idx = I((size_t)M);
        int* subidx = I((size_t)M);
        sddev::sp_c2s_fill(ccoords, bits, off, N, newc, idx, subidx);
        float* hc = F((size_t)M * O);
        sddev::sp_c2s_gather(c1, idx, subidx, M, O, 1, hc);
        float* skip = F((size_t)M * O);
        sddev::sp_c2s_gather(h, idx, subidx, M, C / 8, O / (C / 8), skip);
        // the second branch: LN (no affine), SiLU, conv2 on the new voxels, plus the skip
        float* a2 = F((size_t)M * O);
        sddev::layer_norm_plain(hc, M, O, 1e-6f, a2);
        sddev::silu(a2, (size_t)M * O);
        VoxelMap mp2 = build_map(newc, M);
        float* c2 = F((size_t)M * O);
        conv(a2, O, mp2, u.c2w, u.c2b, O, c2);
        sddev::add(c2, skip, (size_t)M * O);
        h = c2;
        ccoords = newc;
        N = M;
        C = O;
        if (i == 0 && stage) stage("sdec_l0", h, ccoords, (size_t)N, (size_t)C);
    }

    // the output: LN (no affine), the output layer to 7 channels
    float* z = F((size_t)N * C);
    sddev::layer_norm_plain(h, N, C, 1e-5f, z);
    float* y = F((size_t)N * kOut);
    linear16(z, N, C, m.ow, m.ob, kOut, y);
    *out_m = N;  // the count, also when the buffer is too small (the caller retries with it)
    if (N > out_capacity) {
        error_ = "the output has more voxels than the buffer";
        return false;
    }
    std::memcpy(out_feats, y, sizeof(float) * (size_t)N * kOut);
    std::memcpy(out_coords, ccoords, sizeof(int) * (size_t)N * 4);
    *out_m = N;
    return true;
}

}  // namespace trellis
