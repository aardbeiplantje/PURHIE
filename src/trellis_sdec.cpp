// The TRELLIS.2 shape decoder's sparse UNet on the device (trellis_sdec.h). The order is that of
// trellis2/models/sc_vaes/sparse_unet_vae.py (SparseUnetVaeDecoder.forward): from_latent; for each level, its ConvNeXt
// blocks and, except at the last level, the channel-to-spatial block (which predicts the subdivision); then the layer norm
// (no affine) and the output layer. The sparse convolutions are the engine's (sd_gpu.hip), with the neighbour map of each
// voxel set built once.
#include "trellis_sdec.h"

#include "sd_device.h"
#include "zimage_dit.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace trellis {
namespace {

const int kChans[5] = {1024, 512, 256, 128, 64};
const int kBlocks[5] = {4, 16, 8, 4, 0};
constexpr int kLatent = 32;
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
        const uint16_t *c1w, *sw, *c2w;  // sw: the subdivision (absent in the guided decoder)
        float *c1b, *sb, *c2b, *n1w, *n1b;
        int in, out;
    };
    int out_ch = 7;      // the output channels (7: shape; 6: texture)
    bool guided = false;  // the texture decoder: the subdivisions come from the shape decoder
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

ShapeDecoder::ShapeDecoder(const std::string& safetensors_path, int out_channels, bool guided) : p_(new Impl(safetensors_path)) {
    Impl& m = *p_;
    m.out_ch = out_channels;
    m.guided = guided;
    if (!m.st.ok()) {
        error_ = m.st.error();
        return;
    }
    if (!m.st.register_device()) {
        error_ = "the checkpoint could not be registered for the device";
        return;
    }
    bool good = m.f16("from_latent.weight", m.lw, (size_t)kChans[0] * kLatent) && m.vec("from_latent.bias", m.lb, kChans[0]);
    good = good && m.f16("output_layer.weight", m.ow, (size_t)m.out_ch * kChans[4]) && m.vec("output_layer.bias", m.ob, m.out_ch);
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
                   (m.guided || (m.f16(p + "to_subdiv.weight", u.sw, (size_t)8 * C) && m.vec(p + "to_subdiv.bias", u.sb, 8))) &&
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

// a device buffer (sddev::alloc memory, mapped for the GPU and the host), released when it goes out of scope; movable
class DevBuf {
public:
    DevBuf() = default;
    explicit DevBuf(size_t bytes) : p_(sddev::alloc(bytes ? bytes : 1, false)), bytes_(bytes ? bytes : 1) {}
    DevBuf(DevBuf&& o) noexcept : p_(o.p_), bytes_(o.bytes_) {
        o.p_ = nullptr;
        o.bytes_ = 0;
    }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) {
            reset();
            p_ = o.p_;
            bytes_ = o.bytes_;
            o.p_ = nullptr;
            o.bytes_ = 0;
        }
        return *this;
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { reset(); }
    void reset() {
        if (p_) sddev::release(p_, bytes_);
        p_ = nullptr;
        bytes_ = 0;
    }
    float* f() const { return static_cast<float*>(p_); }
    int* i() const { return static_cast<int*>(p_); }
    unsigned char* u8() const { return static_cast<unsigned char*>(p_); }
    unsigned long long* u64() const { return static_cast<unsigned long long*>(p_); }

private:
    void* p_ = nullptr;
    size_t bytes_ = 0;
};

DevBuf floats(size_t n) { return DevBuf(sizeof(float) * n); }
DevBuf ints(size_t n) { return DevBuf(sizeof(int) * n); }

// a voxel set's neighbour map, built on the device: the table (keys, values) and nb[n][27]
struct VoxelMap {
    int n = 0;
    unsigned int tsize = 0;
    DevBuf keys, vals, nb;
};

}  // namespace

bool ShapeDecoder::forward(const int32_t* coords, int n, const float* feats, float* out_feats, int32_t* out_coords, int* out_m,
                           int out_capacity, const StageFn& stage, int upsample_times, std::vector<Subdivision>* subs_out) {
    return run(coords, n, feats, out_feats, out_coords, out_m, out_capacity, stage, upsample_times, nullptr, subs_out);
}

bool ShapeDecoder::forward_guided(const int32_t* coords, int n, const float* feats, const std::vector<Subdivision>& guide, float* out_feats,
                                  int32_t* out_coords, int* out_m, int out_capacity) {
    return run(coords, n, feats, out_feats, out_coords, out_m, out_capacity, nullptr, 0, &guide, nullptr);
}

bool ShapeDecoder::forward(const int32_t* coords, int n, const float* feats, std::vector<float>& out_feats, std::vector<int32_t>& out_coords,
                           const StageFn& stage, int upsample_times, std::vector<Subdivision>* subs_out) {
    out_feats.clear();
    out_coords.clear();
    return run(coords, n, feats, nullptr, nullptr, nullptr, 0, stage, upsample_times, nullptr, subs_out, &out_feats, &out_coords);
}

bool ShapeDecoder::forward_guided(const int32_t* coords, int n, const float* feats, const std::vector<Subdivision>& guide,
                                  std::vector<float>& out_feats, std::vector<int32_t>& out_coords) {
    out_feats.clear();
    out_coords.clear();
    return run(coords, n, feats, nullptr, nullptr, nullptr, 0, nullptr, 0, &guide, nullptr, &out_feats, &out_coords);
}

bool ShapeDecoder::run(const int32_t* coords, int n, const float* feats, float* out_feats, int32_t* out_coords, int* out_m,
                       int out_capacity, const StageFn& stage, int upsample_times, const std::vector<Subdivision>* guide,
                       std::vector<Subdivision>* subs_out, std::vector<float>* vfeats, std::vector<int32_t>* vcoords) {
    if (!ok_) return false;
    if (!sddev::enabled()) {
        error_ = "no device";
        return false;
    }
    if (n <= 0) {
        error_ = "no voxels";
        return false;
    }
    if (guide && guide->size() < 4) {
        error_ = "the guide has fewer than four levels";
        return false;
    }
    Impl& m = *p_;
    // Each buffer lives only while it is needed: a level's block temporaries are allocated once and reused by its blocks (the output
    // alternates between two buffers), and the maps and the subdivision's buffers go as soon as they are used. Only the features and the
    // coordinates of the current voxel set carry over from a level to the next (holding everything until the end needed about 75 GB
    // for the 1024 decode).

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
        mp.keys = DevBuf(sizeof(unsigned long long) * mp.tsize);
        mp.vals = ints(mp.tsize);
        mp.nb = ints((size_t)nv * 27);
        sddev::sp_hash_build(vc, nv, mp.keys.u64(), mp.vals.i(), mp.tsize);
        sddev::sp_neighbor_map(vc, nv, mp.keys.u64(), mp.vals.i(), mp.tsize, mp.nb.i());
        return mp;
    };
    // the sparse 3x3x3 convolution of x [N][cin] on a voxel set: y [N][cout]
    auto conv = [&](const float* x, int cin, const VoxelMap& mp, const uint16_t* W, const float* b, int cout, float* y) {
        for (int o0 = 0; o0 < cout; o0 += kOChunk)
            sddev::sp_subm_conv_f16w(x, mp.n, cin, mp.nb.i(), W, b, cout, y, o0, std::min(kOChunk, cout - o0), 1);
    };
    // the layer norm with the affine gains (fp32), in place from x into y
    auto ln_affine = [&](const float* x, int V, int C, const float* w, const float* bb, float eps, float* y) {
        sddev::layer_norm_plain(x, V, C, eps, y);
        sddev::ss_affine(y, V, C, w, bb);
    };

    // the tokens: from_latent
    DevBuf cc = ints((size_t)n * 4);  // the current voxel set's coordinates
    std::memcpy(cc.i(), coords, sizeof(int) * (size_t)n * 4);
    int N = n;
    DevBuf h = floats((size_t)N * kChans[0]);  // the current features [N][C]
    {
        DevBuf fin = floats((size_t)n * kLatent);
        std::memcpy(fin.f(), feats, sizeof(float) * (size_t)n * kLatent);
        linear16(fin.f(), N, kLatent, m.lw, m.lb, kChans[0], h.f());
    }
    int C = kChans[0];

    const auto dec_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 5; ++i) {
        if (upsample_times > 0 && i == upsample_times) {  // the coordinates of this level, before its blocks (python's upsample)
            if (vcoords) {
                vcoords->assign(cc.i(), cc.i() + (size_t)N * 4);
                return true;
            }
            *out_m = N;  // the count, also when the buffer is too small (the caller retries with it)
            if (N > out_capacity) {
                error_ = "the coordinates have more voxels than the buffer";
                return false;
            }
            std::memcpy(out_coords, cc.i(), sizeof(int) * (size_t)N * 4);
            return true;
        }
        // the level's ConvNeXt blocks: h = mlp(LN(conv(h))) + h
        if (!m.levels[i].empty()) {
            const VoxelMap mp = build_map(cc.i(), N);
            DevBuf t1 = floats((size_t)N * C), t2 = floats((size_t)N * 4 * C), t4 = floats((size_t)N * C), next = floats((size_t)N * C);
            for (const auto& cn : m.levels[i]) {
                conv(h.f(), C, mp, cn.cw, cn.cb, C, t1.f());
                ln_affine(t1.f(), N, C, cn.nw, cn.nb, 1e-6f, t4.f());
                linear16(t4.f(), N, C, cn.m0w, cn.m0b, 4 * C, t2.f());
                sddev::silu(t2.f(), (size_t)N * 4 * C);
                linear16(t2.f(), N, 4 * C, cn.m2w, cn.m2b, C, next.f());
                sddev::add(next.f(), h.f(), (size_t)N * C);  // next += h
                std::swap(h, next);
            }
        }  // the temporaries, the spare features and the map go here
        if (progress_) {
            size_t in_use = 0, cached = 0;
            sddev::pinned_bytes(in_use, cached);
            std::fprintf(stderr, "    decoder level %d: %d voxels, %.1f s, pinned %.1f GB in use, %.1f GB cached\n", i, N,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - dec_t0).count(), in_use / 1e9, cached / 1e9);
        }
        if (i == 4) break;  // the last level has no upsampling
        // the channel-to-spatial block (ResBlockC2S with the predicted subdivision)
        const Impl::C2S& u = m.up[i];
        const int O = u.out;
        DevBuf logits;
        if (!guide) {
            logits = floats((size_t)N * 8);
            linear16(h.f(), N, C, u.sw, u.sb, 8, logits.f());
            if (subs_out) {  // this level's subdivision, for the texture decoder (python's return_subs)
                Subdivision sd;
                sd.coords.assign(cc.i(), cc.i() + (size_t)N * 4);
                sd.logits.assign(logits.f(), logits.f() + (size_t)N * 8);
                subs_out->push_back(std::move(sd));
            }
        }
        // the residual branch: LN (affine), SiLU, conv1 (8 children per voxel), then the upsampling
        DevBuf c1 = floats((size_t)N * 8 * O);
        {
            const VoxelMap mp = build_map(cc.i(), N);
            DevBuf a1 = floats((size_t)N * C);
            ln_affine(h.f(), N, C, u.n1w, u.n1b, 1e-6f, a1.f());
            sddev::silu(a1.f(), (size_t)N * C);
            conv(a1.f(), C, mp, u.c1w, u.c1b, 8 * O, c1.f());
        }
        // the subdivision: the children where the logit is positive, their voxels
        DevBuf bits((size_t)N), cnt = ints((size_t)N), off = ints((size_t)N + 1);
        if (!guide) {
            sddev::sp_c2s_mask(logits.f(), N, bits.u8(), cnt.i());
        } else {  // the guide's subdivision of this level: the children with a positive logit (python: subdiv.feats > 0)
            const Subdivision& g = (*guide)[i];
            if (g.coords.size() != (size_t)N * 4 || std::memcmp(g.coords.data(), cc.i(), sizeof(int) * (size_t)N * 4) != 0) {
                error_ = "the guide's voxels at a level differ from the decoder's";
                return false;
            }
            for (int k = 0; k < N; ++k) {
                unsigned int b = 0;
                for (int c = 0; c < 8; ++c)
                    if (g.logits[(size_t)k * 8 + c] > 0.0f) b |= 1u << c;
                bits.u8()[k] = (unsigned char)b;
                cnt.i()[k] = __builtin_popcount(b);
            }
        }
        logits.reset();
        sddev::sp_exclusive_scan(cnt.i(), N, off.i());
        const int M = off.i()[N];
        if (M <= 0) {
            error_ = "the subdivision produced no voxels";
            return false;
        }
        DevBuf newc = ints((size_t)M * 4), idx = ints((size_t)M), subidx = ints((size_t)M);
        sddev::sp_c2s_fill(cc.i(), bits.u8(), off.i(), N, newc.i(), idx.i(), subidx.i());
        bits.reset();
        cnt.reset();
        off.reset();
        DevBuf hc = floats((size_t)M * O);
        sddev::sp_c2s_gather(c1.f(), idx.i(), subidx.i(), M, O, 1, hc.f());
        c1.reset();
        DevBuf skip = floats((size_t)M * O);
        sddev::sp_c2s_gather(h.f(), idx.i(), subidx.i(), M, C / 8, O / (C / 8), skip.f());
        h.reset();  // the level's features are used up
        idx.reset();
        subidx.reset();
        // the second branch: LN (no affine), SiLU, conv2 on the new voxels, plus the skip
        DevBuf a2 = floats((size_t)M * O);
        sddev::layer_norm_plain(hc.f(), M, O, 1e-6f, a2.f());
        hc.reset();
        sddev::silu(a2.f(), (size_t)M * O);
        DevBuf c2 = floats((size_t)M * O);
        {
            const VoxelMap mp2 = build_map(newc.i(), M);
            conv(a2.f(), O, mp2, u.c2w, u.c2b, O, c2.f());
        }
        a2.reset();
        sddev::add(c2.f(), skip.f(), (size_t)M * O);
        skip.reset();
        h = std::move(c2);
        cc = std::move(newc);
        N = M;
        C = O;
        if (i == 0 && stage) stage("sdec_l0", h.f(), cc.i(), (size_t)N, (size_t)C);
    }

    // the output: LN (no affine), the output layer to 7 channels
    DevBuf y = floats((size_t)N * m.out_ch);
    {
        DevBuf z = floats((size_t)N * C);
        sddev::layer_norm_plain(h.f(), N, C, 1e-5f, z.f());
        h.reset();
        linear16(z.f(), N, C, m.ow, m.ob, m.out_ch, y.f());
    }
    if (vcoords) {
        vfeats->assign(y.f(), y.f() + (size_t)N * m.out_ch);
        vcoords->assign(cc.i(), cc.i() + (size_t)N * 4);
        return true;
    }
    *out_m = N;  // the count, also when the buffer is too small (the caller retries with it)
    if (N > out_capacity) {
        error_ = "the output has more voxels than the buffer";
        return false;
    }
    std::memcpy(out_feats, y.f(), sizeof(float) * (size_t)N * m.out_ch);
    std::memcpy(out_coords, cc.i(), sizeof(int) * (size_t)N * 4);
    *out_m = N;
    return true;
}

}  // namespace trellis
