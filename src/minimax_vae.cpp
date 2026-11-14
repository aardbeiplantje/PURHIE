// MiniMax-H3-Image's VAE decoder (minimax_vae.h; diffusers' AutoencoderKLMiniMaxH3Image with tiling, the decoder part):
// per tile, post_quant_conv (1 x 1), x_embedder (24 -> 1024), 4 register tokens and one zero token, 24 blocks (RMS norm,
// fused QKV with RMS-normed heads and 2-D rotary on the first half of each head, attention, the output projection scaled by
// scale1; RMS norm, SwiGLU feed-forward scaled by scale2), norm_out (LayerNorm), proj_out (1024 -> 3 x 16 x 16) and the
// unpatchify. The tiles follow the reference's _split_tiles, _stitch_tiles and _blend.
#include "minimax_vae.h"

#include "gguf_mmap.h"
#include "minimax_weights.h"
#include "safetensors_mmap.h"
#include "kernel_loader.h"
#include "sd_device.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string g_error;

constexpr int kLayers = 24, kHeads = 16, kHeadDim = 64, kC = 1024, kFF = 4096, kRegs = 4, kLatent = 24, kOut = 3;
constexpr int kPatch = 16, kTilePx = 256, kOverlapPx = 64, kThreads = 256;
constexpr float kEps = 1e-5f;
constexpr double kTheta = 100.0;

struct Kernels {
    hipModule_t mod = nullptr;
    hipFunction_t rmsnorm = nullptr, layernorm = nullptr, rope = nullptr, scale_add = nullptr, swiglu = nullptr,
                  unpatch = nullptr, split_qkv = nullptr;
};
Kernels g_k;

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("minimax vae: ") + what + ": " + hipGetErrorString(e));
}

void load_kernels() {
    if (g_k.mod) return;
    if (kernel_module_load(&g_k.mod, "minimax_vae.co") != hipSuccess) throw std::runtime_error("minimax vae: minimax_vae.co does not load");
    hipFunction_t* fs[] = {&g_k.rmsnorm, &g_k.layernorm, &g_k.rope, &g_k.scale_add, &g_k.swiglu, &g_k.unpatch, &g_k.split_qkv};
    const char* names[] = {"mv_rmsnorm", "mv_layernorm", "mv_rope", "mv_scale_add", "mm_swiglu_native", "mv_unpatch", "mm_split_qkv"};
    for (int i = 0; i < 7; ++i) check(hipModuleGetFunction(fs[i], g_k.mod, names[i]), names[i]);
}

void launch(hipFunction_t f, unsigned blocks, void** args) {
    check(hipModuleLaunchKernel(f, blocks, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr), "launch");
    check(hipStreamSynchronize(nullptr), "stream");
}

unsigned blocks_for(long n) { return (unsigned)((n + kThreads - 1) / kThreads); }

// The tiles of one axis (the reference's _split_tiles): starts, lengths and overlaps, in pixels
void split_tiles(int length, int tile, int min_overlap, std::vector<int>& starts, std::vector<int>& lengths,
                 std::vector<int>& overlaps) {
    starts.clear();
    lengths.clear();
    overlaps.clear();
    if (tile >= length) {
        starts = {0};
        lengths = {length};
        return;
    }
    int num = (length + tile - 1) / tile;
    while (tile * num - min_overlap * (num - 1) - length < 0) ++num;
    overlaps.assign(num - 1, min_overlap);
    const int remaining = tile * num - min_overlap * (num - 1) - length;
    for (int i = 0; i < remaining / kPatch; ++i) overlaps[i % (num - 1)] += kPatch;
    starts.push_back(0);
    for (int i = 0; i < num - 1; ++i) starts.push_back(starts.back() + tile - overlaps[i]);
    lengths.assign(num, tile);
}

// A plane set [C][H][W] on the host (the tiles' pixels; the blend and the crop work on them as the reference does)
struct Planes {
    int c = 0, h = 0, w = 0;
    std::vector<float> d;
    float& at(int ch, int y, int x) { return d[((size_t)ch * h + y) * w + x]; }
    float at(int ch, int y, int x) const { return d[((size_t)ch * h + y) * w + x]; }
};

// the reference's _blend along dim (0: height, 1: width) of a (end of a) into b (start of b), with a linear weight
Planes blend(const Planes& a, const Planes& b, int ext, int dim) {
    const int alen = dim == 0 ? a.h : a.w, blen = dim == 0 ? b.h : b.w;
    ext = std::min({alen, blen, ext});
    Planes out = b;
    for (int ch = 0; ch < b.c; ++ch)
        for (int y = 0; y < b.h; ++y)
            for (int x = 0; x < b.w; ++x) {
                const int p = dim == 0 ? y : x;
                if (p >= ext) continue;
                const float wb = (float)p / (float)ext, wa = 1.0f - wb;
                const int ay = dim == 0 ? a.h - ext + y : y, ax = dim == 0 ? x : a.w - ext + x;
                out.at(ch, y, x) = a.at(ch, ay, ax) * wa + b.at(ch, y, x) * wb;
            }
    return out;
}

Planes crop(const Planes& p, int dim, int cut) {  // drop the last `cut` along dim
    Planes o;
    o.c = p.c;
    o.h = dim == 0 ? p.h - cut : p.h;
    o.w = dim == 1 ? p.w - cut : p.w;
    o.d.resize((size_t)o.c * o.h * o.w);
    for (int ch = 0; ch < o.c; ++ch)
        for (int y = 0; y < o.h; ++y)
            for (int x = 0; x < o.w; ++x) o.at(ch, y, x) = p.at(ch, y, x);
    return o;
}

// the reference's _stitch_tiles: tiles[i][j] in pixels, the overlaps of each axis
Planes stitch(const std::vector<std::vector<Planes>>& tiles, const std::vector<int>& yov, const std::vector<int>& xov) {
    std::vector<Planes> rows;
    const size_t NR = tiles.size();
    for (size_t i = 0; i < NR; ++i) {
        const size_t NC = tiles[i].size();
        std::vector<Planes> row_out;
        for (size_t j = 0; j < NC; ++j) {
            Planes t = tiles[i][j];
            if (i > 0) t = blend(tiles[i - 1][j], t, yov[i - 1], 0);
            if (j > 0) t = blend(tiles[i][j - 1], t, xov[j - 1], 1);
            if (i + 1 < NR) t = crop(t, 0, yov[i]);
            if (j + 1 < NC) t = crop(t, 1, xov[j]);
            row_out.push_back(std::move(t));
        }
        // concatenate the row along the width
        Planes r;
        r.c = row_out[0].c;
        r.h = row_out[0].h;
        for (auto& t : row_out) r.w += t.w;
        r.d.resize((size_t)r.c * r.h * r.w);
        int x0 = 0;
        for (auto& t : row_out) {
            for (int ch = 0; ch < r.c; ++ch)
                for (int y = 0; y < r.h; ++y)
                    for (int x = 0; x < t.w; ++x) r.at(ch, y, x0 + x) = t.at(ch, y, x);
            x0 += t.w;
        }
        rows.push_back(std::move(r));
    }
    // concatenate the rows along the height
    Planes out;
    out.c = rows[0].c;
    out.w = rows[0].w;
    for (auto& r : rows) out.h += r.h;
    out.d.resize((size_t)out.c * out.h * out.w);
    int y0 = 0;
    for (auto& r : rows) {
        for (int ch = 0; ch < out.c; ++ch)
            for (int y = 0; y < r.h; ++y)
                for (int x = 0; x < out.w; ++x) out.at(ch, y0 + y, x) = r.at(ch, y, x);
        y0 += r.h;
    }
    return out;
}

}  // namespace

struct mmvae {
    WeightFile w;  // the decoder's weights: the GGUF, or the original checkpoint
    std::vector<const void*> dev_bases;  // the mappings registered for the device
    sddev::Floats mean_own, std_own;  // the latents' statistics from the checkpoint's config.json (a GGUF has them as tensors)
    ~mmvae() {
        for (const void* base : dev_bases) sddev::unregister_weights(base);
    }
    const float *pq_w = nullptr, *pq_b = nullptr, *proj_in_w = nullptr, *proj_in_b = nullptr, *reg = nullptr,
                *proj_out_w = nullptr, *proj_out_b = nullptr, *norm_out_w = nullptr, *norm_out_b = nullptr,
                *mean = nullptr, *std_ = nullptr;
    struct Block {
        const float *norm1, *qkv_w, *qkv_b, *out_w, *out_b, *scale1, *norm2, *w1_w, *w1_b, *w2_w, *w2_b, *scale2;
    };
    std::vector<Block> blocks;
    int layers = kLayers, heads = kHeads, head_dim = kHeadDim, regs = kRegs, tile_px = kTilePx, overlap_px = kOverlapPx;
    float theta = (float)kTheta;
};

namespace {

const float* tensor(const mmvae* m, const std::string& name) {
    size_t bytes = 0;
    const void* p = m->w.tensor(name, &bytes);
    if (!p) throw std::runtime_error("minimax vae: no tensor " + name);
    return static_cast<const float*>(p);
}

// One tile: zd [24][H][W] (the denormalized latents of the tile) -> out [3][16H][16W]
void decode_tile(const mmvae* m, const float* zd, int H, int W, Planes& out) {
    const int HW = H * W, T = HW + m->regs + 1;
    // the token rows of the latent positions
    sddev::Floats x_in((size_t)HW * kLatent);
    for (int c = 0; c < kLatent; ++c)
        for (int p = 0; p < HW; ++p) x_in[(size_t)p * kLatent + c] = zd[(size_t)c * HW + p];
    sddev::Floats pq((size_t)HW * kLatent);
    sddev::linear_f32(x_in.data(), HW, m->pq_w, kLatent, kLatent, m->pq_b, pq.data());
    sddev::Floats x((size_t)T * kC);
    sddev::linear_f32(pq.data(), HW, m->proj_in_w, kLatent, kC, m->proj_in_b, x.data());
    // the register tokens after the patches, the zero token last
    std::memcpy(x.data() + (size_t)HW * kC, m->reg, sizeof(float) * (size_t)m->regs * kC);
    std::memset(x.data() + (size_t)(HW + m->regs) * kC, 0, sizeof(float) * kC);
    // the rotary tables: positions normalized to [-1, 1) per axis (the cell centres); the first 16 channels are the height
    // and the width frequencies, the other 16 repeat them; the register and zero tokens have position 0 (identity)
    const int rd = m->head_dim / 2;
    sddev::Floats cs((size_t)T * rd), sn((size_t)T * rd);
    std::vector<float> inv(rd / 4);  // num_axes 2: arange(0, 1, 4 / rd) of the exponents
    for (int k = 0; k < rd / 4; ++k) inv[k] = (float)(1.0 / std::pow((double)m->theta, (double)k / (rd / 4.0)));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < rd; ++d) {
            double pos = 0.0;
            if (t < HW) {
                const int i = t / W, j = t % W;
                const int axis = (d % (rd / 2)) / (rd / 4);
                const double cell = axis == 0 ? (double)i : (double)j, size = axis == 0 ? H : W;
                pos = 2.0 * ((cell + 0.5) / size) - 1.0;
            }
            const int k = d % (rd / 4);
            const double ang = 2.0 * M_PI * pos * inv[k];
            cs[(size_t)t * rd + d] = (float)std::cos(ang);
            sn[(size_t)t * rd + d] = (float)std::sin(ang);
        }
    sddev::Floats n1((size_t)T * kC), qkv((size_t)T * 3 * kC), q((size_t)T * kC), k((size_t)T * kC), v((size_t)T * kC), qn((size_t)T * kC),
        kn((size_t)T * kC), a((size_t)T * kC), o((size_t)T * kC), pj((size_t)T * 2 * kFF), g((size_t)T * kFF),
        f((size_t)T * kC);
    for (int l = 0; l < m->layers; ++l) {
        const mmvae::Block& B = m->blocks[l];
        auto rms = [&](const float* in, int n, const float* w, float* outp, int rows) {
            const float* ww = w;
            const float ee = kEps;
            void* args[] = {(void*)&in, (void*)&n, (void*)&ww, (void*)&ee, (void*)&outp};
            launch(g_k.rmsnorm, (unsigned)rows, args);
        };
        rms(x.data(), kC, B.norm1, n1.data(), T);
        // fused QKV: the rows are grouped per head (q, k, v of 64 rows each per head), split by the kernel
        sddev::linear_f32(n1.data(), T, B.qkv_w, kC, 3 * kC, B.qkv_b, qkv.data());
        {
            const int heads = kHeads, D = kHeadDim;
            float* qkvp = qkv.data();
            float* qp = q.data();
            float* kp = k.data();
            float* vp = v.data();
            void* args[] = {(void*)&qkvp, (void*)&T, (void*)&heads, (void*)&D, (void*)&qp, (void*)&kp, (void*)&vp};
            launch(g_k.split_qkv, blocks_for((long)T * kC), args);
        }
        // RMS norm of each head's q and k (no affine), rows of one head are contiguous: T * heads rows of head_dim
        {
            const float ee = kEps;
            const int hd = m->head_dim;
            const float* nullw = nullptr;
            float* qi = q.data();
            float* qo = qn.data();
            void* args[] = {(void*)&qi, (void*)&hd, (void*)&nullw, (void*)&ee, (void*)&qo};
            launch(g_k.rmsnorm, (unsigned)((size_t)T * m->heads), args);
            float* ki = k.data();
            float* ko = kn.data();
            void* args2[] = {(void*)&ki, (void*)&hd, (void*)&nullw, (void*)&ee, (void*)&ko};
            launch(g_k.rmsnorm, (unsigned)((size_t)T * m->heads), args2);
        }
        {
            const int C = kC, heads = m->heads, D = m->head_dim, rdim = rd;
            const long total = (long)T * heads * (rd / 2);
            float* qp = qn.data();
            const float* c = cs.data();
            const float* s = sn.data();
            void* args[] = {(void*)&qp, (void*)&T, (void*)&C, (void*)&heads, (void*)&D, (void*)&rdim, (void*)&c, (void*)&s};
            launch(g_k.rope, blocks_for(total), args);
            float* kp = kn.data();
            void* args2[] = {(void*)&kp, (void*)&T, (void*)&C, (void*)&heads, (void*)&D, (void*)&rdim, (void*)&c, (void*)&s};
            launch(g_k.rope, blocks_for(total), args2);
        }
        sddev::attention(qn.data(), T, kn.data(), v.data(), T, kC, m->heads, a.data());
        sddev::linear_f32(a.data(), T, B.out_w, kC, kC, B.out_b, o.data());
        {
            const int Tt = T, Cc = kC;
            float* xp = x.data();
            const float* op = o.data();
            const float* sc = B.scale1;
            void* args[] = {(void*)&xp, (void*)&op, (void*)&sc, (void*)&Tt, (void*)&Cc};
            launch(g_k.scale_add, blocks_for((long)T * kC), args);
        }
        rms(x.data(), kC, B.norm2, n1.data(), T);
        sddev::linear_f32(n1.data(), T, B.w1_w, kC, 2 * kFF, B.w1_b, pj.data());
        {
            const int Tt = T, F = kFF;
            const float* pp = pj.data();
            float* gp = g.data();
            void* args[] = {(void*)&pp, (void*)&Tt, (void*)&F, (void*)&gp};
            launch(g_k.swiglu, blocks_for((long)T * kFF), args);
        }
        sddev::linear_f32(g.data(), T, B.w2_w, kFF, kC, B.w2_b, f.data());
        {
            const int Tt = T, Cc = kC;
            float* xp = x.data();
            const float* fp = f.data();
            const float* sc = B.scale2;
            void* args[] = {(void*)&xp, (void*)&fp, (void*)&sc, (void*)&Tt, (void*)&Cc};
            launch(g_k.scale_add, blocks_for((long)T * kC), args);
        }
    }
    // norm_out (LayerNorm), proj_out, and the unpatchify of the patch tokens
    sddev::Floats no((size_t)HW * kC), pr((size_t)HW * kOut * kPatch * kPatch);
    {
        const int n = kC;
        const float* w = m->norm_out_w;
        const float* b = m->norm_out_b;
        const float ee = kEps;
        float* xo = x.data();
        float* yo = no.data();
        void* args[] = {(void*)&xo, (void*)&n, (void*)&w, (void*)&b, (void*)&ee, (void*)&yo};
        launch(g_k.layernorm, (unsigned)HW, args);
    }
    sddev::linear_f32(no.data(), HW, m->proj_out_w, kC, kOut * kPatch * kPatch, m->proj_out_b, pr.data());
    out.c = kOut;
    out.h = H * kPatch;
    out.w = W * kPatch;
    sddev::Floats img((size_t)out.c * out.h * out.w);
    {
        const int OH = H, OW = W, P = kPatch;
        const long total = 3L * out.h * out.w;
        const float* tp = pr.data();
        float* ip = img.data();
        void* args[] = {(void*)&tp, (void*)&OH, (void*)&OW, (void*)&P, (void*)&ip};
        launch(g_k.unpatch, blocks_for(total), args);
    }
    out.d.assign(img.data(), img.data() + img.size());  // the host's planes (the blend and the stitch read them there)
}

}  // namespace

// A path of a GGUF (rocm-convert's output), else the original checkpoint's decoder file (vae/diffusion_pytorch_model.safetensors,
// its config.json in the same directory)
static bool is_gguf_file(const char* path) {
    const std::string s = path ? path : "";
    return s.size() >= 5 && s.compare(s.size() - 5, 5, ".gguf") == 0;
}

mmvae_t* mmvae_open(const char* path) {
    try {
        std::unique_ptr<mmvae> m(new mmvae());
        if (!sddev::enabled()) throw std::runtime_error("minimax vae: needs the device (SD_DEVICE unset or gpu)");
        if (is_gguf_file(path)) {
            m->w.g = gguf_mmap_open(path);
            if (!m->w.g) throw std::runtime_error(std::string("cannot open ") + (path ? path : "(null)"));
            if (std::string(gguf_arch(m->w.g)) != "minimax_h3_vae") throw std::runtime_error("not a MiniMax-H3 VAE GGUF");
            if (gguf_arch_int(m->w.g, "decoder.layers", -1) != kLayers || gguf_arch_int(m->w.g, "decoder.heads", -1) != kHeads ||
                gguf_arch_int(m->w.g, "decoder.head_dim", -1) != kHeadDim || gguf_arch_int(m->w.g, "decoder.register_tokens", -1) != kRegs ||
                gguf_arch_int(m->w.g, "latent_channels", -1) != kLatent)
                throw std::runtime_error("minimax vae: the GGUF's decoder is not the 24-layer 1024-wide one this engine has");
            m->theta = gguf_arch_float(m->w.g, "decoder.rope_theta", (float)kTheta);
            if (gguf_n_maps(m->w.g) != 1) throw std::runtime_error("minimax vae: the GGUF is split");
            size_t bytes = 0;
            const void* base = gguf_map_base(m->w.g, &bytes);
            if (!sddev::register_weights(base, bytes)) throw std::runtime_error("minimax vae: cannot register the weights");
            m->dev_bases.push_back(base);
        } else {
            // the original checkpoint: the decoder's tensors by their names, the rope theta and the latents' statistics from
            // config.json (the decoder's own keys: vit_decoder_kwargs.rope_theta; latents_mean, latents_std: 24 values)
            const std::string file = path ? path : "";
            const std::string dir = file.substr(0, file.rfind('/'));
            m->w.st = st_open(file.c_str());
            if (!m->w.st) throw std::runtime_error(std::string("minimax vae: ") + st_last_error());
            for (int i = 0; i < st_n_shards(m->w.st); ++i) {
                size_t bytes = 0;
                const void* base = st_shard_base(m->w.st, i, &bytes);
                if (!sddev::register_weights(base, bytes)) throw std::runtime_error("minimax vae: cannot register the weights");
                m->dev_bases.push_back(base);
            }
            if (st_find(m->w.st, ("decoder.transformer_blocks." + std::to_string(kLayers) + ".norm1.weight").c_str()))
                throw std::runtime_error("minimax vae: the checkpoint's decoder has more layers than this engine's 24");
            char* text = st_read_text(dir.c_str(), "config.json");
            if (!text) throw std::runtime_error("minimax vae: no config.json next to " + file);
            double theta = 0;
            if (st_json_number(text, "rope_theta", &theta) == 1) m->theta = (float)theta;
            double mean[kLatent], sd[kLatent];
            if (st_json_numbers(text, "latents_mean", mean, kLatent) != kLatent || st_json_numbers(text, "latents_std", sd, kLatent) != kLatent) {
                std::free(text);
                throw std::runtime_error("minimax vae: config.json has no 24 latents_mean / latents_std");
            }
            std::free(text);
            m->mean_own = sddev::Floats(kLatent);
            m->std_own = sddev::Floats(kLatent);
            for (int c = 0; c < kLatent; ++c) {
                m->mean_own[c] = (float)mean[c];
                m->std_own[c] = (float)sd[c];
            }
        }
        if (!sddev::enabled()) throw std::runtime_error("minimax vae: needs the device (SD_DEVICE unset or gpu)");
        m->pq_w = tensor(m.get(), "post_quant_conv.weight");
        m->pq_b = tensor(m.get(), "post_quant_conv.bias");
        m->proj_in_w = tensor(m.get(), "decoder.x_embedder.weight");
        m->proj_in_b = tensor(m.get(), "decoder.x_embedder.bias");
        m->reg = tensor(m.get(), "decoder.register_tokens");
        m->proj_out_w = tensor(m.get(), "decoder.proj_out.weight");
        m->proj_out_b = tensor(m.get(), "decoder.proj_out.bias");
        m->norm_out_w = tensor(m.get(), "decoder.norm_out.weight");
        m->norm_out_b = tensor(m.get(), "decoder.norm_out.bias");
        if (m->w.g) {
            m->mean = tensor(m.get(), "latents.mean");
            m->std_ = tensor(m.get(), "latents.std");
        } else {
            m->mean = m->mean_own.data();
            m->std_ = m->std_own.data();
        }
        for (int l = 0; l < kLayers; ++l) {
            const std::string p = "decoder.transformer_blocks." + std::to_string(l) + ".";
            mmvae::Block B;
            B.norm1 = tensor(m.get(), p + "norm1.weight");
            B.qkv_w = tensor(m.get(), p + "attn.to_qkv.weight");
            B.qkv_b = tensor(m.get(), p + "attn.to_qkv.bias");
            B.out_w = tensor(m.get(), p + "attn.to_out.weight");
            B.out_b = tensor(m.get(), p + "attn.to_out.bias");
            B.scale1 = tensor(m.get(), p + "scale1");
            B.norm2 = tensor(m.get(), p + "norm2.weight");
            B.w1_w = tensor(m.get(), p + "ff.w1.weight");
            B.w1_b = tensor(m.get(), p + "ff.w1.bias");
            B.w2_w = tensor(m.get(), p + "ff.w2.weight");
            B.w2_b = tensor(m.get(), p + "ff.w2.bias");
            B.scale2 = tensor(m.get(), p + "scale2");
            m->blocks.push_back(B);
        }
        load_kernels();
        return m.release();
    } catch (const std::exception& e) {
        g_error = e.what();
        return nullptr;
    }
}

void mmvae_close(mmvae_t* model) { delete model; }

const char* mmvae_last_error(void) { return g_error.c_str(); }

int mmvae_decode(mmvae_t* m, const float* latents, int h, int w, float* image) {
    if (!m || !latents || !image || h <= 0 || w <= 0) {
        g_error = "mmvae_decode: bad arguments";
        return -1;
    }
    try {
        std::lock_guard<std::recursive_mutex> lock(sddev::run_mutex());
        const int HP = h * kPatch, WP = w * kPatch;
        // the pipeline's denormalization: z * latents_std + latents_mean, per channel
        std::vector<float> zd((size_t)kLatent * h * w);
        for (int c = 0; c < kLatent; ++c)
            for (int p = 0; p < h * w; ++p) zd[(size_t)c * h * w + p] = latents[(size_t)c * h * w + p] * m->std_[c] + m->mean[c];
        // the tiles in latent units (16 px per latent position): the reference's split of each axis
        std::vector<int> ys, yl, yo, xs, xl, xo;
        split_tiles(HP, m->tile_px, m->overlap_px, ys, yl, yo);
        split_tiles(WP, m->tile_px, m->overlap_px, xs, xl, xo);
        std::vector<std::vector<Planes>> tiles(ys.size());
        for (size_t i = 0; i < ys.size(); ++i)
            for (size_t j = 0; j < xs.size(); ++j) {
                const int ty = ys[i] / kPatch, tx = xs[j] / kPatch, th = yl[i] / kPatch, tw = xl[j] / kPatch;
                std::vector<float> tz((size_t)kLatent * th * tw);
                for (int c = 0; c < kLatent; ++c)
                    for (int y = 0; y < th; ++y)
                        for (int x = 0; x < tw; ++x)
                            tz[((size_t)c * th + y) * tw + x] = zd[((size_t)c * h + ty + y) * w + tx + x];
                Planes out;
                decode_tile(m, tz.data(), th, tw, out);
                tiles[i].push_back(std::move(out));
            }
        const Planes img = tiles.size() == 1 && tiles[0].size() == 1 ? tiles[0][0] : stitch(tiles, yo, xo);
        if (img.c != kOut || img.h != HP || img.w != WP) throw std::runtime_error("mmvae_decode: the tiles do not cover the image");
        std::memcpy(image, img.d.data(), sizeof(float) * img.d.size());
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return -1;
    }
}
