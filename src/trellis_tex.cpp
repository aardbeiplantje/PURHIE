// The TRELLIS.2 texture path on the device (trellis_tex.h): the texture flow sampled with the shared flow sampler (the conditional
// prediction alone), its input rebuilt at each evaluation as the noise and the normalised shape latent, then the guided texture decoder.
#include "trellis_tex.h"

#include "trellis_slat.h"

#include <chrono>
#include <cstring>

namespace trellis {
namespace {

constexpr int kLat = 32;  // the texture latent and the shape latent per voxel
constexpr int kIn = 64;   // the texture flow's input: the noise and the normalised shape latent
constexpr int kOut = 6;   // the texture decoder's outputs

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

struct TexturePipeline::Impl {
    std::unique_ptr<SlatFlow> flow;
    std::unique_ptr<SlatFlow> flow_hr;  // the 1024 texture flow (optional)
    std::unique_ptr<ShapeDecoder> dec;
};

TexturePipeline::TexturePipeline(const std::string& tex_flow, const std::string& tex_dec, const std::string& tex_flow_hr) : p_(new Impl) {
    p_->flow.reset(new SlatFlow(tex_flow, kIn));
    if (!p_->flow->ok()) {
        error_ = "texture flow: " + p_->flow->error();
        return;
    }
    if (!tex_flow_hr.empty()) {
        p_->flow_hr.reset(new SlatFlow(tex_flow_hr, kIn));
        if (!p_->flow_hr->ok()) {
            error_ = "1024 texture flow: " + p_->flow_hr->error();
            return;
        }
    }
    p_->dec.reset(new ShapeDecoder(tex_dec, kOut, true));
    if (!p_->dec->ok()) {
        error_ = "texture decoder: " + p_->dec->error();
        return;
    }
    ok_ = true;
}

TexturePipeline::~TexturePipeline() = default;

bool TexturePipeline::flow_forward(const int32_t* coords, int n, const float* x64, const float* cond, int Lc, float t, float* out, bool hr) {
    error_.clear();
    if (!ok_) {
        error_ = "the texture pipeline is not open";
        return false;
    }
    SlatFlow* fl = hr ? p_->flow_hr.get() : p_->flow.get();
    if (!fl) {
        error_ = "no 1024 texture flow was opened";
        return false;
    }
    if (!fl->forward(coords, n, x64, t, cond, Lc, out)) {
        error_ = "texture flow: " + fl->error();
        return false;
    }
    return true;
}

bool TexturePipeline::decode(const int32_t* coords, int n, const float* tex_slat, const std::vector<Subdivision>& subs, TexResult& out) {
    error_.clear();
    if (!ok_) {
        error_ = "the texture pipeline is not open";
        return false;
    }
    auto t0 = std::chrono::steady_clock::now();
    // the decoder's voxels: a buffer guess of 256 per input voxel; a too-small buffer is reported with its count, and the decoder runs once more
    int cap = 256 * n, M = 0;
    out.raw.resize((size_t)cap * kOut);
    out.coords.resize((size_t)cap * 4);
    bool ok = p_->dec->forward_guided(coords, n, tex_slat, subs, out.raw.data(), out.coords.data(), &M, cap);
    if (!ok && M > cap) {
        cap = M;
        out.raw.resize((size_t)cap * kOut);
        out.coords.resize((size_t)cap * 4);
        ok = p_->dec->forward_guided(coords, n, tex_slat, subs, out.raw.data(), out.coords.data(), &M, cap);
    }
    if (!ok) {
        error_ = "texture decoder: " + p_->dec->error();
        return false;
    }
    out.raw.resize((size_t)M * kOut);
    out.coords.resize((size_t)M * 4);
    out.out.resize((size_t)M * kOut);
    for (size_t i = 0; i < out.raw.size(); ++i) out.out[i] = out.raw[i] * 0.5f + 0.5f;
    out.seconds_dec = seconds_since(t0);
    return true;
}

bool TexturePipeline::run(const int32_t* coords, int n, const float* shape_slat, const std::vector<Subdivision>& subs, const float* cond,
                          int Lc, const NoiseFn& noise, const TexConfig& cfg, TexResult& out, bool hr) {
    error_.clear();
    if (!ok_) {
        error_ = "the texture pipeline is not open";
        return false;
    }
    if (n <= 0 || Lc <= 0) {
        error_ = "no voxels or no image features";
        return false;
    }
    SlatFlow* fl = hr ? p_->flow_hr.get() : p_->flow.get();
    if (!fl) {
        error_ = "no 1024 texture flow was opened";
        return false;
    }
    const size_t cnt = (size_t)n * kLat;

    // the flow's concatenated condition: the normalised shape latent (sample_tex_slat: (shape_slat - mean) / std)
    std::vector<float> shape_n(cnt);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < kLat; ++c) shape_n[(size_t)i * kLat + c] = (shape_slat[(size_t)i * kLat + c] - cfg.shape_mean[c]) / cfg.shape_std[c];

    // the sampler: the noise (the texture latent's start), then the flow on [noise | shape latent] at each evaluation
    std::vector<float> x(cnt);
    noise(x.data(), cnt);
    std::vector<float> in64((size_t)n * kIn);
    auto vel = [&](const float* xx, float t, bool neg, float* v) {
        (void)neg;  // guidance 1: the conditional prediction alone
        for (int i = 0; i < n; ++i) {
            std::memcpy(&in64[(size_t)i * kIn], xx + (size_t)i * kLat, sizeof(float) * kLat);
            std::memcpy(&in64[(size_t)i * kIn + kLat], &shape_n[(size_t)i * kLat], sizeof(float) * kLat);
        }
        const bool ok = fl->forward(coords, n, in64.data(), t, cond, Lc, v);
        if (!ok) error_ = "texture flow: " + fl->error();
        return ok;
    };
    auto t0 = std::chrono::steady_clock::now();
    if (!flow_sample(cfg.sampler, cnt, vel, x.data())) {
        if (error_.empty()) error_ = "texture sampler failed";
        return false;
    }
    out.seconds_flow = seconds_since(t0);

    // the texture latent, de-normalised (the decoder's input)
    out.tex_slat.resize(cnt);
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < kLat; ++c) out.tex_slat[(size_t)i * kLat + c] = x[(size_t)i * kLat + c] * cfg.tex_std[c] + cfg.tex_mean[c];

    // the texture decoder, guided by the shape decoder's subdivisions
    return decode(coords, n, out.tex_slat.data(), subs, out);
}

// The decoder's voxels and the mesh: the colours of the vertices from the texture rows.
static bool colour_vertices(TexturedMesh& out, std::string& error) {
    const size_t M = out.shape.dec_coords.size() / 4;
    if (out.tex.coords != out.shape.dec_coords || out.tex.out.size() != M * 6 || out.shape.verts.size() < M * 3) {
        error = "the texture voxels or the vertices differ from the shape decoder's voxels";
        return false;
    }
    // the decoder's voxels: vertex i is voxel i; the colour is the base colour, clamped to [0, 1]
    const size_t V = out.shape.verts.size() / 3;
    out.colors.assign(V * 3, 0.0f);
    for (size_t i = 0; i < M; ++i)
        for (int c = 0; c < 3; ++c) {
            const float v = out.tex.out[i * 6 + c];
            out.colors[i * 3 + c] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        }
    // the vertices of the holes that fill_holes closed (after the voxels): the mean colour of each fan's boundary vertices
    std::vector<double> acc((V - M) * 3, 0.0);
    std::vector<double> cnt(V - M, 0.0);
    for (size_t t = 0; t < out.shape.faces.size() / 3; ++t) {
        const int32_t* f = &out.shape.faces[t * 3];
        if ((size_t)f[2] < M) continue;
        if ((size_t)f[0] >= M || (size_t)f[1] >= M) continue;
        for (int a = 0; a < 2; ++a) {
            const size_t w = (size_t)f[a], nv = (size_t)f[2] - M;
            for (int c = 0; c < 3; ++c) acc[nv * 3 + c] += out.colors[w * 3 + c];
            cnt[nv] += 1;
        }
    }
    for (size_t nv = 0; nv < V - M; ++nv)
        for (int c = 0; c < 3; ++c)
            out.colors[(M + nv) * 3 + c] = cnt[nv] > 0 ? (float)(acc[nv * 3 + c] / cnt[nv]) : 0.0f;
    return true;
}

bool run_textured(ShapePipeline& shape, TexturePipeline& tex, const float* cond, int Lc, const NoiseFn& noise, const ShapeConfig& scfg,
                  const TexConfig& tcfg, TexturedMesh& out, std::string& error) {
    if (!shape.run(cond, Lc, noise, scfg, out.shape)) {
        error = "shape: " + shape.error();
        return false;
    }
    const int N = (int)(out.shape.coords.size() / 4);
    if (!tex.run(out.shape.coords.data(), N, out.shape.slat.data(), out.shape.subs, cond, Lc, noise, tcfg, out.tex)) {
        error = "texture: " + tex.error();
        return false;
    }
    return colour_vertices(out, error);
}

bool run_textured_cascade(ShapePipeline& shape, TexturePipeline& tex, const float* cond_lr, int Lc_lr, const float* cond_hr, int Lc_hr,
                          const NoiseFn& noise, const ShapeConfig& scfg, const TexConfig& tcfg, TexturedMesh& out, std::string& error) {
    if (!shape.run_cascade(cond_lr, Lc_lr, cond_hr, Lc_hr, noise, scfg, out.shape)) {
        error = "cascade: " + shape.error();
        return false;
    }
    // the 1024 texture pass: the 64^3 voxels, their 1024 shape latent and the 1024 decode's subdivisions, the 1024 image features
    const int K = (int)(out.shape.hr_coords.size() / 4);
    if (!tex.run(out.shape.hr_coords.data(), K, out.shape.hr_slat.data(), out.shape.subs, cond_hr, Lc_hr, noise, tcfg, out.tex, true)) {
        error = "texture (1024): " + tex.error();
        return false;
    }
    return colour_vertices(out, error);
}

}  // namespace trellis
