// The TRELLIS.2 image-to-shape path on the device (trellis_shape.h). The sampler follows trellis2/pipelines/samplers: the Euler
// steps of flow_euler.py over the time warp of its sample(), the classifier-free guidance of the CFG mixin (with the std rescale
// of the predicted x_0) and the guidance interval of the interval mixin. Every flow sample is a flat array of floats; a
// sparse flow's samples are [N][32] in the order of its voxels, and the statistics of the rescale are over all its elements (the
// sparse std of one sequence is the global one).
#include "trellis_shape.h"

#include "trellis_fill.h"
#include "trellis_mesh.h"
#include "trellis_sdec.h"
#include "trellis_slat.h"
#include "trellis_ss.h"
#include "trellis_ss_dec.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <utility>

namespace trellis {
namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

namespace {

constexpr int kSsRes = 16;     // the sparse-structure latent grid
constexpr int kSsC = 8;        // its channels
constexpr int kSsDec = 64;     // the sparse-structure decoder's grid: the occupancy is taken from 2x2x2 blocks of it
constexpr int kOcc = 32;       // the occupied voxels' grid (the shape-SLAT flow's)
constexpr int kLatent = 32;    // the shape-SLAT latent channels
constexpr int kHeadDim = 128;  // the flows' head dim (1536 / 12)
constexpr int kCond = 1024;    // the image features' channels


double stddev_all(const float* x, size_t count) {
    double s = 0, s2 = 0;
    for (size_t i = 0; i < count; ++i) {
        s += x[i];
        s2 += (double)x[i] * x[i];
    }
    const double m = s / (double)count;
    const double var = s2 / (double)count - m * m;
    return std::sqrt(var > 0 ? var : 0.0);
}

// FlowEulerGuidanceIntervalSampler.sample: x holds the noise on entry and the sample on exit.
bool flow_sample_impl(const FlowSampler& p, size_t count, const FlowVelocity& vel, float* x) {
    const double s = p.sigma_min;
    const int steps = p.steps;
    // t_seq = rescale * t / (1 + (rescale - 1) * t) for t = linspace(1, 0, steps + 1)
    std::vector<double> ts((size_t)steps + 1);
    for (int i = 0; i <= steps; ++i) {
        const double t = 1.0 - (double)i / (double)steps;
        ts[i] = p.rescale_t * t / (1.0 + (p.rescale_t - 1.0) * t);
    }
    std::vector<float> vpos(count), vcfg(count);
    for (int i = 0; i < steps; ++i) {
        const double t = ts[i], tp = ts[i + 1];
        const float a = (float)(1.0 - s);            // 1 - sigma_min
        const float b = (float)(s + (1.0 - s) * t);  // sigma_min + (1 - sigma_min) t
        if (!vel(x, (float)(1000.0 * t), false, vpos.data())) return false;
        const float* v = vpos.data();
        // guidance strength 1 is the conditional prediction alone (the python mixins: no negative pass, no rescale)
        if (p.guidance != 1.0f && p.interval_lo <= t && t <= p.interval_hi) {
            // the guidance: strength * conditional + (1 - strength) * negative
            std::vector<float> vneg(count);
            if (!vel(x, (float)(1000.0 * t), true, vneg.data())) return false;
            const float g = p.guidance;
            for (size_t j = 0; j < count; ++j) vcfg[j] = g * vpos[j] + (1.0f - g) * vneg[j];
            if (p.guidance_rescale > 0) {
                // the predicted x_0 of the guided prediction, rescaled to the spread of the conditional one, then blended
                std::vector<float> x0p(count), x0c(count);
                for (size_t j = 0; j < count; ++j) {
                    x0p[j] = a * x[j] - b * vpos[j];
                    x0c[j] = a * x[j] - b * vcfg[j];
                }
                const float ratio = (float)(stddev_all(x0p.data(), count) / stddev_all(x0c.data(), count));
                const float r = p.guidance_rescale;
                for (size_t j = 0; j < count; ++j) {
                    const float x0 = r * (x0c[j] * ratio) + (1.0f - r) * x0c[j];
                    vcfg[j] = (a * x[j] - x0) / b;
                }
            }
            v = vcfg.data();
        }
        const float dt = (float)(t - tp);
        for (size_t j = 0; j < count; ++j) x[j] = x[j] - dt * v[j];
        if (p.on_step) p.on_step(i, x, count);
    }
    return true;
}

}  // namespace

bool flow_sample(const FlowSampler& p, size_t count, const FlowVelocity& vel, float* x) {
    return flow_sample_impl(p, count, vel, x);
}

void ss_rope_table(float* out) {
    // RotaryPositionEmbedder(head_dim, 3) on the grid (x, y, z) in row-major order: 21 frequencies per axis (axis-major), then the
    // identity pair (the head dim's 64 pairs)
    const int freq_dim = kHeadDim / 2 / 3;
    float freqs[64];
    for (int f = 0; f < freq_dim; ++f) freqs[f] = (float)(1.0 / std::pow(10000.0, (double)f / (double)freq_dim));
    for (int x = 0; x < kSsRes; ++x)
        for (int y = 0; y < kSsRes; ++y)
            for (int z = 0; z < kSsRes; ++z) {
                const float pos[3] = {(float)x, (float)y, (float)z};
                float* row = out + (((size_t)x * kSsRes + y) * kSsRes + z) * (kHeadDim / 2) * 2;
                for (int j = 0; j < 3 * freq_dim; ++j) {
                    const float ang = pos[j / freq_dim] * freqs[j % freq_dim];
                    row[2 * j] = std::cos(ang);
                    row[2 * j + 1] = std::sin(ang);
                }
                row[2 * (kHeadDim / 2 - 1)] = 1.0f;  // the identity pair
                row[2 * (kHeadDim / 2 - 1) + 1] = 0.0f;
            }
}

struct ShapePipeline::Impl {
    std::unique_ptr<SsFlow> ss;
    std::unique_ptr<SsDecoder> ss_dec;
    std::unique_ptr<SlatFlow> slat;
    std::unique_ptr<SlatFlow> slat_hr;  // the 1024 flow (the cascade), optional
    std::unique_ptr<ShapeDecoder> shape_dec;
};

ShapePipeline::ShapePipeline(const std::string& ss_flow, const std::string& ss_dec, const std::string& slat_flow,
                             const std::string& shape_dec, const std::string& slat_flow_hr)
    : p_(new Impl) {
    p_->ss.reset(new SsFlow(ss_flow));
    if (!p_->ss->ok()) {
        error_ = "sparse-structure flow: " + p_->ss->error();
        return;
    }
    p_->ss_dec.reset(new SsDecoder(ss_dec));
    if (!p_->ss_dec->ok()) {
        error_ = "sparse-structure decoder: " + p_->ss_dec->error();
        return;
    }
    p_->slat.reset(new SlatFlow(slat_flow));
    if (!p_->slat->ok()) {
        error_ = "shape-SLAT flow: " + p_->slat->error();
        return;
    }
    p_->shape_dec.reset(new ShapeDecoder(shape_dec));
    if (!p_->shape_dec->ok()) {
        error_ = "shape decoder: " + p_->shape_dec->error();
        return;
    }
    if (!slat_flow_hr.empty()) {
        p_->slat_hr.reset(new SlatFlow(slat_flow_hr));
        if (!p_->slat_hr->ok()) {
            error_ = "1024 shape-SLAT flow: " + p_->slat_hr->error();
            return;
        }
    }
    ok_ = true;
}

ShapePipeline::~ShapePipeline() = default;

bool ShapePipeline::sample_sparse_structure(const float* cond, int Lc, const NoiseFn& noise, const FlowSampler& p,
                                            std::vector<float>& z) {
    error_.clear();
    if (!ok_) {
        error_ = "the pipeline is not open";
        return false;
    }
    Impl& m = *p_;
    const std::vector<float> zeros((size_t)Lc * kCond, 0.0f);  // the negative condition (get_cond: zeros_like)
    std::vector<float> rope((size_t)kSsRes * kSsRes * kSsRes * (kHeadDim / 2) * 2);
    ss_rope_table(rope.data());
    const size_t ss_count = (size_t)kSsC * kSsRes * kSsRes * kSsRes;
    z.assign(ss_count, 0.0f);
    noise(z.data(), ss_count);
    auto ss_vel = [&](const float* x, float t, bool neg, float* v) {
        const bool ok = m.ss->forward(x, t, neg ? zeros.data() : cond, Lc, rope.data(), v);
        if (!ok) error_ = "sparse-structure flow: " + m.ss->error();
        return ok;
    };
    if (!flow_sample(p, ss_count, ss_vel, z.data())) {
        if (error_.empty()) error_ = "sparse-structure sampler failed";
        return false;
    }
    return true;
}

bool ShapePipeline::occupancy_32(const float* z, std::vector<int32_t>& coords) {
    Impl& m = *p_;
    std::vector<float> logits((size_t)kSsDec * kSsDec * kSsDec);
    if (!m.ss_dec->forward(z, logits.data())) {
        error_ = "sparse-structure decoder: " + m.ss_dec->error();
        return false;
    }
    coords.clear();
    for (int x = 0; x < kOcc; ++x)
        for (int y = 0; y < kOcc; ++y)
            for (int zz = 0; zz < kOcc; ++zz) {
                bool occupied = false;
                for (int dx = 0; dx < 2 && !occupied; ++dx)
                    for (int dy = 0; dy < 2 && !occupied; ++dy)
                        for (int dz = 0; dz < 2 && !occupied; ++dz)
                            occupied = logits[((size_t)(2 * x + dx) * kSsDec + (2 * y + dy)) * kSsDec + (2 * zz + dz)] > 0;
                if (occupied) {
                    coords.push_back(0);
                    coords.push_back(x);
                    coords.push_back(y);
                    coords.push_back(zz);
                }
            }
    if (coords.empty()) {
        error_ = "no occupied voxels";
        return false;
    }
    return true;
}

bool ShapePipeline::decode_mesh(const std::vector<int32_t>& coords, const std::vector<float>& slat, int grid, const ShapeConfig& cfg,
                                ShapeResult& out) {
    Impl& m = *p_;
    const int N = (int)(coords.size() / 4);
    auto t0 = std::chrono::steady_clock::now();
    // the decoder's voxels: a buffer guess of 256 per input voxel; when it is too small the decoder reports its count and runs once more
    int cap = 256 * N, M = 0;
    out.dec_feats.resize((size_t)cap * 7);
    out.dec_coords.resize((size_t)cap * 4);
    out.subs.clear();
    bool dec_ok = m.shape_dec->forward(coords.data(), N, slat.data(), out.dec_feats.data(), out.dec_coords.data(), &M, cap, nullptr, 0,
                                       &out.subs);
    if (!dec_ok && M > cap) {
        cap = M;
        out.dec_feats.resize((size_t)cap * 7);
        out.dec_coords.resize((size_t)cap * 4);
        out.subs.clear();
        dec_ok = m.shape_dec->forward(coords.data(), N, slat.data(), out.dec_feats.data(), out.dec_coords.data(), &M, cap, nullptr, 0,
                                      &out.subs);
    }
    if (!dec_ok) {
        error_ = "shape decoder: " + m.shape_dec->error();
        return false;
    }
    out.dec_feats.resize((size_t)M * 7);
    out.dec_coords.resize((size_t)M * 4);
    out.seconds_dec = seconds_since(t0);

    t0 = std::chrono::steady_clock::now();
    std::string merr;
    if (!dual_grid_to_mesh(out.dec_coords.data(), out.dec_feats.data(), M, grid, -0.5f, 0.5f, cfg.margin, out.verts, out.faces, merr)) {
        error_ = "mesh: " + merr;
        return false;
    }
    if (cfg.hole_perimeter > 0) {
        FillStats fs;
        fill_small_holes(out.verts, out.faces, cfg.hole_perimeter, fs);
        out.holes_filled = fs.filled;
    }
    out.seconds_mesh = seconds_since(t0);
    return true;
}

bool ShapePipeline::run(const float* cond, int Lc, const NoiseFn& noise, const ShapeConfig& cfg, ShapeResult& out) {
    error_.clear();
    if (!ok_) {
        error_ = "the pipeline is not open";
        return false;
    }
    if (Lc <= 0) {
        error_ = "no image features";
        return false;
    }
    Impl& m = *p_;
    if (m.slat) m.slat->set_progress(cfg.progress);
    if (m.shape_dec) m.shape_dec->set_progress(cfg.progress);
    const std::vector<float> zeros((size_t)Lc * kCond, 0.0f);  // the negative condition (get_cond: zeros_like)

    // 1. the sparse-structure flow, sampled from its noise; 2. its decoder and the occupied voxels at 32^3
    auto t0 = std::chrono::steady_clock::now();
    if (!sample_sparse_structure(cond, Lc, noise, cfg.ss, out.ss_latent)) return false;
    out.seconds_ss = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    if (!occupancy_32(out.ss_latent.data(), out.coords)) return false;
    out.seconds_ss_dec = seconds_since(t0);
    const int N = (int)(out.coords.size() / 4);

    // 3. the shape-SLAT flow on those voxels, sampled from its noise, then de-normalised
    t0 = std::chrono::steady_clock::now();
    const size_t slat_count = (size_t)N * kLatent;
    std::vector<float> s(slat_count);
    noise(s.data(), slat_count);
    auto slat_vel = [&](const float* x, float t, bool neg, float* v) {
        const bool ok = m.slat->forward(out.coords.data(), N, x, t, neg ? zeros.data() : cond, Lc, v);
        if (!ok) error_ = "shape-SLAT flow: " + m.slat->error();
        return ok;
    };
    if (!flow_sample(cfg.slat, slat_count, slat_vel, s.data())) {
        if (error_.empty()) error_ = "shape-SLAT sampler failed";
        return false;
    }
    out.slat.resize(slat_count);
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < kLatent; ++c) out.slat[(size_t)i * kLatent + c] = s[(size_t)i * kLatent + c] * cfg.slat_std[c] + cfg.slat_mean[c];
    out.seconds_slat = seconds_since(t0);

    // 4. the shape decoder and the mesh at the decoder's grid
    return decode_mesh(out.coords, out.slat, cfg.mesh_grid, cfg, out);
}

void cascade_voxels(const std::vector<int32_t>& coords512, std::vector<int32_t>& coords64) {
    std::vector<std::array<int32_t, 4>> q(coords512.size() / 4);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i][0] = coords512[i * 4];
        for (int a = 0; a < 3; ++a) {
            const float c = (float)coords512[i * 4 + 1 + a];
            q[i][1 + a] = (int32_t)((c + 0.5f) / 512.0f * 64.0f);
        }
    }
    std::sort(q.begin(), q.end());
    q.erase(std::unique(q.begin(), q.end()), q.end());
    coords64.resize(q.size() * 4);
    for (size_t i = 0; i < q.size(); ++i)
        for (int a = 0; a < 4; ++a) coords64[i * 4 + a] = q[i][a];
}

bool ShapePipeline::run_cascade(const float* cond_lr, int Lc_lr, const float* cond_hr, int Lc_hr, const NoiseFn& noise,
                                const ShapeConfig& cfg, ShapeResult& out) {
    error_.clear();
    if (!ok_) {
        error_ = "the pipeline is not open";
        return false;
    }
    if (!p_->slat_hr) {
        error_ = "no 1024 shape-SLAT flow was opened";
        return false;
    }
    if (Lc_lr <= 0 || Lc_hr <= 0) {
        error_ = "no image features";
        return false;
    }
    Impl& m = *p_;
    if (m.slat) m.slat->set_progress(cfg.progress);
    if (m.slat_hr) m.slat_hr->set_progress(cfg.progress);
    if (m.shape_dec) m.shape_dec->set_progress(cfg.progress);
    const std::vector<float> zeros_lr((size_t)Lc_lr * kCond, 0.0f), zeros_hr((size_t)Lc_hr * kCond, 0.0f);

    // 1. the sparse-structure flow and its occupied voxels at 32^3 (as run)
    auto t0 = std::chrono::steady_clock::now();
    if (!sample_sparse_structure(cond_lr, Lc_lr, noise, cfg.ss, out.ss_latent)) return false;
    out.seconds_ss = seconds_since(t0);
    t0 = std::chrono::steady_clock::now();
    if (!occupancy_32(out.ss_latent.data(), out.coords)) return false;
    out.seconds_ss_dec = seconds_since(t0);
    const int N = (int)(out.coords.size() / 4);

    // 2. the 512 shape-SLAT flow on the 32^3 voxels, de-normalised (the cascade's low-resolution pass)
    t0 = std::chrono::steady_clock::now();
    const size_t lr_count = (size_t)N * kLatent;
    std::vector<float> s(lr_count);
    noise(s.data(), lr_count);
    auto lr_vel = [&](const float* x, float t, bool neg, float* v) {
        const bool ok = m.slat->forward(out.coords.data(), N, x, t, neg ? zeros_lr.data() : cond_lr, Lc_lr, v);
        if (!ok) error_ = "shape-SLAT flow (512): " + m.slat->error();
        return ok;
    };
    if (!flow_sample(cfg.slat, lr_count, lr_vel, s.data())) {
        if (error_.empty()) error_ = "shape-SLAT sampler failed";
        return false;
    }
    out.slat.resize(lr_count);
    for (int i = 0; i < N; ++i)
        for (int c = 0; c < kLatent; ++c) out.slat[(size_t)i * kLatent + c] = s[(size_t)i * kLatent + c] * cfg.slat_std[c] + cfg.slat_mean[c];
    out.seconds_slat = seconds_since(t0);

    // 3. the upsampled coordinates: the decoder's first four levels (the 512 grid), quantised to the 64^3 grid as python does
    //    ((c + 0.5) / 512 * 64, truncated), then the unique voxels in lexicographic order
    std::vector<int32_t> up;
    int M512 = 0, cap = 64 * N;
    up.resize((size_t)cap * 4);
    bool up_ok = m.shape_dec->forward(out.coords.data(), N, out.slat.data(), nullptr, up.data(), &M512, cap, nullptr, 4);
    if (!up_ok && M512 > cap) {
        cap = M512;
        up.resize((size_t)cap * 4);
        up_ok = m.shape_dec->forward(out.coords.data(), N, out.slat.data(), nullptr, up.data(), &M512, cap, nullptr, 4);
    }
    if (!up_ok) {
        error_ = "shape decoder (upsample): " + m.shape_dec->error();
        return false;
    }
    cascade_voxels(up, out.hr_coords);
    const int M64 = (int)(out.hr_coords.size() / 4);

    // 4. the 1024 shape-SLAT flow on the 64^3 voxels, de-normalised
    t0 = std::chrono::steady_clock::now();
    const size_t hr_count = (size_t)M64 * kLatent;
    std::vector<float> h(hr_count);
    noise(h.data(), hr_count);
    auto hr_vel = [&](const float* x, float t, bool neg, float* v) {
        const bool ok = m.slat_hr->forward(out.hr_coords.data(), M64, x, t, neg ? zeros_hr.data() : cond_hr, Lc_hr, v);
        if (!ok) error_ = "shape-SLAT flow (1024): " + m.slat_hr->error();
        return ok;
    };
    if (!flow_sample(cfg.slat, hr_count, hr_vel, h.data())) {
        if (error_.empty()) error_ = "shape-SLAT sampler (1024) failed";
        return false;
    }
    out.hr_slat.resize(hr_count);
    for (int i = 0; i < M64; ++i)
        for (int c = 0; c < kLatent; ++c) out.hr_slat[(size_t)i * kLatent + c] = h[(size_t)i * kLatent + c] * cfg.slat_std[c] + cfg.slat_mean[c];
    out.seconds_hr = seconds_since(t0);

    // 5. the shape decoder on the 64^3 voxels and the mesh at the 1024 grid
    return decode_mesh(out.hr_coords, out.hr_slat, cfg.mesh_grid, cfg, out);
}

}  // namespace trellis
