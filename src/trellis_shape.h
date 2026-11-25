#pragma once
// The TRELLIS.2 image-to-shape path on the device: the shape half of the 512 pipeline (trellis2/pipelines/trellis2_image_to_3d.py,
// run() with pipeline_type '512'). The sparse-structure flow is sampled with classifier-free guidance on an interval (the
// samplers' FlowEulerGuidanceIntervalSampler), decoded, and its occupied voxels taken at 32^3 (any voxel of a 2x2x2 block at
// 64^3); the shape-SLAT flow samples the features of those voxels, which are de-normalised; the shape decoder's sparse UNet gives
// the dual-grid features; the mesh is o-voxel's flexible_dual_grid_to_mesh at the decoder's grid. The image features come in as
// cond ([Lc][1024]; the DINOv3 encoder is not on the engine yet); the negative condition is zeros, as in get_cond.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include "trellis_sdec.h"

#include <vector>

namespace trellis {

// One flow's sampler settings (pipeline.json's params; the time values are the sampler's own, in (0, 1]).
struct FlowSampler {
    int steps = 12;
    float guidance = 7.5f;           // guidance_strength
    float guidance_rescale = 0.7f;   // guidance_rescale (0.5 for the shape-SLAT flow)
    double rescale_t = 5.0;          // rescale_t (3.0 for the shape-SLAT flow)
    double interval_lo = 0.6;        // guidance_interval
    double interval_hi = 1.0;
    double sigma_min = 1e-5;         // the sampler's sigma_min
    // Debugging: called after each step with the state x (count floats) of that step (0-based index).
    std::function<void(int step, const float* x, size_t count)> on_step;
};

struct ShapeConfig {
    FlowSampler ss;                          // sparse_structure_sampler: 12 steps, 7.5, rescale 0.7, rescale_t 5.0
    FlowSampler slat;                        // shape_slat_sampler: 12 steps, 7.5, rescale 0.5, rescale_t 3.0
    int slat_lr_steps = 0;                   // the cascade's 512 shape-SLAT steps (0: slat.steps; the ComfyUI workflow uses 20)
    std::array<float, 32> slat_mean{};       // shape_slat_normalization
    std::array<float, 32> slat_std{};
    int mesh_grid = 512;                     // the decode's resolution (the 512 pipeline)
    float margin = 0.5f;                     // the dual vertex margin
    float hole_perimeter = 3e-2f;            // fill_holes: the boundary loops below this perimeter are closed (0 disables)
    bool progress = false;                   // stderr: each sampler step, flow block and decoder level as it finishes
};

struct ShapeResult {
    // wall-clock seconds of each stage: the sparse-structure sampling, its decoder, the shape-SLAT sampling, the shape decoder, the mesh
    double seconds_ss = 0, seconds_ss_dec = 0, seconds_slat = 0, seconds_dec = 0, seconds_mesh = 0, seconds_hr = 0;
    // the cascade: the 1024 flow's voxels (64^3, unique) and features; coords/slat/ss_latent/dec are the 512 stages
    std::vector<int32_t> hr_coords;   // [M][4]
    std::vector<float> hr_slat;       // [M][32] de-normalised
    // the shape decoder's subdivisions at its four levels (the texture decoder's guide): the voxels of the 512 (or 1024) decode
    std::vector<Subdivision> subs;
    int holes_filled = 0;  // the mesh's boundary loops closed by fill_holes (their new vertices follow the decoder's voxels)
    std::vector<float> ss_latent;     // [8][16][16][16] the sparse-structure flow's sample
    std::vector<int32_t> coords;      // [N][4] the occupied voxels at 32^3 (batch, x, y, z), in the python order
    std::vector<float> slat;          // [N][32] the shape latent, de-normalised
    std::vector<float> dec_feats;     // [M][7] the shape decoder's output (the dual vertex, the intersections, the split weight)
    std::vector<int32_t> dec_coords;  // [M][4] the decoder's voxels at mesh_grid
    std::vector<float> verts;         // [V][3] the mesh vertices
    std::vector<int32_t> faces;       // [T][3] the mesh faces
};

// Fills dst with count floats of noise; called for the sparse-structure latent (8 * 16^3) and then for the shape latent (N * 32,
// N the occupied voxels).
using NoiseFn = std::function<void(float* dst, size_t count)>;

// The sampler over a flat sample: x holds the noise on entry and the sample on exit. vel gives the velocity of x at the model's time
// (1000 times the sampler's t): the conditional one, or with negative = true the negative one.
using FlowVelocity = std::function<bool(const float* x, float t_model, bool negative, float* v)>;
bool flow_sample(const FlowSampler& p, size_t count, const FlowVelocity& vel, float* x);

// The cascade's high-resolution voxels: the 512-grid coordinates [N][4] quantised to 64^3 the way the python pipeline does it (the
// float (c + 0.5) / 512 * 64, truncated), then the unique voxels in lexicographic order (coords64: [M][4]).
void cascade_voxels(const std::vector<int32_t>& coords512, std::vector<int32_t>& coords64);

// The sparse-structure flow's RoPE table: [4096][64][2] (cos, sin) pairs of the 16^3 grid (RotaryPositionEmbedder, head dim 128).
void ss_rope_table(float* out);

class ShapePipeline {
public:
    // The four checkpoints: the sparse-structure flow and its decoder, the shape-SLAT flow (512) and the shape decoder.
    // slat_flow_hr: the 1024 shape-SLAT flow, for run_cascade (optional).
    ShapePipeline(const std::string& ss_flow, const std::string& ss_dec, const std::string& slat_flow, const std::string& shape_dec,
                  const std::string& slat_flow_hr = std::string());
    ~ShapePipeline();
    ShapePipeline(const ShapePipeline&) = delete;
    ShapePipeline& operator=(const ShapePipeline&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    // The 1024 cascade (the pipeline's '1024_cascade'): the sparse-structure flow and its occupied voxels as run; the 512 shape-SLAT
    // flow on those voxels; the decoder's first four levels give the voxels at 512, quantised to 64^3 (unique); the 1024 flow on
    // them; the decoder and the mesh at cfg.mesh_grid (1024). cond_lr: [Lc_lr][1024] (512 image features), cond_hr: [Lc_hr][1024].
    bool run_cascade(const float* cond_lr, int Lc_lr, const float* cond_hr, int Lc_hr, const NoiseFn& noise, const ShapeConfig& cfg,
                     ShapeResult& out);

    // The sparse-structure flow alone: the sample [8][16][16][16] from the noise (cond: [Lc][1024]).
    bool sample_sparse_structure(const float* cond, int Lc, const NoiseFn& noise, const FlowSampler& p, std::vector<float>& z);

    // cond: [Lc][1024] (the image features). out receives every stage (see ShapeResult).
    bool run(const float* cond, int Lc, const NoiseFn& noise, const ShapeConfig& cfg, ShapeResult& out);

private:
    bool occupancy_32(const float* z, std::vector<int32_t>& coords);
    bool decode_mesh(const std::vector<int32_t>& coords, const std::vector<float>& slat, int grid, const ShapeConfig& cfg, ShapeResult& out);
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool ok_ = false;
    std::string error_;
};

}  // namespace trellis
