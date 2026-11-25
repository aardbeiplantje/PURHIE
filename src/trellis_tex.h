#pragma once
// The TRELLIS.2 texture path of the 512 pipeline (trellis2/pipelines/trellis2_image_to_3d.py: sample_tex_slat and decode_tex_slat)
// on the device. The texture flow samples a latent per shape voxel: its input is the noise and the normalised shape latent
// concatenated (64 channels), conditioned on the image features; the sampler runs with the conditional prediction alone (guidance
// strength 1). The texture decoder is the shape decoder's structure without predicted subdivisions: its channel-to-spatial blocks
// take the shape decoder's subdivisions. Its six outputs go through x * 0.5 + 0.5.

#include "trellis_sdec.h"
#include "trellis_shape.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trellis {

struct TexConfig {
    FlowSampler sampler;                             // tex_slat_sampler: 12 steps, guidance 1.0, rescale_t 3.0
    std::array<float, 32> shape_mean{}, shape_std{};  // shape_slat_normalization (the shape latent the flow is conditioned on)
    std::array<float, 32> tex_mean{}, tex_std{};      // tex_slat_normalization (the texture latent)
    bool progress = false;  // stderr: the texture flow's blocks and the decoder's levels as they finish
};

struct TexResult {
    std::vector<float> tex_slat;  // [N][32] the texture latent, de-normalised (the decoder's input)
    std::vector<float> raw;       // [N][6] the texture decoder's output
    std::vector<float> out;       // [N][6] raw * 0.5 + 0.5 (the pipeline's texture)
    std::vector<int32_t> coords;  // [M][4] the texture voxels: the decoder's output voxels (the shape decoder's output voxels)
    double seconds_flow = 0, seconds_dec = 0;
};

class TexturePipeline {
public:
    // tex_flow: the 512 texture flow (64 input channels); tex_dec: the texture decoder (6 outputs, guided).
    // tex_flow_hr: the 1024 texture flow (optional; the cascade's texture pass uses it with hr = true).
    TexturePipeline(const std::string& tex_flow, const std::string& tex_dec, const std::string& tex_flow_hr = std::string());
    ~TexturePipeline();
    TexturePipeline(const TexturePipeline&) = delete;
    TexturePipeline& operator=(const TexturePipeline&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    // The texture flow's single forward (a stage check): x64 is [n][64] (noise | normalised shape latent), out [n][32].
    bool flow_forward(const int32_t* coords, int n, const float* x64, const float* cond, int Lc, float t, float* out, bool hr = false);

    // The texture decoder alone: tex_slat [n][32] (de-normalised) on the voxels coords, guided by the shape decoder's subdivisions.
    bool decode(const int32_t* coords, int n, const float* tex_slat, const std::vector<Subdivision>& subs, TexResult& out);

    // The texture: coords [n][4] and shape_slat [n][32] (de-normalised) of the shape, its subdivisions subs (ShapeDecoder::forward's
    // subs_out), cond [Lc][1024] (the image features). The noise is drawn once: [n][32].
    bool run(const int32_t* coords, int n, const float* shape_slat, const std::vector<Subdivision>& subs, const float* cond, int Lc,
             const NoiseFn& noise, const TexConfig& cfg, TexResult& out, bool hr = false);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool ok_ = false;
    std::string error_;
};

// The end-to-end textured mesh: the shape (the sparse structure, the 512 shape latent, the shape decoder, the mesh), then the texture on
// the shape decoder's voxels, then each vertex's base colour: vertex i is made from the decoder's voxel i, and the texture decoder returns
// the same voxels in the same order, so its row i is the vertex's texture (base colour: channels 0 to 2, clamped to [0, 1]).
struct TexturedMesh {
    ShapeResult shape;          // the shape stages (the mesh, the voxels, the subdivisions)
    TexResult tex;              // the texture stages
    std::vector<float> colors;  // [V][3] the base colour of each vertex, in [0, 1]
};

// noise is called three times, in order: the sparse-structure latent, the shape latent, the texture latent. cond: [Lc][1024] (the image
// features, which the shape and the texture both take).
bool run_textured(ShapePipeline& shape, TexturePipeline& tex, const float* cond, int Lc, const NoiseFn& noise, const ShapeConfig& scfg,
                  const TexConfig& tcfg, TexturedMesh& out, std::string& error);

// The 1024 cascade with its texture pass: the cascade's shape (run_cascade), the 1024 texture flow on the 64^3 voxels (with the 1024
// image features cond_hr [Lc_hr][1024]) and the decoder's subdivisions of the 1024 decode, then the colours. noise is called four times:
// the sparse-structure latent, the 512 shape latent, the 1024 shape latent, the 1024 texture latent.
bool run_textured_cascade(ShapePipeline& shape, TexturePipeline& tex, const float* cond_lr, int Lc_lr, const float* cond_hr, int Lc_hr,
                          const NoiseFn& noise, const ShapeConfig& scfg, const TexConfig& tcfg, TexturedMesh& out, std::string& error);

}  // namespace trellis
