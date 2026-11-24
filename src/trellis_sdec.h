#pragma once
// The TRELLIS.2 shape decoder's sparse UNet (FlexiDualGridVaeDecoder's SparseUnetVaeDecoder, the checkpoint
// shape_dec_next_dc_f16c32_fp16) on the device: from_latent, five levels of ConvNeXt blocks with channel-to-spatial
// upsampling between them, then the output norm and layer. The mesh step (dual grid to mesh) is not here.
// The fp16 weights are read in place; the norm gains and the biases are converted to fp32 once at open.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace trellis {

class ShapeDecoder {
public:
    explicit ShapeDecoder(const std::string& safetensors_path);
    ~ShapeDecoder();
    ShapeDecoder(const ShapeDecoder&) = delete;
    ShapeDecoder& operator=(const ShapeDecoder&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    // coords: [n][4] int32 (batch, x, y, z) of the latent voxels; feats: [n][32]. The output: out_coords [M][4] and
    // out_feats [M][7] (the vertices, intersection and quad weights per voxel), in the order of the python op.
    // stage (optional) gets each named intermediate: "sdec_l0" (the level 0 output: feats [M0][512], coords [M0][4]).
    using StageFn = std::function<void(const char* name, const float* feats, const int32_t* coords, size_t n, size_t C)>;
    // upsample_times > 0: only the coordinates after that many channel-to-spatial blocks (python's upsample(x, upsample_times): the
    // voxels the cascade's high-resolution pass starts from); out_feats is not written then.
    bool forward(const int32_t* coords, int n, const float* feats, float* out_feats, int32_t* out_coords, int* out_m,
                 int out_capacity, const StageFn& stage = nullptr, int upsample_times = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool ok_ = false;
    std::string error_;
};

}  // namespace trellis
