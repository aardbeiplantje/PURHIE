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
#include <vector>

namespace trellis {

// One level's subdivision logits (python's return_subs): the voxels of the level ([N][4]) and the 8 children's logits per voxel
// ([N][8]). The texture decoder keeps the children whose logit is positive.
struct Subdivision {
    std::vector<int32_t> coords;
    std::vector<float> logits;
};

class ShapeDecoder {
public:
    // out_channels: 7 (the shape decoder: the dual vertex, the intersections, the split weight) or 6 (the texture decoder);
    // guided: the texture decoder, whose channel-to-spatial blocks take their subdivisions from a shape decoder (no to_subdiv).
    explicit ShapeDecoder(const std::string& safetensors_path, int out_channels = 7, bool guided = false);
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
    // subs_out (optional, the shape decoder): receives each level's subdivision logits.
    bool forward(const int32_t* coords, int n, const float* feats, float* out_feats, int32_t* out_coords, int* out_m,
                 int out_capacity, const StageFn& stage = nullptr, int upsample_times = 0,
                 std::vector<Subdivision>* subs_out = nullptr);
    // The guided decoder (the texture decoder): the subdivisions of the shape decoder at its four upsampling levels (guide[i] are the
    // voxels of level i, which the decoder must have).
    bool forward_guided(const int32_t* coords, int n, const float* feats, const std::vector<Subdivision>& guide, float* out_feats,
                        int32_t* out_coords, int* out_m, int out_capacity);

    // progress (stderr): each decoder level as it finishes (off by default)
    void set_progress(bool on) { progress_ = on; }

private:
    bool run(const int32_t* coords, int n, const float* feats, float* out_feats, int32_t* out_coords, int* out_m, int out_capacity,
             const StageFn& stage, int upsample_times, const std::vector<Subdivision>* guide, std::vector<Subdivision>* subs_out);
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool progress_ = false;
    bool ok_ = false;
    std::string error_;
};

}  // namespace trellis
