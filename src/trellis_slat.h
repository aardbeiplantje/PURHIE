#pragma once
// The TRELLIS.2 shape-SLAT flow (slat_flow_img2shape_dit_1_3B_512, the bf16 checkpoint of TRELLIS.2-4B) on the device: the
// stages of trellis2/models/structured_latent_flow.py's forward over the occupied voxels (one sequence). The weights are read
// in place; the small vectors are converted to fp32 once at open. The RoPE table is built from the voxel coordinates.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace trellis {

class SlatFlow {
public:
    // in_channels: the features per voxel the flow takes (32: the shape flow; 64: the texture flow, whose input is the noise and the
    // normalised shape latent concatenated).
    explicit SlatFlow(const std::string& safetensors_path, int in_channels = 32);
    ~SlatFlow();
    SlatFlow(const SlatFlow&) = delete;
    SlatFlow& operator=(const SlatFlow&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    // n: the number of occupied voxels; coords: [n][4] int32 (batch, x, y, z) in the order of the tokens; feats: [n][32]
    // (the noisy latent per voxel); t: the timestep; cond: [Lc][1024] (the image features); out: [n][32] (the prediction).
    // stage (optional) gets the token-major fp32 host copy of each stage: "slat_h0" [n][1536], "slat_temb" [1536],
    // "slat_mod" [9216], "slat_block00", "slat_block01", "slat_block14", "slat_block28", "slat_block29" [n][1536],
    // "slat_out" [n][32].
    bool forward(const int32_t* coords, int n, const float* feats, float t, const float* cond, int Lc, float* out,
                 const std::function<void(const char*, const float*, size_t)>& stage = nullptr);

    // progress (stderr): each block's time with its attention share, as the blocks finish (off by default)
    void set_progress(bool on) { progress_ = on; }

private:
    struct Impl;
    bool progress_ = false;
    std::unique_ptr<Impl> p_;
    bool ok_ = false;
    std::string error_;
};

}  // namespace trellis
