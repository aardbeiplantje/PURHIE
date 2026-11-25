#pragma once
// The image encoder of the TRELLIS.2 pipeline: DINOv3 ViT-L/16 as TRELLIS's DinoV3FeatureExtractor runs it on the device. The patch
// embedding with the CLS and register tokens, 24 blocks (pre-norm attention with the rotary position embedding on the patch tokens,
// layer scale, pre-norm GELU MLP, layer scale), and a layer norm without affine on the output. The weights are read in place from a
// safetensors file of fp32 tensors: ComfyUI's copy (clip_vision/dino_v3_L_naf_fp32.safetensors) has the transformers names, with the
// block list as layer.N and the rotary periods as naf.image_encoder.rope.periods (rope_theta ** (k / 16)).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace trellis {

class DinoV3 {
public:
    explicit DinoV3(const std::string& safetensors_path);
    ~DinoV3();
    DinoV3(const DinoV3&) = delete;
    DinoV3& operator=(const DinoV3&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    // The tokens of a size x size image (size a multiple of 16): x [3][size][size], normalised with the ImageNet mean and std (the
    // pipeline's preprocessing: the resize to size is the caller's). out: [tokens(size)][1024]: the CLS token, the 4 registers, then the
    // patch tokens in row-major order (the pipeline keeps all of them as the image condition).
    bool forward(const float* x, int size, float* out);

    static int tokens(int size) { return 5 + (size / 16) * (size / 16); }

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool ok_ = false;
    std::string error_;
};

}  // namespace trellis
