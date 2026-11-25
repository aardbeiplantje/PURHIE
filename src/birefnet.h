#pragma once
// BiRefNet (background removal: the mask of the salient object) on the device, as the ComfyUI workflow's RemoveBackground and TRELLIS.2's
// rembg step run it: the Hugging Face model (ZhengPeng7/BiRefNet: the Swin-L backbone on the image and on its half-size copy,
// concatenated; the context skips; the squeeze block; the decoder with the image's patches, the deformable ASPP blocks, the gradient
// attention; 1024 x 1024 in, the logits of the mask out). The weights are read in place (F16, the ComfyUI file
// background_removal/birefnet.safetensors); the batch norms are scales and shifts (inference).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace trellis {

class BiRefNet {
public:
    explicit BiRefNet(const std::string& safetensors_path);
    ~BiRefNet();
    BiRefNet(const BiRefNet&) = delete;
    BiRefNet& operator=(const BiRefNet&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    // a stage's output for the checks (tests/birefnet_verify.cpp): the name of the python module it matches, its floats (channel-major
    // maps [C][H][W], host-visible), their count
    using StageFn = std::function<void(const char* name, const float* data, size_t count)>;

    // x: [3][1024][1024], the image resized to 1024 x 1024 and normalised (ImageNet mean and std); logits: [1024][1024]
    bool forward(const float* x, float* logits, const StageFn& stage = nullptr);

    // The mask of an image as the python wrapper makes it: rgb [h][w][3] resized to 1024 x 1024 (PIL's bilinear), normalised, the
    // forward, the sigmoid in 8 bits, resized back to w x h (PIL's bicubic). mask: [h][w] in [0, 1].
    bool mask(const uint8_t* rgb, int w, int h, std::vector<float>& mask);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool ok_ = false;
    std::string error_;
};

}  // namespace trellis
