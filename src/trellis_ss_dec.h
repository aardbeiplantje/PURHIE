#pragma once
// The TRELLIS.2 sparse-structure decoder (SparseStructureDecoder, ss_dec_conv3d_16l8, from microsoft/TRELLIS-image-large) on
// the device: the stages of trellis2/models/sparse_structure_vae.py's decoder. The fp16 convolution weights and the fp32
// norms and input and output convolutions are read in place; the fp16 biases are converted to fp32 once at open.

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

namespace trellis {

class SsDecoder {
public:
    explicit SsDecoder(const std::string& safetensors_path);
    ~SsDecoder();
    SsDecoder(const SsDecoder&) = delete;
    SsDecoder& operator=(const SsDecoder&) = delete;

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }

    // z: [8][16][16][16] (the flow's output, channel-major); out: [64][64][64] (the decoded logits).
    // When stage is set, it is called after each named stage with the token-major fp32 host copy: "dec_in" [4096][512],
    // "dec_mid" [4096][512], "dec_b2" [32768][128], "dec_b5" [262144][32], "dec_out" [262144][1].
    bool forward(const float* z, float* out, const std::function<void(const char*, const float*, size_t)>& stage = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
    bool ok_ = false;
    std::string error_;
};

}  // namespace trellis
