/* Z-Image-Turbo's diffusion transformer (S3-DiT) in the engine, first the host (CPU) path: the weights are read from the
 * ComfyUI-format safetensors file (BF16, names as the file has them: x_embedder.weight, cap_embedder.1.weight, ...). The
 * stages so far are the timestep embedder, the patch embedding and the caption embedder; each is checked against the
 * reference (diffusers' ZImageTransformer2DModel) by tests/zimage_dit_verify.cpp. */
#ifndef ZIMAGE_DIT_H
#define ZIMAGE_DIT_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace zimage {

/* A safetensors file, mapped: the tensors by name (BF16 or F32, read as float). */
class SafeTensors {
public:
    explicit SafeTensors(const std::string& path);
    ~SafeTensors();
    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;
    bool ok() const { return base_ != nullptr; }
    const std::string& error() const { return error_; }
    bool has(const std::string& name) const { return t_.count(name) != 0; }
    /* the tensor as float (BF16 converted), its shape; false if absent */
    bool get(const std::string& name, std::vector<float>& out, std::vector<size_t>& shape) const;

private:
    struct Entry { std::string dtype; std::vector<size_t> shape; size_t begin, end; };
    std::map<std::string, Entry> t_;
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
    size_t data_ = 0;
    std::string error_;
};

/* y[t][o] = b[o] + sum_i x[t][i] W[o][i]: x [T][in] (float), W [out][in], b [out] (both float: converted once by the caller) */
void linear(const float* x, int T, int in, const std::vector<float>& W, int out, const std::vector<float>& b, float* y);

/* RMSNorm over the last dimension: y = x / sqrt(mean(x^2) + eps) * w; x [T][C] */
void rms_norm(const float* x, int T, int C, const std::vector<float>& w, float eps, float* y);

/* The timestep embedding (cos, sin of t * freq) of width dim (256), then Linear(256, 1024), SiLU, Linear(1024, 256). */
void timestep_embed(const SafeTensors& st, float t, float* out256);

/* The patches of a latent [C][F = 1][H][W] (p = 2): tokens [N = (H/p)(W/p)][p*p*C], each token ordered (ph, pw, c) */
void patchify(const float* latent, int C, int H, int W, int p, std::vector<float>& tokens);

/* The caption embedder: RMSNorm(2560), then Linear(2560, 3840). cap [L][2560] padded to a multiple of 32 by repeating
 * its last row (as the reference does), out [Lpad][3840] */
void caption_embed(const SafeTensors& st, const float* cap, int L, std::vector<float>& out, int& Lpad);

/* The image's patch embedding: Linear(64, 3840) on the patches */
void patch_embed(const SafeTensors& st, const std::vector<float>& tokens, int N, std::vector<float>& out);

}  // namespace zimage

#endif /* ZIMAGE_DIT_H */
