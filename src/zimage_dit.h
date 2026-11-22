/* Z-Image-Turbo's diffusion transformer (S3-DiT) in the engine, first the host (CPU) path: the weights are read from the
 * ComfyUI-format safetensors file (BF16, names as the file has them: x_embedder.weight, cap_embedder.1.weight, ...). The
 * stages so far are the timestep embedder, the patch embedding and the caption embedder. The host functions are the reference-only
 * path (the oracle check); the dev_ functions are the device path (AGENTS.md: the weights registered and read in place, the
 * activations in shared memory). tests/zimage_dit_verify.cpp checks both against diffusers' ZImageTransformer2DModel. */
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
    /* the tensor as float (BF16 converted), its shape; false if absent (host path) */
    bool get(const std::string& name, std::vector<float>& out, std::vector<size_t>& shape) const;
    /* device path: the mapping registered for the GPU (read in place, no copy); false if the registration fails */
    bool register_device() const;
    /* a BF16 tensor's bits in the mapping (read in place by the GPU), its shape; false if absent or not BF16 */
    bool bf16_in_place(const std::string& name, const uint16_t*& ptr, std::vector<size_t>& shape) const;
    /* an F8_E4M3 tensor's bytes in the mapping (read in place by the GPU's FP8 linear), its shape; false if absent or not F8 */
    bool fp8_in_place(const std::string& name, const uint8_t*& ptr, std::vector<size_t>& shape) const;
    /* an F32 tensor's floats in the mapping (read in place by the GPU) */
    bool f32_in_place(const std::string& name, const float*& ptr, std::vector<size_t>& shape) const;

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

/* ---- The device path (AGENTS.md, Architecture): the file's mapping is registered (register_device) and its BF16
 * weights are read in place by the GPU's GEMM (sddev::linear_bf16); the activations are in sddev::alloc memory (shared
 * with the CPU, no copies); the host only writes small vectors into that memory (the timestep's frequencies, the
 * patches). Each returns false on an error (its message on stderr). */

/* the timestep embedding into out256 (sddev::alloc memory, 256 floats) */
bool dev_timestep_embed(const SafeTensors& st, float t, float* out256);

/* the patch embedding of a latent [C][1][H][W] (p = 2) into *out ([N][3840] in sddev::alloc memory; N set) */
bool dev_patch_embed(const SafeTensors& st, const float* latent, int C, int H, int W, float** out, int& N);

/* the caption embedder (RMS norm on the GPU, then the linear) of cap [L][2560] into *out ([Lpad][3840], Lpad = L padded
 * to 32 by repeating the last row) */
bool dev_caption_embed(const SafeTensors& st, const float* cap, int L, float** out, int& Lpad);

/* The RoPE table of the image tokens of an Ht x Wt patch grid (ids (start, h, w) as the reference: the time axis is start):
 * fr [N][64][2] = (cos, sin) per complex pair; axes (32, 48, 48) at theta 256 */
void rope_table_image(int Ht, int Wt, int start, float* fr);

/* the RoPE table of N tokens with ids (ids [N][3]: the time axis, h, w) */
void rope_table_ids(const int* ids, int N, float* fr);

/* the caption's pad rows (the rows from L on, of the padded [Lpad][3840] in shared memory) take the learned pad token
 * (cap_pad_token, BF16 [1][3840]) as the reference does before the refiner */
bool dev_caption_pad_rows(const SafeTensors& st, float* cap, int L, int Lpad);

/* One noise refiner block (the modulated attention and SwiGLU blocks, as the reference's ZImageTransformerBlock) on the
 * device: x [N][3840] in shared memory, updated in place; t256: the timestep embedding (shared memory); fr: the RoPE table
 * of the N tokens. prefix: "noise_refiner.0" (the file's names). */
bool dev_refiner_block(const SafeTensors& st, const std::string& prefix, float* x, int N, const float* t256, const float* fr,
                       bool modulated = true);


/* The output: the final layer (LayerNorm, the scale 1 + Linear(SiLU(t)), the linear to 64) over the image rows of J (the
 * first 256 rows of the joint sequence, [288][3840]), then unpatchify into lat [16][32][32] (host memory, the result) */
bool dev_output(const SafeTensors& st, const float* J, const float* t256, float* lat);

/* One forward of the DiT on the device: lat [16][32][32] (the latent: the sampler's), cap [L][2560] the caption's embeddings
 * (L 1 to 32; the rows past L take the pad token), t the model's timestep (1 - sigma), vel [16][32][32] the velocity (host) */
bool dev_dit_forward(const SafeTensors& st, const float* lat, const float* cap, int L, float t, float* vel);

/* The sampler: the flow-matching Euler steps from the noise [16][32][32]; sigmas linspace(1, 1/steps, steps) shifted by shift,
 * then 0; each step's velocity is negated (the pipeline's convention) and x += (sigma_next - sigma) * (-velocity). out: the final
 * latent [16][32][32] */
bool dev_sample(const SafeTensors& st, const float* noise, const float* cap, int L, int steps, float shift, float* out);

}  // namespace zimage

#endif /* ZIMAGE_DIT_H */
