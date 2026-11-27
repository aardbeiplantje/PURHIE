/* Qwen-Image 2.1's VAE decoder and encoder (diffusers AutoencoderKLQwenImage21, the residual variant: 64 latent channels, base width 144, channel
 * plan 1152, 1152, 1152, 576, 288, 144, the up blocks with a parameter-free duplicate-upsample shortcut; one frame, so the causal
 * temporal kernels see only the current frame and the time upsampling does nothing except its shortcut's slot) on the device, from the
 * ComfyUI BF16 file (read in place). The latent is channel-major [64][gh][gw] (the DiT's tokens, patch 1); the image is 16 gh x 16 gw. */
#ifndef QWENIMAGE21_VAE_H
#define QWENIMAGE21_VAE_H

#include "zimage_dit.h"

namespace qwenimage21 {

/* latent [64][gh][gw] (the DiT's, before the latent normalisation) -> out [4][16 gh][16 gw] (channel-major, clamped to [-1, 1]); the
 * normalisation (z * std + mean, the pipeline's) is applied first. Returns false on an error. */
bool dev_vae21_decode(const zimage::SafeTensors& st, const float* latent, int gh, int gw, float* out);

/* The encoder (the edit's condition image; the residual down blocks with the average-down shortcut, 96, 96, 192, 384, 768, 768):
 * image [4][H][W] (RGBA in [-1, 1], channel-major; H and W multiples of 16) -> latent [(H / 16) (W / 16)][64], the packed, normalised
 * mean ((mu - mean) / std: the pipeline's _encode_vae_image with argmax, then _pack_latents). Returns false on an error. */
bool dev_vae21_encode(const zimage::SafeTensors& st, const float* image, int H, int W, float* latent);

}  // namespace qwenimage21

#endif /* QWENIMAGE21_VAE_H */
