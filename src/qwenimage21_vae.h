/* Qwen-Image 2.1's VAE decoder (diffusers AutoencoderKLQwenImage21, the residual variant: 64 latent channels, base width 144, channel
 * plan 1152, 1152, 1152, 576, 288, 144, the up blocks with a parameter-free duplicate-upsample shortcut; one frame, so the causal
 * temporal kernels see only the current frame and the time upsampling does nothing except its shortcut's slot) on the device, from the
 * ComfyUI BF16 file (read in place). The latent is channel-major [64][16][16] (the DiT's 16 x 16 tokens, patch 1). */
#ifndef QWENIMAGE21_VAE_H
#define QWENIMAGE21_VAE_H

#include "zimage_dit.h"

namespace qwenimage21 {

/* latent [64][16][16] (the DiT's, before the latent normalisation) -> out [4][256][256] (channel-major, clamped to [-1, 1]); the
 * normalisation (z * std + mean, the pipeline's) is applied first. Returns false on an error. */
bool dev_vae21_decode(const zimage::SafeTensors& st, const float* latent, float* out);

}  // namespace qwenimage21

#endif /* QWENIMAGE21_VAE_H */
