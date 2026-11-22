/* Qwen-Image's VAE decoder (the Wan-style 3D causal autoencoder, one frame: the causal convolutions see no history, so their
 * current-frame taps are the only ones used; the time upsampling does nothing on the first frame) on the device, AGENTS.md
 * rules: the BF16 weights of the ComfyUI file read in place, the activations in shared memory. */
#ifndef QWENIMAGE_VAE_H
#define QWENIMAGE_VAE_H

#include "zimage_dit.h"

namespace qwenimage {

/* latent [16][32][32] (the sampler's, channel-major, before the latent normalisation) -> out [3][256][256] (channel-major, clamped to
 * [-1, 1]); the normalisation (latents * std + mean, the pipeline's) is applied first. Returns false on an error. */
bool dev_vae_decode(const zimage::SafeTensors& st, const float* latent, float* out);

}  // namespace qwenimage

#endif /* QWENIMAGE_VAE_H */
