/* Z-Image's VAE decoder (the Flux autoencoder: 16 latent channels, the file's F32 weights) on the device, AGENTS.md rules:
 * the weights mapped and read in place (SafeTensors::f32_in_place), the activations in shared memory. The decoder's blocks:
 * conv_in, the mid block (ResNet, single-head attention, ResNet), four up levels (three ResNets each; nearest upsampling and a
 * 3x3 convolution after the first three), the output (GroupNorm, SiLU, the 3x3 convolution to 3 channels). */
#ifndef ZIMAGE_VAE_H
#define ZIMAGE_VAE_H

#include "zimage_dit.h"

namespace zimage {

/* latent [16][32][32] (the sampler's output, before the scaling) -> out [3][256][256] (host memory); the file's scaling
 * (0.3611) and shift (0.1159) are applied first, as the pipeline does before the decode. Returns false on an error. */
bool dev_vae_decode(const SafeTensors& st, const float* latent, float* out, int H0 = 32, int W0 = 32);

}  // namespace zimage

#endif /* ZIMAGE_VAE_H */
