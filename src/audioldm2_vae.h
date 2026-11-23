/* AudioLDM 2's audio VAE decoder (vae/, AutoencoderKL: post_quant_conv, the decoder with 512, 256 and 128 channels, mid-block
 * attention, nearest upsampling): the UNet's latent [8][h][w] to the mel spectrogram [1][4h][4w]. The latent is divided by the
 * scaling factor first, as the pipeline does. The device path: the fp32 weights read in place. Stage check:
 * tests/audioldm2_vae_verify.cpp against tools/audioldm2_stage_vae.py. */
#ifndef AUDIOLDM2_VAE_H
#define AUDIOLDM2_VAE_H

#include "zimage_dit.h"

namespace audioldm2 {

/* z [8][h][w] (the UNet's output); mel [4h][4w] (one channel). h and w: any size >= 1. The mapping of st is registered for the device
 * by the caller. False on any error. */
bool vae_decode(const zimage::SafeTensors& st, const float* z, int h, int w, float* mel);

}  // namespace audioldm2

#endif  /* AUDIOLDM2_VAE_H */
