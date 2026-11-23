/* AudioLDM 2's UNet (unet/, AudioLDM2UNet2DConditionModel): 8 latent channels, blocks of 128, 256, 384, 640 channels; a time embedding
 * of 512; resnets with the time embedding added per channel; three transformers per resnet, each a self-attention, a cross-attention
 * to encoder_hidden_states (the GPT-2 tokens, 768 wide) and one to encoder_hidden_states_1 (the T5 tokens, 1024 wide); 8 heads.
 * The device path: the fp32 weights read in place; the time embedding on the host and the rest on the GPU. Stage check:
 * tests/audioldm2_unet_verify.cpp against tools/audioldm2_stage_unet.py. */
#ifndef AUDIOLDM2_UNET_H
#define AUDIOLDM2_UNET_H

#include "zimage_dit.h"

#include <cstdint>

namespace audioldm2 {

/* x [8][H][W] (any H, W >= 8: the sizes follow diffusers' upsample_size); t: the timestep; ctx1 [T1][768] (encoder_hidden_states); ctx2
 * [T2][1024] (encoder_hidden_states_1) with mask2 (T2 entries, 0 for a padding token; null: all real). out [8][H][W]. The mapping of st is
 * registered for the device by the caller. False on any error. */
bool unet_forward(const zimage::SafeTensors& st, const float* x, int H, int W, float t, const float* ctx1, int T1, const float* ctx2, int T2,
                  const int32_t* mask2, float* out);

}  // namespace audioldm2

#endif  /* AUDIOLDM2_UNET_H */
