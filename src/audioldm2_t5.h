/* AudioLDM 2's second text encoder: T5 (text_encoder_2/, the encoder of T5EncoderModel): 24 pre-norm blocks of 16 heads (64 wide),
 * RMS norms without bias, unscaled scores with the bucketed relative position bias (layer 0's table, shared by the blocks), the
 * gated GELU (gelu_new) feed-forward, a final RMS norm. The device path: the blocks on the GPU (the fp32 weights read in place);
 * the token embedding rows and the bias table's lookup on the host. Stage check: tests/audioldm2_t5_verify.cpp. */
#ifndef AUDIOLDM2_T5_H
#define AUDIOLDM2_T5_H

#include "zimage_dit.h"

#include <cstdint>

namespace audioldm2 {

/* ids: T tokens (no padding: one prompt). hidden [T][1024] (the encoder's last hidden state). The mapping of st is registered for the
 * device by the caller. False on any error. */
bool t5_encoder_forward(const zimage::SafeTensors& st, const int32_t* ids, int T, float* hidden);

}  // namespace audioldm2

#endif  /* AUDIOLDM2_T5_H */
