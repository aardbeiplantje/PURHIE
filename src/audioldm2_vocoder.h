/* AudioLDM 2's vocoder (vocoder/, SpeechT5's HiFi-GAN): the mel [T][64] to the waveform: conv_pre (7), five stages of leaky ReLU,
 * a transposed convolution (rates 5, 4, 2, 2, 2) and three residual blocks averaged (kernels 3, 7, 11; dilations 1, 3, 5), then a
 * leaky ReLU (slope 0.01), conv_post (7) and tanh. The device path: the fp32 weights read in place; the mel's transpose on the host.
 * Stage check: tests/audioldm2_vocoder_verify.cpp against tools/audioldm2_stage_vocoder.py. */
#ifndef AUDIOLDM2_VOCODER_H
#define AUDIOLDM2_VOCODER_H

#include "zimage_dit.h"

#include <vector>

namespace audioldm2 {

/* mel [T][64] (frames of 64 bins) -> wav (the samples: T * 160 + the transposed convolutions' edges). The mapping of st is registered
 * for the device by the caller. False on any error. */
bool vocoder_forward(const zimage::SafeTensors& st, const float* mel, int T, std::vector<float>& wav);

}  // namespace audioldm2

#endif  /* AUDIOLDM2_VOCODER_H */
