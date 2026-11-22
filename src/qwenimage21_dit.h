/* Qwen-Image 2.1's diffusion transformer (32 single-stream blocks, width 4096, 32 heads of 128; diffusers' QwenImage21Transformer2DModel)
 * from the ComfyUI int8 ConvRot file (the linears read in place as int8 with the row scales and the Hadamard rotation; the other
 * weights BF16 in place), on the device. Text-to-image layout: the text tokens (the encoder's output, causal among themselves), then
 * the image tokens (each the packed latent of the 16 x 16 grid, attending to everything); the text takes the t = 0 modulation row. */
#ifndef QWENIMAGE21_DIT_H
#define QWENIMAGE21_DIT_H

#include "zimage_dit.h"

namespace qwenimage21 {

/* packed [gh * gw][64] (the image tokens), enc [L][4096] (the text tokens), t the timestep in [0, 1]; out [gh * gw][64] (host).
 * Returns false on an error. */
bool dev_dit21_forward(const zimage::SafeTensors& st, const float* packed, int gh, int gw, const float* enc, int L, float t, float* out);

}  // namespace qwenimage21

#endif /* QWENIMAGE21_DIT_H */
