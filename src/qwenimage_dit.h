/* Qwen-Image's diffusion transformer (MMDiT: 60 double-stream blocks, dim 3072, 24 heads of 128) from the ComfyUI FP8 file,
 * on the device (AGENTS.md rules: the E4M3 weights read in place, the activations in shared memory). The image and text streams
 * have their own projections; the attention is joint over [text, image] tokens with one RoPE (axes 16, 56, 56 at theta 10000,
 * centred image positions, text positions after the image's largest half-grid). */
#ifndef QWENIMAGE_DIT_H
#define QWENIMAGE_DIT_H

#include "zimage_dit.h"

namespace qwenimage {

/* One forward: packed [N][64] (N = gh * gw patches of 2 x 2 latent cells, the pipeline's packing), enc [L][3584] (the text
 * encoder's output, all L tokens), t the model's timestep (sigma-based, in [0, 1]: the pipeline passes timestep / 1000), out
 * [N][64] (host memory). Returns false on an error. */
bool dev_dit_forward(const zimage::SafeTensors& st, const float* packed, int gh, int gw, const float* enc, int L, float t, float* out);

}  // namespace qwenimage

#endif /* QWENIMAGE_DIT_H */
