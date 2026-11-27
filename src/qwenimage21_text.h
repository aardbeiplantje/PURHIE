/* Qwen-Image 2.1's text encoder: the Qwen3-VL 8B language model (36 layers, width 4096, 32 query and 8 key/value heads of 128, QK
 * norms, rotate-half RoPE at theta 5e6, SwiGLU 12288) from the ComfyUI int8 ConvRot file (every linear and the embedding table
 * read in place as int8 with the row scales and the Hadamard rotation); with a condition image, its vision tower's tokens at the
 * image pads, the M-RoPE positions and the deepstack features (qwenimage21_vision.h). The output is the last decoder layer's hidden state
 * before the final norm (the pipeline's hidden_states[-1]) of every token; the caller drops the system tokens. */
#ifndef QWENIMAGE21_TEXT_H
#define QWENIMAGE21_TEXT_H

#include "zimage_dit.h"

namespace qwenimage21 {

/* ids [T] (1 to 8192 tokens) -> out [T][4096] (host memory). Returns false on an error. */
bool dev_text21_encode(const zimage::SafeTensors& st, const int* ids, int T, float* out);

/* The edit's input (Qwen3VLModel with images): the image pad tokens (id 151655) take the vision tower's merged tokens img [n_img][4096]
 * in their order (the ids' pads must number n_img); pos [3][T] the M-RoPE positions (temporal, height, width: mrope.h's, interleaved
 * over the 64 frequencies as mrope_section [24, 20, 20] with mrope_interleaved); ds [3][n_img][4096] the deepstack features, added at
 * the pads after layers 0, 1 and 2. */
bool dev_text21_encode_vl(const zimage::SafeTensors& st, const int* ids, int T, const int* pos, const float* img, const float* ds,
                          int n_img, float* out);

}  // namespace qwenimage21

#endif /* QWENIMAGE21_TEXT_H */
