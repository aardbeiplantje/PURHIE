/* Qwen-Image 2.1's text encoder: the Qwen3-VL 8B language model (36 layers, width 4096, 32 query and 8 key/value heads of 128, QK
 * norms, rotate-half RoPE at theta 5e6, SwiGLU 12288) from the ComfyUI int8 ConvRot file (every linear and the embedding table
 * read in place as int8 with the row scales and the Hadamard rotation). The output is the last decoder layer's hidden state
 * before the final norm (the pipeline's hidden_states[-1]) of every token; the caller drops the system tokens. */
#ifndef QWENIMAGE21_TEXT_H
#define QWENIMAGE21_TEXT_H

#include "zimage_dit.h"

namespace qwenimage21 {

/* ids [T] (1 to 64 tokens) -> out [T][4096] (host memory). Returns false on an error. */
bool dev_text21_encode(const zimage::SafeTensors& st, const int* ids, int T, float* out);

}  // namespace qwenimage21

#endif /* QWENIMAGE21_TEXT_H */
