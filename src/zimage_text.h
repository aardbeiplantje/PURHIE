/* Z-Image's text encoder (Qwen3-4B, from the ComfyUI safetensors file: BF16, HF names) on the device, AGENTS.md rules: the weights
 * read in place from the mapping, the activations in shared memory. The output is the hidden state before the last layer (the
 * pipeline's hidden_states[-2]) of the prompt's tokens (the chat-templated ids; no padding). */
#ifndef ZIMAGE_TEXT_H
#define ZIMAGE_TEXT_H

#include "zimage_dit.h"

namespace zimage {

/* ids [T] (1 to 64 tokens) -> out [T][2560] (host memory): the hidden state after 35 of the 36 layers, before the final norm */
bool dev_text_encode(const SafeTensors& st, const int* ids, int T, float* out);

}  // namespace zimage

#endif /* ZIMAGE_TEXT_H */
