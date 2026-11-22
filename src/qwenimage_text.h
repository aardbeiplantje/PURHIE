/* Qwen-Image's text encoder: the language model of Qwen2.5-VL 7B (from the ComfyUI FP8 file: the linears are E4M3 with the
 * per-tensor scale_weight, read in place; the embeddings and norms BF16), on the device (AGENTS.md rules). The output is the
 * final norm's hidden state (the pipeline's hidden_states[-1]) of the prompt's tokens; the caller drops the template's first 34. */
#ifndef QWENIMAGE_TEXT_H
#define QWENIMAGE_TEXT_H

#include "zimage_dit.h"

namespace qwenimage {

/* ids [T] (1 to 64 tokens) -> out [T][3584] (host memory) */
bool dev_text_encode(const zimage::SafeTensors& st, const int* ids, int T, float* out);

}  // namespace qwenimage

#endif /* QWENIMAGE_TEXT_H */
