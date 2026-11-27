/* Qwen-Image 2.1's condition-image encoder: the vision tower of its Qwen3-VL 8B text encoder (transformers' Qwen3VLVisionModel: the
 * patch embedding of 3 x 2 x 16 x 16 pixels, the learned 48 x 48 position table bilinearly interpolated to the grid, 27 blocks of
 * width 1152 with 16 heads of 72, the 2-D rotary positions and the GELU-tanh MLP of 4304; the 2 x 2 merger to the language model's
 * width 4096, and the deepstack mergers after blocks 8, 16 and 24), from the ComfyUI file's BF16 model.visual.* weights in place,
 * on the device. The products are BF16 in and out (hipBLASLt), the residual stream FP32. */
#ifndef QWENIMAGE21_VISION_H
#define QWENIMAGE21_VISION_H

#include "zimage_dit.h"

namespace qwenimage21 {

const int VISION_OUT = 4096;   /* the merged tokens' width: the language model's */
const int VISION_DEEPSTACK = 3;  /* the deepstack features, added after the language model's layers 0, 1 and 2 */

/* pixels [gh * gw][1536]: the processor's pixel rows in merge order (image_qwen_pixels with patch 16, merge 2, temporal 2, mean and
 * std 0.5); gh and gw even, at most 256. merged [(gh / 2) (gw / 2)][4096] and ds [3][(gh / 2) (gw / 2)][4096] (host). Returns false
 * on an error. */
bool dev_vision21_encode(const zimage::SafeTensors& st, const float* pixels, int gh, int gw, float* merged, float* ds);

}  // namespace qwenimage21

#endif /* QWENIMAGE21_VISION_H */
