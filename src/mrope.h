/* Qwen3.5's M-RoPE positions (transformers' Qwen3_5Model.get_rope_index): every token's (temporal, height, width)
 * position. Text tokens take one index on all three axes; the image pad tokens of an image take (start, start + row,
 * start + col) over the merged grid in raster order, and the text after an image continues from start + max(h, w) /
 * merge. delta is the decode offset: the position of the token at index i > n - 1 is i + delta on all three axes. */
#ifndef MROPE_H
#define MROPE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ids: n tokens; grid_hw: [n_images][2] the patch grid (h, w) of each image in the order their pad runs come;
 * merge: the spatial merge; image_token: the pad token id. pos_out: [n][3]. delta_out: may be NULL.
 * Returns 0, or -1 when the image pad runs and the grids disagree. */
int mrope_positions(const int32_t* ids, int n, const int32_t* grid_hw, int n_images, int merge, int32_t image_token,
                    int64_t* pos_out, int64_t* delta_out);

#ifdef __cplusplus
}
#endif

#endif /* MROPE_H */
