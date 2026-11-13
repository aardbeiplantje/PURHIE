/* Images for the vision tower (phase V5 of PLAN.md): decoding (PNG, JPEG, ... by stb_image, public domain) and the
 * Qwen-VL image processor's steps: smart_resize to multiples of patch x merge within the pixel bounds, bicubic
 * resampling as PIL does it, rescale and normalize, and the patches as the vision tower's pixel rows (merge order). */
#ifndef IMAGE_H
#define IMAGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decodes an image in memory to RGB8 (w * h * 3 bytes, row-major). 0, or -1. Free with image_free. */
int image_decode_rgb(const unsigned char* data, size_t n, unsigned char** rgb_out, int* w_out, int* h_out);
void image_free(void* p);

/* The processor: the image (rgb, w x h) resized to (h', w') with h', w' multiples of factor = patch * merge whose product
 * lies within [min_pixels, max_pixels] (smart_resize), then the pixel rows: *rows_out holds (h' / patch) * (w' / patch)
 * rows of 3 * temporal * patch * patch floats (channel, temporal, row, column; the image repeated over the temporal
 * dimension), in merge order (merge x merge blocks, raster over the blocks), normalized as (x / 255 - mean) / std.
 * grid_h / grid_w: the patch grid. 0, or -1. Free *rows_out with image_free. */
int image_qwen_pixels(const unsigned char* rgb, int w, int h, int patch, int merge, int temporal, float mean, float stdev,
                      long long min_pixels, long long max_pixels, float** rows_out, int* grid_h, int* grid_w);

#ifdef __cplusplus
}
#endif

#endif /* IMAGE_H */
