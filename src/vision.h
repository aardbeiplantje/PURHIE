/* The vision tower of a checkpoint (Qwen3.5's, from its mmproj GGUF: rocm-convert --mmproj): an image's pixel rows in,
 * the LM's image token embeddings out. Host float32 reference (phase V3 of PLAN.md): the order of operations is
 * transformers' (Qwen3_5VisionModel), so the engine's GPU path can be checked against it. */
#ifndef VISION_H
#define VISION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vision_model vision_model_t;

/* Opens an mmproj GGUF (general.architecture qwen3_5_vision, v.* tensors in F32). NULL on error. */
vision_model_t* vision_open(const char* path);
void            vision_close(vision_model_t* vm);
const char*     vision_last_error(void);

/* Hyperparameters: the LM's embedding width (projection dim), the spatial merge, the patch size. */
int vision_n_out(const vision_model_t* vm);
int vision_merge(const vision_model_t* vm);
int vision_patch(const vision_model_t* vm);
int vision_temporal(const vision_model_t* vm);
/* The image processor's normalization (the mmproj's image_mean / image_std, the first channel's; Qwen: all 0.5). */
void vision_image_norm(const vision_model_t* vm, float* mean, float* stdev);

/* A stage's values for checks, called once per stage of vision_encode: "patch_embed" (the patch embedding, before
 * the position embedding), "block0" (the first block's output), "merged" (the LM's image tokens). */
typedef void (*vision_stage_fn)(const char* name, const float* data, int64_t n_rows, int64_t n_cols, void* ud);

/* One image (one temporal step): pixels is [h * w][3 * 2 * P * P] as the image processor gives its pixel rows (h and
 * w in patches of P pixels, the merge applies to them: h and w divisible by it). out receives [(h / m) * (w / m)]
 * [n_out] rows in the order the LM takes its image tokens (merged blocks of m x m patches). stage may be NULL.
 * Returns 0, or -1 (vision_last_error). */
int vision_encode(vision_model_t* vm, const float* pixels, int h, int w, float* out, vision_stage_fn stage, void* ud);

#ifdef __cplusplus
}
#endif

#endif /* VISION_H */
