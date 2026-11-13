/* Stable Diffusion (sd-turbo, SD 2.x architecture) in the engine on the host (PLAN.md image phase, I2-I5). The GGUF is
 * rocm-convert of the diffusers pipeline (Convert::Diffusers: the fp16 tensors, named by component and index; the names as
 * a KV table). The weights are read in place from the mapping (F16), the activations float32. */
#ifndef SD_H
#define SD_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sd_model sd_model_t;

sd_model_t* sd_open(const char* gguf_path);
void sd_close(sd_model_t* model);
const char* sd_last_error(void);

int sd_text_ctx(const sd_model_t* model);     /* 77 */
int sd_text_dim(const sd_model_t* model);     /* 1024 (the cross-attention width) */

/* The CLIP tokens of a prompt: ids[sd_text_ctx] (BOS, the prompt's tokens, EOS, then EOS as padding, the prompt cut to fit).
 * Returns the number of real tokens (BOS and EOS included), or -1. */
int sd_tokenize(sd_model_t* model, const char* prompt, int* ids);

/* The text encoder's last hidden state of the prompt: out[sd_text_ctx][sd_text_dim] (diffusers' encode_prompt without
 * classifier-free guidance). Returns 0, or -1. */
int sd_encode_prompt(sd_model_t* model, const char* prompt, float* out);

/* The VAE decoder: latents [4][h][w] (the UNet's output, before the scaling factor's division) to the image planes
 * [3][8h][8w] (float, not clamped: diffusers' decode output, in about [-1, 1]). Returns 0, or -1. */
int sd_vae_decode(sd_model_t* model, const float* latents, int h, int w, float* image);

#ifdef __cplusplus
}
#endif

#endif /* SD_H */
