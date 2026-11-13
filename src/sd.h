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

/* The UNet's one prediction (the UNet2DConditionModel, sd-turbo): the sample [4][h][w] (the scheduler's model input) at the
 * timestep t, the context (the prompt's [sd_text_ctx][sd_text_dim] of sd_encode_prompt); out [4][h][w] (the epsilon). h and
 * w are multiples of 8. Returns 0, or -1. */
int sd_unet(sd_model_t* model, const float* sample, int h, int w, float t, const float* ctx, float* out);

/* The VAE decoder: latents [4][h][w] (the UNet's output, before the scaling factor's division) to the image planes
 * [3][8h][8w] (float, not clamped: diffusers' decode output, in about [-1, 1]). Returns 0, or -1. */
int sd_vae_decode(sd_model_t* model, const float* latents, int h, int w, float* image);

/* The image: one Euler step (sd-turbo: one step, guidance off). init: the initial latents [4][h/8][w/8] (the caller's:
 * noise times sd_init_sigma's sigma for the scheduler's convention); prompt -> the text encoder; the UNet's epsilon at
 * t = 999; the Euler step to sigma 0; the VAE; rgb receives h x w x 3 bytes (diffusers' postprocess: round(255 * clamp(v / 2
 * + 0.5))). Returns 0, or -1. */
int sd_generate(sd_model_t* model, const char* prompt, const float* init, int h, int w, uint8_t* rgb);
/* Standard normal latents [4][h/8][w/8] from a seed (the endpoint's noise): not torch's sequence. Returns 0, or -1. */
int sd_random_latents(uint64_t seed, int h, int w, float* out);
int sd_init_sigma(sd_model_t* model, float* sigma);  /* the noise level of the one step (14.61 for sd-turbo) */

/* PNG (RGB, 8 bits, zlib): *out is malloc'd (sd_free_png). Returns 0, or -1. */
int sd_encode_png(const uint8_t* rgb, int w, int h, uint8_t** out, size_t* n);
void sd_free_png(uint8_t* png);

#ifdef __cplusplus
}
#endif

#endif /* SD_H */
