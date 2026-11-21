/* Z-Image-Turbo, text to image, in the engine: the prompt through the Qwen3 chat template and the tokenizer, the Qwen3-4B text
 * encoder (zimage_text.h), the 8-step sampler over the DiT (zimage_dit.h), the Flux autoencoder's decoder (zimage_vae.h) and the
 * output's pixels. The three files are the ComfyUI safetensors (diffusion model, VAE, text encoder: BF16 or F32, read in place,
 * AGENTS.md rules); the tokenizer is the Qwen3 vocabulary and merges (one token per line, the merges as "a b" lines). */
#ifndef ZIMAGE_PIPE_H
#define ZIMAGE_PIPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct zi_pipe zi_pipe_t;

/* vocab: nv token strings by id; merges: nm merge rules "a b". Returns NULL on error (zi_last_error). */
zi_pipe_t* zi_open(const char* dit_path, const char* vae_path, const char* te_path, const char** vocab, size_t nv,
                   const char** merges, size_t nm);
void zi_close(zi_pipe_t* pipe);
const char* zi_last_error(void);

/* prompt -> rgb (height * width * 3 bytes, 8 bits). Only 256 x 256 for now (the DiT's latent grid is 32 x 32). The prompt's
 * chat-templated tokens must be at most 32 (the caption length of the DiT). The seed draws the initial noise (not torch's
 * sequence). Returns 0, or -1. */
int zi_generate(zi_pipe_t* pipe, const char* prompt, int height, int width, unsigned long long seed, unsigned char* rgb);

/* the same with the initial noise given: latent [16][32][32] floats (the reference's noise in the checks). Returns 0, or -1. */
int zi_generate_with_noise(zi_pipe_t* pipe, const char* prompt, const float* noise, unsigned char* rgb);

#ifdef __cplusplus
}
#endif

#endif /* ZIMAGE_PIPE_H */
