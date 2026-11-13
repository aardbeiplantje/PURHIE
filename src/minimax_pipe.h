/* MiniMax-H3-Image, text to image, in the engine (PLAN.md M5): the prompt through the Qwen3-VL text encoder (its hidden state after
 * layer 50), the six-step DiT sampler (minimax_dit.h), the VAE decoder (minimax_vae.h) and the output's pixels. The four GGUFs are
 * rocm-convert outputs of /hf/MiniMax-H3-Image (text, transformer, PDD heads, VAE decoder). The text encoder and the DiT are not
 * resident together (their mappings are pinned): the encoder is loaded for the prompt and released before the DiT. */
#ifndef MINIMAX_PIPE_H
#define MINIMAX_PIPE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mm_pipe mm_pipe_t;

mm_pipe_t* mm_open(const char* text_gguf, const char* dit_gguf, const char* pdd_gguf, const char* vae_gguf);
void mm_close(mm_pipe_t* pipe);
const char* mm_last_error(void);

/* rgb: height * width * 3 bytes. height and width: multiples of 32 (the latent grid is 16 px and the patch is 2 latent cells).
 * The seed draws the initial noise (not torch's sequence). Returns 0, or -1. */
int mm_generate(mm_pipe_t* pipe, const char* prompt, int height, int width, unsigned long long seed, unsigned char* rgb);

#ifdef __cplusplus
}
#endif

#endif /* MINIMAX_PIPE_H */
