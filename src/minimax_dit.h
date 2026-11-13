/* MiniMax-H3-Image's diffusion transformer in the engine (PLAN.md M3, the t2i path): the token refiner over the prompt's
 * hidden states, the 50 AdaLN blocks over the packed sequence (text rows, then the image rows), the step's output head, and
 * the six-step Euler sampler of the PDD schedule. The BF16 weights are read in place from the DiT GGUF (rocm-convert of
 * /hf/MiniMax-H3-Image/transformer); the device path only. */
#ifndef MINIMAX_DIT_H
#define MINIMAX_DIT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmdit mmdit_t;

/* dit: the transformer's GGUF; pdd: the PDD heads' GGUF (the sigmas and the six output heads) */
mmdit_t* mmdit_open(const char* dit_gguf, const char* pdd_gguf);
void mmdit_close(mmdit_t* model);
const char* mmdit_last_error(void);

/* The number of sampling steps (six) and their sigmas (sigmas[0] = 1 ... sigmas[steps] = 0). */
int mmdit_steps(const mmdit_t* model);
float mmdit_sigma(const mmdit_t* model, int i);

/* text [T][5120]: the prompt's hidden state after the text encoder's layer 50 (M1). sample [24][h][w]: the sampler's latents
 * (h, w: the latent grid, even). step: the sampling step (0 .. steps - 1). out [24][h][w]: the velocity. Returns 0, or -1. */
int mmdit_velocity(mmdit_t* model, const float* text, int T, const float* sample, int h, int w, int step, float* out);

/* The whole sampler from the initial noise [24][h][w] (N(0, 1)): the six Euler steps; out [24][h][w] are the denoised
 * latents (the VAE's input, before its normalization). Returns 0, or -1. */
int mmdit_sample(mmdit_t* model, const float* text, int T, const float* noise, int h, int w, float* out);

#ifdef __cplusplus
}
#endif

#endif /* MINIMAX_DIT_H */
