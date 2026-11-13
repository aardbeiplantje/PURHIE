/* MiniMax-H3-Image's VAE decoder in the engine (PLAN.md M2): the ViT over the latent positions, the reference's tiles (256 px,
 * overlap 64 px, blended linearly) and the latents' normalization. The F32 weights are read in place from the GGUF
 * (rocm-convert of /hf/MiniMax-H3-Image/vae); the device path only (the image engine's device, sd_device). */
#ifndef MINIMAX_VAE_H
#define MINIMAX_VAE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mmvae mmvae_t;

mmvae_t* mmvae_open(const char* gguf_path);
void mmvae_close(mmvae_t* model);
const char* mmvae_last_error(void);

/* latents [24][h][w]: the pipeline's normalized latents (before z * latents_std + latents_mean, which this applies).
 * image [3][16h][16w] float, not clamped (the pipeline's decode output). Returns 0, or -1. */
int mmvae_decode(mmvae_t* model, const float* latents, int h, int w, float* image);

#ifdef __cplusplus
}
#endif

#endif /* MINIMAX_VAE_H */
