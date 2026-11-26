/* Qwen-Image 2.1, text to image, in the engine (diffusers' QwenImage21Pipeline): the prompt through the text-to-image template (a
 * system turn, "Comprehend and analyze the provided prompt.", whose tokens are dropped from the embeddings) and the Qwen tokenizer,
 * the Qwen3-VL 8B text encoder (qwenimage21_text.h), the flow-matching Euler sampler over the DiT (qwenimage21_dit.h; no guidance:
 * the model is sampled without it), and the VAE decoder (qwenimage21_vae.h), which draws RGBA. The files are the ComfyUI ones (the
 * int8 ConvRot DiT and text encoder, the BF16 VAE), read in place (AGENTS.md rules); the tokenizer is the Qwen vocabulary and
 * merges (the same as Qwen3's). The schedule is the model's scheduler_config.json (Qwen/Qwen-Image-2.1): linspace(1, 1/steps, steps)
 * with the exponential dynamic shift (base 256 -> 0.5, max 8192 -> 0.9) and the stretch to 0.02, as Qwen-Image's. */
#ifndef QWENIMAGE21_PIPE_H
#define QWENIMAGE21_PIPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qi21_pipe qi21_pipe_t;

/* vocab: nv token strings by id; merges: nm merge rules "a b". Returns NULL on error (qi21_last_error). */
qi21_pipe_t* qi21_open(const char* dit_path, const char* vae_path, const char* te_path, const char** vocab, size_t nv,
                       const char** merges, size_t nm);
void qi21_close(qi21_pipe_t* pipe);
const char* qi21_last_error(void);

/* prompt -> rgba (height * width * 4 bytes, 8 bits: the model draws an alpha channel). height and width: multiples of 32 from 256 to
 * 2048; steps 1 to 100 (the pipeline's default is 40). The seed draws the noise as torch.randn(1, 1, 64, height / 16, width / 16)
 * after manual_seed(seed), the CPU generator (diffusers' noise for the seed). progress (may be NULL) is called after each step with
 * (step + 1, steps, user). Returns 0, or -1. */
int qi21_generate(qi21_pipe_t* pipe, const char* prompt, int height, int width, int steps, unsigned long long seed, unsigned char* rgba,
                  void (*progress)(int step, int steps, void* user), void* user);

#ifdef __cplusplus
}

#include <functional>
#include <string>
#include <vector>

#include "tokenizer.h"
#include "zimage_dit.h"

namespace qwenimage21 {

/* the prompt's ids (the whole template) and the count of the system turn's tokens at its start (dropped from the embeddings) */
bool prompt_ids(tokenizer_t* tok, const std::string& prompt, std::vector<int>& ids, int& drop);

/* the sigmas of the schedule for the image's tokens (steps + 1 values, the last 0) */
void sigmas(int steps, int image_tokens, std::vector<float>& sig);

/* the sampler: the packed noise [gh * gw][64] to the packed latent [gh * gw][64] (host memory), the text embeddings emb [L][4096].
 * on_step (may be empty): after each step, with its index, the velocity and the new latent. */
bool sample(const zimage::SafeTensors& dit, const float* noise, int gh, int gw, const float* emb, int L, int steps, float* latent,
            const std::function<void(int, const float*, const float*)>& on_step);

/* the packed latent [gh * gw][64] to the image [4][16 gh][16 gw] (RGBA in [-1, 1]) */
bool decode(const zimage::SafeTensors& vae, const float* latent, int gh, int gw, float* out);

}  // namespace qwenimage21
#endif

#endif /* QWENIMAGE21_PIPE_H */
