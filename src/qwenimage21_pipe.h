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

/* For the server: one image at a time on a background thread, so the caller's loop keeps serving (an image takes a minute or more). The
 * job holds the device's run mutex (sd_device.h) while it runs: the other models of the device layer must not run meanwhile.
 * qi21_start: the options as qi21_generate's; 0 started, 1 a job is running (none started), -1 an error (qi21_last_error).
 * qi21_poll: 0 running, 1 done (its PNG waits for qi21_take), -1 failed (qi21_take gives its error), 2 no job; step / steps: the
 * sampler's (the text encoder and the decoder count as step 0 and the last); seconds: since the start (to the end when done).
 * qi21_take: the finished job's image as RGBA PNG bytes (*png malloc'ed: qi21_free), and the job is cleared; 0, or -1 when it failed
 * (qi21_last_error: its error; the job is cleared) or no job is done. qi21_close waits for a running job. */
int qi21_start(qi21_pipe_t* pipe, const char* prompt, int height, int width, int steps, unsigned long long seed);

/* The image edit (the pipeline with a condition image): image the condition image's file bytes (PNG, JPEG, ...: image_decode_rgba;
 * the alpha kept, as the VAE reads RGBA), prompt the instruction (it may name the image as <image1>). resolution: the side of the
 * pixel budget (256 to 2048; the pipeline's output_resolution, 1024 by default): the output and the resized condition are
 * calculate_dimensions(resolution^2, the image's aspect ratio) in multiples of 32. qi21_edit: *rgba (width * height * 4, malloc'ed:
 * qi21_free) and its size; 0, or -1. qi21_start_edit: the background job as qi21_start's (qi21_poll, qi21_take). */
int qi21_edit(qi21_pipe_t* pipe, const char* prompt, const unsigned char* image, size_t n, int resolution, int steps,
              unsigned long long seed, unsigned char** rgba, int* width, int* height, void (*progress)(int step, int steps, void* user),
              void* user);
int qi21_start_edit(qi21_pipe_t* pipe, const char* prompt, const unsigned char* image, size_t n, int resolution, int steps,
                    unsigned long long seed);
/* the output size of an edit of a w x h image at the resolution (calculate_dimensions); 0, or -1 */
int qi21_edit_size(int w, int h, int resolution, int* width, int* height);
int qi21_poll(qi21_pipe_t* pipe, int* step, int* steps, double* seconds);
int qi21_take(qi21_pipe_t* pipe, unsigned char** png, size_t* n);
void qi21_free(void* ptr);

#ifdef __cplusplus
}

#include <functional>
#include <string>
#include <vector>

#include "qwenimage21_dit.h"
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
/* the same with a conditioning (the text, and for an edit the condition images: qwenimage21_dit.h): its prefix once, then the steps */
bool sample(const zimage::SafeTensors& dit, const float* noise, int gh, int gw, const Condition& c, int steps, float* latent,
            const std::function<void(int, const float*, const float*)>& on_step);

/* the packed latent [gh * gw][64] to the image [4][16 gh][16 gw] (RGBA in [-1, 1]) */
bool decode(const zimage::SafeTensors& vae, const float* latent, int gh, int gw, float* out);

/* The image edit's steps (diffusers' QwenImage21Pipeline with a condition image). */

/* the condition image's pixel rows for the vision tower: rgba (w x h, multiples of 32) composited over white as the pipeline does
 * it (PIL's paste with the alpha as the mask), then the Qwen-VL processor's rows (patch 16, merge 2, temporal 2, mean and std 0.5;
 * the size kept: a multiple of 32 within its bounds). rows [gh * gw][1536] in merge order. */
bool condition_pixels(const unsigned char* rgba, int w, int h, std::vector<float>& rows, int& gh, int& gw);

/* calculate_dimensions(resolution^2, w / h): the width and the height in multiples of 32 (Python's round, half to even) */
void edit_size(int w, int h, int resolution, int& width, int& height);
/* the condition image (rgba, w x h) resized to ow x oh as VaeImageProcessor.resize does it: PIL's LANCZOS on the premultiplied
 * RGBa, back to RGBA (a copy when the size is the same) */
std::vector<unsigned char> resize_condition(const unsigned char* rgba, int w, int h, int ow, int oh);
/* the edit's ids: the image template with one image of n_pad pads ("<image1><|vision_start|>", the pads, "<|vision_end|>" before
 * the prompt), and the count of the system turn's tokens at its start */
bool edit_ids(tokenizer_t* tok, const std::string& prompt, int n_pad, std::vector<int>& ids, int& drop);
/* the M-RoPE positions [3][T] of the ids with one image of gh x gw patches (merge 2) */
bool edit_positions(const std::vector<int>& ids, int gh, int gw, std::vector<int>& pos);
/* the condition: rgba (cw x ch, the resized condition image) through the vision tower and, with the prompt, the text encoder: emb
 * [L][4096] (the system tokens dropped) and pad [L] (the image pads); and through the VAE encoder: cond [(ch / 16) (cw / 16)][64] */
bool encode_condition(const zimage::SafeTensors& te, const zimage::SafeTensors& vae, tokenizer_t* tok, const std::string& prompt,
                      const unsigned char* rgba, int cw, int ch, std::vector<float>& emb, std::vector<unsigned char>& pad,
                      std::vector<float>& cond);

}  // namespace qwenimage21
#endif

#endif /* QWENIMAGE21_PIPE_H */
