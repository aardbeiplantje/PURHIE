/* Qwen-Image's sampler (the pipeline's FlowMatchEulerDiscreteScheduler with the exponential dynamic shift and the terminal stretch)
 * and the latent's unpacking: the device DiT for each step, the latent state (256 x 64 values) on the host. */
#ifndef QWENIMAGE_PIPE_H
#define QWENIMAGE_PIPE_H

#include "zimage_dit.h"

#include <vector>

namespace qwenimage {

/* the sigmas of the schedule: linspace(1, 1/steps, steps), the exponential shift by mu, the stretch to the terminal value, then 0
 * (steps + 1 values). mu from the image's token count (calculate_shift: base 256 -> 0.5, max 8192 -> 0.9). */
void sigmas_for(int steps, int image_tokens, std::vector<float>& sig);

/* the Euler sampler from the packed noise [gh * gw][64] with the text embeddings enc [L][3584]; out: the final packed latent
 * [gh * gw][64] (host memory). Returns false on an error. */
bool dev_sample(const zimage::SafeTensors& dit, const float* noise, int gh, int gw, const float* enc, int L, int steps, float* out);

/* the packed latent [gh * gw][64] to the channel-major latent [16][2 gh][2 gw] (the pipeline's _unpack_latents) */
void unpack(const float* packed, int gh, int gw, float* lat);

}  // namespace qwenimage

#endif /* QWENIMAGE_PIPE_H */
