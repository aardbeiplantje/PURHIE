/* Qwen-Image 2.1's diffusion transformer (32 single-stream blocks, width 4096, 32 heads of 128; diffusers' QwenImage21Transformer2DModel)
 * from the ComfyUI int8 ConvRot file (the linears read in place as int8 with the row scales and the Hadamard rotation; the other
 * weights BF16 in place) or the BF16 file, on the device. The joint sequence: the text tokens (the encoder's output, the system turn
 * dropped) with the condition images' latent tokens in place of their image pads (each pad stands for a 2 x 2 group of latent tokens),
 * then the target image's tokens. The attention is block-causal: causal along the sequence, each image block bidirectional within
 * itself; the text and the condition images take the t = 0 modulation row (causal_condition), so they do not depend on the step: their
 * keys and values are computed once (prefix) and every step runs only the target's tokens against them (the pipeline's KV cache). */
#ifndef QWENIMAGE21_DIT_H
#define QWENIMAGE21_DIT_H

#include "zimage_dit.h"

#include <utility>
#include <vector>

namespace qwenimage21 {

/* The conditioning: enc [L][4096] the text encoder's tokens; pad [L] (may be null: text only) marks the image pads, each run of pads
 * one condition image's block of 4 x its pads latent tokens; cond [sum gh_i gw_i][64] the condition images' packed latents in order,
 * grids their (gh_i, gw_i) in latent tokens. */
struct Condition {
    const float* enc = nullptr;
    int L = 0;
    const unsigned char* pad = nullptr;
    const float* cond = nullptr;
    std::vector<std::pair<int, int>> grids;
};

class Dit21 {
public:
    explicit Dit21(const zimage::SafeTensors& st);
    ~Dit21();
    Dit21(const Dit21&) = delete;
    Dit21& operator=(const Dit21&) = delete;
    bool ok() const;
    /* the prefix for a target of gh x gw latent tokens: every block's keys and values of the text and condition tokens */
    bool prefix(const Condition& c, int gh, int gw);
    /* one step after prefix(): packed [gh gw][64] the target's latent, t the timestep in [0, 1] -> out [gh gw][64] (host) */
    bool step(const float* packed, float t, float* out);

private:
    struct Impl;
    Impl* p_;
};

/* the text-to-image forward of one step (a prefix of the text alone, then the step): packed [gh * gw][64] (the image tokens), enc
 * [L][4096] (the text tokens), t the timestep in [0, 1]; out [gh * gw][64] (host). Returns false on an error. */
bool dev_dit21_forward(const zimage::SafeTensors& st, const float* packed, int gh, int gw, const float* enc, int L, float t, float* out);

}  // namespace qwenimage21

#endif /* QWENIMAGE21_DIT_H */
