/* AudioLDM 2's CLAP text encoder (text_encoder/, the text tower of ClapModel and its text projection): a RoBERTa-style model
 * (word, position and token-type embeddings, 12 post-LN blocks of 12 heads, exact GELU, a tanh pooler on the first token) and the
 * projection (Linear 768 -> 512, ReLU, Linear 512 -> 512). The device path: the blocks and the linears on the GPU (the fp32 weights
 * read in place); the embedding rows and the pooler's tanh and the ReLU on the host (single rows). The audio tower is not used.
 * Stage check: tests/audioldm2_clap_verify.cpp against tools/audioldm2_stage_clap.py. */
#ifndef AUDIOLDM2_CLAP_H
#define AUDIOLDM2_CLAP_H

#include "zimage_dit.h"

#include <cstdint>

namespace audioldm2 {

/* ids, mask: T tokens (padding positions have mask 0). hidden [T][768] (the text model's last hidden state, may be null),
 * pooled [768] (the pooler's output, may be null), features [512] (get_text_features, may be null). The mapping of st is registered
 * for the device by the caller. False on any error. */
bool clap_text_forward(const zimage::SafeTensors& st, const int32_t* ids, const int32_t* mask, int T, float* hidden, float* pooled,
                       float* features);

}  // namespace audioldm2

#endif  /* AUDIOLDM2_CLAP_H */
