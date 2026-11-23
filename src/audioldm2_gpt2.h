/* AudioLDM 2's language model (GPT-2: 12 blocks, 768 wide, 12 heads, GELU (tanh), from language_model/model.safetensors): the
 * device path, its weights read in place from the mapping (fp32). The stage check is tests/audioldm2_gpt2_verify.cpp against the
 * diffusers oracle (/data/audioldm2-stages, the teacher-forced GPT-2 of the pipeline). */
#ifndef AUDIOLDM2_GPT2_H
#define AUDIOLDM2_GPT2_H

#include "zimage_dit.h"

#include <cstdint>

namespace audioldm2 {

/* mask: T entries (0 for a padding position: its keys are not attended; null: all real). x [T][768] (the inputs_embeds, float; T <= 1024): hidden6 [T][768] = the output of the sixth block (HF's hidden_states[6]),
 * last [T][768] = the final norm of the twelfth block (HF's last_hidden_state). Both may be null. The mapping of st is registered
 * for the device by the caller (st.register_device()). False on any error. */
bool gpt2_forward(const zimage::SafeTensors& st, const float* x, int T, const int32_t* mask, float* hidden6, float* last);

/* the pipeline's generate_language_model: each of max_new steps runs the GPT-2 on the sequence so far and appends the last
 * hidden state (HF's hidden_states[-1]); out [max_new][768] gets the generated rows. embeds [T][768], T + max_new <= 1024. */
bool gpt2_generate(const zimage::SafeTensors& st, const float* embeds, const int32_t* mask, int T, int max_new, float* out);

}  // namespace audioldm2

#endif  /* AUDIOLDM2_GPT2_H */
