/* AudioLDM 2's projection model (projection_model/): the CLAP and T5 text embeddings projected to the language model's width (768),
 * each sequence wrapped in its learned start and end tokens, the two concatenated. The device path: the linears on the GPU (the
 * weights read in place, fp32), the token rows copied on the host. Stage check: tests/audioldm2_projection_verify.cpp. */
#ifndef AUDIOLDM2_PROJECTION_H
#define AUDIOLDM2_PROJECTION_H

#include "zimage_dit.h"

namespace audioldm2 {

/* clap [Tc][512], t5 [Tt][1024] (float) -> out [(Tc + Tt + 4)][768]: sos, the CLAP rows, eos, sos_1, the T5 rows, eos_1.
 * The mapping of st is registered for the device by the caller. False on any error. */
bool projection_forward(const zimage::SafeTensors& st, const float* clap, int Tc, const float* t5, int Tt, float* out);

}  // namespace audioldm2

#endif  /* AUDIOLDM2_PROJECTION_H */
