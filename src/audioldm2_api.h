/* AudioLDM 2 for the Perl service (C API over audioldm2::Pipeline): the directory of the model, its tokenizers (the CLAP's
 * vocabulary and merges, the T5's pieces and scores), and the text to audio. */
#ifndef AUDIOLDM2_API_H
#define AUDIOLDM2_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aud_pipe aud_pipe_t;

/* the model directory (the diffusers layout) and the tokenizers: the arrays are copied. NULL on failure (aud_last_error). */
aud_pipe_t* aud_open(const char* dir, const char** clap_vocab, size_t n_clap_vocab, const char** clap_merges, size_t n_clap_merges,
                     const char** t5_pieces, const double* t5_scores, size_t n_t5);
void aud_close(aud_pipe_t* p);
const char* aud_last_error(void);

/* the waveform (16 kHz, mono; *out is malloc'ed, free it with aud_free) of the prompt; negative: the negative prompt ("" for none).
 * seconds of audio, steps of the sampler, guidance the classifier-free scale, seed the initial latent's seed. 0 on success. */
int aud_generate(aud_pipe_t* p, const char* prompt, const char* negative, float seconds, int steps, float guidance,
                 unsigned long long seed, float** out, size_t* n);
void aud_free(float* samples);

#ifdef __cplusplus
}
#endif

#endif  /* AUDIOLDM2_API_H */
