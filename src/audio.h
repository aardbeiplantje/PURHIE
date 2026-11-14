/* Audio in the engine (PLAN.md A5): a WAV file's samples as mono float32 at the rate asked for, through a band-limited
 * resampler (a Kaiser-windowed sinc, the same kind of filter as soxr's high quality that librosa uses). The WAV: PCM 8,
 * 16, 24 and 32 bits or 32-bit float, any channel count (averaged to mono), the RIFF chunks in any order. */
#ifndef AUDIO_H
#define AUDIO_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The samples of a WAV file at to_rate (samples_out: malloc'd, free with audio_free). Returns 0, or -1 (audio_last_error). */
int audio_decode_wav(const unsigned char* data, size_t n, int to_rate, float** samples_out, size_t* count_out);

/* The resampled copy of x (n samples at from_rate) at to_rate: ceil(n * to_rate / from_rate) samples. Returns 0, or -1. */
int audio_resample(const float* x, size_t n, int from_rate, int to_rate, float** out, size_t* m);

/* A FLAC stream of n float samples at rate: 16-bit mono (clipped to [-1, 1], rounded as the WAV writer rounds), lossless.
 * out: malloc'd (free with audio_free). Returns 0, or -1 (audio_last_error). */
int audio_flac16(const float* x, size_t n, int rate, unsigned char** out, size_t* out_n);

void audio_free(float* samples);
const char* audio_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_H */
