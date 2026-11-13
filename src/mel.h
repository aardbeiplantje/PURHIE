// Whisper's log-mel features (phase A2 of PLAN.md): the log-mel spectrogram transformers' WhisperFeatureExtractor
// computes (5.5.1: the audio is zero-padded or truncated to 30 s, a periodic Hann window of 400 samples, hop 160,
// the reflect-padded centred frames, the power spectrum, 80 Slaney-normalized triangular mel filters, log10 with a
// floor of 1e-10, the last frame dropped, the values clamped to the maximum less 8, then (x + 4) / 4).
#ifndef ROCM_INFERENCE_MEL_H
#define ROCM_INFERENCE_MEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WHISPER_N_FFT 400
#define WHISPER_HOP 160
#define WHISPER_CHUNK_SAMPLES 480000  // 30 s at 16 kHz
#define WHISPER_N_FRAMES 3000         // the frames of a 30 s chunk (1 + 3000 frames, the last dropped)

// The triangular mel filters, Slaney scale and normalization (transformers' mel_filter_bank): out[n_mels][n_fft / 2 + 1],
// row m is filter m over the FFT bins. Returns 0, or -1 on bad arguments.
int mel_filter_bank(int n_mels, int n_fft, int sample_rate, double fmin, double fmax, float* out);

// The log-mel spectrogram of n mono samples (16 kHz, float in [-1, 1]): out[n_mels][WHISPER_N_FRAMES] row-major (as
// transformers' (1, 80, 3000)). Samples past 30 s are truncated (the extractor's default); shorter audio is zero-padded.
// Returns 0, or -1 on bad arguments.
int whisper_log_mel(const float* samples, int n, int n_mels, float* out);

// The speaker encoder's log-mel of 24 kHz samples (qwen-tts's mel_spectrogram: 128 bins, 1024-point frames, hop 256, Slaney
// filters to 12 kHz; the reflect-padded, uncentred frames): out[128][frames] row-major, malloc'd (speaker_free_mel). Returns 0,
// or -1 on bad arguments (fewer than 257 samples).
int speaker_log_mel(const float* samples, size_t n, float** out, int* frames);
void speaker_free_mel(float* mel);

#ifdef __cplusplus
}
#endif

#endif  // ROCM_INFERENCE_MEL_H
