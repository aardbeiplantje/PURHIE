/* Qwen3-TTS-Tokenizer-12Hz's decoder (phase T3 of PLAN.md): the 12.5 Hz codec frames (16 codebooks each) to a 24 kHz
 * waveform, on the host in float32, the order of operations of qwen-tts 0.1.1 (Qwen3TTSTokenizerV2Decoder: the residual
 * quantizer's decode, the causal pre-convolution, the 8-layer sliding-window transformer, two upsamplers with ConvNeXt
 * blocks, four decoder blocks with Snake activations and dilated residual units, the output convolution, the clamp). The
 * GGUF is rocm-convert of the checkpoint (the decoder's F32 tensors). */
#ifndef QWEN3TTS_CODEC_H
#define QWEN3TTS_CODEC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen3tts_codec qwen3tts_codec_t;

qwen3tts_codec_t* qwen3tts_codec_open(const char* gguf_path);
void qwen3tts_codec_close(qwen3tts_codec_t* codec);
const char* qwen3tts_codec_last_error(void);

int qwen3tts_codec_n_quantizers(const qwen3tts_codec_t* codec); /* 16 */
int qwen3tts_codec_samples_per_frame(const qwen3tts_codec_t* codec); /* 1920 */
int qwen3tts_codec_sample_rate(const qwen3tts_codec_t* codec); /* 24000 */

/* The waveform of T frames: codes [T][n_quantizers] (frame-major, each 0 .. codebook_size - 1). Frames past 300 are decoded
 * in chunks with 25 frames of left context, as qwen-tts's chunked_decode does. *wav receives T * samples_per_frame floats
 * (free with qwen3tts_codec_free_wav). Returns 0, or -1. */
int qwen3tts_codec_decode(qwen3tts_codec_t* codec, const int* codes, int T, float** wav, size_t* n_out);
void qwen3tts_codec_free_wav(float* wav);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_CODEC_H */
