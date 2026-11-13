/* Speech (phase T4 of PLAN.md): a text to a 24 kHz waveform in a voice, the talker sampled (qwen3tts_generate) and the codec
 * decoder (qwen3tts_codec_decode), as qwen-tts's generate_voice_clone does: the reference's frames and the talker's are
 * decoded together and the reference's share of the samples is cut off. The voice is the reference's codec frames, its
 * transcript and its speaker embedding (computed outside the engine: see tools/qwen3tts_voice.py). */
#ifndef QWEN3TTS_SPEECH_H
#define QWEN3TTS_SPEECH_H

#include "qwen3tts.h"
#include "qwen3tts_codec.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* text: the words to say; ref_text: the transcript of the voice's reference clip; ref_codes [R][16] and spk [n_embd] its
 * frames and embedding; language_id (qwen3tts_language_id). *wav receives *n_out samples (free with
 * qwen3tts_speech_free_wav), *n_frames the talker's frames (0: no audio). max_frames bounds the talker. Returns 0, or -1. */
int qwen3tts_speak(qwen3tts_model_t* model, qwen3tts_codec_t* codec, const char* text, const char* ref_text,
                   const int* ref_codes, int R, const float* spk, int language_id, const qwen3tts_sampling_t* s,
                   int max_frames, float** wav, size_t* n_out, int* n_frames);
void qwen3tts_speech_free_wav(float* wav);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_SPEECH_H */
