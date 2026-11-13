/* Whisper, the speech recognizer's encoder-decoder (phase A3 of PLAN.md): the GGUF of rocm-convert (its whisper.*
 * keys and encoder.* / decoder.* tensors, F32) on the host in float32, the order of operations of transformers'
 * WhisperForConditionalGeneration. The mel features come from mel.h (whisper_log_mel). */
#ifndef WHISPER_H
#define WHISPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct whisper_model whisper_model_t;

/* Opens a whisper GGUF (its weights stay mapped). NULL on failure: whisper_last_error. */
whisper_model_t* whisper_open(const char* gguf_path);
void whisper_close(whisper_model_t* model);
const char* whisper_last_error(void);

int whisper_n_mels(const whisper_model_t* model);
int whisper_n_audio_ctx(const whisper_model_t* model);    /* the encoder's positions: 1500 for a 30 s chunk */
int whisper_n_audio_state(const whisper_model_t* model);  /* the width: 384 for tiny */
int whisper_n_vocab(const whisper_model_t* model);
int whisper_eot(const whisper_model_t* model);            /* the end-of-transcript token */

/* The encoder: mel [n_mels][WHISPER_N_FRAMES] (whisper_log_mel's output) to out [n_audio_ctx][n_audio_state]. 0, or -1. */
int whisper_encode(whisper_model_t* model, const float* mel, float* out);

/* Greedy decoding (transformers' generate with the generation_config's suppress_tokens and begin_suppress_tokens): the
 * prompt tokens (e.g. <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>) and then the most likely token
 * until the end-of-transcript token or max_new tokens. enc is whisper_encode's output. out_tokens receives the tokens
 * after the prompt, without the end-of-transcript token. Returns their count, or -1. */
int whisper_greedy(whisper_model_t* model, const float* enc, const int* prompt, int n_prompt, int max_new, int* out_tokens);

#ifdef __cplusplus
}
#endif

#endif /* WHISPER_H */
