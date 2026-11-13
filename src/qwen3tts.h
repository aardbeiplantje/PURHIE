/* Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-0.6B-Base), the talker and its code predictor (phase T2 of PLAN.md): the GGUF of
 * rocm-convert (F32, talker.* and pred.* tensors, qwen3tts.* keys) on the host in float32, the order of operations of
 * qwen-tts 0.1.1 (Qwen3TTSForConditionalGeneration.generate: the in-context voice clone's prefill; the talker, a Qwen3
 * decoder with M-RoPE; the code predictor for the 15 codebooks after the first). The speaker embedding (the speaker
 * encoder) and the codec (the speech tokenizer) are not in the engine yet: the speaker embedding is an input. */
#ifndef QWEN3TTS_H
#define QWEN3TTS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen3tts_model qwen3tts_model_t;

qwen3tts_model_t* qwen3tts_open(const char* gguf_path);
void qwen3tts_close(qwen3tts_model_t* model);
const char* qwen3tts_last_error(void);

int qwen3tts_n_embd(const qwen3tts_model_t* model);       /* the talker's width: 1024 */
int qwen3tts_n_codebooks(const qwen3tts_model_t* model);  /* 16 */
int qwen3tts_codec_vocab(const qwen3tts_model_t* model);  /* 3072 */
int qwen3tts_pred_vocab(const qwen3tts_model_t* model);   /* 2048 */
int qwen3tts_language_id(const qwen3tts_model_t* model, const char* name); /* e.g. "english" -> 2050, or -1 */

/* The speaker encoder (T5 of PLAN.md): the x-vector of a reference clip (float samples at 24 kHz, n of them, at least 257) as
 * the talker's speaker embedding (qwen-tts's extract_speaker_embedding). spk receives qwen3tts_speaker_dim floats (1024).
 * Needs the GGUF's spk.* tensors (rocm-convert of the checkpoint has them). Returns 0, or -1. */
int qwen3tts_speaker_dim(const qwen3tts_model_t* model);  /* 0 when the GGUF has no speaker encoder */
int qwen3tts_speaker_embedding(qwen3tts_model_t* model, const float* samples, size_t n, float* spk);

/* The in-context voice clone's prefill (qwen-tts's generate with non-streaming off, one sequence): text_ids (the assistant
 * text's ids: <|im_start|>assistant\n TEXT <|im_end|>\n<|im_start|>assistant\n, n_text of them), ref_ids (the reference
 * text's ids, the same template, n_ref), ref_codes (R x 16 codec frames of the reference clip), spk (the speaker
 * embedding, n_embd floats), language_id. out receives the prefill's embeddings [P][n_embd] (P returned in *positions,
 * at most max_positions); trailing receives the trailing text vectors (n_trailing rows, at most max_trailing; the text left
 * after the reference's frames, or one row of the TTS pad) and pad the TTS pad vector (n_embd floats), which follows the
 * trailing rows once they run out. Returns 0, or -1. */
int qwen3tts_build_prefill(qwen3tts_model_t* model, const int* text_ids, int n_text, const int* ref_ids, int n_ref,
                           const int* ref_codes, int R, const float* spk, int language_id, float* out, int max_positions,
                           int* positions, float* trailing, int max_trailing, int* n_trailing, float* pad);

/* One talker frame's input: the sum of its 16 codebooks' embeddings (codes[0] through the talker's table, codes[1..15]
 * through the code predictor's tables) plus the trailing text vector (NULL: none): out[n_embd]. */
int qwen3tts_frame_input(qwen3tts_model_t* model, const int* codes, const float* trailing, float* out);

/* The talker over L positions (a causal pass from position 0): X [L][n_embd] (the prefill's embeddings then the frames'
 * inputs), hidden [L][n_embd] (the normed last state), logits [L][codec_vocab]. Returns 0, or -1. */
int qwen3tts_talker_forward(qwen3tts_model_t* model, const float* X, int L, float* hidden, float* logits);

/* The code predictor, teacher-forced: for each of the T frames, the talker's hidden state H[T][n_embd] of the position
 * that predicted the frame's first codebook, and the frame's codes [T][16]; logits [T][15][pred_vocab] for codebooks
 * 1..15 (the codes 1..14 are the inputs). Returns 0, or -1. */
int qwen3tts_code_predictor(qwen3tts_model_t* model, const float* H, const int* codes, int T, float* logits);

/* The sampling of the codec frames (qwen-tts's generation config: do_sample, top_k 50, temperature 0.9, repetition_penalty
 * 1.05; the code predictor's the same without the penalty). temperature <= 0 is the argmax; top_k 0 keeps every token.
 * seed starts the generator (splitmix64). */
typedef struct qwen3tts_sampling {
    float temperature;
    int top_k;
    float repetition_penalty;
    float sub_temperature;  /* the code predictor's */
    int sub_top_k;
    uint64_t seed;
} qwen3tts_sampling_t;

qwen3tts_sampling_t qwen3tts_sampling_default(void);  /* the generation config above, seed 0 */

/* One token from logits[n] (the temperature, then the top_k most likely; the state advances by one draw). Returns the
 * token, or -1 on a null argument. */
int qwen3tts_sample(const float* logits, int n, float temperature, int top_k, uint64_t* state);

/* The chat template's ids of a text, as qwen-tts tokenizes it: "<|im_start|>assistant\n" TEXT "<|im_end|>\n<|im_start|>assistant\n"
 * (the model's GGUF tokenizer). Returns the number of ids (at most max_ids), or -1. */
int qwen3tts_tokenize_chat(qwen3tts_model_t* model, const char* text, int* ids, int max_ids);

/* Generation of the codec frames with the sampling s (non-null), the same prefill and loop as qwen3tts_generate_greedy:
 * the talker samples the first codebook, the code predictor the other 15. Returns the number of frames, or -1. */
int qwen3tts_generate(qwen3tts_model_t* model, const int* text_ids, int n_text, const int* ref_ids, int n_ref,
                      const int* ref_codes, int R, const float* spk, int language_id, const qwen3tts_sampling_t* s,
                      int max_frames, int* out_codes);

/* Greedy generation of the codec frames (qwen-tts's generate with do_sample off, the sub-talker greedy too): the prefill,
 * then per frame the talker's argmax of the first codebook (the logits after the repetition penalty, the codec range
 * above the codec's text range suppressed except EOS, EOS not before the second frame), the code predictor's argmax of
 * the other 15 codebooks, and the next frame's input. Stops at the end-of-speech token (not written) or after max_frames.
 * out_codes receives the frames [n][16]. Returns the number of frames, or -1. */
int qwen3tts_generate_greedy(qwen3tts_model_t* model, const int* text_ids, int n_text, const int* ref_ids, int n_ref,
                             const int* ref_codes, int R, const float* spk, int language_id, float repetition_penalty,
                             int max_frames, int* out_codes);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_H */
