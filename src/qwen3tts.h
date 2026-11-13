/* Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-0.6B-Base), the talker and its code predictor (phase T2 of PLAN.md): the GGUF of
 * rocm-convert (F32, talker.* and pred.* tensors, qwen3tts.* keys) on the host in float32, the order of operations of
 * qwen-tts 0.1.1 (Qwen3TTSForConditionalGeneration.generate: the in-context voice clone's prefill; the talker, a Qwen3
 * decoder with M-RoPE; the code predictor for the 15 codebooks after the first). The speaker embedding (the speaker
 * encoder) and the codec (the speech tokenizer) are not in the engine yet: the speaker embedding is an input. */
#ifndef QWEN3TTS_H
#define QWEN3TTS_H

#include <stddef.h>

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
