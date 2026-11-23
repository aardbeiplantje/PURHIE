/* AudioLDM 2 (text to audio) on the engine: the stages of src/audioldm2_*.cpp chained as diffusers' AudioLDM2Pipeline runs them:
 * the CLAP and T5 text encoders, the projection, the GPT-2's 8 generated tokens (the UNet's encoder_hidden_states), the UNet with
 * classifier-free guidance (the empty prompt as the negative) and DDIM (eta 0, the leading timesteps), the VAE decoder, the vocoder,
 * the waveform cut to the audio length. The tokens come in as ids (the tokenizers are not in the engine yet). */
#ifndef AUDIOLDM2_PIPELINE_H
#define AUDIOLDM2_PIPELINE_H

#include "zimage_dit.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace audioldm2 {

/* the tokens of one prompt: the CLAP text encoder's ids and mask (512 each, padded as the pipeline pads), the T5 ids */
struct Tokens {
    std::vector<int32_t> clap_ids, clap_mask, t5_ids;
};

/* the AudioLDM 2 directory (the diffusers layout: language_model/, projection_model/, text_encoder/, text_encoder_2/, unet/, vae/,
 * vocoder/), each component's file read in place */
class Pipeline {
public:
    explicit Pipeline(const std::string& dir);
    ~Pipeline();
    bool ok() const { return error_.empty(); }
    const std::string& error() const { return error_; }
    /* the waveform (16 kHz, mono) of the prompt: seconds of audio, the sampler's steps, the classifier-free guidance scale, the seed of
     * the initial latent (torch's CPU randn). neg: the negative prompt's tokens (the empty prompt). False on any error. */
    bool generate(const Tokens& pos, const Tokens& neg, float seconds, int steps, float guidance, uint64_t seed, std::vector<float>& wav);
    /* the tokenizers (the RoBERTa BPE of the CLAP text encoder: its vocabulary and merges; the T5 unigram: its pieces and scores, the
     * unknown id 2 and the end id 1). Read by the caller from the tokenizer.json files. */
    bool load_tokenizers(const std::vector<std::string>& clap_vocab, const std::vector<std::string>& clap_merges,
                         const std::vector<std::string>& t5_pieces, const std::vector<double>& t5_scores);
    /* the same from texts: the prompt and the negative prompt (empty: no negative text) */
    bool generate_text(const std::string& prompt, const std::string& negative, float seconds, int steps, float guidance, uint64_t seed,
                       std::vector<float>& wav);

private:
    struct Files;
    std::unique_ptr<Files> f_;
    std::unique_ptr<class ClapTokenizer> clap_tok_;
    std::unique_ptr<class T5Tokenizer> t5_tok_;
    std::string error_;
};

}  // namespace audioldm2

#endif  /* AUDIOLDM2_PIPELINE_H */
