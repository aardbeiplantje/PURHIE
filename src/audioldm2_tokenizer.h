/* AudioLDM 2's tokenizers on the engine: the CLAP text encoder's (RoBERTa: the GPT-2 byte-level BPE, tokenizer_create_gpt2, with <s>
 * and </s> and the padding to 512) and the T5 encoder's (SentencePiece's unigram: whitespace collapsed and trimmed, the spaces as
 * U+2581 with a prefix, the best segmentation by Viterbi over the pieces' scores, unknown characters fused and mapped to <unk>, </s>).
 * The vocabularies come from the tokenizer.json files (read by the caller). */
#ifndef AUDIOLDM2_TOKENIZER_H
#define AUDIOLDM2_TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct tokenizer;

namespace audioldm2 {

/* the RoBERTa side: the vocabulary by id and the merges ("a b"), the special ids */
class ClapTokenizer {
public:
    ClapTokenizer(const std::vector<std::string>& vocab, const std::vector<std::string>& merges);
    ~ClapTokenizer();
    ClapTokenizer(const ClapTokenizer&) = delete;
    ClapTokenizer& operator=(const ClapTokenizer&) = delete;
    bool ok() const { return tok_ != nullptr; }
    /* the ids of the text as the pipeline's tokenizer gives them: <s>, the pieces, </s>, padded with <pad> to 512, and the mask */
    void encode(const std::string& text, std::vector<int32_t>& ids, std::vector<int32_t>& mask) const;

private:
    tokenizer* tok_ = nullptr;
};

/* the T5 side: the pieces by id and their scores, the unknown id */
class T5Tokenizer {
public:
    T5Tokenizer(const std::vector<std::string>& pieces, const std::vector<double>& scores, int unk_id, int eos_id);
    /* the ids of the text (no padding), </s> last */
    std::vector<int32_t> encode(const std::string& text) const;

private:
    std::vector<std::string> pieces_;
    std::unordered_map<std::string, int> index_;
    std::vector<double> scores_;
    size_t max_chars_ = 1;
    int unk_id_, eos_id_;
    double unk_score_ = -1e9;
};

}  // namespace audioldm2

#endif  /* AUDIOLDM2_TOKENIZER_H */
