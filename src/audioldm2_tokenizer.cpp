#include "audioldm2_tokenizer.h"

#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <limits>

namespace audioldm2 {
namespace {

constexpr int kClapMax = 512, kPad = 1, kBos = 0, kEos = 2;

/* the code points of a UTF-8 string (an invalid byte is its own code point) */
std::vector<uint32_t> cpts_of(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + n > s.size()) n = 1;
        uint32_t v = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
        for (int k = 1; k < n; ++k) v = (v << 6) | ((unsigned char)s[i + k] & 0x3F);
        out.push_back(v);
        i += n;
    }
    return out;
}

void append_cpt(std::string& s, uint32_t c) {
    if (c < 0x80) s += (char)c;
    else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 0x3F)); }
    else if (c < 0x10000) { s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
    else { s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 0x3F)); s += (char)(0x80 | ((c >> 6) & 0x3F)); s += (char)(0x80 | (c & 0x3F)); }
}

}  // namespace

ClapTokenizer::ClapTokenizer(const std::vector<std::string>& vocab, const std::vector<std::string>& merges) {
    std::vector<const char*> vp, mp;
    for (const auto& s : vocab) vp.push_back(s.c_str());
    for (const auto& s : merges) mp.push_back(s.c_str());
    tok_ = tokenizer_create_gpt2(vp.data(), vp.size(), mp.data(), mp.size());
    if (!tok_) std::fprintf(stderr, "audioldm2: the CLAP tokenizer could not be created\n");
}

ClapTokenizer::~ClapTokenizer() {
    if (tok_) tokenizer_free(tok_);
}

void ClapTokenizer::encode(const std::string& text, std::vector<int32_t>& ids, std::vector<int32_t>& mask) const {
    ids.assign(kClapMax, kPad);
    mask.assign(kClapMax, 0);
    std::vector<int> pieces((size_t)text.size() + 16);
    const int n = tok_ ? tokenizer_encode(tok_, text.c_str(), pieces.data(), (int)pieces.size()) : 0;
    /* <s>, the pieces (at most 510 of them), </s> */
    int k = 0;
    ids[k] = kBos; mask[k] = 1; ++k;
    for (int i = 0; i < n && k < kClapMax - 1; ++i, ++k) { ids[k] = pieces[i]; mask[k] = 1; }
    ids[k] = kEos; mask[k] = 1;
}

T5Tokenizer::T5Tokenizer(const std::vector<std::string>& pieces, const std::vector<double>& scores, int unk_id, int eos_id)
    : pieces_(pieces), scores_(scores), unk_id_(unk_id), eos_id_(eos_id) {
    double lowest = 0.0;
    for (size_t i = 0; i < pieces_.size(); ++i) {
        index_[pieces_[i]] = (int)i;
        lowest = std::min(lowest, scores_[i]);
        max_chars_ = std::max(max_chars_, cpts_of(pieces_[i]).size());
    }
    unk_score_ = lowest - 10.0;  /* sentencepiece's unknown penalty */
}

std::vector<int32_t> T5Tokenizer::encode(const std::string& text) const {
    /* the normalizer: tabs and newlines are spaces, runs of spaces are one (the ends keep one space) */
    std::vector<uint32_t> in = cpts_of(text), norm;
    for (uint32_t c : in) {
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
        if (c == ' ' && !norm.empty() && norm.back() == ' ') continue;
        norm.push_back(c);
    }
    if (norm.empty()) return {eos_id_};
    /* the Metaspace pre-tokenizer: a prefix unless the text starts with a space, and every space as U+2581 */
    std::vector<uint32_t> chars;
    if (norm[0] != ' ') chars.push_back(0x2581);
    for (uint32_t c : norm) chars.push_back(c == ' ' ? 0x2581 : c);
    const size_t n = chars.size();
    /* Viterbi: best[i] = best score of the first i characters */
    const double neg = -std::numeric_limits<double>::infinity();
    std::vector<double> best(n + 1, neg);
    std::vector<int> from(n + 1, -1), id(n + 1, -1);
    std::vector<bool> fused_unk(n + 1, false);
    best[0] = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (best[i] == neg) continue;
        std::string piece;
        const size_t lim = std::min(n, i + max_chars_);
        for (size_t j = i; j < lim; ++j) {
            append_cpt(piece, chars[j]);
            auto it = index_.find(piece);
            if (it == index_.end()) continue;
            const double s = best[i] + scores_[it->second];
            if (s > best[j + 1]) { best[j + 1] = s; from[j + 1] = (int)i; id[j + 1] = it->second; fused_unk[j + 1] = false; }
        }
        /* the single character as unknown (fused with an unknown character before it) */
        const double us = best[i] + unk_score_;
        const bool prev_unk = i > 0 && id[i] == -2;
        if (us > best[i + 1] || (prev_unk && us >= best[i + 1])) {
            best[i + 1] = us;
            from[i + 1] = (int)i;
            id[i + 1] = -2;
            fused_unk[i + 1] = true;
        }
    }
    /* the path back, the unknown characters together as one <unk> */
    std::vector<int32_t> out;
    size_t at = n;
    while (at > 0) {
        const int i = from[at];
        if (id[at] >= 0) out.push_back(id[at]);
        else if (id[at] == -2) {
            /* a run of unknown characters is one unknown token */
            if (out.empty() || out.back() != unk_id_) out.push_back(unk_id_);
        }
        at = (size_t)i;
    }
    std::reverse(out.begin(), out.end());
    out.push_back(eos_id_);
    return out;
}

}  // namespace audioldm2
