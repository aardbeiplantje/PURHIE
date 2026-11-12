// perplexity: a model's perplexity on a text, computed as llama.cpp's llama-perplexity does, so the two can be
// compared (and kept as a regression baseline: tests/regress.sh); and the KL divergence of a quantized model's
// predictions from a reference model's, the quality measure for a quantization recipe (perplexity alone moves by
// +-2% on small per-tensor choices without following the precision).
//
//   perplexity <model.gguf> <text file> [--ctx 512] [--chunks N] [--kv f16|q8_0|q4_0] [--batch N]
//              [--save-logits FILE | --kl FILE]
//
// The text is tokenized once and cut into chunks of --ctx tokens (the first replaced by BOS when the model's
// tokenizer adds one). Each chunk is a new sequence, prefilled with the logits of every position
// (inference_prefill_logits); the second half of its tokens is scored, each given at least --ctx / 2 tokens before
// it: nll += -log softmax(logits[i])[tokens[i + 1]] for i = ctx / 2 .. ctx - 2 (the scored half prefilled in pieces,
// with logits; the first half without). Prints the running perplexity per
// chunk and, last, "perplexity: P +/- E (N tokens, C chunks)" (E: the standard error of the mean log-likelihood,
// carried through exp). Run from build/co.
//
// --save-logits FILE writes the scored positions' log-probabilities (the reference model's run; the file is
// llama-perplexity's --kl-divergence-base format: "_logits_", n_ctx, n_vocab, n_chunk, the tokens, then per chunk a
// row per scored position of two floats (scale, min) and the vocabulary as uint16, the logits quantized to 16 bits
// over max - 16 .. max; about n_vocab x 2 bytes a position, 130 MB a chunk of 512 at a 250k vocabulary).
// --kl FILE (the quantized model's run) reads such a file, of this run's --ctx and text, and reports besides the
// perplexity: the reference's perplexity, ln(PPL / PPL_ref), the KL divergence KL(ref || model) per position (its
// mean, median, 90 / 95 / 99 / 99.9th percentiles and maximum), the difference of the scored token's probability
// (mean, RMS) and how often both models' top token is the same. The reference's perplexity here is from the stored
// 16-bit rows (a scored token under e^-16 of the top one is capped there), so it is a little under the reference
// run's own figure: compare quantizations with each other on one file. Qwen3-0.6B from its Q8_0 (40 chunks): recipe
// strix mean KL 0.188, the same top token 78.5%; recipe plain 0.221 and 75.9% (a 0.6B model loses a lot at 4 bits).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "inference.h"

// llama-perplexity's row: two floats (the scale of a step, the log-probability of 0) and the quantized logits
static int row_u16(int V) { return 2 * ((V + 1) / 2) + 4; }

static inline int nearest_int(float f) {
    float v = f + 12582912.f;
    int i;
    memcpy(&i, &v, sizeof i);
    return (i & 0x007fffff) - 0x00400000;
}

// A row of logits as the file stores it: logits below max - 16 are 0 (a probability under e^-16 of the top one)
static void encode_row(int V, const float* lg, uint16_t* out) {
    float mx = lg[0], mn = lg[0];
    for (int v = 1; v < V; ++v) { mx = std::max(mx, lg[v]); mn = std::min(mn, lg[v]); }
    mn = std::max(mn, mx - 16);
    double sum = 0.0;
    for (int v = 0; v < V; ++v) sum += expf(lg[v] - mx);
    const float lse = log(sum);
    float* d = (float*)out;
    d[0] = (mx - mn) / 65535.f;  // the scale
    d[1] = mn - mx - lse;        // the log-probability a 0 stands for
    out += 4;
    if (d[0] > 0) {
        const float inv = 1 / d[0];
        for (int v = 0; v < V; ++v) out[v] = lg[v] > mn ? nearest_int(inv * (lg[v] - mn)) : 0;
    } else {
        memset(out, 0, sizeof(uint16_t) * V);
    }
}

struct RowStats { double nll, nll_ref, kl; float p_diff; bool same_top; };

// This model's logits row against the reference's stored row, at the scored token tok
static RowStats compare_row(int V, const float* lg, const uint16_t* ref, int tok) {
    float mx = lg[0];
    int top = 0;
    for (int v = 1; v < V; ++v) if (lg[v] > mx) { mx = lg[v]; top = v; }
    double sum = 0.0;
    for (int v = 0; v < V; ++v) sum += expf(lg[v] - mx);
    const double lse = mx + log(sum);  // log p(v) = lg[v] - lse
    const float* d = (const float*)ref;
    const float scale = d[0], lp0 = d[1];
    ref += 4;
    RowStats s;
    s.nll = lse - lg[tok];
    s.nll_ref = -(scale * ref[tok] + lp0);
    double kl = 0.0;
    int top_ref = 0;
    uint16_t qmax = ref[0];
    for (int v = 0; v < V; ++v) {
        if (ref[v] > qmax) { qmax = ref[v]; top_ref = v; }
        const float lp = scale * ref[v] + lp0;
        if (lp > -16.f) kl += expf(lp) * (lp - (lg[v] - lse));
    }
    s.kl = kl;
    s.same_top = top == top_ref;
    s.p_diff = expf(-s.nll) - expf(-s.nll_ref);
    return s;
}

static double percentile(std::vector<float>& v, double q) {
    if (v.empty()) return 0.0;
    const size_t i = std::min(v.size() - 1, (size_t)(q * v.size()));
    std::nth_element(v.begin(), v.begin() + i, v.end());
    return v[i];
}

static bool read_exact(FILE* f, void* p, size_t n) { return fread(p, 1, n, f) == n; }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <text file> [--ctx 512] [--chunks N] [--kv f16|q8_0|q4_0] [--batch N]"
                        " [--save-logits FILE | --kl FILE]\n", argv[0]);
        return 2;
    }
    int n_ctx = 512, max_chunks = 0, batch = 0, kv = INFERENCE_KV_F16;
    std::string save_file, kl_file;
    for (int i = 3; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--ctx") n_ctx = atoi(v.c_str());
        else if (a == "--chunks") max_chunks = atoi(v.c_str());
        else if (a == "--batch") batch = atoi(v.c_str());
        else if (a == "--kv") kv = v == "q8_0" ? INFERENCE_KV_Q8_0 : v == "q4_0" ? INFERENCE_KV_Q4_0 : INFERENCE_KV_F16;
        else if (a == "--save-logits") save_file = v;
        else if (a == "--kl") kl_file = v;
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (n_ctx < 16) { fprintf(stderr, "--ctx must be at least 16\n"); return 2; }
    if (!save_file.empty() && !kl_file.empty()) { fprintf(stderr, "--save-logits or --kl, not both\n"); return 2; }

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
    std::string text;
    char buf[65536];
    for (size_t got; (got = fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, got);
    fclose(f);

    inference_model_t* m = inference_load_model(argv[1]);
    if (!m) { fprintf(stderr, "load: %s\n", inference_get_last_error()); return 1; }
    if ((kv != INFERENCE_KV_F16 && inference_set_kv_type(m, kv) != 0) ||
        // the logits of a whole prefill batch are kept (batch x vocab floats): 256 rows unless asked
        inference_set_prefill_batch(m, batch > 0 ? batch : std::min(n_ctx, 256)) != 0 ||
        inference_kv_cache_init(m, n_ctx) != 0) {
        fprintf(stderr, "setup: %s\n", inference_get_last_error());
        return 1;
    }
    int n_tok = 0;
    int32_t* toks = inference_tokenize(m, text.c_str(), &n_tok);
    if (!toks || n_tok < 2 * n_ctx) {
        fprintf(stderr, "the text has %d tokens: at least 2 chunks of %d needed\n", n_tok, n_ctx);
        return 1;
    }
    const int bos = inference_bos_id(m);
    const bool add_bos = bos >= 0 && toks[0] == bos;  // the tokenizer prepends it: so does every chunk
    int n_chunks = n_tok / n_ctx;
    if (max_chunks > 0 && max_chunks < n_chunks) n_chunks = max_chunks;
    const int V = inference_vocab_size(m), first = n_ctx / 2, scored = n_ctx - 1 - first;
    // the chunks' tokens as scored (BOS first when the tokenizer adds one): the file's token table
    std::vector<int32_t> all(n_chunks * (size_t)n_ctx);
    for (int c = 0; c < n_chunks; ++c) {
        memcpy(&all[(size_t)c * n_ctx], toks + (size_t)c * n_ctx, sizeof(int32_t) * n_ctx);
        if (add_bos) all[(size_t)c * n_ctx] = bos;
    }

    // --save-logits: the file's header; --kl: the reference's header checked against this run
    FILE* ref = nullptr;
    const int nv = row_u16(V);
    if (!save_file.empty()) {
        ref = fopen(save_file.c_str(), "wb");
        if (!ref) { fprintf(stderr, "cannot write %s\n", save_file.c_str()); return 1; }
        fwrite("_logits_", 1, 8, ref);
        fwrite(&n_ctx, sizeof n_ctx, 1, ref);
        fwrite(&V, sizeof V, 1, ref);
        fwrite(&n_chunks, sizeof n_chunks, 1, ref);
        fwrite(all.data(), sizeof(int32_t), all.size(), ref);
    } else if (!kl_file.empty()) {
        ref = fopen(kl_file.c_str(), "rb");
        if (!ref) { fprintf(stderr, "cannot read %s\n", kl_file.c_str()); return 1; }
        char magic[8];
        int f_ctx = 0, f_v = 0, f_chunks = 0;
        if (!read_exact(ref, magic, 8) || memcmp(magic, "_logits_", 8) != 0 || !read_exact(ref, &f_ctx, sizeof f_ctx) ||
            !read_exact(ref, &f_v, sizeof f_v) || !read_exact(ref, &f_chunks, sizeof f_chunks)) {
            fprintf(stderr, "%s is not a logits file (perplexity --save-logits, llama-perplexity --kl-divergence-base)\n",
                    kl_file.c_str());
            return 1;
        }
        if (f_ctx != n_ctx) { fprintf(stderr, "%s was saved with --ctx %d, this run has %d\n", kl_file.c_str(), f_ctx, n_ctx); return 1; }
        if (f_v != V) { fprintf(stderr, "%s has a vocabulary of %d, this model %d\n", kl_file.c_str(), f_v, V); return 1; }
        if (f_chunks < n_chunks) n_chunks = f_chunks;
        std::vector<int32_t> f_toks((size_t)f_chunks * n_ctx);
        if (!read_exact(ref, f_toks.data(), sizeof(int32_t) * f_toks.size())) { fprintf(stderr, "%s: short read of the tokens\n", kl_file.c_str()); return 1; }
        if (memcmp(f_toks.data(), all.data(), sizeof(int32_t) * (size_t)n_chunks * n_ctx) != 0) {
            fprintf(stderr, "%s was saved on other tokens than this text gives with this model's tokenizer\n", kl_file.c_str());
            return 1;
        }
    }
    printf("perplexity: %s, %d tokens, %d chunks of %d (scoring %d each), vocab %d%s%s%s\n", argv[1], n_tok, n_chunks,
           n_ctx, scored, V, add_bos ? ", BOS per chunk" : "",
           save_file.empty() ? "" : ", saving the log-probabilities to ", save_file.c_str());
    if (!kl_file.empty()) printf("KL divergence from the reference in %s\n", kl_file.c_str());

    // The first half of a chunk is prefilled without logits, the scored half in pieces of PIECE positions with them,
    // so that the logits held stay at PIECE x vocab floats at any --ctx (16k x 248k would be 16 GB)
    const int PIECE = 512;
    std::vector<float> logits((size_t)PIECE * V);
    std::vector<uint16_t> rows(ref ? (size_t)scored * nv : 0);  // a chunk's rows, written or read
    std::vector<float> kls, p_diffs;
    if (!kl_file.empty()) { kls.reserve((size_t)n_chunks * scored); p_diffs.reserve((size_t)n_chunks * scored); }
    double nll = 0.0, nll2 = 0.0, nll_ref = 0.0, nll_ref2 = 0.0, d_nll = 0.0, d_nll2 = 0.0, kl_sum = 0.0, kl2 = 0.0;
    double pd_sum = 0.0, pd2 = 0.0;
    long long count = 0, same_top = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < n_chunks; ++c) {
        const int32_t* chunk = &all[(size_t)c * n_ctx];
        if (!kl_file.empty() && !read_exact(ref, rows.data(), sizeof(uint16_t) * rows.size())) {
            fprintf(stderr, "%s: short read at chunk %d\n", kl_file.c_str(), c);
            return 1;
        }
        inference_reset(m);
        if (inference_prefill(m, chunk, first) != 0) {
            fprintf(stderr, "chunk %d: %s\n", c, inference_get_last_error());
            return 1;
        }
        for (int p0 = first; p0 < n_ctx - 1; p0 += PIECE) {  // (token n_ctx - 1 predicts nothing scored)
            const int np = std::min(PIECE, n_ctx - 1 - p0);
            if (inference_prefill_logits(m, chunk + p0, np, logits.data()) != 0) {
                fprintf(stderr, "chunk %d: %s\n", c, inference_get_last_error());
                return 1;
            }
            // (rows on all CPU cores: the softmax over the vocabulary is the slow part)
            double cn = 0.0, cn2 = 0.0, cr = 0.0, cr2 = 0.0, cd = 0.0, cd2 = 0.0, ck = 0.0, ck2 = 0.0, cp = 0.0, cp2 = 0.0;
            long long cs = 0;
            std::vector<RowStats> st(kl_file.empty() ? 0 : np);
#pragma omp parallel for reduction(+ : cn, cn2, cr, cr2, cd, cd2, ck, ck2, cp, cp2, cs) schedule(static)
            for (int i = 0; i < np; ++i) {  // logits row i: after token p0 + i, scoring token p0 + i + 1
                const float* lg = &logits[(size_t)i * V];
                const int tok = chunk[p0 + i + 1], row = p0 + i - first;
                double l;
                if (!kl_file.empty()) {
                    const RowStats s = compare_row(V, lg, &rows[(size_t)row * nv], tok);
                    st[i] = s;
                    l = s.nll;
                    cr += s.nll_ref; cr2 += s.nll_ref * s.nll_ref;
                    cd += s.nll - s.nll_ref; cd2 += (s.nll - s.nll_ref) * (s.nll - s.nll_ref);
                    ck += s.kl; ck2 += s.kl * s.kl;
                    cp += s.p_diff; cp2 += (double)s.p_diff * s.p_diff;
                    cs += s.same_top;
                } else {
                    if (!save_file.empty()) encode_row(V, lg, &rows[(size_t)row * nv]);
                    float mx = lg[0];
                    for (int v = 1; v < V; ++v) mx = std::max(mx, lg[v]);
                    double sum = 0.0;
                    for (int v = 0; v < V; ++v) sum += exp((double)(lg[v] - mx));
                    l = -((double)(lg[tok] - mx) - log(sum));
                }
                cn += l;
                cn2 += l * l;
            }
            for (const RowStats& s : st) { kls.push_back(s.kl); p_diffs.push_back(fabsf(s.p_diff)); }
            nll += cn; nll2 += cn2; nll_ref += cr; nll_ref2 += cr2; d_nll += cd; d_nll2 += cd2;
            kl_sum += ck; kl2 += ck2; pd_sum += cp; pd2 += cp2; same_top += cs;
            count += np;
        }
        if (!save_file.empty()) fwrite(rows.data(), sizeof(uint16_t), rows.size(), ref);
        if (kl_file.empty()) printf("[%d]%.4f%s", c + 1, exp(nll / count), (c + 1) % 10 == 0 || c + 1 == n_chunks ? "\n" : " ");
        else printf("[%d]%.4f kl %.4f%s", c + 1, exp(nll / count), kl_sum / count, (c + 1) % 5 == 0 || c + 1 == n_chunks ? "\n" : " ");
        fflush(stdout);
    }
    if (ref) fclose(ref);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // the standard error of a mean over the scored positions
    auto err = [count](double sum, double sum2) {
        const double mean = sum / count, var = sum2 / count - mean * mean;
        return sqrt(var > 0.0 ? var / (count - 1) : 0.0);
    };
    const double mean = nll / count;
    printf("perplexity: %.4f +/- %.4f (%lld tokens, %d chunks, %.1f s, %.0f tok/s)\n", exp(mean), exp(mean) * err(nll, nll2), count,
           n_chunks, s, (double)n_chunks * n_ctx / s);
    if (!kl_file.empty()) {
        const double mr = nll_ref / count;
        printf("reference perplexity: %.4f +/- %.4f; ln(PPL / PPL_ref) %.5f +/- %.5f\n", exp(mr), exp(mr) * err(nll_ref, nll_ref2),
               d_nll / count, err(d_nll, d_nll2));
        printf("KL divergence: mean %.6f +/- %.6f, median %.6f, 90%% %.6f, 95%% %.6f, 99%% %.6f, 99.9%% %.6f, max %.6f\n",
               kl_sum / count, err(kl_sum, kl2), percentile(kls, 0.5), percentile(kls, 0.9), percentile(kls, 0.95),
               percentile(kls, 0.99), percentile(kls, 0.999), *std::max_element(kls.begin(), kls.end()));
        printf("token probability: mean difference %+.3f%% +/- %.3f%%, RMS %.3f%%, 99%% |diff| %.3f%%; same top token %.3f%%\n",
               100.0 * pd_sum / count, 100.0 * err(pd_sum, pd2), 100.0 * sqrt(pd2 / count), 100.0 * percentile(p_diffs, 0.99),
               100.0 * same_top / count);
    }
    inference_free_tokens(toks, n_tok);
    inference_free_model(m);
    return 0;
}
