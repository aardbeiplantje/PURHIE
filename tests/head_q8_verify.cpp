// head_q8_verify.cpp <model.gguf> <text file> [chunks = 200] [tokens a chunk = 128] -- what the int8 (Q8_0) copy of
// a float LM head (inference_set_lm_head_q8) does to the next-token distribution: over the chunks' last positions,
// the float head's logits against the copy's: the argmax agreement, the KL divergence, and the negative log
// likelihood of the text's real next token under each. Exit 1 if the argmax differs in more than 2% of the chunks
// or the mean KL exceeds 1e-3.
#include "inference.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void log_softmax(const std::vector<float>& lg, std::vector<double>& lp) {
    const float mx = *std::max_element(lg.begin(), lg.end());
    double s = 0.0;
    for (float v : lg) s += exp((double)v - mx);
    const double ls = log(s) + mx;
    for (size_t i = 0; i < lg.size(); ++i) lp[i] = lg[i] - ls;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model.gguf> <text> [chunks] [tokens]\n", argv[0]); return 2; }
    const int chunks = argc > 3 ? atoi(argv[3]) : 200, T = argc > 4 ? atoi(argv[4]) : 128;
    FILE* f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 2; }
    std::string text((size_t)chunks * T * 8, '\0');
    text.resize(fread(&text[0], 1, text.size(), f));
    fclose(f);
    inference_model_t* m = inference_load_model(argv[1]);
    if (!m || inference_kv_cache_init(m, T + 16) != 0) { fprintf(stderr, "%s\n", inference_get_last_error()); return 2; }
    int n = 0;
    const int32_t* ids = inference_tokenize(m, text.c_str(), &n);
    if (!ids || n < (T + 1) * 2) { fprintf(stderr, "too little text (%d tokens)\n", n); return 2; }
    const int rc = inference_set_lm_head_q8(m, 1);
    if (rc != 0) { printf("the LM head is not a float one (%d): nothing to compare\n", rc); return 0; }
    const int V = inference_vocab_size(m), N = std::min(chunks, n / (T + 1));
    std::vector<float> a(V), b(V);
    std::vector<int> vid(V);
    std::vector<double> la(V), lb(V);
    double kl = 0.0, klmax = 0.0, nll_a = 0.0, nll_b = 0.0, dmax = 0.0;
    int same = 0;
    for (int c = 0; c < N; ++c) {
        const int32_t* t = ids + (size_t)c * (T + 1);
        for (int q8 = 0; q8 < 2; ++q8) {
            inference_set_lm_head_q8(m, q8);
            inference_reset(m);
            if (inference_prefill(m, t, T) != 0 || inference_get_logits(m, vid.data(), (q8 ? b : a).data()) != 0) {
                fprintf(stderr, "%s\n", inference_get_last_error());
                return 2;
            }
        }
        log_softmax(a, la);
        log_softmax(b, lb);
        double k = 0.0;
        for (int i = 0; i < V; ++i) {
            k += exp(la[i]) * (la[i] - lb[i]);
            dmax = std::max(dmax, (double)fabsf(a[i] - b[i]));
        }
        kl += k;
        klmax = std::max(klmax, k);
        nll_a -= la[t[T]];
        nll_b -= lb[t[T]];
        same += std::max_element(a.begin(), a.end()) - a.begin() == std::max_element(b.begin(), b.end()) - b.begin();
    }
    printf("%d positions: argmax the same at %d (%.2f%%); KL(float || int8) mean %.3g, max %.3g; max |logit difference| %.3g\n",
           N, same, 100.0 * same / N, kl / N, klmax, dmax);
    printf("next-token NLL: float %.5f, int8 %.5f (perplexity %.4f vs %.4f, %+.3f%%)\n", nll_a / N, nll_b / N,
           exp(nll_a / N), exp(nll_b / N), 100.0 * (exp(nll_b / N) / exp(nll_a / N) - 1.0));
    inference_free_model(m);
    return same * 100 < N * 98 || kl / N > 1e-3;
}
