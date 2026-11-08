// split_verify -- prefill is split-invariant: a prompt prefilled at once and prefilled in two calls cut at any point
// (as a reused prefix, a store resume or --parallel's chunks cut it) gives the same bits, every layer's output of
// every token and the last token's logits (qwen_model's prefill mode: every batch size on the kernels of a large
// batch, attention over position ranges fixed per context). Usage (from build/co, for the kernels):
//   split_verify MODEL.gguf [n_tokens = 1200] [cut ...]
// Without cuts: a set around the engine's batch sizes and tile widths, odd and even. Per cut: PASS, or the first
// layer and token that differ and how many tokens differ there. Exit 1 if any cut differs.

#include "inference.h"
#include "../src/qwen_model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>


struct Trace {
    int n_layer = 0, n = 0;
    std::vector<uint64_t> h;  // [layer][pos]: FNV-1a of the residual stream's bits
    void init(int layers, int tokens) {
        n_layer = layers;
        n = tokens;
        h.assign((size_t)layers * tokens, 0);
    }
};

static void on_layer(int l, int pos, const float* x, int n, void* user) {
    Trace* t = static_cast<Trace*>(user);
    if (l < 0 || l >= t->n_layer || pos < 0 || pos >= t->n) return;
    uint64_t h = 1469598103934665603ull;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(x);
    for (size_t i = 0; i < (size_t)n * sizeof(float); ++i) h = (h ^ b[i]) * 1099511628211ull;
    t->h[(size_t)l * t->n + pos] = h;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: split_verify MODEL.gguf [n_tokens] [cut ...]\n");
        return 2;
    }
    const int want = argc > 2 ? atoi(argv[2]) : 1200;
    inference_model_t* m = inference_load_model(argv[1]);
    if (!m || inference_kv_cache_init(m, 4096) != 0) {
        fprintf(stderr, "cannot load %s: %s\n", argv[1], inference_get_last_error());
        return 2;
    }
    // a prompt of `want` tokens: numbered lines of varied words (no file needed)
    std::string text;
    for (int i = 0; (int)text.size() < want * 8; ++i)
        text += "Line " + std::to_string(i) + ": the " + (i % 3 ? "quick" : "slow") + " step " +
                std::to_string(i * 7 % 97) + " of the build " + (i % 5 ? "passed" : "failed") + ".\n";
    int n = 0;
    int32_t* all = inference_tokenize(m, text.c_str(), &n);
    if (!all || n < want) {
        fprintf(stderr, "tokenize failed (%d tokens)\n", n);
        return 2;
    }
    n = want;
    std::vector<int32_t> ids(all, all + n);
    free(all);
    std::vector<int> cuts;
    for (int i = 3; i < argc; ++i) cuts.push_back(atoi(argv[i]));
    if (cuts.empty())
        for (int c : {1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 255, 256, 257, 511, 513, 1023, 1024, 1025})
            if (c < n) cuts.push_back(c);
    cuts.push_back(n - 1);
    cuts.push_back(n - 16);
    cuts.push_back(n - 33);

    qwen_model_t* qm = inference_get_qwen_model(m);
    const int V = inference_vocab_size(m);
    int layers = 0;
    for (;;) {  // the number of layers: the callback's highest index on a one-token prefill
        Trace probe;
        probe.init(512, 1);
        qwen_model_set_layer_callback(qm, on_layer, &probe);
        inference_reset(m);
        if (inference_prefill(m, ids.data(), 1) != 0) return 2;
        for (int l = 0; l < 512; ++l)
            if (probe.h[(size_t)l]) layers = l + 1;
        break;
    }
    std::vector<int> vid(V);
    std::vector<float> ref_logits(V), logits(V);
    auto run = [&](Trace& t, int cut, std::vector<float>& lg) {
        t.init(layers, n);
        qwen_model_set_layer_callback(qm, on_layer, &t);
        inference_reset(m);
        int rc = 0;
        if (cut > 0) rc = inference_prefill(m, ids.data(), cut);
        if (rc == 0) rc = inference_prefill(m, ids.data() + cut, n - cut);
        if (rc == 0) rc = inference_get_logits(m, vid.data(), lg.data());
        qwen_model_set_layer_callback(qm, nullptr, nullptr);
        return rc;
    };
    Trace ref, got;
    if (run(ref, 0, ref_logits) != 0) {
        fprintf(stderr, "prefill failed: %s\n", inference_get_last_error());
        return 2;
    }
    printf("split_verify: %d tokens, %d layers, %zu cuts%s\n", n, layers, cuts.size(),
           getenv("QMATVEC_PREFILL_SPLIT_VARIANT") ? " (QMATVEC_PREFILL_SPLIT_VARIANT: the old batch-size paths)" : "");
    int fails = 0;
    for (int cut : cuts) {
        if (run(got, cut, logits) != 0) {
            printf("  [FAIL] cut %d: prefill failed: %s\n", cut, inference_get_last_error());
            ++fails;
            continue;
        }
        int fl = -1, fp = -1, count = 0;
        for (int l = 0; l < layers && fl < 0; ++l)
            for (int p = 0; p < n; ++p)
                if (ref.h[(size_t)l * n + p] != got.h[(size_t)l * n + p]) {
                    if (fl < 0) fl = l, fp = p;
                    ++count;
                }
        float md = 0.0f;
        for (int i = 0; i < V; ++i) md = std::fmax(md, std::fabs(ref_logits[i] - logits[i]));
        if (fl < 0 && md == 0.0f) {
            printf("  [PASS] cut %d\n", cut);
        } else {
            ++fails;
            if (fl >= 0)
                printf("  [FAIL] cut %d: layer %d differs first, at token %d (%d tokens of that layer differ); logits max |d| %.4g\n",
                       cut, fl, fp, count, md);
            else
                printf("  [FAIL] cut %d: the layers agree, the logits differ (max |d| %.4g)\n", cut, md);
        }
    }
    printf("%s (%d of %zu cuts differ)\n", fails ? "FAILED" : "PASSED", fails, cuts.size());
    inference_free_model(m);
    return fails ? 1 : 0;
}
