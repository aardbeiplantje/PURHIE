// mtp_own_verify -- a model's own MTP block as its draft head (inference_set_draft_model with the model's file).
//
//   mtp_own_verify <model.gguf> <its-mtp-head.gguf> [no-mtp-model.gguf]
//
// <model.gguf> carries an MTP block after its decoder layers (as converted from Qwen3.8-Flash-Next's BF16 release:
// blk.N.nextn.*, the final mixer nextn.hc_head_*, fused hc_*_down_inject) and <its-mtp-head.gguf> is the same block
// as a sidecar (the release's layout); tools/gen_qwen4exp_tiny.py writes both with TINY_MTP=<head>.
// 1. The same drafts: greedy decoding, the head's single and chained (3) drafts at every step, with the model's
//    own block and with the sidecar: equal at every step.
// 2. inference_generate_stream with the own block (1 and 3 drafts a pass, adaptive) = without (greedy and sampled).
// 3. Errors: a model's own file without an MTP block ([no-mtp-model.gguf]) is refused; unloading and loading the
//    own block again works.
// MTP_OWN_STAGE=<dir>: the model staged in <dir> (inference_load_opts.stage_dir).

#include "inference.h"
#include "qwen_model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static int argmax(const float* x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}

// Greedy decoding of n tokens after the prompt; per step the head's draft and its 3 chained drafts
static bool drafts_of(inference_model_t* m, const std::vector<int>& prompt, int n, std::vector<int>& out) {
    qwen_model_t* q = inference_get_qwen_model(m);
    const int V = inference_vocab_size(m);
    std::vector<float> logits((size_t)V);
    out.clear();
    if (qwen_model_prefill(q, prompt.data(), (int)prompt.size(), 0, logits.data()) != 0) return false;
    int pos = (int)prompt.size(), next = argmax(logits.data(), V);
    for (int i = 0; i < n; ++i) {
        int ch[3] = {-1, -1, -1};
        const int nd = qwen_model_mtp_drafts(q, next, pos, ch, 3);
        const int d = qwen_model_mtp_draft(q, next, pos);
        if (nd < 1 || d < 0) return false;
        out.push_back(d);
        out.insert(out.end(), ch, ch + 3);
        if (qwen_model_forward(q, next, pos, logits.data()) != 0) return false;
        ++pos;
        next = argmax(logits.data(), V);
        out.push_back(next);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <its-mtp-head.gguf> [no-mtp-model.gguf]\n", argv[0]);
        return 2;
    }
    // MTP_OWN_STAGE=<dir>: the model staged there (the GPU reads the stage; the MTP block is pinned on its own)
    inference_load_opts lo = {getenv("MTP_OWN_STAGE")};
    inference_model_t* m = inference_load_model_ex(argv[1], &lo);
    if (m) printf("  %s\n", inference_stage_info(m));
    if (!m || inference_kv_cache_init(m, 2048) != 0) {
        fprintf(stderr, "setup failed: %s\n", inference_get_last_error());
        return 1;
    }
    const char* text = "<|im_start|>user\nWrite a Python function that returns the n-th Fibonacci number."
                       "<|im_end|>\n<|im_start|>assistant\n";
    int n = 0;
    int32_t* t = inference_tokenize(m, text, &n);
    std::vector<int> prompt(t, t + n);
    std::vector<int32_t> prompt32(t, t + n);
    inference_free_tokens(t, n);
    char what[256];

    // 1. own block vs sidecar: the same drafts
    std::vector<int> own, side;
    const bool a = inference_set_draft_model(m, argv[1]) == 0 && drafts_of(m, prompt, 48, own);
    if (!a) printf("    own block: %s\n", inference_get_last_error());
    const bool b = inference_set_draft_model(m, argv[2]) == 0 && drafts_of(m, prompt, 48, side);
    if (!b) printf("    sidecar: %s\n", inference_get_last_error());
    int diff = -1;
    for (size_t i = 0; i < own.size() && i < side.size() && diff < 0; ++i)
        if (own[i] != side[i]) diff = (int)i;
    snprintf(what, sizeof what, "the own MTP block drafts what the sidecar drafts: %zu drafts and tokens (first "
             "difference %d)", own.size(), diff);
    check(a && b && own.size() == 48 * 5 && own == side, what);

    // 2. generation with the own block = without
    std::vector<int32_t> ref;
    for (float temp : {0.0f, 0.8f}) {
        for (int with : {0, 1, 3, -1}) {  // -1: adaptive
            if (inference_set_draft_model(m, with ? argv[1] : nullptr) != 0 ||
                (with && inference_set_draft_tokens(m, with < 0 ? 0 : with) != 0)) {
                check(false, inference_get_last_error());
                continue;
            }
            inference_reset(m);
            inference_gen_options o = inference_gen_default_options();
            o.sampler.temperature = temp;
            o.sampler.seed = 7;
            o.max_new_tokens = 96;
            o.ignore_eog = 1;
            std::vector<int32_t> out(96);
            int nout = 0;
            inference_stop_reason why;
            if (inference_generate_stream(m, prompt32.data(), n, &o, out.data(), 96, &nout, &why) != 0) {
                check(false, inference_get_last_error());
                continue;
            }
            out.resize((size_t)nout);
            if (with == 0) { ref = out; continue; }
            snprintf(what, sizeof what, "temperature %.1f, %s: %d tokens with the own block = without%s", temp,
                     with < 0 ? "adaptive drafts" : with == 1 ? "1 draft a pass" : "3 drafts a pass", nout,
                     out == ref ? "" : " -- DIFFER");
            check(out == ref && nout == 96, what);
        }
    }

    // 3. errors
    if (argc > 3) {
        inference_model_t* p = inference_load_model(argv[3]);
        const int rc = p ? inference_set_draft_model(p, argv[3]) : 0;
        snprintf(what, sizeof what, "a model without an MTP block: refused (%s)", p ? inference_get_last_error() : "?");
        check(p && rc != 0 && strstr(inference_get_last_error(), "no MTP block"), what);
        if (p) inference_free_model(p);
    }
    check(inference_set_draft_model(m, nullptr) == 0 && inference_set_draft_model(m, argv[1]) == 0 &&
              inference_set_draft_model(m, argv[1]) == 0, "unloaded and loaded again");
    inference_kv_cache_free(m);
    inference_free_model(m);
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
