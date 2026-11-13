// mm_te_verify: the MiniMax-H3-Image text conditioning (PLAN.md, MiniMax M1): the converted Qwen3-VL text encoder run by the
// engine (qwen3 layout) against transformers' hidden_states[50] (the output after decoder layer 50, no final norm) for the
// reference prompts (tokens with no special tokens; /data/minimax-ref/te, from mm_te_ref.py).
//
//   mm_te_verify <text-encoder.gguf> <reference dir>
//
// Per prompt: the engine's prefill over the reference token ids, the residual after layer 49 (0-based) captured per token
// by the layer callback, compared row by row: cosine, and the largest difference against the peak of the reference.
#include "inference.h"
#include "inference_internal.h"
#include "qwen_model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

// A little-endian .npy of float32 or int32 (version 1.0): the header's shape and the data after it
struct Npy {
    std::vector<size_t> shape;
    std::vector<char> data;
    bool read(const std::string& path, const char* descr) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        std::vector<char> b;
        char buf[1 << 16];
        size_t k;
        while ((k = std::fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + k);
        std::fclose(f);
        if (b.size() < 10 || std::memcmp(b.data(), "\x93NUMPY", 6) != 0) return false;
        const size_t hl = (uint8_t)b[8] | ((uint8_t)b[9] << 8);
        const std::string hd(b.data() + 10, hl);
        if (hd.find(descr) == std::string::npos) return false;
        const size_t lp = hd.find('('), rp = hd.find(')');
        std::string dims = hd.substr(lp + 1, rp - lp - 1);
        size_t pos = 0;
        while (pos < dims.size()) {
            while (pos < dims.size() && (dims[pos] == ' ' || dims[pos] == ',')) ++pos;
            if (pos >= dims.size()) break;
            shape.push_back((size_t)std::stoull(dims.substr(pos)));
            while (pos < dims.size() && dims[pos] != ',') ++pos;
        }
        data.assign(b.begin() + 10 + (long)hl, b.end());
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <text-encoder.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    inference_model* m = inference_load_model(argv[1]);
    if (!m) {
        std::fprintf(stderr, "load failed: %s\n", inference_get_last_error());
        return 1;
    }
    if (inference_kv_cache_init(m, 256) != 0) {  // the qwen model is created with the cache (the prompts are short)
        std::fprintf(stderr, "kv init: %s\n", inference_get_last_error());
        return 1;
    }
    qwen_model_t* qm = inference_get_qwen_model(m);
    const int E = m->n_embd, L = 50;  // hidden_states[50]: the residual after layer index 49
    const int V = m->vocab_size;
    if (!qm || E != 5120) {
        std::fprintf(stderr, "not the Qwen3-VL text encoder (width %d)\n", E);
        return 1;
    }
    int failures = 0;
    for (int i = 0; i < 3; ++i) {
        Npy ids, want;
        if (!ids.read(dir + "/ids_" + std::to_string(i) + ".npy", "<i4") ||
            !want.read(dir + "/hs50_" + std::to_string(i) + ".npy", "<f4")) {
            std::fprintf(stderr, "cannot read reference %d in %s\n", i, dir.c_str());
            return 2;
        }
        const int T = (int)ids.shape[0];
        const float* ref = reinterpret_cast<const float*>(want.data.data());
        const int32_t* tok = reinterpret_cast<const int32_t*>(ids.data.data());
        std::vector<int> tokens(tok, tok + T);
        std::vector<float> got((size_t)T * E);
        struct Ctx {
            std::vector<float>* out;
            int T, E;
        } ctx{&got, T, E};
        qwen_model_set_layer_callback(qm, [](int l, int pos, const float* x, int n, void* u) {
            Ctx* c = static_cast<Ctx*>(u);
            if (l == 49) std::memcpy(c->out->data() + (size_t)pos * c->E, x, sizeof(float) * (size_t)n);
        }, &ctx);
        std::vector<float> logits(V);
        const int rc = qwen_model_prefill(qm, tokens.data(), T, 0, logits.data());
        qwen_model_set_layer_callback(qm, nullptr, nullptr);
        if (rc != 0) {
            std::fprintf(stderr, "prefill failed: %s\n", inference_get_last_error());
            return 1;
        }
        double worst_cos = 1.0, peak = 0.0, worst_diff = 0.0;
        for (int t = 0; t < T; ++t) {
            const float* a = got.data() + (size_t)t * E;
            const float* b = ref + (size_t)t * E;
            double dot = 0, na = 0, nb = 0;
            for (int j = 0; j < E; ++j) {
                dot += (double)a[j] * b[j];
                na += (double)a[j] * a[j];
                nb += (double)b[j] * b[j];
                peak = std::fmax(peak, std::fabs((double)b[j]));
                worst_diff = std::fmax(worst_diff, std::fabs((double)a[j] - b[j]));
            }
            const double cos = dot / std::sqrt(na * nb);
            if (cos < worst_cos) worst_cos = cos;
        }
        const bool ok = worst_cos > 0.999;
        if (!ok) ++failures;
        std::printf("prompt %d: %d tokens, layer 50 (hidden_states[50]) worst row cosine %.7f, max |diff| %.3e of the peak %.2f  %s\n",
                    i, T, worst_cos, worst_diff, peak, ok ? "ok" : "FAIL");
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
