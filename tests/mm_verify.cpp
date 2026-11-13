// mm_verify: an image chat through the engine (phase V4 of PLAN.md), against transformers' reference for the same
// prompt: the vision tower's features (src/vision.cpp) at the image pad tokens, the M-RoPE positions (src/mrope.cpp),
// the prefill with the overrides (inference_set_mm), the last position's logits, then 32 greedy tokens.
//
//   mm_verify <text model.gguf> <mmproj.gguf> <reference dir>   (run from build/co for the kernels)
//
// The reference directory: the V1 dump (input_ids, image_grid_thw, pixel_values, last_logits, reply_ids .npy). Pass
// when the logits' cosine is at least 0.99 and the argmax equals the reference's, and the 32 tokens are the same
// (a near-tie can differ after the first; reported).
#include "inference.h"
#include "mrope.h"
#include "vision.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Npy {
    std::vector<long long> shape;
    std::string descr;
    std::vector<char> data;
};

Npy load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    char magic[8];
    f.read(magic, 8);
    uint16_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 2);
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    Npy n;
    const size_t d = hdr.find("'descr': '") + 10;
    n.descr = hdr.substr(d, hdr.find('\'', d) - d);
    const size_t s = hdr.find("'shape': (") + 10;
    const std::string body = hdr.substr(s, hdr.find(')', s) - s);
    for (size_t p = 0; p < body.size();) {
        size_t c = body.find(',', p);
        if (c == std::string::npos) c = body.size();
        const std::string tok = body.substr(p, c - p);
        if (tok.find_first_of("0123456789") != std::string::npos) n.shape.push_back(std::atoll(tok.c_str()));
        p = c + 1;
    }
    long long count = 1;
    for (long long v : n.shape) count *= v;
    const size_t esz = (n.descr == "<f4" || n.descr == "<i4") ? 4 : 8;
    n.data.resize((size_t)count * esz);
    f.read(n.data.data(), (std::streamsize)n.data.size());
    return n;
}

std::vector<int64_t> as_i64(const Npy& n) {
    std::vector<int64_t> v(n.data.size() / 8);
    std::memcpy(v.data(), n.data.data(), n.data.size());
    return v;
}

std::vector<float> as_f32(const Npy& n) {
    std::vector<float> v(n.data.size() / 4);
    std::memcpy(v.data(), n.data.data(), n.data.size());
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <text model.gguf> <mmproj.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[3];
    const Npy ids_n = load(dir + "/input_ids.npy");
    const Npy grid_n = load(dir + "/image_grid_thw.npy");
    const Npy pix_n = load(dir + "/pixel_values.npy");
    const Npy want_logits_n = load(dir + "/last_logits.npy");
    const Npy reply_n = load(dir + "/reply_ids.npy");
    const std::vector<int64_t> ids64 = as_i64(ids_n);
    const int n = (int)ids64.size();
    const int64_t* grid = reinterpret_cast<const int64_t*>(grid_n.data.data());
    const int gh = (int)grid[1], gw = (int)grid[2];
    const std::vector<float> pixels = as_f32(pix_n);
    const std::vector<float> want = as_f32(want_logits_n);
    const std::vector<int64_t> reply = as_i64(reply_n);
    const int image_token = 248056, merge = 2;

    // the image's features
    vision_model_t* vm = vision_open(argv[2]);
    if (!vm) { std::fprintf(stderr, "vision_open: %s\n", vision_last_error()); return 1; }
    const int E = vision_n_out(vm);
    std::vector<float> feats((size_t)(gh / merge) * (gw / merge) * E);
    if (vision_encode(vm, pixels.data(), gh, gw, feats.data(), nullptr, nullptr) != 0) {
        std::fprintf(stderr, "vision_encode: %s\n", vision_last_error());
        return 1;
    }
    vision_close(vm);

    // the overrides: the image rows take the features in order; the M-RoPE positions of the whole prompt
    std::vector<int32_t> ids32(n);
    std::vector<unsigned char> use(n, 0);
    std::vector<float> emb((size_t)n * E, 0.0f);
    int k = 0;
    for (int t = 0; t < n; ++t) {
        ids32[t] = (int32_t)ids64[t];
        if (ids32[t] == image_token) {
            use[t] = 1;
            std::memcpy(&emb[(size_t)t * E], &feats[(size_t)k * E], sizeof(float) * E);
            ++k;
        }
    }
    if ((size_t)k != feats.size() / E) { std::fprintf(stderr, "%d image pads, %zu features\n", k, feats.size() / E); return 1; }
    const int32_t hw[2] = {gh, gw};
    std::vector<int64_t> pos3(3 * (size_t)n);
    int64_t delta = 0;
    if (mrope_positions(ids32.data(), n, hw, 1, merge, image_token, pos3.data(), &delta) != 0) {
        std::fprintf(stderr, "mrope_positions failed\n");
        return 1;
    }

    inference_model_t* m = inference_load_model(argv[1]);
    if (!m) { std::fprintf(stderr, "load: %s\n", inference_get_last_error()); return 1; }
    if (inference_kv_cache_init(m, 1024) != 0) { std::fprintf(stderr, "kv: %s\n", inference_get_last_error()); return 1; }
    const int V = inference_vocab_size(m);
    if (inference_set_mm(m, 0, n, emb.data(), use.data(), pos3.data()) != 0) {
        std::fprintf(stderr, "set_mm: %s\n", inference_get_last_error());
        return 1;
    }
    if (inference_prefill(m, ids32.data(), n) != 0) { std::fprintf(stderr, "prefill: %s\n", inference_get_last_error()); return 1; }
    if (inference_mm_rope_delta(m) != (int)delta) {
        std::fprintf(stderr, "rope delta %d, transformers %lld\n", inference_mm_rope_delta(m), (long long)delta);
        return 1;
    }

    std::vector<int> vocab_ids(V);
    std::vector<float> logits(V);
    if (inference_get_logits(m, vocab_ids.data(), logits.data()) != 0) {
        std::fprintf(stderr, "logits: %s\n", inference_get_last_error());
        return 1;
    }
    double dot = 0, na = 0, nb = 0;
    int am = 0;
    for (int i = 0; i < V; ++i) {
        dot += (double)logits[i] * want[i];
        na += (double)logits[i] * logits[i];
        nb += (double)want[i] * want[i];
        if (logits[i] > logits[am]) am = i;
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    int wam = 0;
    for (int i = 1; i < V; ++i)
        if (want[i] > want[wam]) wam = i;
    std::printf("prefill %d tokens (%d image rows, rope delta %lld): last logits cosine %.6f, argmax %d (reference %d)\n", n,
                k, (long long)delta, cos, am, wam);

    // 32 greedy tokens, as the reference's reply
    int next = am, same = 0;
    std::vector<int> out;
    for (int i = 0; i < (int)reply.size(); ++i) {
        out.push_back(next);
        same += next == (int)reply[i];
        int nxt = 0;
        if (inference_decode_token(m, next, &nxt) != 0) { std::fprintf(stderr, "decode: %s\n", inference_get_last_error()); return 1; }
        next = nxt;
    }
    std::printf("greedy: %d of %zu tokens equal the reference's reply\n", same, reply.size());
    inference_free_model(m);
    const bool pass = cos >= 0.99 && am == wam && same == (int)reply.size();
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
