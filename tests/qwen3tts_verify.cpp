// qwen3tts_verify: the talker and its code predictor (src/qwen3tts.cpp, phase T2 of PLAN.md) against qwen-tts 0.1.1 for the
// same in-context voice clone (phase T1/T2's reference): the prefill of the reference clip's codes and the text, then
// the talker over the teacher-forced frames (the reference's own codes), and the code predictor's logits for them.
//
//   qwen3tts_verify <qwen3tts.gguf> <reference dir>
//
// The GGUF: rocm-convert of the checkpoint (perl/ROCm-Inference: the Qwen3-TTS converter, F32). The reference directory
// (/data/audio-ref/tts): text_ids.npy, ref_ids.npy, ref_codes.npy (74 x 16), talker_codes.npy (50 x 16), t2_spk.npy,
// t2_prefill.npy (84 x 1024), t2_trailing.npy (1 x 1024), t2_pad.npy, t2_hidden.npy (133 x 1024), t2_logits.npy (133 x
// 3072), t2_pred_logits.npy (50 x 15 x 2048). Each stage passes when every row's cosine is at least 0.9999 and the largest
// difference is at most 1e-4 of the reference's largest value (float32 accumulation over 28 blocks).
#include "qwen3tts.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Npy {
    std::vector<long long> shape;
    std::string descr;
    std::vector<char> data;
};

Npy load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) { std::fprintf(stderr, "%s: not a .npy\n", path.c_str()); std::exit(2); }
    uint16_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 2);
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    Npy n;
    const size_t d = hdr.find("'descr': '") + 10;
    n.descr = hdr.substr(d, hdr.find('\'', d) - d);
    const size_t s = hdr.find("'shape': (") + 10;
    std::stringstream ss(hdr.substr(s, hdr.find(')', s) - s));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.find_first_of("0123456789") != std::string::npos) n.shape.push_back(std::atoll(tok.c_str()));
    }
    long long count = 1;
    for (long long v : n.shape) count *= v;
    const size_t esz = (n.descr == "<f4" || n.descr == "<i4") ? 4 : 8;
    if (n.descr != "<f4" && n.descr != "<i8") { std::fprintf(stderr, "%s: dtype %s\n", path.c_str(), n.descr.c_str()); std::exit(2); }
    n.data.resize((size_t)count * esz);
    f.read(n.data.data(), (std::streamsize)n.data.size());
    if (hdr.find("'fortran_order': True") != std::string::npos && n.shape.size() == 2) {
        // column-major on disk: transpose to the row-major layout the reads below expect
        const size_t R = (size_t)n.shape[0], Cc = (size_t)n.shape[1];
        std::vector<char> t(n.data.size());
        for (size_t r = 0; r < R; ++r)
            for (size_t c = 0; c < Cc; ++c) std::memcpy(&t[(r * Cc + c) * esz], &n.data[(c * R + r) * esz], esz);
        n.data.swap(t);
    }
    return n;
}

std::vector<float> as_f32(const Npy& n) {
    std::vector<float> v(n.data.size() / 4);
    std::memcpy(v.data(), n.data.data(), n.data.size());
    return v;
}

std::vector<int> as_ints(const Npy& n) {
    std::vector<int> v;
    if (n.descr == "<i8") {
        const size_t c = n.data.size() / 8;
        v.resize(c);
        for (size_t i = 0; i < c; ++i) v[i] = (int)reinterpret_cast<const int64_t*>(n.data.data())[i];
    } else {
        const size_t c = n.data.size() / 4;
        v.resize(c);
        std::memcpy(v.data(), n.data.data(), n.data.size());
    }
    return v;
}

// Each row's worst cosine and the largest difference of got against want, then the verdict
bool compare(const char* name, const float* got, const float* want, size_t rows, size_t cols) {
    double min_cos = 1.0, max_abs = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < rows * cols; ++i) max_ref = std::fmax(max_ref, std::fabs(want[i]));
    for (size_t r = 0; r < rows; ++r) {
        double dot = 0, na = 0, nb = 0;
        for (size_t c = 0; c < cols; ++c) {
            const double a = got[r * cols + c], b = want[r * cols + c];
            dot += a * b;
            na += a * a;
            nb += b * b;
            max_abs = std::fmax(max_abs, std::fabs(a - b));
        }
        min_cos = std::fmin(min_cos, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
    }
    const bool ok = min_cos >= 0.9999 && max_abs <= 1e-4 * max_ref;
    std::printf("%-22s %4zu x %4zu  worst row cosine %.7f  max |diff| %.3e (%.1e of %.2f)  %s\n", name, rows, cols, min_cos,
                max_abs, max_ref > 0 ? max_abs / max_ref : 0.0, max_ref, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <qwen3tts.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    const std::vector<int> text_ids = as_ints(load_npy(dir + "/text_ids.npy"));
    const std::vector<int> ref_ids = as_ints(load_npy(dir + "/ref_ids.npy"));
    const std::vector<int> ref_codes = as_ints(load_npy(dir + "/ref_codes.npy"));
    const std::vector<int> codes = as_ints(load_npy(dir + "/talker_codes.npy"));
    const std::vector<float> spk = as_f32(load_npy(dir + "/t2_spk.npy"));
    const std::vector<float> want_prefill = as_f32(load_npy(dir + "/t2_prefill.npy"));
    const std::vector<float> want_trailing = as_f32(load_npy(dir + "/t2_trailing.npy"));
    const std::vector<float> want_pad = as_f32(load_npy(dir + "/t2_pad.npy"));
    const std::vector<float> want_hidden = as_f32(load_npy(dir + "/t2_hidden.npy"));
    const std::vector<float> want_logits = as_f32(load_npy(dir + "/t2_logits.npy"));
    const std::vector<float> want_pred = as_f32(load_npy(dir + "/t2_pred_logits.npy"));

    qwen3tts_model_t* m = qwen3tts_open(argv[1]);
    if (!m) { std::fprintf(stderr, "qwen3tts_open: %s\n", qwen3tts_last_error()); return 1; }
    const int C = qwen3tts_n_embd(m), ncb = qwen3tts_n_codebooks(m), V = qwen3tts_codec_vocab(m), VP = qwen3tts_pred_vocab(m);
    const int T = (int)(codes.size() / ncb), R = (int)(ref_codes.size() / ncb);
    bool pass = true;
    const int lang = qwen3tts_language_id(m, "english");
    std::printf("english id %d (expected 2050)%s\n", lang, lang == 2050 ? "" : "  FAIL");
    pass = pass && lang == 2050;

    // the prefill (the in-context voice clone's), the trailing text and the pad
    std::vector<float> prefill(128 * (size_t)C), trailing(8 * (size_t)C), pad(C);
    int P = 0, nt = 0;
    if (qwen3tts_build_prefill(m, text_ids.data(), (int)text_ids.size(), ref_ids.data(), (int)ref_ids.size(), ref_codes.data(), R,
                               spk.data(), lang, prefill.data(), 128, &P, trailing.data(), 8, &nt, pad.data()) != 0) {
        std::fprintf(stderr, "qwen3tts_build_prefill: %s\n", qwen3tts_last_error());
        return 1;
    }
    if ((size_t)P * C != want_prefill.size()) { std::fprintf(stderr, "prefill has %d positions, the reference %zu\n", P, want_prefill.size() / C); return 1; }
    pass = compare("prefill", prefill.data(), want_prefill.data(), P, C) && pass;
    pass = compare("trailing", trailing.data(), want_trailing.data(), nt, C) && pass;
    pass = compare("pad", pad.data(), want_pad.data(), 1, C) && pass;

    // the talker over the prefill and the frames' inputs (frame t's input is the summed embeddings of its codes plus the
    // trailing text, the pad after the trailing rows run out; the last frame's input is not needed)
    const int L = P + T - 1;
    std::vector<float> X((size_t)L * C), hidden((size_t)L * C), logits((size_t)L * V);
    std::memcpy(X.data(), prefill.data(), sizeof(float) * (size_t)P * C);
    for (int t = 0; t + 1 < T; ++t) {
        const float* tr = t < nt ? &trailing[(size_t)t * C] : pad.data();
        if (qwen3tts_frame_input(m, &codes[(size_t)t * ncb], tr, &X[(size_t)(P + t) * C]) != 0) {
            std::fprintf(stderr, "qwen3tts_frame_input: %s\n", qwen3tts_last_error());
            return 1;
        }
    }
    if (qwen3tts_talker_forward(m, X.data(), L, hidden.data(), logits.data()) != 0) {
        std::fprintf(stderr, "qwen3tts_talker_forward: %s\n", qwen3tts_last_error());
        return 1;
    }
    pass = compare("talker hidden", hidden.data(), want_hidden.data(), L, C) && pass;
    pass = compare("talker logits", logits.data(), want_logits.data(), L, V) && pass;

    // the code predictor for the frames: H = the hidden states of the positions that predicted each frame's first code
    std::vector<float> H((size_t)T * C), pred((size_t)T * (ncb - 1) * VP);
    std::memcpy(H.data(), &hidden[(size_t)(P - 1) * C], sizeof(float) * (size_t)T * C);
    if (qwen3tts_code_predictor(m, H.data(), codes.data(), T, pred.data()) != 0) {
        std::fprintf(stderr, "qwen3tts_code_predictor: %s\n", qwen3tts_last_error());
        return 1;
    }
    pass = compare("predictor logits", pred.data(), want_pred.data(), (size_t)T * (ncb - 1), VP) && pass;

    // the greedy generation (the talker's and the sub-talker's argmax, the repetition penalty 1.05, the suppressions): its
    // frames equal qwen-tts's greedy run frame for frame (talker_greedy_codes.npy: 199 frames, no end-of-speech within)
    const std::vector<int> greedy_ref = as_ints(load_npy(dir + "/talker_greedy_codes.npy"));
    const int G = (int)(greedy_ref.size() / ncb);
    std::vector<int> gen((size_t)200 * ncb);
    const int ng = qwen3tts_generate_greedy(m, text_ids.data(), (int)text_ids.size(), ref_ids.data(), (int)ref_ids.size(),
                                            ref_codes.data(), R, spk.data(), lang, 1.05f, 200, gen.data());
    if (ng < 0) {
        std::fprintf(stderr, "qwen3tts_generate_greedy: %s\n", qwen3tts_last_error());
        return 1;
    }
    int first_bad = -1;
    for (int t = 0; t < std::min(ng, G) && first_bad < 0; ++t)
        for (int c = 0; c < ncb; ++c)
            if (gen[(size_t)t * ncb + c] != greedy_ref[(size_t)t * ncb + c]) { first_bad = t; break; }
    const bool greedy_ok = first_bad < 0 && std::min(ng, G) == G;
    std::printf("greedy generation: %d frames (qwen-tts: %d), %s\n", ng, G,
                greedy_ok ? "every frame equal  ok" : (first_bad < 0 ? "FAIL (too few frames)" : "FAIL: first different frame follows"));
    if (first_bad >= 0) {
        std::printf("   first different frame %d\n", first_bad);
        for (int t = std::max(0, first_bad - 1); t <= first_bad; ++t) {
            std::printf("   frame %d engine:", t);
            for (int c = 0; c < ncb; ++c) std::printf(" %d", gen[(size_t)t * ncb + c]);
            std::printf("\n   frame %d qwen-tts:", t);
            for (int c = 0; c < ncb; ++c) std::printf(" %d", greedy_ref[(size_t)t * ncb + c]);
            std::printf("\n");
        }
    }
    pass = pass && greedy_ok;

    qwen3tts_close(m);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
