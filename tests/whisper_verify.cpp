// whisper_verify: whisper's encoder and greedy decoder (src/whisper.cpp, phase A3 of PLAN.md) against transformers for the
// same clip (phase A1's reference): the mel features (src/mel.cpp), the encoder output, then the greedy tokens.
//
//   whisper_verify <whisper.gguf> <reference dir>
//
// The GGUF: rocm-convert of the checkpoint (perl/ROCm-Inference: the Whisper converter, F32). The reference directory:
// samples.npy (the clip's waveform), encoder_out.npy (1 x 1500 x 384 float32), tokens.npy (int64: the greedy tokens after
// the prompt, without the end-of-transcript token). Passes when every encoder row's cosine is at least 0.9999, its largest
// difference at most 1e-4 of the reference's largest value (float32 accumulation over the blocks; the reference is torch
// float32), and the decoder's tokens equal the reference's.
#include "mel.h"
#include "whisper.h"

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
    return n;
}

std::vector<float> as_f32(const Npy& n) {
    std::vector<float> v(n.data.size() / 4);
    std::memcpy(v.data(), n.data.data(), n.data.size());
    return v;
}

std::vector<int64_t> as_i64(const Npy& n) {
    std::vector<int64_t> v(n.data.size() / 8);
    std::memcpy(v.data(), n.data.data(), n.data.size());
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <whisper.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    const std::vector<float> samples = as_f32(load_npy(dir + "/samples.npy"));
    const std::vector<float> want_enc = as_f32(load_npy(dir + "/encoder_out.npy"));
    const std::vector<int64_t> want_tokens = as_i64(load_npy(dir + "/tokens.npy"));

    whisper_model_t* m = whisper_open(argv[1]);
    if (!m) { std::fprintf(stderr, "whisper_open: %s\n", whisper_last_error()); return 1; }
    const int T = whisper_n_audio_ctx(m), C = whisper_n_audio_state(m);
    if ((size_t)T * C != want_enc.size()) { std::fprintf(stderr, "encoder reference has %zu values, %d x %d expected\n", want_enc.size(), T, C); return 2; }

    std::vector<float> mel(80 * (size_t)WHISPER_N_FRAMES);
    if (whisper_log_mel(samples.data(), (int)samples.size(), 80, mel.data()) != 0) { std::fprintf(stderr, "whisper_log_mel failed\n"); return 1; }
    std::vector<float> enc((size_t)T * C);
    if (whisper_encode(m, mel.data(), enc.data()) != 0) { std::fprintf(stderr, "whisper_encode: %s\n", whisper_last_error()); return 1; }

    // the encoder: the worst row's cosine and the largest difference
    double min_cos = 1.0, max_abs = 0.0, max_ref = 0.0;
    for (float v : want_enc) max_ref = std::fmax(max_ref, std::fabs(v));
    for (int t = 0; t < T; ++t) {
        double dot = 0, na = 0, nb = 0;
        for (int c = 0; c < C; ++c) {
            const double a = enc[(size_t)t * C + c], b = want_enc[(size_t)t * C + c];
            dot += a * b;
            na += a * a;
            nb += b * b;
            max_abs = std::fmax(max_abs, std::fabs(a - b));
        }
        min_cos = std::fmin(min_cos, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30));
    }
    const bool enc_ok = min_cos >= 0.9999 && max_abs <= 1e-4 * max_ref;
    std::printf("encoder %d x %d: worst row cosine %.7f, max |diff| %.3e (%.1e of the largest value %.2f)  %s\n", T, C, min_cos,
                max_abs, max_abs / max_ref, max_ref, enc_ok ? "ok" : "FAIL");

    // the decoder: the prompt of transcribe in English without timestamps (the reference's prompt)
    const int eot = whisper_eot(m);
    const int prompt[4] = {50258, 50259, 50359, 50363};  // <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
    std::vector<int> got(448);
    const int n = whisper_greedy(m, enc.data(), prompt, 4, 100, got.data());
    if (n < 0) { std::fprintf(stderr, "whisper_greedy: %s\n", whisper_last_error()); return 1; }
    size_t same = 0;
    for (int i = 0; i < n && i < (int)want_tokens.size(); ++i) same += got[i] == want_tokens[i];
    const bool tok_ok = n == (int)want_tokens.size() && same == want_tokens.size();
    std::printf("greedy: %d tokens (the reference's %zu), %zu equal, eot %d%s\n", n, want_tokens.size(), same, eot,
                tok_ok ? "" : "  FAIL");
    if (!tok_ok) {
        std::printf("got:");
        for (int i = 0; i < n; ++i) std::printf(" %d", got[i]);
        std::printf("\nwant:");
        for (int64_t t : want_tokens) std::printf(" %lld", (long long)t);
        std::printf("\n");
    }
    whisper_close(m);
    const bool pass = enc_ok && tok_ok && eot == 50257;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
