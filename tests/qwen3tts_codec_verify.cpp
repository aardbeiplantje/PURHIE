// qwen3tts_codec_verify: the 12.5 Hz codec's decoder (src/qwen3tts_codec.cpp, phase T3 of PLAN.md) against qwen-tts 0.1.1 on
// the same frames (the T1 reference): the talker's codes (50 frames) decode to the reference wave (wave.npy), and the
// reference clip's own codes (74 frames) decode to the speech qwen-tts gives them (diag_ref_decode.npy).
//
//   qwen3tts_codec_verify <qwen3tts-codec.gguf> <reference dir>
//
// Each wave passes when its cosine is at least 0.9999 and its largest difference at most 1e-4 of the reference's largest value.
#include "qwen3tts_codec.h"

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
        v.resize(n.data.size() / 4);
        std::memcpy(v.data(), n.data.data(), n.data.size());
    }
    return v;
}

// The waveform against the reference: cosine, largest difference, the verdict
bool compare_wave(const char* name, const float* got, size_t n_got, const std::vector<float>& want) {
    if (n_got != want.size()) {
        std::printf("%-24s %zu samples, the reference %zu  FAIL\n", name, n_got, want.size());
        return false;
    }
    double dot = 0, na = 0, nb = 0, max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < n_got; ++i) {
        const double a = got[i], b = want[i];
        dot += a * b;
        na += a * a;
        nb += b * b;
        max_abs = std::fmax(max_abs, std::fabs(a - b));
        max_ref = std::fmax(max_ref, std::fabs(b));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const bool ok = cos >= 0.9999 && max_abs <= 1e-4 * max_ref;
    std::printf("%-24s %zu samples  cosine %.7f  max |diff| %.3e (%.1e of %.3f)  %s\n", name, n_got, cos, max_abs,
                max_abs / max_ref, max_ref, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <qwen3tts-codec.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    qwen3tts_codec_t* c = qwen3tts_codec_open(argv[1]);
    if (!c) { std::fprintf(stderr, "qwen3tts_codec_open: %s\n", qwen3tts_codec_last_error()); return 1; }
    const int nq = qwen3tts_codec_n_quantizers(c);
    bool pass = qwen3tts_codec_samples_per_frame(c) == 1920 && qwen3tts_codec_sample_rate(c) == 24000;
    std::printf("%d quantizers, %d samples a frame, %d Hz\n", nq, qwen3tts_codec_samples_per_frame(c),
                qwen3tts_codec_sample_rate(c));

    // the talker's codes (T1's sampled run) against its wave: qwen-tts decodes the reference clip's frames followed by the
    // talker's, and keeps the samples after the reference's share, int(R / (R + T) * samples)
    {
        const std::vector<int> ref = as_ints(load_npy(dir + "/ref_codes.npy"));
        const std::vector<int> talker = as_ints(load_npy(dir + "/talker_codes.npy"));
        const std::vector<float> want = as_f32(load_npy(dir + "/wave.npy"));
        std::vector<int> codes = ref;
        codes.insert(codes.end(), talker.begin(), talker.end());
        const size_t R = ref.size() / nq, T = talker.size() / nq;
        float* wav = nullptr;
        size_t n = 0;
        if (qwen3tts_codec_decode(c, codes.data(), (int)(R + T), &wav, &n) != 0) {
            std::fprintf(stderr, "qwen3tts_codec_decode: %s\n", qwen3tts_codec_last_error());
            return 1;
        }
        const size_t cut = (size_t)((double)R / (double)(R + T) * (double)n);
        pass = compare_wave("talker frames -> wave", wav + cut, n - cut, want) && pass;
        qwen3tts_codec_free_wav(wav);
    }
    // the reference clip's own codes against the speech qwen-tts decodes them to
    {
        const std::vector<int> codes = as_ints(load_npy(dir + "/ref_codes.npy"));
        const std::vector<float> want = as_f32(load_npy(dir + "/diag_ref_decode.npy"));
        float* wav = nullptr;
        size_t n = 0;
        if (qwen3tts_codec_decode(c, codes.data(), (int)(codes.size() / nq), &wav, &n) != 0) {
            std::fprintf(stderr, "qwen3tts_codec_decode: %s\n", qwen3tts_codec_last_error());
            return 1;
        }
        pass = compare_wave("reference codes -> wave", wav, n, want) && pass;
        qwen3tts_codec_free_wav(wav);
    }
    qwen3tts_codec_close(c);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
