// mel_verify: whisper's log-mel features (src/mel.cpp, phase A2 of PLAN.md) against transformers' WhisperFeatureExtractor
// for the same clip (phase A1's reference): the mel filter bank, then the 80 x 3000 features.
//
//   mel_verify <reference dir>
//
// The reference directory: samples.npy (float32, the clip's waveform), input_features.npy (1 x 80 x 3000 float32, the
// extractor's output: its torch path, float32), input_features_f64.npy (80 x 3000 float64, the same steps in numpy
// double: the check of the arithmetic), mel_filters.npy (201 x 80 float64, the extractor's filter bank). Passes when the
// filter bank's largest difference is at most 1e-6, the features' largest difference from the double replica at most
// 1e-6, and from the float32 reference at most 5e-5 (the reference's own rounding, about 1.2e-5 here).
#include "mel.h"

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
    const size_t esz = (n.descr == "<f4") ? 4 : 8;
    if (n.descr != "<f4" && n.descr != "<f8") { std::fprintf(stderr, "%s: dtype %s\n", path.c_str(), n.descr.c_str()); std::exit(2); }
    n.data.resize((size_t)count * esz);
    f.read(n.data.data(), (std::streamsize)n.data.size());
    return n;
}

std::vector<float> as_f32(const Npy& n) {
    const size_t count = n.data.size() / (n.descr == "<f4" ? 4 : 8);
    std::vector<float> v(count);
    for (size_t i = 0; i < count; ++i) {
        v[i] = n.descr == "<f4" ? reinterpret_cast<const float*>(n.data.data())[i]
                                : (float)reinterpret_cast<const double*>(n.data.data())[i];
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[1];
    const Npy samples_n = load_npy(dir + "/samples.npy");
    const Npy feats_n = load_npy(dir + "/input_features.npy");
    const Npy filt_n = load_npy(dir + "/mel_filters.npy");
    const Npy f64_n = load_npy(dir + "/input_features_f64.npy");
    const std::vector<float> samples = as_f32(samples_n);
    const std::vector<float> want = as_f32(feats_n);
    const std::vector<float> want_filt = as_f32(filt_n);  // [bins 201][mels 80]
    const std::vector<float> want64 = as_f32(f64_n);
    const int bins = WHISPER_N_FFT / 2 + 1, mels = 80, T = WHISPER_N_FRAMES;
    if (filt_n.shape.size() != 2 || filt_n.shape[0] != bins || filt_n.shape[1] != mels || want.size() != (size_t)mels * T) {
        std::fprintf(stderr, "unexpected reference shapes\n");
        return 2;
    }

    // the filter bank: ours is [mels][bins]; the reference's [bins][mels]
    std::vector<float> filt((size_t)mels * bins);
    if (mel_filter_bank(mels, WHISPER_N_FFT, 16000, 0.0, 8000.0, filt.data()) != 0) {
        std::fprintf(stderr, "mel_filter_bank failed\n");
        return 1;
    }
    double filt_max = 0.0, filt_ref_max = 0.0;
    for (int m = 0; m < mels; ++m)
        for (int k = 0; k < bins; ++k) {
            const double a = filt[(size_t)m * bins + k], b = want_filt[(size_t)k * mels + m];
            filt_max = std::fmax(filt_max, std::fabs(a - b));
            filt_ref_max = std::fmax(filt_ref_max, std::fabs(b));
        }
    std::printf("mel filters: max |diff| %.3e (reference's largest value %.4f)\n", filt_max, filt_ref_max);

    // the features
    std::vector<float> got((size_t)mels * T);
    if (whisper_log_mel(samples.data(), (int)samples.size(), mels, got.data()) != 0) {
        std::fprintf(stderr, "whisper_log_mel failed\n");
        return 1;
    }
    if (want64.size() != got.size()) { std::fprintf(stderr, "unexpected f64 reference shape\n"); return 2; }
    double max_abs = 0.0, max64 = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double a = got[i], b = want[i];
        const double d = std::fabs(a - b);
        if (d > max_abs) { max_abs = d; worst = i; }
        max64 = std::fmax(max64, std::fabs(a - want64[i]));
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    std::printf("features %d x %d: cosine %.9f against the float32 reference, max |diff| %.3e at mel %zu, frame %zu (got %.6f, "
                "want %.6f)\n", mels, T, dot / (std::sqrt(na) * std::sqrt(nb)), max_abs, worst / T, worst % T, got[worst],
                want[worst]);
    std::printf("features vs the double replica: max |diff| %.3e\n", max64);
    const bool pass = filt_max <= 1e-6 && max64 <= 1e-6 && max_abs <= 5e-5;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
