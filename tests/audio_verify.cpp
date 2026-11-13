// audio_verify: the engine's band-limited resampler and WAV decode (src/audio.cpp, PLAN.md A5) against librosa 0.11's
// soxr_hq (the references qwen-tts loads its audio with). The reference directory (/data/audio-ref/resample, written by
// its dump.py): in16.f32 (the clip at 16 kHz), ref24_from16.f32, ref16_from24.f32, clip44.wav (16-bit at 44.1 kHz) and
// ref16_from44.f32 (librosa.load(sr=16000) of that file). A check passes when the largest difference is at most 3e-3 of the
// reference's peak and the cosine is at least 0.999995 (the edges, where the filter repeats the end sample, are left out of
// the largest difference and counted in the cosine).
//
//   audio_verify <reference dir>
#include "audio.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<float> x(b.size() / 4);
    std::memcpy(x.data(), b.data(), x.size() * 4);
    return x;
}

bool compare(const char* name, const float* got, size_t ng, const float* want, size_t nw) {
    const size_t n = std::min(ng, nw);
    const size_t edge = 64;
    double peak = 0.0, dot = 0.0, gg = 0.0, ww = 0.0, worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        peak = std::fmax(peak, std::fabs((double)want[i]));
        dot += (double)got[i] * want[i];
        gg += (double)got[i] * got[i];
        ww += (double)want[i] * want[i];
        if (i >= edge && i + edge < n) worst = std::fmax(worst, std::fabs((double)got[i] - want[i]));
    }
    const double cos = dot / std::sqrt(gg * ww);
    const double rel = worst / peak;
    const bool ok = ng == nw && rel <= 3e-3 && cos >= 0.999995;
    std::printf("%-28s %zu vs %zu samples, cosine %.7f, max |diff| (interior) %.3e = %.2e of the peak  %s\n", name, ng, nw, cos,
                worst, rel, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <reference dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    bool pass = true;

    const std::vector<float> in16 = read_f32(dir + "/in16.f32");
    float* up = nullptr;
    size_t nup = 0;
    if (audio_resample(in16.data(), in16.size(), 16000, 24000, &up, &nup) != 0) { std::fprintf(stderr, "%s\n", audio_last_error()); return 1; }
    const std::vector<float> ref24 = read_f32(dir + "/ref24_from16.f32");
    pass = compare("16 kHz -> 24 kHz", up, nup, ref24.data(), ref24.size()) && pass;
    audio_free(up);

    const std::vector<float> ref16 = read_f32(dir + "/ref16_from24.f32");
    float* down = nullptr;
    size_t ndown = 0;
    if (audio_resample(ref24.data(), ref24.size(), 24000, 16000, &down, &ndown) != 0) { std::fprintf(stderr, "%s\n", audio_last_error()); return 1; }
    pass = compare("24 kHz -> 16 kHz", down, ndown, ref16.data(), ref16.size()) && pass;
    audio_free(down);

    const std::vector<char> wav = [&] {
        std::ifstream f(dir + "/clip44.wav", std::ios::binary);
        return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }();
    float* dec = nullptr;
    size_t ndec = 0;
    if (audio_decode_wav(reinterpret_cast<const unsigned char*>(wav.data()), wav.size(), 16000, &dec, &ndec) != 0) {
        std::fprintf(stderr, "audio_decode_wav: %s\n", audio_last_error());
        return 1;
    }
    const std::vector<float> ref44 = read_f32(dir + "/ref16_from44.f32");
    pass = compare("WAV 44.1 kHz -> 16 kHz", dec, ndec, ref44.data(), ref44.size()) && pass;
    audio_free(dec);

    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
