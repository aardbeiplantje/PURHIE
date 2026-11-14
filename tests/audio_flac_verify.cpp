// audio_flac_verify: the FLAC encoder (src/audio.cpp, audio_flac16) on signals of several lengths: its stream is written
// to the given directory with the 16-bit samples the WAV writer would give (<name>.flac and <name>.i16), and the
// Python check (decode with libsndfile) compares the two exactly. Usage: audio_flac_verify <dir> [speech.f32]
#include "audio.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void write_file(const std::string& path, const void* p, size_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(p, 1, n, f) != n) { std::perror(path.c_str()); std::exit(1); }
    std::fclose(f);
}

static void check(const std::string& dir, const std::string& name, const std::vector<float>& x) {
    unsigned char* flac = nullptr;
    size_t n = 0;
    if (audio_flac16(x.data(), x.size(), 24000, &flac, &n) != 0) {
        std::fprintf(stderr, "%s: %s\n", name.c_str(), audio_last_error());
        std::exit(1);
    }
    write_file(dir + "/" + name + ".flac", flac, n);
    std::vector<int16_t> i16(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const double v = x[i] > 1.0 ? 1.0 : x[i] < -1.0 ? -1.0 : (double)x[i];
        i16[i] = (int16_t)std::nearbyint(v * 32767.0);
    }
    write_file(dir + "/" + name + ".i16", i16.data(), i16.size() * 2);
    std::printf("%s: %zu samples -> %zu bytes (%.1f%% of 16-bit PCM)\n", name.c_str(), x.size(), n,
                x.empty() ? 0.0 : 100.0 * n / (2.0 * x.size()));
    audio_free((float*)flac);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <dir> [speech.f32]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    std::vector<float> sine(24000 * 2), noise(9000), tiny(1), empty, full(4096), clip(5000), last(4097);
    for (size_t i = 0; i < sine.size(); ++i) sine[i] = 0.5f * std::sin(2 * M_PI * 440.0 * i / 24000.0);
    unsigned s = 12345;
    for (auto& v : noise) { s = s * 1103515245u + 12345u; v = ((s >> 8) / 8388608.0f - 1.0f) * 0.7f; }
    tiny[0] = -0.25f;
    for (size_t i = 0; i < full.size(); ++i) full[i] = 0.9f * std::sin(0.01 * i) + 0.05f * std::cos(0.7 * i);
    for (size_t i = 0; i < clip.size(); ++i) clip[i] = (i / 50) % 2 ? 1.7f : -1.3f;  // beyond the range: clipped
    for (size_t i = 0; i < last.size(); ++i) last[i] = 0.3f * std::sin(0.05 * i);
    check(dir, "sine", sine);
    check(dir, "noise", noise);
    check(dir, "tiny", tiny);
    check(dir, "empty", empty);
    check(dir, "full_block", full);
    check(dir, "clipped", clip);
    check(dir, "last_block", last);
    if (argc > 2) {  // the speech samples (packed float32, 24 kHz) of qwen3tts_speech_verify's output
        FILE* f = std::fopen(argv[2], "rb");
        if (!f) { std::perror(argv[2]); return 1; }
        std::vector<float> sp;
        float v;
        while (std::fread(&v, sizeof v, 1, f) == 1) sp.push_back(v);
        std::fclose(f);
        check(dir, "speech", sp);
    }
    std::printf("written to %s\n", dir.c_str());
    return 0;
}
