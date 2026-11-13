// qwen3tts_speech_verify: the speech of a text in a voice (src/qwen3tts_speech.cpp, phase T4 of PLAN.md): the talker sampled
// with qwen-tts's generation config, the codec decoder, the cut of the reference's share. The sampled frames cannot equal
// torch's (its generator), so the checks are what the sampling must hold: the same seed gives the same waves, another
// seed gives other frames, every sample is finite and within the clamp, and the speech has the talker's length.
//
//   qwen3tts_speech_verify <qwen3tts.gguf> <qwen3tts-codec.gguf> <voice dir> [out.wav]
//
// The voice directory (tools/qwen3tts_voice.py writes it): voice.json (ref_text, frames, codebooks, n_embd), codes.i32
// (frames x codebooks, little-endian int32), spk.f32 (n_embd float32, little-endian). With out.wav the seed-0 wave is
// written as 16-bit PCM at 24 kHz (check it with whisper: the text should come back).
#include "qwen3tts.h"
#include "qwen3tts_codec.h"
#include "qwen3tts_speech.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// the string value of "key" in a flat JSON object (the voice's own file: no escapes needed)
std::string json_string(const std::string& js, const std::string& key) {
    const size_t k = js.find("\"" + key + "\"");
    if (k == std::string::npos) { std::fprintf(stderr, "voice.json: no %s\n", key.c_str()); std::exit(2); }
    const size_t a = js.find('"', js.find(':', k) + 1) + 1;
    return js.substr(a, js.find('"', a) - a);
}

long json_int(const std::string& js, const std::string& key) {
    const size_t k = js.find("\"" + key + "\"");
    if (k == std::string::npos) { std::fprintf(stderr, "voice.json: no %s\n", key.c_str()); std::exit(2); }
    return std::atol(js.c_str() + js.find(':', k) + 1);
}

void write_wav16(const char* path, const float* x, size_t n, int rate) {
    std::ofstream f(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("RIFF", 4); u32((uint32_t)(36 + 2 * n)); f.write("WAVEfmt ", 8); u32(16); u16(1); u16(1);
    u32((uint32_t)rate); u32((uint32_t)rate * 2); u16(2); u16(16); f.write("data", 4); u32((uint32_t)(2 * n));
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        const int16_t s = (int16_t)std::lrint(v * 32767.0f);
        f.write(reinterpret_cast<const char*>(&s), 2);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <qwen3tts.gguf> <qwen3tts-codec.gguf> <voice dir> [out.wav]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[3];
    const std::string js = slurp(dir + "/voice.json");
    const std::string ref_text = json_string(js, "ref_text");
    const int R = (int)json_int(js, "frames"), ncb = (int)json_int(js, "codebooks"), C = (int)json_int(js, "n_embd");
    const std::string cbytes = slurp(dir + "/codes.i32"), sbytes = slurp(dir + "/spk.f32");
    if (cbytes.size() != (size_t)R * ncb * 4 || sbytes.size() != (size_t)C * 4) {
        std::fprintf(stderr, "voice: the files do not match voice.json\n");
        return 2;
    }
    std::vector<int> ref_codes((size_t)R * ncb);
    std::memcpy(ref_codes.data(), cbytes.data(), cbytes.size());
    std::vector<float> spk((size_t)C);
    std::memcpy(spk.data(), sbytes.data(), sbytes.size());

    qwen3tts_model_t* m = qwen3tts_open(argv[1]);
    if (!m) { std::fprintf(stderr, "qwen3tts_open: %s\n", qwen3tts_last_error()); return 1; }
    qwen3tts_codec_t* codec = qwen3tts_codec_open(argv[2]);
    if (!codec) { std::fprintf(stderr, "qwen3tts_codec_open: %s\n", qwen3tts_codec_last_error()); return 1; }
    if (C != qwen3tts_n_embd(m) || ncb != qwen3tts_n_codebooks(m)) { std::fprintf(stderr, "voice: the model's sizes differ\n"); return 1; }
    const int lang = qwen3tts_language_id(m, "english");
    const char* text = "Hello, this is a short test of the speech synthesis.";

    bool pass = true;
    auto run = [&](uint64_t seed, float** wav, size_t* n, int* frames) {
        qwen3tts_sampling_t s = qwen3tts_sampling_default();
        s.seed = seed;
        return qwen3tts_speak(m, codec, text, ref_text.c_str(), ref_codes.data(), R, spk.data(), lang, &s, 400, wav, n, frames);
    };
    float *w0 = nullptr, *w0b = nullptr, *w1 = nullptr;
    size_t n0 = 0, n0b = 0, n1 = 0;
    int f0 = 0, f0b = 0, f1 = 0;
    if (run(0, &w0, &n0, &f0) != 0 || run(0, &w0b, &n0b, &f0b) != 0 || run(1, &w1, &n1, &f1) != 0) {
        std::fprintf(stderr, "qwen3tts_speak: %s\n", qwen3tts_last_error());
        return 1;
    }
    const bool same = n0 == n0b && f0 == f0b && std::memcmp(w0, w0b, sizeof(float) * n0) == 0;
    std::printf("seed 0: %d talker frames, %zu samples (%.2f s); seed 0 again %s\n", f0, n0, (double)n0 / 24000.0,
                same ? "identical  ok" : "DIFFERENT  FAIL");
    pass = pass && same && f0 > 0;

    double peak = 0.0, sq = 0.0;
    bool finite = true;
    for (size_t i = 0; i < n0; ++i) {
        finite = finite && std::isfinite(w0[i]);
        peak = std::fmax(peak, std::fabs((double)w0[i]));
        sq += (double)w0[i] * w0[i];
    }
    const double rms = n0 ? std::sqrt(sq / (double)n0) : 0.0;
    const bool level_ok = finite && peak <= 1.0 && rms > 0.005;
    std::printf("the wave: finite %s, peak %.3f, rms %.4f  %s\n", finite ? "yes" : "NO", peak, rms, level_ok ? "ok" : "FAIL");
    pass = pass && level_ok;

    const bool other = !(n1 == n0 && f1 == f0 && std::memcmp(w1, w0, sizeof(float) * n0) == 0);
    std::printf("seed 1: %d talker frames, %zu samples; %s\n", f1, n1, other ? "other than seed 0  ok" : "the same as seed 0  FAIL");
    pass = pass && other && f1 > 0;

    if (argc >= 5) {
        write_wav16(argv[4], w0, n0, 24000);
        std::printf("wrote %s\n", argv[4]);
    }
    qwen3tts_speech_free_wav(w0);
    qwen3tts_speech_free_wav(w0b);
    qwen3tts_speech_free_wav(w1);
    qwen3tts_codec_close(codec);
    qwen3tts_close(m);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
