// speaker_verify: the speaker encoder of the Qwen3-TTS talker GGUF (src/qwen3tts.cpp, src/mel.cpp; PLAN.md T5) against
// qwen-tts 0.1.1 on the same 24 kHz clip: librosa's Slaney filter bank (128 bins, 1024-point frames, 0 to 12 kHz), the
// speaker mel (mel_spectrogram, the uncentred reflect-padded frames) and extract_speaker_embedding (the 1024-dim x-vector).
// A stage passes at cosine 0.99999 and a largest difference at most 1e-3 of the reference's largest value (the mel's
// log is compared in its own units: the same tolerance on the log values).
//
//   speaker_verify <talker.gguf> <reference dir>   (the reference dir: /data/audio-ref/speaker, dump.py there)
#include "mel.h"
#include "qwen3tts.h"

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

bool compare(const char* name, const float* got, const float* want, size_t n, size_t rows = 1) {
    double peak = 0.0, dot = 0.0, gg = 0.0, ww = 0.0, worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        peak = std::fmax(peak, std::fabs((double)want[i]));
        dot += (double)got[i] * want[i];
        gg += (double)got[i] * got[i];
        ww += (double)want[i] * want[i];
        worst = std::fmax(worst, std::fabs((double)got[i] - want[i]));
    }
    const double cos = dot / std::sqrt(gg * ww);
    const double rel = worst / peak;
    const bool ok = cos >= 0.99999 && rel <= 1e-3;
    std::printf("%-26s %zu row(s), cosine %.7f, max |diff| %.3e = %.2e of the peak  %s\n", name, rows, cos, worst, rel,
                ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <talker.gguf> <reference dir>\n", argv[0]); return 2; }
    const std::string dir = argv[2];
    const std::vector<float> samples = read_f32(dir + "/samples24.f32");
    const std::vector<float> mel_want = read_f32(dir + "/mel_ref.f32");
    const std::vector<float> spk_want = read_f32(dir + "/spk_ref.f32");
    const std::vector<float> fb_want = read_f32(dir + "/mel_basis_ref.f32");
    bool pass = true;

    std::vector<float> fb(128 * 513);
    mel_filter_bank(128, 1024, 24000, 0.0, 12000.0, fb.data());
    pass = compare("Slaney filter bank", fb.data(), fb_want.data(), fb.size(), 128) && pass;

    float* mel = nullptr;
    int T = 0;
    if (speaker_log_mel(samples.data(), samples.size(), &mel, &T) != 0) { std::fprintf(stderr, "speaker_log_mel failed\n"); return 1; }
    const size_t T_want = mel_want.size() / 128;
    if ((size_t)T != T_want) { std::printf("frames %d, reference %zu  FAIL\n", T, T_want); return 1; }
    pass = compare("speaker log-mel", mel, mel_want.data(), mel_want.size(), 128) && pass;
    speaker_free_mel(mel);

    qwen3tts_model_t* m = qwen3tts_open(argv[1]);
    if (!m) { std::fprintf(stderr, "qwen3tts_open: %s\n", qwen3tts_last_error()); return 1; }
    std::vector<float> spk(spk_want.size());
    if (qwen3tts_speaker_embedding(m, samples.data(), samples.size(), spk.data()) != 0) {
        std::fprintf(stderr, "qwen3tts_speaker_embedding: %s\n", qwen3tts_last_error());
        return 1;
    }
    pass = compare("speaker embedding", spk.data(), spk_want.data(), spk.size()) && pass;
    qwen3tts_close(m);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
