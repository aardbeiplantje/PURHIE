// whisper_fallback_verify: the long form's temperature fallback (src/whisper.cpp, PLAN.md item 11).
//
//   whisper_fallback_verify <whisper.gguf> <clip.wav>
//
// Checks: (a) the zlib compression ratio of one English sentence is below 2.4 and of the sentence repeated ten times above
// 2.4; (b) the clip repeated six times with 0.5 s of silence between the repeats (over 30 s: the long form) transcribes, with
// the default settings, to the text the long form gave before the fallback (the sentence six times); (c) with thresholds
// that fail every attempt (compression_ratio_max 0) and keep_last 1, the sampled temperatures run, the text is not empty
// and two runs give the same text (the sampling is deterministic); (d) the same thresholds with the default keep_last 0
// give the text of (b): the greedy attempt has the best average log-probability there.
#include "audio.h"
#include "whisper.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

const char* kSentence = " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.";

std::string repeat(const std::string& s, int n) {
    std::string out;
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

// the long clip: the clip's samples six times, with 0.5 s of silence (8000 samples at 16 kHz) between the repeats
std::vector<float> long_clip(const std::vector<float>& clip) {
    std::vector<float> out;
    for (int r = 0; r < 6; ++r) {
        out.insert(out.end(), clip.begin(), clip.end());
        if (r < 5) out.insert(out.end(), 8000, 0.0f);
    }
    return out;
}

// the transcription of samples by the long form, and the decoder passes it made
bool transcribe(whisper_model_t* m, const std::vector<float>& samples, std::string& text, int& passes) {
    static char buf[1 << 16];
    const int n = whisper_transcribe(m, samples.data(), samples.size(), "en", buf, sizeof buf);
    if (n < 0) {
        std::fprintf(stderr, "whisper_transcribe: %s\n", whisper_last_error());
        return false;
    }
    text = buf;
    passes = whisper_fallback_attempts(m);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <whisper.gguf> <clip.wav>\n", argv[0]);
        return 2;
    }
    // (a) the compression ratio of the text
    const std::string one = kSentence;
    const std::string ten = repeat(one, 10);
    const float cr_one = whisper_compression_ratio(one.data(), one.size());
    const float cr_ten = whisper_compression_ratio(ten.data(), ten.size());
    const bool a_ok = cr_one < 2.4f && cr_ten > 2.4f;
    std::printf("compression ratio: one sentence %.2f, ten repeats %.2f  %s\n", cr_one, cr_ten, a_ok ? "ok" : "FAIL");

    // the clip: the WAV's samples (audio.h's decoder, 16 kHz mono)
    std::vector<unsigned char> wav;
    {
        std::ifstream f(argv[2], std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "cannot read %s\n", argv[2]);
            return 2;
        }
        wav.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    float* raw = nullptr;
    size_t raw_n = 0;
    if (audio_decode_wav(wav.data(), wav.size(), 16000, &raw, &raw_n) != 0) {
        std::fprintf(stderr, "audio_decode_wav: %s\n", audio_last_error());
        return 2;
    }
    const std::vector<float> clip(raw, raw + raw_n);
    audio_free(raw);
    const std::vector<float> samples = long_clip(clip);

    whisper_model_t* m = whisper_open(argv[1]);
    if (!m) {
        std::fprintf(stderr, "whisper_open: %s\n", whisper_last_error());
        return 1;
    }

    // (b) the default settings: the text of the long form before the fallback
    std::string text_b;
    int passes_b = 0;
    if (!transcribe(m, samples, text_b, passes_b)) return 1;
    const std::string want_b = repeat(kSentence, 6);
    const bool b_ok = text_b == want_b;
    std::printf("long clip (%.1f s, default settings): %d decoder passes, \"%s\"  %s\n", samples.size() / 16000.0, passes_b,
                text_b.c_str(), b_ok ? "ok" : "FAIL");

    // (c) every attempt fails, the last one is kept: the sampled temperatures run, twice, with the same result
    whisper_set_fallback(m, 0.0f, -1.0f, 1);
    std::string text_c1, text_c2;
    int passes_c1 = 0, passes_c2 = 0;
    const bool run_ok = transcribe(m, samples, text_c1, passes_c1) && transcribe(m, samples, text_c2, passes_c2);
    const bool c_ok = run_ok && passes_c1 > 0 && passes_c1 % 6 == 0 && !text_c1.empty() && text_c1 == text_c2;
    std::printf("forced fallback, keep_last 1: %d decoder passes, text \"%s\", the second run %s  %s\n", passes_c1,
                text_c1.c_str(), text_c1 == text_c2 ? "identical" : "different", c_ok ? "ok" : "FAIL");

    // (d) every attempt fails, the default keeps the best average log-probability: the greedy text of (b)
    whisper_set_fallback(m, 0.0f, -1.0f, 0);
    std::string text_d;
    int passes_d = 0;
    const bool d_run = transcribe(m, samples, text_d, passes_d);
    const bool d_ok = d_run && passes_d > 0 && passes_d % 6 == 0 && text_d == text_b;
    std::printf("forced fallback, keep_last 0: %d decoder passes, text %s the default's  %s\n", passes_d,
                text_d == text_b ? "equal to" : "different from", d_ok ? "ok" : "FAIL");

    whisper_set_fallback(m, 2.4f, -1.0f, 0);
    whisper_close(m);
    const bool pass = a_ok && b_ok && c_ok && d_ok;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
