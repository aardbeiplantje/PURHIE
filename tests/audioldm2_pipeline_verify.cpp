/* The AudioLDM 2 pipeline (src/audioldm2_pipeline.cpp) against diffusers' output: /data/audioldm2/ref_ocean.wav (5 s, 16 kHz, 16-bit,
 * "ocean waves crashing on a beach", seed 0, 100 steps, guidance 3.5). Usage: audioldm2_pipeline_verify [steps] [seconds] [guidance]
 * [seed] [out.f32]. The engine's waveform is written as float32 to out.f32 when given. */
#include "audioldm2_pipeline.h"
#include "sd_device.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static bool read_bytes(const std::string& path, std::vector<unsigned char>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    v.resize((size_t)std::ftell(f));
    std::fseek(f, 0, SEEK_SET);
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    return got == v.size();
}

/* the 16-bit PCM data of a WAV file's "data" chunk (the reference is written by the diffusers run: a plain 44-byte header) */
static bool read_ref(const std::string& path, std::vector<float>& out) {
    std::vector<unsigned char> b;
    if (!read_bytes(path, b) || b.size() < 44) return false;
    size_t pos = 12;
    while (pos + 8 <= b.size()) {
        const uint32_t sz = b[pos + 4] | (b[pos + 5] << 8) | (b[pos + 6] << 16) | ((uint32_t)b[pos + 7] << 24);
        if (std::memcmp(&b[pos], "data", 4) == 0) {
            const size_t n = sz / 2;
            out.resize(n);
            for (size_t i = 0; i < n; ++i) out[i] = (int16_t)(b[pos + 8 + 2 * i] | (b[pos + 9 + 2 * i] << 8)) / 32767.0f;
            return true;
        }
        pos += 8 + sz;
    }
    return false;
}

static bool read_lines(const std::string& path, std::vector<std::string>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string cur;
    for (int c; (c = std::fgetc(f)) != EOF;) {
        if (c == '\n') { out.push_back(cur); cur.clear(); }
        else cur += (char)c;
    }
    if (!cur.empty()) out.push_back(cur);
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 100;
    const float seconds = argc > 2 ? (float)std::atof(argv[2]) : 5.0f;
    const float guidance = argc > 3 ? (float)std::atof(argv[3]) : 3.5f;
    const uint64_t seed = argc > 4 ? (uint64_t)std::atoll(argv[4]) : 0;
    const std::string out_path = argc > 5 ? argv[5] : "";
    const std::string dir = "/data/audioldm2";
    /* the tokenizers from the text exports (tools/audioldm2_export_tokenizers.py); the prompt's ids are checked against the
     * diffusers tokenization in the test (the reference is the ocean prompt) */
    const std::string tdir = "/data/audioldm2-stages/";
    std::vector<std::string> clap_vocab, clap_merges, t5_lines, t5_pieces;
    std::vector<double> t5_scores;
    if (!read_lines(tdir + "clap_vocab.txt", clap_vocab) || !read_lines(tdir + "clap_merges.txt", clap_merges) ||
        !read_lines(tdir + "t5_pieces.tsv", t5_lines)) {
        std::fprintf(stderr, "cannot read the tokenizer exports (tools/audioldm2_export_tokenizers.py)\n");
        return 1;
    }
    for (const auto& l : t5_lines) {
        const size_t tab = l.rfind('\t');
        if (tab == std::string::npos) continue;
        t5_pieces.push_back(l.substr(0, tab));
        t5_scores.push_back(std::atof(l.c_str() + tab + 1));
    }
    audioldm2::Pipeline pipe(dir);
    if (!pipe.ok()) {
        std::fprintf(stderr, "the pipeline: %s\n", pipe.error().c_str());
        return 1;
    }
    if (!pipe.load_tokenizers(clap_vocab, clap_merges, t5_pieces, t5_scores)) {
        std::fprintf(stderr, "the tokenizers\n");
        return 1;
    }
    std::vector<float> wav;
    if (!pipe.generate_text("ocean waves crashing on a beach", "", seconds, steps, guidance, seed, wav)) {
        std::fprintf(stderr, "the pipeline failed\n");
        return 1;
    }
    std::printf("engine: %zu samples (%.2f s at 16 kHz), %d steps, guidance %.2f, seed %llu\n", wav.size(), wav.size() / 16000.0, steps, guidance,
                (unsigned long long)seed);
    if (!out_path.empty()) {
        FILE* f = std::fopen(out_path.c_str(), "wb");
        if (f) { std::fwrite(wav.data(), 4, wav.size(), f); std::fclose(f); }
    }
    if (steps != 100 || seconds != 5.0f || guidance != 3.5f || seed != 0) return 0;  /* the reference is one setting */
    std::vector<float> ref;
    if (!read_ref(dir + "/ref_ocean.wav", ref)) {
        std::fprintf(stderr, "cannot read the reference\n");
        return 1;
    }
    const size_t n = ref.size() < wav.size() ? ref.size() : wav.size();
    double num = 0, den = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)wav[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
    }
    std::printf("vs diffusers (%zu samples): rel L2 %.3e, SNR %.1f dB\n", n, std::sqrt(num / den), -10.0 * std::log10(num / den));
    return 0;
}
