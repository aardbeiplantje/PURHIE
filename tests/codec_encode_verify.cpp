// codec_encode_verify: the speech tokenizer's encoder (src/qwen3tts_codec.cpp, T6 of PLAN.md) against qwen-tts 0.1.1's
// encode on the reference clip at 24 kHz: the codes (frames x 16, every integer). The reference (/data/audio-ref/speaker,
// codec_dump.py): samples24.f32 and enc_codes_ref.i32. The check passes when every code equals the reference's; a mismatch
// prints the first differing frames (a nearest-entry tie flips a code, a wrong layer flips many).
//
//   codec_encode_verify <qwen3tts-codec.gguf> <reference dir>
#include "qwen3tts_codec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_raw(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    std::vector<char> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<T> x(b.size() / sizeof(T));
    std::memcpy(x.data(), b.data(), x.size() * sizeof(T));
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <codec.gguf> <reference dir>\n", argv[0]); return 2; }
    const std::string dir = argv[2];
    const std::vector<float> samples = read_raw<float>(dir + "/samples24.f32");
    const std::vector<int> want = read_raw<int>(dir + "/enc_codes_ref.i32");
    qwen3tts_codec_t* c = qwen3tts_codec_open(argv[1]);
    if (!c) { std::fprintf(stderr, "qwen3tts_codec_open: %s\n", qwen3tts_codec_last_error()); return 1; }
    int* codes = nullptr;
    int T = 0;
    if (qwen3tts_codec_encode(c, samples.data(), samples.size(), &codes, &T) != 0) {
        std::fprintf(stderr, "qwen3tts_codec_encode: %s\n", qwen3tts_codec_last_error());
        return 1;
    }
    const int V = qwen3tts_codec_n_quantizers(c);
    const size_t n = (size_t)T * V;
    size_t diff = 0;
    int first = -1;
    for (size_t i = 0; i < n && i < want.size(); ++i)
        if (codes[i] != want[i]) {
            if (diff == 0) first = (int)(i / V);
            ++diff;
        }
    const bool shape_ok = n == want.size();
    std::printf("codes: %d frames x %d codebooks, reference %zu frames: %zu codes differ  %s\n", T, V, want.size() / V, diff,
                (shape_ok && diff == 0) ? "ok" : "FAIL");
    if (first >= 0) {
        std::printf("   first differing frame %d (engine / qwen-tts):\n", first);
        for (int t = first; t < T && t < first + 2; ++t) {
            std::printf("   frame %d engine:", t);
            for (int q = 0; q < V; ++q) std::printf(" %d", codes[(size_t)t * V + q]);
            std::printf("\n   frame %d qwen-tts:", t);
            for (int q = 0; q < V; ++q) std::printf(" %d", want[(size_t)t * V + q]);
            std::printf("\n");
        }
    }
    qwen3tts_codec_free_codes(codes);
    qwen3tts_codec_close(c);
    const bool pass = shape_ok && diff == 0;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
