// codec_after_verify: qwen3tts_codec_decode_after (src/qwen3tts_codec.cpp) against the whole sequence's decode. For a
// sequence of 300 codes (fixed random values), the samples of the frames after R reference frames must equal, bit for bit,
// the whole sequence's samples after the first R frames, for R in {1, 20, 74, 249, 250, 260} (the transformer's window is
// 249 frames: references longer than the window and one that fills it), and again on a cache hit and with a second reference.
// Usage: codec_after_verify <codec.gguf>
#include "qwen3tts_codec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <codec.gguf>\n", argv[0]);
        return 2;
    }
    qwen3tts_codec_t* c = qwen3tts_codec_open(argv[1]);
    if (!c) {
        std::fprintf(stderr, "%s\n", qwen3tts_codec_last_error());
        return 1;
    }
    const int N = 300, S = qwen3tts_codec_samples_per_frame(c), nq = qwen3tts_codec_n_quantizers(c);
    std::vector<int> codes((size_t)N * nq);
    unsigned s = 1;
    for (auto& v : codes) {
        s = s * 1103515245u + 12345u;
        v = (int)((s >> 16) % 2048);
    }
    float* whole = nullptr;
    size_t nw = 0;
    if (qwen3tts_codec_decode(c, codes.data(), N, &whole, &nw) != 0) {
        std::fprintf(stderr, "%s\n", qwen3tts_codec_last_error());
        return 1;
    }
    int fails = 0;
    auto check = [&](const char* what, int R, const int* ref, const std::vector<int>& gen_codes, int T) {
        float* after = nullptr;
        size_t na = 0;
        if (qwen3tts_codec_decode_after(c, ref, R, gen_codes.data(), T, &after, &na) != 0) {
            std::fprintf(stderr, "%s: %s\n", what, qwen3tts_codec_last_error());
            fails++;
            return;
        }
        const size_t from = (size_t)R * S;
        bool same = na == (size_t)T * S && std::memcmp(after, whole + from, sizeof(float) * na) == 0;
        std::printf("%-26s R %3d, T %3d: %s\n", what, R, T, same ? "identical to the whole sequence  ok" : "DIFFERS  FAIL");
        if (!same) fails++;
        qwen3tts_codec_free_wav(after);
    };
    for (int R : {1, 20, 74, 249, 250, 260}) {
        std::vector<int> gen(codes.begin() + (size_t)R * nq, codes.begin() + (size_t)N * nq);
        check("reference of R frames", R, codes.data(), gen, N - R);
    }
    {  // a cache hit: the same reference again
        std::vector<int> gen(codes.begin() + (size_t)74 * nq, codes.begin() + (size_t)N * nq);
        check("cache hit", 74, codes.data(), gen, N - 74);
    }
    {  // a second reference (other frames): its own state, the first one still cached; checked against its own whole decode
        std::vector<int> part(codes.begin() + (size_t)100 * nq, codes.begin() + (size_t)160 * nq);  // 60 frames
        float* whole2 = nullptr;
        size_t n2 = 0;
        if (qwen3tts_codec_decode(c, part.data(), 60, &whole2, &n2) != 0) fails++;
        std::vector<int> gen(part.begin() + (size_t)20 * nq, part.end());
        float* after = nullptr;
        size_t na = 0;
        if (qwen3tts_codec_decode_after(c, part.data(), 20, gen.data(), 40, &after, &na) != 0) fails++;
        bool same = na == (size_t)40 * S && std::memcmp(after, whole2 + (size_t)20 * S, sizeof(float) * na) == 0;
        std::printf("%-26s R %3d, T %3d: %s\n", "second reference", 20, 40, same ? "identical to the whole sequence  ok" : "DIFFERS  FAIL");
        if (!same) fails++;
        qwen3tts_codec_free_wav(after);
        qwen3tts_codec_free_wav(whole2);
        check("first reference again", 74, codes.data(), std::vector<int>(codes.begin() + (size_t)74 * nq, codes.begin() + (size_t)N * nq), N - 74);
    }
    qwen3tts_codec_free_wav(whole);
    qwen3tts_codec_close(c);
    std::printf(fails ? "FAIL\n" : "PASS\n");
    return fails ? 1 : 0;
}
