// sd_text_verify: the Stable Diffusion engine's CLIP tokenizer and text encoder (src/sd.cpp, PLAN.md I2) against transformers'
// CLIPTokenizer and CLIPTextModel (diffusers 0.35.2's encode_prompt) for sd-turbo. The reference directory
// (/data/image-ref/sd-turbo): ids.i32 (the prompt's 77 ids, padded with the tokenizer's pad token) and prompt_embeds.f32
// (77 x 1024 floats of the prompt "a photo of a red fox in the snow").
//
//   sd_text_verify <sd.gguf> <reference dir>
#include "sd.h"

#include <cmath>
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

// the prompt's ids as transformers' CLIPTokenizer gives them, without padding (BOS, the tokens, EOS)
bool check_ids(sd_model_t* m, const char* prompt, const std::vector<int>& want) {
    std::vector<int> ids(sd_text_ctx(m));
    const int n = sd_tokenize(m, prompt, ids.data());
    bool ok = n == (int)want.size();
    for (int i = 0; ok && i < n; ++i) ok = ids[i] == want[i];
    std::printf("tokens of \"%s\": %d ids, %s\n", prompt, n, ok ? "equal to transformers'  ok" : "FAIL");
    if (!ok) {
        std::printf("   engine:");
        for (int i = 0; i < n; ++i) std::printf(" %d", ids[i]);
        std::printf("\n   transformers:");
        for (int v : want) std::printf(" %d", v);
        std::printf("\n");
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s <sd.gguf> <reference dir>\n", argv[0]); return 2; }
    const std::string dir = argv[2];
    bool pass = true;
    sd_model_t* m = sd_open(argv[1]);
    if (!m) { std::fprintf(stderr, "sd_open: %s\n", sd_last_error()); return 1; }

    // the reference prompt: the ids (padded to 77) and the embeddings
    const std::vector<int> ids_want = read_raw<int>(dir + "/ids.i32");
    std::vector<int> ids(sd_text_ctx(m));
    const int real = sd_tokenize(m, "a photo of a red fox in the snow", ids.data());
    bool ids_ok = real > 0 && std::equal(ids.begin(), ids.end(), ids_want.begin());
    std::printf("ids of the reference prompt: %d real tokens, all 77 %s\n", real, ids_ok ? "equal to transformers'  ok" : "FAIL");
    pass = pass && ids_ok;
    // a prompt with capitals, a contraction and punctuation (transformers' tokenizer, without the padding)
    pass = check_ids(m, "A  Photo, of CAT's!!", {49406, 320, 1125, 267, 539, 2368, 568, 0, 0, 49407}) && pass;
    pass = check_ids(m, "hello   world 123", {49406, 3306, 1002, 272, 273, 274, 49407}) && pass;
    // punctuation runs (transformers' fast tokenizer): "?!" is "?" with the marker and a bare "!", "..." one piece
    pass = check_ids(m, "wow?! x...y", {49406, 2781, 286, 0, 343, 678, 344, 49407}) && pass;

    const std::vector<float> want = read_raw<float>(dir + "/prompt_embeds.f32");
    std::vector<float> got((size_t)sd_text_ctx(m) * sd_text_dim(m));
    if (sd_encode_prompt(m, "a photo of a red fox in the snow", got.data()) != 0) {
        std::fprintf(stderr, "sd_encode_prompt: %s\n", sd_last_error());
        return 1;
    }
    double peak = 0.0, worst = 0.0, dot = 0.0, gg = 0.0, ww = 0.0;
    const int C = sd_text_dim(m);
    double worst_row_cos = 1.0;
    for (size_t i = 0; i < got.size(); ++i) {
        peak = std::fmax(peak, std::fabs((double)want[i]));
        worst = std::fmax(worst, std::fabs((double)got[i] - want[i]));
        dot += (double)got[i] * want[i];
        gg += (double)got[i] * got[i];
        ww += (double)want[i] * want[i];
    }
    for (size_t r = 0; r < got.size() / C; ++r) {
        double d = 0.0, a = 0.0, b = 0.0;
        for (int i = 0; i < C; ++i) {
            d += (double)got[r * C + i] * want[r * C + i];
            a += (double)got[r * C + i] * got[r * C + i];
            b += (double)want[r * C + i] * want[r * C + i];
        }
        worst_row_cos = std::fmin(worst_row_cos, d / std::sqrt(a * b));
    }
    const double cos = dot / std::sqrt(gg * ww);
    const bool emb_ok = worst_row_cos >= 0.99999 && worst / peak <= 1e-3;
    std::printf("text encoder 77 x %d: worst row cosine %.7f (whole %.7f), max |diff| %.3e = %.2e of the peak %.2f  %s\n", C,
                worst_row_cos, cos, worst, worst / peak, peak, emb_ok ? "ok" : "FAIL");
    pass = pass && emb_ok;
    sd_close(m);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
