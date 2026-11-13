// mm_canvas_verify: the engine's six-step sampler on the RELEASED pipeline's inputs (PLAN.md item 0): the prompt's embeddings,
// the initial noise (torch's seed 0, shape 24 x h x w) and the released run's final latent (the VAE's input), for the canvas
// the released pipeline generates on (a 256 request is generated at 2048 x 2048: h = w = 128).
//
//   mm_canvas_verify <dit.gguf> <pdd.gguf> <text.f32 [T x 5120]> <noise.f32> <released_latent.f32> <h> <w>
#include "minimax_dit.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool read_f32(const char* path, std::vector<float>& v) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)n / 4);
    const size_t got = std::fread(v.data(), 1, (size_t)n, f);
    std::fclose(f);
    return got == (size_t)n;
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr, "usage: %s <dit.gguf> <pdd.gguf> <text.f32> <noise.f32> <released_latent.f32> <h> <w>\n", argv[0]);
        return 2;
    }
    const int h = std::atoi(argv[6]), w = std::atoi(argv[7]);
    std::vector<float> text, noise, want;
    if (!read_f32(argv[3], text) || !read_f32(argv[4], noise) || !read_f32(argv[5], want)) {
        std::fprintf(stderr, "cannot read the inputs\n");
        return 2;
    }
    const int T = (int)(text.size() / 5120);
    mmdit_t* m = mmdit_open(argv[1], argv[2]);
    if (!m) {
        std::fprintf(stderr, "open: %s\n", mmdit_last_error());
        return 1;
    }
    std::vector<float> got(want.size());
    if (mmdit_sample(m, text.data(), T, noise.data(), h, w, got.data()) != 0) {
        std::fprintf(stderr, "sample: %s\n", mmdit_last_error());
        return 1;
    }
    mmdit_close(m);
    double dot = 0, na = 0, nb = 0, peak = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        dot += (double)got[i] * want[i];
        na += (double)got[i] * got[i];
        nb += (double)want[i] * want[i];
        if (std::fabs(want[i]) > peak) peak = std::fabs(want[i]);
    }
    std::printf("sampler vs the released pipeline (%dx%d latent, %d text tokens): cosine %.6f, released peak %.2f, engine norm %.1f, released norm %.1f\n",
                h, w, T, dot / std::sqrt(na * nb), peak, std::sqrt(na), std::sqrt(nb));
    return 0;
}
