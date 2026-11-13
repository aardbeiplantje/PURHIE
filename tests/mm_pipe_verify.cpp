// mm_pipe_verify: MiniMax-H3-Image end to end in the engine (src/minimax_pipe.cpp, PLAN.md M5): one image from a prompt, written as a PNG.
//
//   mm_pipe_verify <text.gguf> <dit.gguf> <pdd.gguf> <vae.gguf> <height> <width> <seed> <out.png> [prompt]
#include "minimax_pipe.h"
#include "sd.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 9 && argc != 10) {
        std::fprintf(stderr, "usage: %s <text.gguf> <dit.gguf> <pdd.gguf> <vae.gguf> <height> <width> <seed> <out.png> [prompt]\n", argv[0]);
        return 2;
    }
    const int H = std::atoi(argv[5]), W = std::atoi(argv[6]);
    const unsigned long long seed = std::strtoull(argv[7], nullptr, 10);
    const char* prompt = argc == 10 ? argv[9] : "a photo of a red fox in the snow";
    mm_pipe_t* p = mm_open(argv[1], argv[2], argv[3], argv[4]);
    if (!p) {
        std::fprintf(stderr, "open: %s\n", mm_last_error());
        return 1;
    }
    std::vector<unsigned char> rgb((size_t)H * W * 3);
    auto t0 = std::chrono::steady_clock::now();
    if (mm_generate(p, prompt, H, W, seed, rgb.data()) != 0) {
        std::fprintf(stderr, "generate: %s\n", mm_last_error());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    mm_close(p);
    uint8_t* png = nullptr;
    size_t n = 0;
    if (sd_encode_png(rgb.data(), W, H, &png, &n) != 0) {
        std::fprintf(stderr, "png: %s\n", sd_last_error());
        return 1;
    }
    std::ofstream f(argv[8], std::ios::binary);
    f.write(reinterpret_cast<const char*>(png), (std::streamsize)n);
    sd_free_png(png);
    std::printf("%dx%d, seed %llu: %.1f s, %zu bytes of PNG -> %s\n", W, H, seed,
                std::chrono::duration<double>(t1 - t0).count(), n, argv[8]);
    return 0;
}
