/* Z-Image-Turbo's text to image in the engine (src/zimage_pipe.h) against the reference pipeline (diffusers 0.43.0, the ComfyUI
 * files, 'a red square on a blue background', 256 x 256, 8 steps; /data/zimage-ref/ref_gen.py). The reference's noise is given
 * (zi_generate_with_noise), so the images are comparable: the check is the whole chain, the text encoder, the sampler and the
 * decoder, in 8 bits. Thresholds: the stages' own errors add up to a mean of about 1 level (the text encoder 2.2%, the sampler
 * 6.3%, the decoder 3e-6). A seeded run (zi_generate) must also produce an image that is not flat. Reads the vocabulary and merges
 * dumped from the Qwen3 tokenizer.json (/data/zimage-ref/te/vocab.txt, merges.txt). Run from build/co (the kernels).
 *
 * Build: see the Makefile's ZIMAGE_PIPE_VERIFY. */
#include "zimage_pipe.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::vector<std::string> lines(const char* path) {
    std::ifstream f(path);
    std::vector<std::string> v;
    std::string s;
    while (std::getline(f, s)) v.push_back(s);
    return v;
}

static bool read_floats(const char* path, std::vector<float>& v) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    size_t n = fread(v.data(), 4, v.size(), f);
    fclose(f);
    return n == v.size();
}

static bool read_bytes(const char* path, std::vector<unsigned char>& v) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    size_t n = fread(v.data(), 1, v.size(), f);
    fclose(f);
    return n == v.size();
}

int main() {
    const char* prompt = "a red square on a blue background";
    const char* dit = "/comfyui/diffusion_models/z_image_turbo_bf16.safetensors";
    const char* vae = "/comfyui/vae/ae.safetensors";
    const char* te = "/comfyui/text_encoders/qwen_3_4b.safetensors";

    auto vocab = lines("/data/zimage-ref/te/vocab.txt");
    auto merges = lines("/data/zimage-ref/te/merges.txt");
    std::vector<const char*> vp, mp;
    for (auto& s : vocab) vp.push_back(s.c_str());
    for (auto& s : merges) mp.push_back(s.c_str());

    zi_pipe_t* p = zi_open(dit, vae, te, vp.data(), vp.size(), mp.data(), mp.size());
    if (!p) {
        printf("[FAIL] zi_open: %s\n", zi_last_error());
        return 1;
    }

    std::vector<float> noise(16 * 32 * 32);
    std::vector<unsigned char> ref(256 * 256 * 3), rgb(256 * 256 * 3);
    if (!read_floats("/data/zimage-ref/dit/raw/noise.f32", noise) || !read_bytes("/data/zimage-ref/dit/raw/image_u8.bin", ref)) {
        printf("[FAIL] the reference's noise or image is missing\n");
        zi_close(p);
        return 1;
    }

    if (zi_generate_with_noise(p, prompt, noise.data(), rgb.data()) != 0) {
        printf("[FAIL] zi_generate_with_noise: %s\n", zi_last_error());
        zi_close(p);
        return 1;
    }
    long exact = 0, within1 = 0;
    double sumd = 0;
    for (size_t i = 0; i < rgb.size(); ++i) {
        int d = (int)rgb[i] - (int)ref[i];
        if (d == 0) ++exact;
        if (std::abs(d) <= 1) ++within1;
        sumd += std::abs(d);
    }
    const double n = (double)rgb.size();
    const double mean_abs = sumd / n, pct_within1 = 100.0 * within1 / n;
    const bool pass = mean_abs <= 2.0 && pct_within1 >= 70.0;
    printf("%s reference image (given noise): exact %.2f%%, within 1 level %.2f%%, mean abs %.3f levels\n", pass ? "[PASS]" : "[FAIL]",
           100.0 * exact / n, pct_within1, mean_abs);

    /* a seeded run: the image is not flat (its spread of values) */
    if (zi_generate(p, prompt, 256, 256, 1234, rgb.data()) != 0) {
        printf("[FAIL] zi_generate: %s\n", zi_last_error());
        zi_close(p);
        return 1;
    }
    double mean = 0, var = 0;
    for (unsigned char v : rgb) mean += v;
    mean /= n;
    for (unsigned char v : rgb) var += (v - mean) * (v - mean);
    const double sd = std::sqrt(var / n);
    const bool flat_ok = sd > 10.0;
    printf("%s seeded run (seed 1234): mean %.1f, spread %.1f\n", flat_ok ? "[PASS]" : "[FAIL]", mean, sd);

    zi_close(p);
    return pass && flat_ok ? 0 : 1;
}
