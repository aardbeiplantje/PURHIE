/* Z-Image-Turbo at 1024 x 1024 in the engine against diffusers (the same prompt, the same noise: /data/zimage-ref/z1024, ref_1024.py),
 * with the time of the run. Usage: zimage_1024_verify [steps]; run from build/co. Prints the match and the seconds. */
#include "zimage_pipe.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool read_bytes(const std::string& path, void* dst, size_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.read(static_cast<char*>(dst), (std::streamsize)n);
    return (size_t)f.gcount() == n;
}

int main(int argc, char** argv) {
    const int steps = argc > 1 ? std::atoi(argv[1]) : 9;
    const std::string dir = "/data/zimage-ref/z1024/";
    std::ifstream pf(dir + "prompt.txt", std::ios::binary);
    std::stringstream ps;
    ps << pf.rdbuf();
    const std::string prompt = ps.str();

    auto lines = [](const char* p) {
        std::ifstream f(p);
        std::vector<std::string> v;
        std::string s;
        while (std::getline(f, s)) v.push_back(s);
        return v;
    };
    auto vocab = lines("/data/zimage-ref/te/vocab.txt");
    auto merges = lines("/data/zimage-ref/te/merges.txt");
    std::vector<const char*> vp, mp;
    for (auto& s : vocab) vp.push_back(s.c_str());
    for (auto& s : merges) mp.push_back(s.c_str());
    zi_pipe_t* p = zi_open("/comfyui/diffusion_models/z_image_turbo_bf16.safetensors", "/comfyui/vae/ae.safetensors",
                           "/comfyui/text_encoders/qwen_3_4b.safetensors", vp.data(), vp.size(), mp.data(), mp.size());
    if (!p) {
        printf("[FAIL] zi_open: %s\n", zi_last_error());
        return 1;
    }
    std::vector<float> noise(16 * 128 * 128);
    std::vector<unsigned char> ref(1024 * 1024 * 3), rgb(1024 * 1024 * 3);
    if (!read_bytes(dir + "noise.f32", noise.data(), noise.size() * 4) ||
        !read_bytes(dir + "image_" + std::to_string(steps) + ".u8", ref.data(), ref.size())) {
        printf("[FAIL] the noise or the reference image (run ref_1024.py %d)\n", steps);
        zi_close(p);
        return 1;
    }
    /* the first run includes the kernel loads and the caches; the second is the timed one */
    if (zi_generate_noise_steps(p, prompt.c_str(), 1024, 1024, steps, noise.data(), rgb.data()) != 0) {
        printf("[FAIL] zi_generate_noise_steps: %s\n", zi_last_error());
        zi_close(p);
        return 1;
    }
    auto t0 = std::chrono::steady_clock::now();
    const int rc = zi_generate_noise_steps(p, prompt.c_str(), 1024, 1024, steps, noise.data(), rgb.data());
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (rc != 0) {
        printf("[FAIL] zi_generate_noise_steps: %s\n", zi_last_error());
        zi_close(p);
        return 1;
    }
    long exact = 0, within1 = 0;
    double sumd = 0;
    for (size_t i = 0; i < rgb.size(); ++i) {
        const int d = (int)rgb[i] - (int)ref[i];
        exact += d == 0;
        within1 += std::abs(d) <= 1;
        sumd += std::abs(d);
    }
    const double n = (double)rgb.size();
    printf("  %d steps, 1024 x 1024: %.2f s\n", steps, secs);
    printf("  vs diffusers: exact %.2f%%, within 1 level %.2f%%, mean abs %.3f levels\n", 100.0 * exact / n, 100.0 * within1 / n,
           sumd / n);
    zi_close(p);
    return 0;
}
