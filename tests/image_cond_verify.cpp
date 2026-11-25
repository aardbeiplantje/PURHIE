// Check of the engine's image conditioning (src/trellis_image.cpp: the PNG decoded, resized with PIL's Lanczos, normalised, through the
// image encoder) against the python preprocessing and the transformers reference of the same image (tools/dinov3_oracle.py, IMAGE =
// comfy/example.png). Usage: image_verify [oracle_dino] [image.png] [weights]
#include "dinov3.h"
#include "image.h"
#include "trellis_image.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool read_bytes(const std::string& path, std::vector<char>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)bytes);
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    return got == v.size();
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_dino";
    const std::string image = argc > 2 ? argv[2] : "/work/rocm-hip/comfy/example.png";
    const std::string weights = argc > 3 ? argv[3] : "/comfyui/clip_vision/dino_v3_L_naf_fp32.safetensors";
    std::vector<char> png, tb;
    if (!read_bytes(image, png) || !read_bytes(dir + "/tokens.f32", tb)) {
        std::fprintf(stderr, "cannot read %s or the oracle in %s\n", image.c_str(), dir.c_str());
        return 1;
    }
    unsigned char* rgb = nullptr;
    int w = 0, h = 0;
    if (image_decode_rgb(reinterpret_cast<const unsigned char*>(png.data()), png.size(), &rgb, &w, &h) != 0) {
        std::fprintf(stderr, "cannot decode %s\n", image.c_str());
        return 1;
    }
    trellis::DinoV3 enc(weights);
    if (!enc.ok()) {
        std::fprintf(stderr, "open: %s\n", enc.error().c_str());
        return 1;
    }
    std::vector<float> cond;
    std::string error;
    const bool cond_ok = trellis::image_condition(enc, rgb, w, h, 512, cond, error);
    image_free(rgb);
    if (!cond_ok) {
        std::fprintf(stderr, "condition: %s\n", error.c_str());
        return 1;
    }
    const float* ref = reinterpret_cast<const float*>(tb.data());
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < cond.size(); ++i) {
        const double e = (double)cond[i] - ref[i];
        num += e * e;
        den += (double)ref[i] * ref[i];
        mx = std::fmax(mx, std::fabs(e));
    }
    const double rel = std::sqrt(num / den);
    std::printf("%s (%dx%d) -> 512 px: %zu tokens, rel L2 %.3e vs python, max abs %.3e\n", image.c_str(), w, h, cond.size() / 1024, rel, mx);
    const bool ok = rel < 5e-3;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
