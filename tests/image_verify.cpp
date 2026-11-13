// image_verify: an image file through the image processor (src/image.cpp) against transformers' Qwen2VLImageProcessor
// for the same image (phase V5 of PLAN.md): the pixel rows, then the vision tower's features on them (src/vision.cpp).
//
//   image_verify <image file> <mmproj.gguf> <reference dir>
//
// The reference directory: the V1 dump (pixel_values.npy, image_grid_thw.npy, image_features.npy, test.png). Prints the
// pixel rows' largest difference and cosine, the features' worst row cosine; passes when the features' worst row is at
// least 0.99 (the pixels differ slightly from PIL's by resampling rounding) and the grid is equal.
#include "image.h"
#include "vision.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Npy {
    std::vector<long long> shape;
    std::string descr;
    std::vector<char> data;
};

Npy load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    char magic[8];
    f.read(magic, 8);
    uint16_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 2);
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    Npy n;
    const size_t d = hdr.find("'descr': '") + 10;
    n.descr = hdr.substr(d, hdr.find('\'', d) - d);
    const size_t s = hdr.find("'shape': (") + 10;
    const std::string body = hdr.substr(s, hdr.find(')', s) - s);
    for (size_t p = 0; p < body.size();) {
        size_t c = body.find(',', p);
        if (c == std::string::npos) c = body.size();
        const std::string tok = body.substr(p, c - p);
        if (tok.find_first_of("0123456789") != std::string::npos) n.shape.push_back(std::atoll(tok.c_str()));
        p = c + 1;
    }
    long long count = 1;
    for (long long v : n.shape) count *= v;
    const size_t esz = (n.descr == "<f4" || n.descr == "<i4") ? 4 : 8;
    n.data.resize((size_t)count * esz);
    f.read(n.data.data(), (std::streamsize)n.data.size());
    return n;
}

std::vector<float> as_f32(const Npy& n) {
    std::vector<float> v(n.data.size() / 4);
    std::memcpy(v.data(), n.data.data(), n.data.size());
    return v;
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <image file> <mmproj.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::vector<char> file = read_file(argv[1]);
    unsigned char* rgb = nullptr;
    int w = 0, h = 0;
    if (image_decode_rgb(reinterpret_cast<const unsigned char*>(file.data()), file.size(), &rgb, &w, &h) != 0) {
        std::fprintf(stderr, "cannot decode %s\n", argv[1]);
        return 1;
    }
    float* rows = nullptr;
    int gh = 0, gw = 0;
    // the processor's defaults for Qwen3.5: patch 16, merge 2, temporal 2, mean and std 0.5, the pixel bounds
    if (image_qwen_pixels(rgb, w, h, 16, 2, 2, 0.5f, 0.5f, 65536, 16777216, &rows, &gh, &gw) != 0) {
        std::fprintf(stderr, "image_qwen_pixels failed\n");
        return 1;
    }
    image_free(rgb);

    const std::string dir = argv[3];
    const Npy ref_pix = load(dir + "/pixel_values.npy");
    const Npy ref_grid = load(dir + "/image_grid_thw.npy");
    const Npy ref_feat = load(dir + "/image_features.npy");
    const std::vector<float> want = as_f32(ref_pix);
    const size_t n = want.size();
    const int64_t* g = reinterpret_cast<const int64_t*>(ref_grid.data.data());
    bool pass = gh == (int)g[1] && gw == (int)g[2] && (size_t)gh * gw * 1536 == n;
    std::printf("grid %d x %d (reference %lld x %lld)\n", gh, gw, (long long)g[1], (long long)g[2]);
    if (pass) {
        double max_abs = 0, dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < n; ++i) {
            const double a = rows[i], b = want[i];
            max_abs = std::fmax(max_abs, std::fabs(a - b));
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        std::printf("pixel rows: cosine %.6f, max |diff| %.4f (the reference's range +-1)\n", dot / std::sqrt(na * nb), max_abs);
    }

    vision_model_t* vm = vision_open(argv[2]);
    if (!vm) { std::fprintf(stderr, "vision_open: %s\n", vision_last_error()); return 1; }
    const int E = vision_n_out(vm);
    const size_t rows_out = (size_t)(gh / 2) * (gw / 2);
    std::vector<float> feats(rows_out * E);
    if (vision_encode(vm, rows, gh, gw, feats.data(), nullptr, nullptr) != 0) {
        std::fprintf(stderr, "vision_encode: %s\n", vision_last_error());
        return 1;
    }
    vision_close(vm);
    const std::vector<float> want_f = as_f32(ref_feat);
    double worst = 1.0;
    for (size_t r = 0; r < rows_out && want_f.size() == feats.size(); ++r) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < E; ++i) {
            const double a = feats[r * E + i], b = want_f[r * E + i];
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        worst = std::fmin(worst, dot / (std::sqrt(na * nb) + 1e-30));
    }
    pass = pass && want_f.size() == feats.size() && worst >= 0.99;
    std::printf("features: worst row cosine %.6f against the reference\n", worst);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    std::free(rows);
    return pass ? 0 : 1;
}
