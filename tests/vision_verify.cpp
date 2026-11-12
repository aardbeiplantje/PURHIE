// vision_verify: the vision tower (src/vision.cpp) against transformers' outputs for the same image (phase V3 of
// PLAN.md). The reference directory holds .npy files from the dump of tests' reference run (the processor's
// pixel_values and image_grid_thw; the patch embedding, block 0's output and the merged features from transformers).
//
//   vision_verify <mmproj.gguf> <reference dir>
//
// Per stage: the worst row's cosine against the reference and the largest absolute difference; passes when every
// row's cosine is at least 0.999 (the bar for the other architectures' checks). Exit 0 on pass.
#include "vision.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Npy {
    std::vector<int64_t> shape;
    std::string descr;
    std::vector<char> data;
};

Npy load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) { std::fprintf(stderr, "%s: not a .npy\n", path.c_str()); std::exit(2); }
    const int major = (unsigned char)magic[6];
    uint32_t hlen = 0;
    if (major == 1) {
        uint16_t l;
        f.read(reinterpret_cast<char*>(&l), 2);
        hlen = l;
    } else {
        f.read(reinterpret_cast<char*>(&hlen), 4);
    }
    std::string hdr(hlen, ' ');
    f.read(&hdr[0], hlen);
    Npy n;
    const size_t d = hdr.find("'descr': '") + 10;
    n.descr = hdr.substr(d, hdr.find('\'', d) - d);
    const size_t s = hdr.find("'shape': (") + 10;
    std::stringstream ss(hdr.substr(s, hdr.find(')', s) - s));
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.find_first_of("0123456789") != std::string::npos) n.shape.push_back(std::atoll(tok.c_str()));
    }
    int64_t count = 1;
    for (int64_t v : n.shape) count *= v;
    const size_t esz = n.descr == "<f4" ? 4 : 8;
    n.data.resize((size_t)count * esz);
    f.read(n.data.data(), (std::streamsize)n.data.size());
    if (n.descr != "<f4" && n.descr != "<i8") { std::fprintf(stderr, "%s: dtype %s\n", path.c_str(), n.descr.c_str()); std::exit(2); }
    return n;
}

std::vector<float> as_f32(const Npy& n) {
    std::vector<float> v(n.data.size() / 4);
    std::memcpy(v.data(), n.data.data(), n.data.size());
    return v;
}

struct Reference {
    std::map<std::string, Npy> by_stage;
};

struct Compare {
    std::string name;
    int64_t rows = 0, cols = 0;
    double min_cos = 1.0, max_abs = 0.0;
    bool shape_ok = true;
};

Compare compare(const std::string& name, const float* got, int64_t rows, int64_t cols, const Npy& ref) {
    Compare c;
    c.name = name;
    c.rows = rows;
    c.cols = cols;
    if (ref.shape.size() != 2 || ref.shape[0] != rows || ref.shape[1] != cols) {
        c.shape_ok = false;
        return c;
    }
    const std::vector<float> want = as_f32(ref);
    for (int64_t r = 0; r < rows; ++r) {
        double dot = 0, na = 0, nb = 0;
        for (int64_t i = 0; i < cols; ++i) {
            const double a = got[r * cols + i], b = want[r * cols + i];
            dot += a * b;
            na += a * a;
            nb += b * b;
            c.max_abs = std::fmax(c.max_abs, std::fabs(a - b));
        }
        const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
        c.min_cos = std::fmin(c.min_cos, cos);
    }
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <mmproj.gguf> <reference dir>\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    vision_model_t* vm = vision_open(argv[1]);
    if (!vm) { std::fprintf(stderr, "vision_open: %s\n", vision_last_error()); return 1; }

    const Npy pix = load_npy(dir + "/pixel_values.npy");
    const Npy grid = load_npy(dir + "/image_grid_thw.npy");
    if (pix.shape.size() != 2 || grid.shape.size() != 2 || grid.shape[1] != 3) { std::fprintf(stderr, "bad shapes\n"); return 2; }
    const int64_t T = reinterpret_cast<const int64_t*>(grid.data.data())[0];
    const int h = (int)reinterpret_cast<const int64_t*>(grid.data.data())[1];
    const int w = (int)reinterpret_cast<const int64_t*>(grid.data.data())[2];
    if (T != 1 || (int64_t)h * w != pix.shape[0]) { std::fprintf(stderr, "one image of h x w patches expected\n"); return 2; }
    const std::vector<float> pixels = as_f32(pix);
    const int m = vision_merge(vm);
    const int n_out = vision_n_out(vm);
    std::vector<float> out((size_t)(h / m) * (w / m) * n_out);

    struct Ctx {
        const std::string* dir;
        std::vector<Compare>* res;
    } ctx{&dir, nullptr};
    std::vector<Compare> res;
    ctx.res = &res;
    auto stage = [](const char* name, const float* data, int64_t rows, int64_t cols, void* ud) {
        Ctx* c = static_cast<Ctx*>(ud);
        const char* file = std::strcmp(name, "patch_embed") == 0 ? "patch_embed.npy"
                         : std::strcmp(name, "block0") == 0     ? "block0.npy"
                                                               : "image_features.npy";
        c->res->push_back(compare(name, data, rows, cols, load_npy(*c->dir + "/" + file)));
    };
    if (vision_encode(vm, pixels.data(), h, w, out.data(), stage, &ctx) != 0) {
        std::fprintf(stderr, "vision_encode: %s\n", vision_last_error());
        return 1;
    }
    vision_close(vm);

    bool pass = true;
    for (const Compare& c : res) {
        const bool ok = c.shape_ok && c.min_cos >= 0.999;
        pass = pass && ok;
        if (!c.shape_ok) {
            std::printf("%-12s shape %lld x %lld does not match the reference\n", c.name.c_str(), (long long)c.rows, (long long)c.cols);
            continue;
        }
        std::printf("%-12s %lld x %lld  worst row cosine %.6f  max |diff| %.3e  %s\n", c.name.c_str(), (long long)c.rows,
                    (long long)c.cols, c.min_cos, c.max_abs, ok ? "ok" : "FAIL");
    }
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
