// mrope_verify: the M-RoPE positions (src/mrope.cpp) against transformers' get_rope_index for the vision reference's
// prompt. usage: mrope_verify <reference dir>   (input_ids.npy, image_grid_thw.npy, positions.npy, rope_delta.npy)
#include "mrope.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<char> read_npy(const std::string& path, std::vector<long long>* shape, std::string* descr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(2); }
    char magic[8];
    f.read(magic, 8);
    uint16_t hl = 0;
    f.read(reinterpret_cast<char*>(&hl), 2);
    std::string hdr(hl, ' ');
    f.read(&hdr[0], hl);
    const size_t d = hdr.find("'descr': '") + 10;
    *descr = hdr.substr(d, hdr.find('\'', d) - d);
    shape->clear();
    size_t s = hdr.find("'shape': (") + 10;
    const size_t e = hdr.find(')', s);
    std::string body = hdr.substr(s, e - s);
    for (size_t p = 0; p < body.size();) {
        size_t c = body.find(',', p);
        if (c == std::string::npos) c = body.size();
        std::string tok = body.substr(p, c - p);
        if (tok.find_first_of("0123456789") != std::string::npos) shape->push_back(std::atoll(tok.c_str()));
        p = c + 1;
    }
    long long n = 1;
    for (long long v : *shape) n *= v;
    const size_t esz = (*descr == "<i8" || *descr == "<f8") ? 8 : (*descr == "<i4" ? 4 : 4);
    std::vector<char> data((size_t)n * esz);
    f.read(data.data(), (std::streamsize)data.size());
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <reference dir>\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    std::vector<long long> sh;
    std::string de;
    const std::vector<char> ids_raw = read_npy(dir + "/input_ids.npy", &sh, &de);
    const int n = (int)sh.back() == (int)sh.back() ? (int)(ids_raw.size() / 8) : 0;  // int64 ids
    std::vector<int32_t> ids(n);
    for (int i = 0; i < n; ++i) { int64_t v; std::memcpy(&v, &ids_raw[8 * i], 8); ids[i] = (int32_t)v; }
    const std::vector<char> grid_raw = read_npy(dir + "/image_grid_thw.npy", &sh, &de);
    std::vector<int32_t> grid(grid_raw.size() / 8);
    for (size_t i = 0; i < grid.size(); ++i) { int64_t v; std::memcpy(&v, &grid_raw[8 * i], 8); grid[i] = (int32_t)v; }
    const int n_img = (int)grid.size() / 3;
    std::vector<int32_t> hw(2 * n_img);
    for (int i = 0; i < n_img; ++i) { hw[2 * i] = grid[3 * i + 1]; hw[2 * i + 1] = grid[3 * i + 2]; }
    const std::vector<char> ref_raw = read_npy(dir + "/positions.npy", &sh, &de);
    std::vector<int64_t> ref(ref_raw.size() / 8);
    std::memcpy(ref.data(), ref_raw.data(), ref_raw.size());
    std::vector<int64_t> got(3 * (size_t)n);
    int64_t delta = 0;
    if (mrope_positions(ids.data(), n, hw.data(), n_img, 2, 248056, got.data(), &delta) != 0) {
        std::fprintf(stderr, "mrope_positions failed\n");
        return 1;
    }
    int bad = 0;
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < 3; ++a)
            if (got[3 * i + a] != ref[3 * i + a]) {
                if (bad < 5) std::printf("token %d axis %d: got %lld want %lld\n", i, a, (long long)got[3 * i + a], (long long)ref[3 * i + a]);
                ++bad;
            }
    const std::vector<char> dl = read_npy(dir + "/rope_delta.npy", &sh, &de);
    int64_t want_delta;
    std::memcpy(&want_delta, dl.data(), 8);
    if (delta != want_delta) { std::printf("delta: got %lld want %lld\n", (long long)delta, (long long)want_delta); ++bad; }
    std::printf("%d tokens, %d images: %s (delta %lld)\n", n, n_img, bad == 0 ? "PASS" : "FAIL", (long long)delta);
    return bad == 0 ? 0 : 1;
}
