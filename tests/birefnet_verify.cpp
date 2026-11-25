// BiRefNet on the device (src/birefnet.cpp) against the python model's stage dumps (tools/bg_crop_birefnet.py --dump DIR: DIR/NAME.f32
// and DIR/index.json). The dumped input runs through BiRefNet::forward; each stage it reports is compared with the dump of the same name
// (the error relative to the dump's norm, the largest difference relative to its largest value). With IMAGE and MASK (the tool's
// out_dir/mask.png), BiRefNet::mask on the image is compared with the tool's mask too (8-bit values: the mean and largest difference).
// Usage: birefnet_verify WEIGHTS DUMP_DIR [IMAGE MASK]
#include "src/birefnet.h"
#include "src/image.h"
#include "sd_device.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static bool read_file(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

static bool read_f32(const std::string& path, size_t n, std::vector<float>& out) {
    std::vector<unsigned char> b;
    if (!read_file(path, b) || b.size() != n * sizeof(float)) return false;
    out.resize(n);
    std::memcpy(out.data(), b.data(), b.size());
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        std::fprintf(stderr, "usage: %s WEIGHTS DUMP_DIR [IMAGE MASK]\n", argv[0]);
        return 2;
    }
    if (!sddev::enabled()) {
        std::fprintf(stderr, "no device\n");
        return 1;
    }
    const std::string dir = argv[2];
    auto t0 = std::chrono::steady_clock::now();
    trellis::BiRefNet net(argv[1]);
    if (!net.ok()) {
        std::fprintf(stderr, "BiRefNet: %s\n", net.error().c_str());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    std::printf("loaded in %.2f s\n", std::chrono::duration<double>(t1 - t0).count());
    std::vector<float> x;
    if (!read_f32(dir + "/input.f32", (size_t)3 * 1024 * 1024, x)) {
        std::fprintf(stderr, "no %s/input.f32 of 3 x 1024 x 1024 floats\n", dir.c_str());
        return 1;
    }
    // the stage tolerances: the relative error of the norm (the backbone's long chain of F16-weight GEMMs in FP32 against torch's)
    const double tol = 2e-3;
    int bad = 0, seen = 0;
    std::vector<float> ref;
    auto stage = [&](const char* name, const float* d, size_t n) {
        ++seen;
        if (!read_f32(dir + "/" + name + ".f32", n, ref)) {
            std::printf("  %-18s no dump of %zu floats\n", name, n);
            ++bad;
            return;
        }
        double se = 0, sr = 0, maxd = 0, maxr = 0;
        for (size_t i = 0; i < n; ++i) {
            const double e = (double)d[i] - ref[i];
            se += e * e;
            sr += (double)ref[i] * ref[i];
            maxd = std::max(maxd, std::fabs(e));
            maxr = std::max(maxr, (double)std::fabs(ref[i]));
        }
        const double rel = std::sqrt(se / std::max(sr, 1e-30));
        const bool ok = std::isfinite(rel) && rel < tol;
        if (!ok) ++bad;
        std::printf("  %-18s %9zu  rel %.2e  max|d| %.3e (of %.3e)  %s\n", name, n, rel, maxd, maxr, ok ? "ok" : "FAIL");
        std::fflush(stdout);
    };
    std::vector<float> logits((size_t)1024 * 1024);
    if (!net.forward(x.data(), logits.data(), stage)) {
        std::fprintf(stderr, "forward: %s\n", net.error().c_str());
        return 1;
    }
    auto t2 = std::chrono::steady_clock::now();
    std::printf("forward (with the checks) in %.2f s; %d stages, %d off\n", std::chrono::duration<double>(t2 - t1).count(), seen, bad);
    // the forward alone, timed
    auto t3 = std::chrono::steady_clock::now();
    net.forward(x.data(), logits.data());
    auto t4 = std::chrono::steady_clock::now();
    std::printf("forward in %.2f s\n", std::chrono::duration<double>(t4 - t3).count());
    // the mask's pixels against the python mask's (sigmoid > 0.5)
    {
        std::vector<float> rl;
        if (read_f32(dir + "/conv_out1.f32", logits.size(), rl)) {
            size_t diff = 0;
            for (size_t i = 0; i < rl.size(); ++i) diff += (logits[i] > 0) != (rl[i] > 0);
            std::printf("mask pixels that differ (at 0.5): %zu of %zu\n", diff, rl.size());
        }
    }
    if (argc == 5) {
        std::vector<unsigned char> ib, mb;
        unsigned char *rgb = nullptr, *mrgb = nullptr;
        int w = 0, h = 0, mw = 0, mh = 0;
        if (!read_file(argv[3], ib) || image_decode_rgb(ib.data(), ib.size(), &rgb, &w, &h) != 0 || !read_file(argv[4], mb) ||
            image_decode_rgb(mb.data(), mb.size(), &mrgb, &mw, &mh) != 0 || mw != w || mh != h) {
            std::fprintf(stderr, "cannot read %s and %s (the same size)\n", argv[3], argv[4]);
            return 1;
        }
        std::vector<float> m;
        auto t5 = std::chrono::steady_clock::now();
        if (!net.mask(rgb, w, h, m)) {
            std::fprintf(stderr, "mask: %s\n", net.error().c_str());
            return 1;
        }
        auto t6 = std::chrono::steady_clock::now();
        double sum = 0;
        int maxd = 0;
        size_t off = 0;
        for (size_t i = 0; i < m.size(); ++i) {
            const int a = (int)std::lround(m[i] * 255.0f), b = mrgb[i * 3];
            sum += std::abs(a - b);
            maxd = std::max(maxd, std::abs(a - b));
            off += std::abs(a - b) > 8;
        }
        std::printf("mask %d x %d in %.2f s: mean |d| %.3f, max |d| %d (8-bit), %zu pixels off by more than 8\n", w, h,
                    std::chrono::duration<double>(t6 - t5).count(), sum / m.size(), maxd, off);
        if (off > m.size() / 1000) ++bad;
        image_free(rgb);
        image_free(mrgb);
    }
    std::printf("%s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
