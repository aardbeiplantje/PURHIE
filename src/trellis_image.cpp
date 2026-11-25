// The image conditioning of the TRELLIS.2 pipeline (trellis_image.h). The resize is PIL's: the Lanczos-3 taps of each output
// position (the filter stretched by the reduction, weights normalised), a horizontal pass and then a vertical pass, each rounded and
// clipped to 8 bits.
#include "trellis_image.h"

#include <algorithm>
#include <cmath>

namespace trellis {
namespace {

double lanczos3(double x) {
    x = std::fabs(x);
    if (x >= 3.0) return 0.0;
    if (x < 1e-12) return 1.0;
    const double px = M_PI * x, px3 = px / 3.0;
    return std::sin(px) / px * std::sin(px3) / px3;
}

struct Taps {
    int first = 0;
    std::vector<double> w;
};

double pil_filter(PilFilter f, double x) {
    x = std::fabs(x);
    switch (f) {
        case PilFilter::Bilinear:
            return x < 1.0 ? 1.0 - x : 0.0;
        case PilFilter::Bicubic: {  // a = -0.5, PIL's
            const double a = -0.5;
            if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
            if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
            return 0.0;
        }
        default:
            return lanczos3(x);
    }
}

double pil_support(PilFilter f) { return f == PilFilter::Bilinear ? 1.0 : f == PilFilter::Bicubic ? 2.0 : 3.0; }

std::vector<Taps> pil_taps(int in, int out, PilFilter f) {
    const double scale = (double)in / out, fscale = std::fmax(1.0, scale), support = pil_support(f) * fscale;
    std::vector<Taps> taps(out);
    for (int o = 0; o < out; ++o) {
        const double center = (o + 0.5) * scale;
        const int lo = std::max(0, (int)(center - support + 0.5)), hi = std::min(in, (int)(center + support + 0.5));
        taps[o].first = lo;
        double sum = 0;
        for (int i = lo; i < hi; ++i) {
            const double wt = pil_filter(f, (i - center + 0.5) / fscale);
            taps[o].w.push_back(wt);
            sum += wt;
        }
        for (double& wt : taps[o].w) wt /= sum;
    }
    return taps;
}

uint8_t clip8(double v) {
    const double r = std::floor(v + 0.5);
    return (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
}

}  // namespace

std::vector<uint8_t> resize_pil8(const uint8_t* px, int w, int h, int ch, int ow, int oh, PilFilter f) {
    std::vector<uint8_t> tmp((size_t)h * ow * ch), out((size_t)oh * ow * ch);
    const std::vector<Taps> tx = pil_taps(w, ow, f), ty = pil_taps(h, oh, f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < ch; ++c) {
                double s = 0;
                for (size_t k = 0; k < tx[x].w.size(); ++k) s += tx[x].w[k] * px[((size_t)y * w + tx[x].first + k) * ch + c];
                tmp[((size_t)y * ow + x) * ch + c] = clip8(s);
            }
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x)
            for (int c = 0; c < ch; ++c) {
                double s = 0;
                for (size_t k = 0; k < ty[y].w.size(); ++k) s += ty[y].w[k] * tmp[((size_t)(ty[y].first + k) * ow + x) * ch + c];
                out[((size_t)y * ow + x) * ch + c] = clip8(s);
            }
    return out;
}

std::vector<uint8_t> resize_lanczos_rgb8(const uint8_t* rgb, int w, int h, int ow, int oh) {
    return resize_pil8(rgb, w, h, 3, ow, oh, PilFilter::Lanczos);
}

bool image_condition(DinoV3& enc, const uint8_t* rgb, int w, int h, int size, std::vector<float>& cond, std::string& error) {
    if (!rgb || w <= 0 || h <= 0 || size <= 0 || size % 16 != 0) {
        error = "the image must be non-empty and the size a multiple of 16";
        return false;
    }
    const std::vector<uint8_t> r = (w == size && h == size) ? std::vector<uint8_t>(rgb, rgb + (size_t)w * h * 3)
                                                             : resize_lanczos_rgb8(rgb, w, h, size, size);
    // [0, 1], then the ImageNet mean and std, as channels-first planes
    const float mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> x((size_t)3 * size * size);
    for (int y = 0; y < size; ++y)
        for (int xx = 0; xx < size; ++xx)
            for (int c = 0; c < 3; ++c) {
                const float v = r[((size_t)y * size + xx) * 3 + c] / 255.0f;
                x[((size_t)c * size + y) * size + xx] = (v - mean[c]) / stdv[c];
            }
    const int T = DinoV3::tokens(size);
    cond.assign((size_t)T * 1024, 0.0f);
    if (!enc.forward(x.data(), size, cond.data())) {
        error = "image encoder: " + enc.error();
        return false;
    }
    return true;
}

}  // namespace trellis
