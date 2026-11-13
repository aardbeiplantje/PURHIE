/* Images for the vision tower (image.h). The resampling follows PIL's (ImagingResample with BICUBIC, a = -0.5: the
 * filter's support widens with the downscale, the weights normalized per output pixel, each pass rounded to 8 bits),
 * and the patch layout follows transformers' Qwen2VLImageProcessor (the reshape / transpose of its _preprocess). */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include "third_party/stb_image.h"

#include "image.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// PIL's bicubic kernel (a = -0.5)
double bicubic(double x) {
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

// One axis of PIL's resample: for each output position, the first input index, the count and the weights
struct Coeffs {
    std::vector<int> first, count;
    std::vector<double> w;  // [out][width]
    int width = 0;
};

Coeffs coefficients(int in, int out) {
    Coeffs c;
    const double scale = (double)in / (double)out;
    const double filterscale = scale > 1.0 ? scale : 1.0;
    const double support = 2.0 * filterscale;
    c.width = (int)std::ceil(support) * 2 + 1;
    c.first.resize(out);
    c.count.resize(out);
    c.w.assign((size_t)out * c.width, 0.0);
    for (int o = 0; o < out; ++o) {
        const double center = (o + 0.5) * scale;
        const int xmin = std::max(0, (int)(center - support + 0.5));
        const int xmax = std::min(in, (int)(center + support + 0.5));
        const int n = xmax - xmin;
        c.first[o] = xmin;
        c.count[o] = n;
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            const double v = bicubic((j + xmin - center + 0.5) / filterscale);
            c.w[(size_t)o * c.width + j] = v;
            sum += v;
        }
        if (sum != 0.0)
            for (int j = 0; j < n; ++j) c.w[(size_t)o * c.width + j] /= sum;
    }
    return c;
}

inline unsigned char clip8(double v) {
    const double r = std::floor(v + 0.5);
    return (unsigned char)(r < 0 ? 0 : r > 255 ? 255 : r);
}

// PIL's two passes (horizontal, then vertical), each rounded to 8 bits, RGB
std::vector<unsigned char> resize_bicubic(const unsigned char* src, int w, int h, int nw, int nh) {
    const Coeffs cx = coefficients(w, nw), cy = coefficients(h, nh);
    std::vector<unsigned char> mid((size_t)nw * h * 3), dst((size_t)nw * nh * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < nw; ++x) {
            double acc[3] = {0, 0, 0};
            for (int j = 0; j < cx.count[x]; ++j) {
                const double wt = cx.w[(size_t)x * cx.width + j];
                const unsigned char* p = src + ((size_t)y * w + cx.first[x] + j) * 3;
                for (int c = 0; c < 3; ++c) acc[c] += wt * p[c];
            }
            for (int c = 0; c < 3; ++c) mid[((size_t)y * nw + x) * 3 + c] = clip8(acc[c]);
        }
    for (int y = 0; y < nh; ++y)
        for (int x = 0; x < nw; ++x) {
            double acc[3] = {0, 0, 0};
            for (int j = 0; j < cy.count[y]; ++j) {
                const double wt = cy.w[(size_t)y * cy.width + j];
                const unsigned char* p = &mid[((size_t)(cy.first[y] + j) * nw + x) * 3];
                for (int c = 0; c < 3; ++c) acc[c] += wt * p[c];
            }
            for (int c = 0; c < 3; ++c) dst[((size_t)y * nw + x) * 3 + c] = clip8(acc[c]);
        }
    return dst;
}

// transformers' smart_resize: the nearest multiples of factor (Python's round: ties to even), then the pixel bounds
bool smart_size(int h, int w, int factor, long long min_pixels, long long max_pixels, int* hb, int* wb) {
    if (h < factor || w < factor) return false;
    int h_bar = (int)std::nearbyint((double)h / factor) * factor;
    int w_bar = (int)std::nearbyint((double)w / factor) * factor;
    if ((long long)h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt((double)h * w / (double)max_pixels);
        h_bar = std::max(factor, (int)std::floor(h / beta / factor) * factor);
        w_bar = std::max(factor, (int)std::floor(w / beta / factor) * factor);
    } else if ((long long)h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt((double)min_pixels / ((double)h * w));
        h_bar = (int)std::ceil(h * beta / factor) * factor;
        w_bar = (int)std::ceil(w * beta / factor) * factor;
    }
    *hb = h_bar;
    *wb = w_bar;
    return true;
}

}  // namespace

extern "C" {

int image_decode_rgb(const unsigned char* data, size_t n, unsigned char** rgb_out, int* w_out, int* h_out) {
    if (!data || !rgb_out || !w_out || !h_out || n == 0 || n > 0x7fffffffu) return -1;
    int w = 0, h = 0, comp = 0;
    unsigned char* px = stbi_load_from_memory(data, (int)n, &w, &h, &comp, 3);
    if (!px || w <= 0 || h <= 0) {
        if (px) stbi_image_free(px);
        return -1;
    }
    *rgb_out = px;  // (stb's allocation: image_free releases it)
    *w_out = w;
    *h_out = h;
    return 0;
}

void image_free(void* p) {
    if (p) stbi_image_free(p);
}

int image_qwen_pixels(const unsigned char* rgb, int w, int h, int patch, int merge, int temporal, float mean, float stdev,
                      long long min_pixels, long long max_pixels, float** rows_out, int* grid_h, int* grid_w) {
    if (!rgb || !rows_out || !grid_h || !grid_w || patch <= 0 || merge <= 0 || temporal <= 0 || stdev == 0.0f) return -1;
    int nh = 0, nw = 0;
    if (!smart_size(h, w, patch * merge, min_pixels, max_pixels, &nh, &nw)) return -1;
    const std::vector<unsigned char> img = (nh == h && nw == w)
                                               ? std::vector<unsigned char>(rgb, rgb + (size_t)w * h * 3)
                                               : resize_bicubic(rgb, w, h, nw, nh);
    const int gh = nh / patch, gw = nw / patch;
    const int C = 3, row = C * temporal * patch * patch;
    float* out = static_cast<float*>(std::malloc(sizeof(float) * (size_t)gh * gw * row));
    if (!out) return -1;
    // rows in merge order: (hb, wb) blocks raster, then (ir, ic) inside a block; features (c, t, py, px)
    for (int hb = 0; hb < gh / merge; ++hb)
        for (int wb = 0; wb < gw / merge; ++wb)
            for (int ir = 0; ir < merge; ++ir)
                for (int ic = 0; ic < merge; ++ic) {
                    const int r = ((hb * (gw / merge) + wb) * merge + ir) * merge + ic;
                    float* o = out + (size_t)r * row;
                    const int gy = hb * merge + ir, gx = wb * merge + ic;  // the patch's grid position
                    for (int c = 0; c < C; ++c)
                        for (int t = 0; t < temporal; ++t)
                            for (int py = 0; py < patch; ++py)
                                for (int px = 0; px < patch; ++px) {
                                    const int y = gy * patch + py, x = gx * patch + px;
                                    const float v = img[((size_t)y * nw + x) * 3 + c] / 255.0f;
                                    *o++ = (v - mean) / stdev;
                                }
                }
    *rows_out = out;
    *grid_h = gh;
    *grid_w = gw;
    return 0;
}

}  // extern "C"
