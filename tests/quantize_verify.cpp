// quantize_verify -- src/quantize.cpp's encoders against the reference's, byte for byte.
//
//   quantize_verify [model.gguf ...]
//
// The reference is ROCmFPX's ggml-base (ggml_quantize_chunk: rocmfp4.c, quantize_row_q8_0_ref, fp16 / bf16 rows),
// linked into this test only. Inputs: synthetic rows (Gaussian, uniform, heavy-tailed, all-zero / tiny / huge
// blocks, NaN / Inf, values on the codebook's decision boundaries) and, per GGUF given, up to 64 rows of every 2-D
// tensor of a type quant_types.h decodes. Every target type: Q8_0, Q4_0_ROCMFP4, Q4_0_ROCMFP4_FAST, F16, BF16.
// Also: qz_quantize with threads = the single-row encoders, and a decode of the encoded rows round-trips through
// quant_types.h's decoders (the engine reads what the encoder writes).

#include "ggml.h"
#include "gguf_mmap.h"
#include "quant_types.h"
#include "quantize.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
}

static const int kTypes[] = {QT_Q8_0, QT_Q4_0_ROCMFP4, QT_Q4_0_ROCMFP4_FAST, QT_F16, QT_BF16, QT_Q4_0, QT_Q4_1, QT_Q5_0, QT_Q5_1, QT_Q4_K, QT_Q5_K, QT_Q6_K, QT_Q2_K, QT_Q3_K, QT_MXFP4, QT_NVFP4, QT_TQ1_0, QT_TQ2_0, QT_IQ4_NL, QT_IQ4_XS};

// Bytes differing between ours and the reference's for nrows x ncols; first differing row in *first
static size_t compare(int t, const float* x, int64_t nrows, int64_t ncols, int64_t* first) {
    const size_t rs = qt_row_size(t, ncols);
    std::vector<uint8_t> ref(rs * nrows), ours(rs * nrows);
    ggml_quantize_chunk((ggml_type)t, x, ref.data(), 0, nrows, ncols, nullptr);
    char err[256];
    if (qz_quantize(t, ours.data(), QT_F32, x, nrows, ncols, 8, err, sizeof err) != 0) {
        printf("    %s\n", err);
        *first = 0;
        return rs * nrows;
    }
    size_t nd = 0;
    *first = -1;
    for (size_t i = 0; i < ref.size(); ++i)
        if (ref[i] != ours[i]) {
            if (*first < 0) *first = (int64_t)(i / rs);
            ++nd;
        }
    return nd;
}

static void check_all(const char* what, const std::vector<float>& x, int64_t ncols, bool finite_only = false) {
    const int64_t nrows = (int64_t)x.size() / ncols;
    for (int t : kTypes) {
        if (finite_only && (t == QT_Q4_K || t == QT_Q5_K || t == QT_Q6_K || t == QT_Q2_K || t == QT_Q3_K)) continue;  // (see the NaN / Inf case)
        if ((t == QT_Q4_K || t == QT_Q5_K || t == QT_Q6_K || t == QT_Q2_K || t == QT_Q3_K || t == QT_TQ1_0 || t == QT_TQ2_0 || t == QT_IQ4_XS) && ncols % 256) continue;
        if (t == QT_NVFP4 && ncols % 64) continue;  // a block of 64 (the reference reads past a shorter row)  // a K-quant row is whole super-blocks (the reference has no row of 32)
        int64_t first;
        const size_t nd = compare(t, x.data(), nrows, ncols, &first);
        char buf[256];
        snprintf(buf, sizeof buf, "%-28s %4lld x %-6lld %-18s %zu bytes differ%s", what, (long long)nrows,
                 (long long)ncols, qt_get_traits(t)->name, nd, nd ? (" (row " + std::to_string(first) + ")").c_str() : "");
        check(nd == 0, buf);
    }
}

// Encoded rows decode (quant_types.h) to what the reference's dequantizer gives: the engine reads the encoder's output
static void check_roundtrip(const std::vector<float>& x, int64_t ncols) {
    const int64_t nrows = (int64_t)x.size() / ncols;
    for (int t : kTypes) {
        const size_t rs = qt_row_size(t, ncols);
        std::vector<uint8_t> q(rs * nrows);
        char err[256];
        qz_quantize(t, q.data(), QT_F32, x.data(), nrows, ncols, 4, err, sizeof err);
        std::vector<float> a(x.size()), b(x.size());
        qt_to_float(t, q.data(), a.data(), (int64_t)x.size());
        ggml_get_type_traits((ggml_type)t)->to_float(q.data(), b.data(), (int64_t)x.size());
        check(memcmp(a.data(), b.data(), a.size() * 4) == 0,
              std::string("decode of the encoded rows = the reference's: ") + qt_get_traits(t)->name);
    }
}

int main(int argc, char** argv) {
    ggml_quantize_init(GGML_TYPE_Q8_0);
    std::mt19937 rng(1234);
    const int64_t C = 4096;

    printf("synthetic rows\n");
    {
        std::normal_distribution<float> g(0.0f, 0.02f);
        std::vector<float> x(64 * C);
        for (auto& v : x) v = g(rng);
        check_all("gaussian sigma 0.02", x, C);
        check_roundtrip(x, C);
    }
    {
        std::uniform_real_distribution<float> u(-3.0f, 3.0f);
        std::vector<float> x(32 * C);
        for (auto& v : x) v = u(rng);
        check_all("uniform +-3", x, C);
    }
    {  // heavy tails: Gaussian / uniform, scaled per row across 12 decades
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::uniform_real_distribution<float> u(0.05f, 1.0f);
        std::vector<float> x(48 * 1024);
        for (size_t i = 0; i < x.size(); ++i) x[i] = g(rng) / u(rng) * std::pow(10.0f, (float)((i / 1024) % 12) - 8.0f);
        check_all("heavy-tailed, 1e-8 .. 1e3", x, 1024);
    }
    {  // blocks that are all zero, subnormal-scale tiny, beyond the largest scale, one outlier, constant
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> x(16 * 256, 0.0f);
        for (int r = 0; r < 16; ++r)
            for (int j = 0; j < 256; ++j) {
                float& v = x[r * 256 + j];
                switch ((r * 8 + j / 32) % 8) {
                case 0: v = 0.0f; break;
                case 1: v = g(rng) * 1e-7f; break;
                case 2: v = g(rng) * 1e5f; break;
                case 3: v = (j % 32 == 5) ? 50.0f : g(rng) * 0.01f; break;
                case 4: v = 0.75f; break;
                case 5: v = -1e-30f * (float)(j + 1); break;
                case 6: v = (j & 1) ? 1.0f : -1.0f; break;
                default: v = g(rng); break;
                }
            }
        check_all("zero / tiny / huge / outlier", x, 256);
    }
    {  // NaN and Inf among finite values
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> x(8 * 256);
        for (size_t i = 0; i < x.size(); ++i) x[i] = g(rng);
        for (size_t i = 3; i < x.size(); i += 97) x[i] = NAN;
        for (size_t i = 11; i < x.size(); i += 131) x[i] = (i & 1) ? INFINITY : -INFINITY;
        // ggml's K-quant reference indexes its tables with nearest_int of a NaN and crashes (the reference's own
        // undefined behaviour): the K-quants are checked on the finite cases only
        check_all("NaN / Inf", x, 256, true);
    }
    {  // values on the Codebook10 decision boundaries (a = 0.5, 1.5, ... 9.0) for the block's own scale
        std::vector<float> x(16 * 32);
        const float b[] = {0.5f, 1.5f, 2.5f, 3.5f, 5.0f, 7.0f, 9.0f, 10.0f};
        for (int r = 0; r < 16; ++r)
            for (int j = 0; j < 32; ++j) x[r * 32 + j] = b[j % 8] * std::ldexp(1.0f, r - 8) * ((j & 8) ? -1.0f : 1.0f);
        check_all("decision boundaries", x, 32);
    }
    {  // BF16-rounded values (a converted checkpoint's)
        std::normal_distribution<float> g(0.0f, 0.05f);
        std::vector<float> x(32 * C);
        for (auto& v : x) v = qt_bf16_to_fp32(qz_fp32_to_bf16(g(rng)));
        check_all("bf16-rounded gaussian", x, C);
    }

    for (int a = 1; a < argc; ++a) {
        gguf_model_t* g = gguf_mmap_open(argv[a]);
        if (!g) {
            check(false, std::string("open ") + argv[a]);
            continue;
        }
        printf("%s\n", argv[a]);
        int n = 0;
        for (int64_t i = 0; i < gguf_n_tensors(g); ++i) {
            const int type = gguf_tensor_type(g, i);
            const int64_t* ne = gguf_tensor_ne(g, i);
            if (gguf_tensor_dims(g, i) < 2 || ne[0] % 32 || !qt_get_traits(type)) continue;
            int64_t rows = 1;
            for (int d = 1; d < gguf_tensor_dims(g, i); ++d) rows *= ne[d];
            rows = std::min<int64_t>(rows, 64);
            size_t nb = 0;
            const uint8_t* p = (const uint8_t*)gguf_mmap_get_tensor(g, gguf_tensor_name(g, i), &nb);
            std::vector<float> x((size_t)(rows * ne[0]));
            if (qt_to_float(type, p, x.data(), (int64_t)x.size()) != 0) continue;
            for (int t : kTypes) {
                int64_t first;
                const size_t nd = compare(t, x.data(), rows, ne[0], &first);
                if (nd) {
                    char buf[256];
                    snprintf(buf, sizeof buf, "%s (%s) -> %s: %zu bytes differ (row %lld)", gguf_tensor_name(g, i),
                             qt_get_traits(type)->name, qt_get_traits(t)->name, nd, (long long)first);
                    check(false, buf);
                }
            }
            ++n;
        }
        check(n > 0, std::to_string(n) + " tensors x 5 types: the same bytes as the reference");
        gguf_mmap_close(g);
    }
    printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
