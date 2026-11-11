// expert_cpu_bench.cpp <model.gguf> [layer] -- the CPU expert kernel (src/expert_cpu.h) on a model's real experts:
// the AVX-512 path against the scalar one and against the weights decoded to floats, and its time an expert, on
// one thread and on the pool.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "expert_cpu.h"
#include "gguf_mmap.h"
#include "quant_types.h"

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf> [layer]\n", argv[0]); return 2; }
    gguf_model_t* g = gguf_mmap_open(argv[1]);
    if (!g) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    const int il = argc > 2 ? atoi(argv[2]) : 3, E = 4096, FE = 2048, NE = 288;
    char name[96];
    size_t bytes[3];
    const void* w[3];
    const char* tn[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    for (int t = 0; t < 3; ++t) {
        snprintf(name, sizeof name, "blk.%d.%s.weight", il, tn[t]);
        w[t] = gguf_get_tensor_ptr(g, name, &bytes[t]);
        if (!w[t]) { fprintf(stderr, "no %s\n", name); return 2; }
    }
    const size_t rb_gu = (size_t)E / 32 * 17, rb_d = (size_t)FE / 32 * 17;
    if (bytes[0] != rb_gu * FE * NE || bytes[2] != rb_d * E * NE) { fprintf(stderr, "not GLM-5.3's expert shapes\n"); return 2; }
    auto expert = [&](int t, int e) { return (const char*)w[t] + (size_t)e * (bytes[t] / NE); };

    std::vector<float> x(E), d(E / 32);
    std::vector<int8_t> q(E);
    unsigned r = 12345;
    for (float& v : x) { r = r * 1664525u + 1013904223u; v = ((int)(r >> 8) % 2001 - 1000) * 1e-3f; }
    cpu_q8_0_quantize(x.data(), E, q.data(), d.data());

    // correctness: one matvec against the scalar path (same blocks: equal to float rounding) and against floats
    std::vector<float> y(FE), ys(FE), row(E);
    cpu_matvec_fp4fast_q8(expert(0, 5), rb_gu, FE, E, q.data(), d.data(), y.data());
    double worst = 0, dot = 0, na = 0, nb = 0;
    for (int n = 0; n < FE; ++n) {
        float acc = 0;  // decoded weights x dequantized activations
        qt_to_float(QT_Q4_0_ROCMFP4_FAST, expert(0, 5) + (size_t)n * rb_gu, row.data(), E);
        for (int k = 0; k < E; ++k) acc += row[k] * ((float)q[k] * d[k / 32]);
        ys[n] = acc;
        worst = fmax(worst, fabs(acc - y[n]) / (fabs(acc) + 1e-3));
        dot += (double)acc * y[n]; na += (double)acc * acc; nb += (double)y[n] * y[n];
    }
    printf("matvec vs decoded floats: cos %.9f, worst relative difference %.2e\n", dot / sqrt(na * nb), worst);
    const bool ok = dot / sqrt(na * nb) > 0.999999;

    // time: whole experts (gate, up, SwiGLU, down), each a different expert so the weights are not in the CPU cache
    std::vector<float> out((size_t)64 * E), work((size_t)64 * (3 * FE + FE));
    for (int e = 0; e < NE; ++e) {  // in memory first: this times the math, not the disk
        volatile char s = 0;
        for (int t = 0; t < 3; ++t)
            for (size_t o = 0; o < bytes[t] / NE; o += 4096) s += expert(t, e)[o];
    }
    double t0 = now();
    for (int e = 0; e < 32; ++e)
        cpu_expert_ffn(expert(0, e), expert(1, e), rb_gu, expert(2, e), rb_d, FE, E, q.data(), d.data(), 10.0f,
                       out.data(), work.data());
    const double one = (now() - t0) / 32;
    printf("one thread: %.2f ms an expert (%.1f GB/s of weights)\n", one * 1e3, 3.0 * bytes[0] / NE / one / 1e9);
    for (int n : {8, 16, 32, 64}) {
        double best = 1e9;
        for (int rep = 0; rep < 5; ++rep) {
            t0 = now();
            cpu_parallel_for(n, [&](int i) {
                const int e = 40 + (i * 7 + rep * 13) % 240;
                cpu_expert_ffn(expert(0, e), expert(1, e), rb_gu, expert(2, e), rb_d, FE, E, q.data(), d.data(), 10.0f,
                               out.data() + (size_t)i * E, work.data() + (size_t)i * 4 * FE);
            });
            best = fmin(best, now() - t0);
        }
        printf("pool: %2d experts in %.2f ms (%.2f ms an expert, %.1f GB/s)\n", n, best * 1e3, best * 1e3 / n,
               3.0 * n * bytes[0] / NE / best / 1e9);
    }
    gguf_mmap_close(g);
    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
