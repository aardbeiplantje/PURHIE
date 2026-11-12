// attn_bench [T = 1024] [reps = 5] -- prefill attention alone (attention.hip attn_prep + attn_flash + attn_combine in
// the prefill mode): a batch of T tokens at several positions of a cache of random K / V, for the head shapes of
// the models; best of reps. TFLOPS counts Q K^T and P V over the positions each token sees (4 n_q D per position).
// No model. Run from build/co (INFERENCE_KERNEL_DIR=. for a rebuilt attention.co).

#include "../src/attention.h"

#include <hip/hip_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(2); } } while (0)

struct Shape { int D, n_q, n_kv, gated; const char* name; };

static uint32_t g_rng = 12345;
static float rnd() { g_rng = g_rng * 1664525u + 1013904223u; return ((g_rng >> 8) & 0xffff) / 32768.0f - 1.0f; }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 1024, reps = argc > 2 ? atoi(argv[2]) : 5;
    if (attention_init("attention.co") != 0) { fprintf(stderr, "cannot load attention.co (run from build/co)\n"); return 2; }
    const int n_ctx = 32768;
    for (const Shape& sh : {Shape{256, 16, 2, 1, "D 256, 16 / 2 heads, gated (Qwen3.6-35B)"},
                            Shape{256, 16, 4, 1, "D 256, 16 / 4 heads, gated (Qwen3.5-4B)"},
                            Shape{128, 16, 8, 0, "D 128, 16 / 8 heads (Qwen3-0.6B)"},
                            Shape{64, 32, 8, 0, "D 64, 32 / 8 heads (Llama-3.2-1B)"}}) {
        const int D = sh.D, NQ = sh.n_q, NKV = sh.n_kv;
        const size_t qrow = (size_t)(sh.gated ? 2 : 1) * NQ * D, kvrow = (size_t)NKV * D;
        float *qfull, *k, *v, *q, *ml, *acc, *out;
        __half *K, *V;
        const size_t ml_n = attention_part_ml_floats(NQ, n_ctx, T), acc_n = attention_part_acc_floats_d(NQ, n_ctx, T, D);
        CK(hipHostMalloc((void**)&qfull, T * qrow * 4, 0));
        CK(hipHostMalloc((void**)&k, T * kvrow * 4, 0));
        CK(hipHostMalloc((void**)&v, T * kvrow * 4, 0));
        CK(hipHostMalloc((void**)&q, T * (size_t)NQ * D * 4, 0));
        CK(hipHostMalloc((void**)&ml, ml_n * 4, 0));
        CK(hipHostMalloc((void**)&acc, acc_n * 4, 0));
        CK(hipHostMalloc((void**)&out, T * (size_t)NQ * D * 4, 0));
        CK(hipHostMalloc((void**)&K, (size_t)n_ctx * kvrow * 2, 0));
        CK(hipHostMalloc((void**)&V, (size_t)n_ctx * kvrow * 2, 0));
        for (size_t i = 0; i < T * qrow; ++i) qfull[i] = 2.0f * rnd();
        for (size_t i = 0; i < T * kvrow; ++i) k[i] = 2.0f * rnd(), v[i] = rnd();
        for (size_t i = 0; i < (size_t)n_ctx * kvrow; ++i) K[i] = __float2half(rnd()), V[i] = __float2half(rnd());
        attention_opts o = {};
        o.head_dim = D;
        o.gated = sh.gated;
        o.no_qk_norm = 1;
        o.part_slots = (long long)attention_part_slots(n_ctx, T);
        attention_set_prefill(n_ctx);
        printf("%s, %d tokens a batch\n", sh.name, T);
        for (int pos0 : {0, 4096, 16384, n_ctx - T}) {
            double best = 1e30;
            for (int r = 0; r < reps + 1; ++r) {
                const double t0 = now();
                CK(attention_forward_opts(qfull, k, v, nullptr, nullptr, K, V, 30, 0, NQ, NKV, 0, pos0, T, 10000.0f, 1e-6f,
                                          q, ml, acc, out, &o, nullptr, 0));
                CK(hipDeviceSynchronize());
                if (r) best = std::min(best, now() - t0);
            }
            const double flops = 4.0 * NQ * D * T * (pos0 + (T + 1) / 2.0);
            double cs = 0.0;
            for (size_t i = 0; i < T * (size_t)NQ * D; i += 97) cs += out[i];
            printf("  at %5d: %8.2f ms  %7.1f us/token  %5.1f TFLOPS  (sum %.6g)\n", pos0, best * 1e3, best * 1e6 / T,
                   flops / best / 1e12, cs);
        }
        attention_set_prefill(0);
        for (void* p : {(void*)qfull, (void*)k, (void*)v, (void*)q, (void*)ml, (void*)acc, (void*)out, (void*)K, (void*)V})
            CK(hipHostFree(p));
    }
    return 0;
}
