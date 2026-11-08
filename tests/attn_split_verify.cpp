// attn_split_verify -- prefill attention is the same however the batch is cut: in the prefill mode
// (attention_set_prefill), a batch of n tokens at positions 0 .. n-1 and the batch of its first k tokens give the same
// bits for those k tokens, for every k (the KV cache rows past the shorter batch hold NaN: never read). No model: random
// Q / K / V of the shapes of Qwen3.5 / 3.6 (head_dim 256) and Qwen3 (128). Run from build/co (the kernels).
//   attn_split_verify [n = 40]
// ATTN_SPLIT_VERBOSE=1: each differing batch's token and size. ATTN_SPLIT_VMOD=0 / n / 3: token 2's output when the
// V of the tokens after it (P = 0 there) is zeroed, negated or tripled: the WMMA's P V depends on the sign of such
// zero products (zero / negated differ, tripled does not), which the kernel's diagonal block works around

#include "../src/attention.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); exit(2); } } while (0)

struct Shape { int D, n_q, n_kv, gated; const char* name; };

static uint32_t g_rng = 12345;
static float rnd() { g_rng = g_rng * 1664525u + 1013904223u; return ((g_rng >> 8) & 0xffff) / 32768.0f - 1.0f; }

int main(int argc, char** argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 40;
    if (attention_init("attention.co") != 0) { fprintf(stderr, "cannot load attention.co (run from build/co)\n"); return 2; }
    const int n_ctx = 4096;
    int fails = 0, checks = 0;
    for (const Shape& sh : {Shape{256, 16, 4, 1, "head_dim 256, 16 / 4 heads, gated (Qwen3.5)"},
                            Shape{256, 16, 2, 1, "head_dim 256, 16 / 2 heads, gated (Qwen3.6-35B)"},
                            Shape{128, 16, 8, 0, "head_dim 128, 16 / 8 heads (Qwen3)"}}) {
        const int D = sh.D, NQ = sh.n_q, NKV = sh.n_kv;
        const size_t qrow = (size_t)(sh.gated ? 2 : 1) * NQ * D, kvrow = (size_t)NKV * D;
        float *qfull, *k, *v, *q, *ml, *acc, *outA, *outB;
        __half *K, *V;
        const size_t ml_n = attention_part_ml_floats(NQ, n_ctx, n), acc_n = attention_part_acc_floats_d(NQ, n_ctx, n, D);
        CK(hipHostMalloc((void**)&qfull, n * qrow * 4, 0));
        CK(hipHostMalloc((void**)&k, n * kvrow * 4, 0));
        CK(hipHostMalloc((void**)&v, n * kvrow * 4, 0));
        CK(hipHostMalloc((void**)&q, n * (size_t)NQ * D * 4, 0));
        CK(hipHostMalloc((void**)&ml, ml_n * 4, 0));
        CK(hipHostMalloc((void**)&acc, acc_n * 4, 0));
        CK(hipHostMalloc((void**)&outA, n * (size_t)NQ * D * 4, 0));
        CK(hipHostMalloc((void**)&outB, n * (size_t)NQ * D * 4, 0));
        CK(hipHostMalloc((void**)&K, (size_t)n_ctx * kvrow * 2, 0));
        CK(hipHostMalloc((void**)&V, (size_t)n_ctx * kvrow * 2, 0));
        for (size_t i = 0; i < n * qrow; ++i) qfull[i] = 2.0f * rnd();
        for (size_t i = 0; i < n * kvrow; ++i) k[i] = 2.0f * rnd(), v[i] = rnd();
        attention_opts o = {};
        o.head_dim = D;
        o.gated = sh.gated;
        o.no_qk_norm = 1;
        o.part_slots = (long long)attention_part_slots(n_ctx, n);
        auto run = [&](int T, float* out) {
            const __half nan_h = __float2half(NAN);
            for (size_t i = 0; i < (size_t)n_ctx * kvrow; ++i) K[i] = V[i] = nan_h;  // rows not written: NaN
            CK(attention_forward_opts(qfull, k, v, nullptr, nullptr, K, V, 30, 0, NQ, NKV, 0, 0, T, 10000.0f, 1e-6f,
                                      q, ml, acc, out, &o, nullptr, 0));
            CK(hipDeviceSynchronize());
        };
        attention_set_prefill(n_ctx);
        if (getenv("ATTN_SPLIT_VMOD")) {  // token 2's output when V (and K) of the tokens after it change: P is 0 there
            run(n, outA);
            std::vector<float> keep(v + 3 * kvrow, v + n * kvrow);
            for (size_t i = 3 * kvrow; i < n * kvrow; ++i) v[i] = getenv("ATTN_SPLIT_VMOD")[0] == '0' ? 0.0f : getenv("ATTN_SPLIT_VMOD")[0] == 'n' ? -v[i] : 3.0f * v[i];
            run(n, outB);
            std::copy(keep.begin(), keep.end(), v + 3 * kvrow);
            const char m0 = getenv("ATTN_SPLIT_VMOD")[0];
            printf("    token 2 with V of tokens 3.. %s: %s\n", m0 == '0' ? "zero" : m0 == 'n' ? "negated" : "tripled",
                   memcmp(outA + 2 * (size_t)NQ * D, outB + 2 * (size_t)NQ * D, (size_t)NQ * D * 4) ? "DIFFERS" : "same");
        }
        run(n, outA);
        int bad = 0, first_k = -1, first_z = -1;
        for (int kk = 1; kk < n; ++kk) {
            run(kk, outB);
            ++checks;
            for (int z = 0; z < kk; ++z)
                if (memcmp(outA + (size_t)z * NQ * D, outB + (size_t)z * NQ * D, (size_t)NQ * D * 4) != 0) {
                    if (first_k < 0) first_k = kk, first_z = z;
                    ++bad;
                    if (getenv("ATTN_SPLIT_VERBOSE")) {
                        float md = 0.0f;
                        int nd = 0;
                        for (size_t i = 0; i < (size_t)NQ * D; ++i) {
                            const float a = outA[(size_t)z * NQ * D + i], b = outB[(size_t)z * NQ * D + i];
                            if (memcmp(&a, &b, 4) != 0) ++nd, md = std::fmax(md, std::fabs(a - b));
                        }
                        printf("    k %d: token %d of %d differs (%d values, max |d| %.3g)\n", kk, z, kk, nd, md);
                    }
                    break;
                }
        }
        attention_set_prefill(0);
        if (bad == 0) printf("  [PASS] %s: every batch of the first k of %d tokens agrees\n", sh.name, n);
        else printf("  [FAIL] %s: %d of %d batch lengths differ (first: k = %d, token %d)\n", sh.name, bad, n - 1,
                    first_k, first_z);
        fails += bad > 0;
        for (void* p : {(void*)qfull, (void*)k, (void*)v, (void*)q, (void*)ml, (void*)acc, (void*)outA, (void*)outB,
                        (void*)K, (void*)V})
            CK(hipHostFree(p));
    }
    printf("%s (%d checks)\n", fails ? "FAILED" : "PASSED", checks);
    return fails ? 1 : 0;
}
