// expert_cpu.cpp -- see expert_cpu.h.

#include "expert_cpu.h"

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <immintrin.h>

// Signed FP4 codebook at half scale and the UE4M3 scale byte (quant_types.cpp)
static const int8_t kCodebook[16] = {0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10};

static float ue4m3_half(uint8_t e) {
    if (e > 0x7e) return 0.0f;
    const int ex = e >> 3, m = e & 7;
    return ex == 0 ? (float)m * 0x1p-10f : (float)(8 + m) * ldexpf(1.0f, ex - 11);
}

static const float* scale_table() {
    static float t[256];
    static std::once_flag once;
    std::call_once(once, [] { for (int i = 0; i < 256; ++i) t[i] = ue4m3_half((uint8_t)i); });
    return t;
}

static bool have_vnni() {
    static const bool v = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                          __builtin_cpu_supports("avx512vnni") && !getenv("INFERENCE_CPU_SCALAR");
    return v;
}

__attribute__((target("f16c")))  // (every x86-64 CPU since 2012)
void cpu_q8_0_quantize(const float* x, int K, int8_t* q, float* d) {
    for (int b = 0; b < K / 32; ++b, x += 32, q += 32) {
        float am = 0.0f;
        for (int j = 0; j < 32; ++j) am = fmaxf(am, fabsf(x[j]));
        const float id = am != 0.0f ? 127.0f / am : 0.0f;
        for (int j = 0; j < 32; ++j) q[j] = (int8_t)(int)rintf(x[j] * id);
        d[b] = _cvtsh_ss(_cvtss_sh(am / 127.0f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
}

static void matvec_scalar(const uint8_t* W, size_t row_bytes, int N, int K, const int8_t* q, const float* d, float* y) {
    const float* st = scale_table();
    for (int n = 0; n < N; ++n, W += row_bytes) {
        float acc = 0.0f;
        const uint8_t* b = W;
        for (int ib = 0; ib < K / 32; ++ib, b += 17) {
            int s = 0;
            const int8_t* x = q + ib * 32;
            for (int j = 0; j < 16; ++j) s += kCodebook[b[j] & 15] * x[j] + kCodebook[b[j] >> 4] * x[j + 16];
            acc += st[b[16]] * d[ib] * (float)s;
        }
        y[n] = acc;
    }
}

// Two blocks (34 bytes) a 512-bit vector: [low nibbles 0 | high nibbles 0 | low 1 | high 1] through the codebook,
// |w| . (x with w's sign) by VPDPBUSD: 8 partial sums a block, each times the block's scale, summed as floats
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")))
static void matvec_vnni(const uint8_t* W, size_t row_bytes, int N, int K, const int8_t* q, const float* d, float* y) {
    const float* st = scale_table();
    const __m512i cb = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)kCodebook));
    const __m128i m4 = _mm_set1_epi8(15);
    const int np = K / 64;
    for (int n = 0; n < N; ++n, W += row_bytes) {
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const uint8_t* b = W;
        for (int p = 0; p < np; ++p, b += 34) {
            const __m128i n0 = _mm_loadu_si128((const __m128i*)b), n1 = _mm_loadu_si128((const __m128i*)(b + 17));
            __m512i nib = _mm512_castsi128_si512(_mm_and_si128(n0, m4));
            nib = _mm512_inserti32x4(nib, _mm_and_si128(_mm_srli_epi16(n0, 4), m4), 1);
            nib = _mm512_inserti32x4(nib, _mm_and_si128(n1, m4), 2);
            nib = _mm512_inserti32x4(nib, _mm_and_si128(_mm_srli_epi16(n1, 4), m4), 3);
            const __m512i w = _mm512_shuffle_epi8(cb, nib);
            const __m512i x = _mm512_loadu_si512((const void*)(q + (size_t)p * 64));
            const __mmask64 neg = _mm512_movepi8_mask(w);
            const __m512i xs = _mm512_mask_sub_epi8(x, neg, _mm512_setzero_si512(), x);
            const __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(w), xs);
            const __m512 sc = _mm512_insertf32x8(_mm512_set1_ps(st[b[16]] * d[2 * p]),
                                                 _mm256_set1_ps(st[b[33]] * d[2 * p + 1]), 1);
            if (p & 1) acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), sc, acc1);
            else acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), sc, acc0);
        }
        y[n] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

void cpu_matvec_fp4fast_q8(const void* W, size_t row_bytes, int N, int K, const int8_t* q, const float* d, float* y) {
    if (have_vnni()) matvec_vnni((const uint8_t*)W, row_bytes, N, K, q, d, y);
    else matvec_scalar((const uint8_t*)W, row_bytes, N, K, q, d, y);
}

// Q8_0 weights: a block is its F16 scale and 32 int8; two blocks a vector
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,f16c")))
static void matvec_q8_vnni(const uint8_t* W, size_t row_bytes, int N, int K, const int8_t* q, const float* d, float* y) {
    const int np = K / 64;
    for (int n = 0; n < N; ++n, W += row_bytes) {
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const uint8_t* b = W;
        for (int p = 0; p < np; ++p, b += 68) {
            const __m512i w = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(b + 2))),
                                                 _mm256_loadu_si256((const __m256i*)(b + 36)), 1);
            const __m512i x = _mm512_loadu_si512((const void*)(q + (size_t)p * 64));
            const __mmask64 neg = _mm512_movepi8_mask(w);
            const __m512i xs = _mm512_mask_sub_epi8(x, neg, _mm512_setzero_si512(), x);
            const __m512i s = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(w), xs);
            uint16_t h0, h1;
            memcpy(&h0, b, 2);
            memcpy(&h1, b + 34, 2);
            const __m512 sc = _mm512_insertf32x8(_mm512_set1_ps(_cvtsh_ss(h0) * d[2 * p]),
                                                 _mm256_set1_ps(_cvtsh_ss(h1) * d[2 * p + 1]), 1);
            if (p & 1) acc1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), sc, acc1);
            else acc0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), sc, acc0);
        }
        y[n] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    }
}

__attribute__((target("f16c")))
void cpu_matvec_q8_q8(const void* W, size_t row_bytes, int N, int K, const int8_t* q, const float* d, float* y) {
    if (have_vnni()) {
        matvec_q8_vnni((const uint8_t*)W, row_bytes, N, K, q, d, y);
        return;
    }
    const uint8_t* r = (const uint8_t*)W;
    for (int n = 0; n < N; ++n, r += row_bytes) {
        float acc = 0.0f;
        for (int ib = 0; ib < K / 32; ++ib) {
            const uint8_t* b = r + (size_t)ib * 34;
            uint16_t h;
            memcpy(&h, b, 2);
            int s = 0;
            for (int j = 0; j < 32; ++j) s += (int)(int8_t)b[2 + j] * q[ib * 32 + j];
            acc += _cvtsh_ss(h) * d[ib] * (float)s;
        }
        y[n] = acc;
    }
}

void cpu_expert_ffn(const void* gate, const void* up, size_t row_bytes_gu, const void* down, size_t row_bytes_d,
                    int FE, int E, const int8_t* q, const float* d, float limit, float* out, float* work) {
    float* g = work;
    float* u = work + FE;
    float* dq = work + 2 * FE;          // FE / 32 scales
    int8_t* aq = (int8_t*)(work + 3 * FE);
    cpu_matvec_fp4fast_q8(gate, row_bytes_gu, FE, E, q, d, g);
    cpu_matvec_fp4fast_q8(up, row_bytes_gu, FE, E, q, d, u);
    for (int i = 0; i < FE; ++i) {      // glm_swiglu_clamp (glm5_gpu.hip)
        float gv = g[i], uv = u[i];
        if (limit > 0.0f) {
            gv = fminf(gv, limit);
            uv = fminf(fmaxf(uv, -limit), limit);
        }
        g[i] = gv / (1.0f + expf(-gv)) * uv;
    }
    cpu_q8_0_quantize(g, FE, aq, dq);
    cpu_matvec_fp4fast_q8(down, row_bytes_d, E, FE, aq, dq, out);
}

// ---------------------------------------------------------------- the thread pool

namespace {
// A loop: its items are taken by index; a thread that wakes late finds its own loop's items gone, never another's
struct Job {
    const std::function<void(int)>* fn;
    int n;
    std::atomic<int> next{0}, done{0};
};

struct Pool {
    std::vector<std::thread> threads;
    std::mutex m;
    std::condition_variable cv, done_cv;
    std::shared_ptr<Job> job;
    int gen = 0;
    bool stop = false;

    Pool() {
        int nt = (int)std::thread::hardware_concurrency() - 1;
        if (const char* e = getenv("INFERENCE_CPU_THREADS")) nt = atoi(e) - 1;
        nt = nt < 0 ? 0 : nt > 15 ? 15 : nt;
        for (int i = 0; i < nt; ++i) threads.emplace_back([this] { work(); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> lk(m); stop = true; }
        cv.notify_all();
        for (auto& t : threads) t.join();
    }
    void drain(Job& j) {
        for (int i; (i = j.next.fetch_add(1)) < j.n;) {
            (*j.fn)(i);
            if (j.done.fetch_add(1) + 1 == j.n) {
                std::lock_guard<std::mutex> lk(m);
                done_cv.notify_all();
            }
        }
    }
    void work() {
        int seen = 0;
        std::unique_lock<std::mutex> lk(m);
        for (;;) {
            cv.wait(lk, [&] { return stop || gen != seen; });
            if (stop) return;
            seen = gen;
            const std::shared_ptr<Job> j = job;
            lk.unlock();
            if (j) drain(*j);
            lk.lock();
        }
    }
    void run(int count, const std::function<void(int)>& f) {
        const std::shared_ptr<Job> j = std::make_shared<Job>();
        j->fn = &f;
        j->n = count;
        {
            std::lock_guard<std::mutex> lk(m);
            job = j;
            ++gen;
        }
        cv.notify_all();
        drain(*j);
        std::unique_lock<std::mutex> lk(m);
        done_cv.wait(lk, [&] { return j->done.load() == j->n; });  // (f is not called after this)
        job.reset();
    }
};
}  // namespace

void cpu_parallel_for(int n, const std::function<void(int)>& fn) {
    static Pool pool;
    static std::mutex one;  // one loop at a time
    if (n <= 0) return;
    std::lock_guard<std::mutex> lk(one);
    if (n == 1 || pool.threads.empty()) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    pool.run(n, fn);
}
