// talker_sample_verify: the samplers on the device (talker_gpu.hip's tk_sample_next, tk_sample_c0 and tk_gather_row) against
// a host reference of the same rule: the top_k most likely of the logits by (logit descending, index ascending) (top_k 0:
// all), the draw from their float softmax at the temperature with the cumulative sum in double, the splitmix state advanced
// once per draw (temperature <= 0: the first maximum, no draw). Checks 3000 random rows (vocabularies 2048, 2000, 100, 7, 1;
// temperatures 0, 0.7, 1, 1.3; top_k 0, 1, 50, 2; ties and -inf entries), the advanced state, the position's increment and
// the gathered row (BF16 to float); then 600 rows of the first codebook's draw (3072 logits: the repetition penalty on the
// seen tokens, the suppressed range except eos, eos allowed or not, the token's seen flag, the position's reset).
//
//   talker_sample_verify [talker_gpu.co]   (default build/co/talker_gpu.co, the kernel object of the make target)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint64_t host_next(uint64_t* state) {  // qwen3tts.cpp's next_u64
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Entry {
    float v;
    int i;
};

bool before(const Entry& a, const Entry& b) { return a.v > b.v || (a.v == b.v && a.i < b.i); }

int ref_sample(const std::vector<float>& lg, float T, int top_k, uint64_t& st) {
    const int n = (int)lg.size();
    std::vector<Entry> p(n);
    for (int i = 0; i < n; ++i) p[i] = {lg[i], i};
    std::sort(p.begin(), p.end(), before);
    if (!(p[0].v > -INFINITY)) return 0;
    if (!(T > 0.0f)) return p[0].i;
    const int K = (top_k > 0 && top_k < n) ? top_k : n;
    std::vector<float> ev(K);
    for (int i = 0; i < K; ++i) ev[i] = p[i].v > -INFINITY ? std::exp((p[i].v - p[0].v) / T) : 0.0f;
    double sum = 0.0;
    for (int i = 0; i < K; ++i) sum += (double)ev[i];
    double r = (double)(host_next(&st) >> 11) * (1.0 / 9007199254740992.0) * sum;
    int tok = p[K - 1].i;
    for (int i = 0; i < K; ++i) {
        r -= (double)ev[i];
        if (r < 0.0) {
            tok = p[i].i;
            break;
        }
    }
    return tok;
}

#define CHECK(e)                                                                    \
    do {                                                                            \
        hipError_t err_ = (e);                                                      \
        if (err_ != hipSuccess) {                                                   \
            std::printf("hip error %s at line %d\n", hipGetErrorString(err_), __LINE__); \
            return 1;                                                               \
        }                                                                           \
    } while (0)

}  // namespace

int main(int argc, char** argv) {
    const char* co = argc > 1 ? argv[1] : "build/co/talker_gpu.co";
    hipModule_t mod;
    CHECK(hipModuleLoad(&mod, co));
    hipFunction_t fs, fg, fc;
    CHECK(hipModuleGetFunction(&fs, mod, "tk_sample_next"));
    CHECK(hipModuleGetFunction(&fc, mod, "tk_sample_c0"));
    CHECK(hipModuleGetFunction(&fg, mod, "tk_gather_row"));

    const int C = 1024, ROWS = 2048;
    std::mt19937 rng(12345);
    std::vector<uint16_t> table((size_t)ROWS * C);
    for (auto& x : table) x = (uint16_t)(rng() & 0xffff);
    uint16_t* d_table;
    CHECK(hipMalloc(&d_table, table.size() * sizeof(uint16_t)));
    CHECK(hipMemcpy(d_table, table.data(), table.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    float *d_lg, *d_row;
    CHECK(hipMalloc(&d_lg, 2048 * sizeof(float)));
    CHECK(hipMalloc(&d_row, C * sizeof(float)));
    int *d_tok, *d_pos;
    uint64_t* d_state;
    CHECK(hipMalloc(&d_tok, sizeof(int)));
    CHECK(hipMalloc(&d_pos, sizeof(int)));
    CHECK(hipMalloc(&d_state, sizeof(uint64_t)));

    int fails = 0, trials = 0, tie_rows = 0;
    const int ns[] = {2048, 2000, 100, 7, 1};
    const float Ts[] = {0.0f, 0.7f, 1.0f, 1.3f};
    const int tks[] = {0, 1, 50, 2};
    std::normal_distribution<float> nd(0.0f, 3.0f);
    for (int trial = 0; trial < 3000; ++trial) {
        const int n = ns[trial % 5];
        const float T = Ts[(trial / 5) % 4];
        const int top_k = tks[(trial / 20) % 4];
        std::vector<float> lg(n);
        for (int i = 0; i < n; ++i) lg[i] = nd(rng);
        if (trial % 7 == 3) {  // ties: coarse values
            for (int i = 0; i < n; ++i) lg[i] = std::round(lg[i]);
            tie_rows++;
        }
        if (trial % 11 == 5 && n > 3) {  // -inf entries
            lg[1] = -INFINITY;
            lg[n - 1] = -INFINITY;
        }
        uint64_t st_host = rng(), st_dev = st_host;
        const int want = ref_sample(lg, T, top_k, st_host);
        CHECK(hipMemcpy(d_lg, lg.data(), n * sizeof(float), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d_state, &st_dev, sizeof(uint64_t), hipMemcpyHostToDevice));
        const int pos = 5;
        CHECK(hipMemcpy(d_pos, &pos, sizeof(int), hipMemcpyHostToDevice));
        const int nn = n, C_ = C, next = 1;
        const float Tt = T;
        const int tkk = top_k;
        float* lgp = d_lg;
        int* tokp = d_tok;
        uint64_t* stp = d_state;
        uint16_t* tabp = d_table;
        float* rowp = d_row;
        int* posp = d_pos;
        void* args[] = {(void*)&lgp, (void*)&nn, (void*)&Tt, (void*)&tkk, (void*)&stp, (void*)&tokp, (void*)&tabp,
                        (void*)&C_, (void*)&rowp, (void*)&posp, (void*)&next};
        CHECK(hipModuleLaunchKernel(fs, 1, 1, 1, 1024, 1, 1, 0, nullptr, args, nullptr));
        CHECK(hipDeviceSynchronize());
        int got = -1, pos_after = -1;
        uint64_t st_after = 0;
        std::vector<float> row(C);
        CHECK(hipMemcpy(&got, d_tok, sizeof(int), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&st_after, d_state, sizeof(uint64_t), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&pos_after, d_pos, sizeof(int), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(row.data(), d_row, C * sizeof(float), hipMemcpyDeviceToHost));
        bool ok = got == want && st_after == st_host && pos_after == pos + 1;
        for (int i = 0; ok && i < C; i += 97) {
            const uint32_t u = (uint32_t)table[(size_t)got * C + i] << 16;
            float f;
            std::memcpy(&f, &u, 4);
            if (std::memcmp(&f, &row[i], 4) != 0) ok = false;
        }
        trials++;
        if (!ok) {
            fails++;
            if (fails <= 10)
                std::printf("trial %d n %d T %.1f top_k %d: want %d got %d, state %s, position %d\n", trial, n, T, top_k, want,
                            got, st_after == st_host ? "equal" : "differs", pos_after);
        }
    }
    // the first codebook's draw (tk_sample_c0) against the host's processing of the logits, then the same reference draw
    int c0_trials = 0, c0_fails = 0;
    {
        const int n = 3072, sup_lo = n - 1024;
        int* d_seen;
        float* d_lg3;
        CHECK(hipMalloc(&d_seen, n * sizeof(int)));
        CHECK(hipMalloc(&d_lg3, n * sizeof(float)));
        for (int trial = 0; trial < 600; ++trial) {
            const float T = Ts[trial % 4];
            const int top_k = tks[(trial / 4) % 4];
            const float rep = trial % 3 == 0 ? 1.05f : 1.3f;
            const int eos = sup_lo + (int)(rng() % 1024);
            const int allow_eos = (trial / 2) % 2;
            std::vector<float> lg(n);
            for (int i = 0; i < n; ++i) lg[i] = nd(rng);
            if (trial % 7 == 3)
                for (int i = 0; i < n; ++i) lg[i] = std::round(lg[i]);
            if (trial % 5 == 0) lg[eos] = 20.0f;  // eos the most likely: drawn when allowed, never when not
            std::vector<int> seen(n, 0);
            const int n_seen = (int)(rng() % 200);
            for (int i = 0; i < n_seen; ++i) seen[rng() % n] = 1;
            std::vector<float> pl(lg);
            for (int i = 0; i < n; ++i)
                if (seen[i]) pl[i] = pl[i] < 0.0f ? pl[i] * rep : pl[i] / rep;
            for (int v = sup_lo; v < n; ++v)
                if (v != eos) pl[v] = -INFINITY;
            if (!allow_eos) pl[eos] = -INFINITY;
            uint64_t st_host = rng(), st_dev = st_host;
            const int want = ref_sample(pl, T, top_k, st_host);
            const int pos = 9;
            CHECK(hipMemcpy(d_lg3, lg.data(), n * sizeof(float), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(d_seen, seen.data(), n * sizeof(int), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(d_state, &st_dev, sizeof(uint64_t), hipMemcpyHostToDevice));
            CHECK(hipMemcpy(d_pos, &pos, sizeof(int), hipMemcpyHostToDevice));
            float* lgp = d_lg3;
            uint64_t* stp = d_state;
            int *tokp = d_tok, *seenp = d_seen, *posp = d_pos;
            void* args[] = {(void*)&lgp, (void*)&n, (void*)&T, (void*)&top_k, (void*)&stp, (void*)&tokp, (void*)&seenp,
                            (void*)&rep, (void*)&sup_lo, (void*)&eos, (void*)&allow_eos, (void*)&posp};
            CHECK(hipModuleLaunchKernel(fc, 1, 1, 1, 1024, 1, 1, 0, nullptr, args, nullptr));
            CHECK(hipDeviceSynchronize());
            int got = -1, pos_after = -1;
            uint64_t st_after = 0;
            std::vector<int> seen_after(n);
            CHECK(hipMemcpy(&got, d_tok, sizeof(int), hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(&st_after, d_state, sizeof(uint64_t), hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(&pos_after, d_pos, sizeof(int), hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(seen_after.data(), d_seen, n * sizeof(int), hipMemcpyDeviceToHost));
            if (want >= 0 && want < n) seen[want] = 1;
            const bool ok = got == want && st_after == st_host && pos_after == 0 && seen_after == seen &&
                            (allow_eos || got != eos);
            c0_trials++;
            if (!ok) {
                c0_fails++;
                if (c0_fails <= 10)
                    std::printf("c0 trial %d T %.1f top_k %d eos %d (%s): want %d got %d, state %s, position %d\n", trial, T, top_k,
                                eos, allow_eos ? "allowed" : "suppressed", want, got, st_after == st_host ? "equal" : "differs",
                                pos_after);
            }
        }
        (void)hipFree(d_seen);
        (void)hipFree(d_lg3);
        std::printf("first codebook: %d rows, %d mismatches %s\n", c0_trials, c0_fails, c0_fails ? "FAIL" : "ok");
        fails += c0_fails;
    }
    // the gather alone (tk_gather_row): the row of the token on the device as floats, and the position advances
    {
        const int tok = 777, pos = 3, C_ = C;
        uint16_t* tabp = d_table;
        float* rowp = d_row;
        int *posp = d_pos, *tokp = d_tok;
        CHECK(hipMemcpy(d_pos, &pos, sizeof(int), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(d_tok, &tok, sizeof(int), hipMemcpyHostToDevice));
        void* args[] = {(void*)&tabp, (void*)&tokp, (void*)&C_, (void*)&rowp, (void*)&posp};
        CHECK(hipModuleLaunchKernel(fg, 1, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
        CHECK(hipDeviceSynchronize());
        std::vector<float> row(C);
        int pos_after = -1;
        CHECK(hipMemcpy(row.data(), d_row, C * sizeof(float), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&pos_after, d_pos, sizeof(int), hipMemcpyDeviceToHost));
        bool ok = pos_after == pos + 1;
        for (int i = 0; i < C; ++i) {
            const uint32_t u = (uint32_t)table[(size_t)tok * C + i] << 16;
            float f;
            std::memcpy(&f, &u, 4);
            if (std::memcmp(&f, &row[i], 4) != 0) ok = false;
        }
        std::printf("gather: the row and the position %s\n", ok ? "ok" : "FAIL");
        if (!ok) fails++;
    }
    (void)hipFree(d_table);
    (void)hipFree(d_lg);
    (void)hipFree(d_row);
    (void)hipFree(d_tok);
    (void)hipFree(d_pos);
    (void)hipFree(d_state);
    std::printf("sampler: %d rows (%d with ties), %d mismatches %s\n", trials, tie_rows, fails, fails ? "FAIL" : "ok");
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
