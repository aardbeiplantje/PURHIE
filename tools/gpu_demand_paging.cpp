// gpu_demand_paging.cpp -- can a GPU kernel read a mapped file that was never registered (hipHostRegister), the
// driver paging it in on the GPU's faults? Not as this host boots (amdgpu noretry=1: a GPU fault is fatal to the
// process, perhaps to more); to be run after booting with amdgpu.noretry=0 (kernel command line), with the
// deployment down. If it works, a model larger than memory needs no expert pager (src/expert_pager.h): the kernel's
// own LRU decides what stays.
//
//   /opt/rocm/bin/hipcc -O2 -w tools/gpu_demand_paging.cpp -o /tmp/gpu_demand_paging
//   /tmp/gpu_demand_paging <file of at least 2 GB, e.g. /data/conv/glm53-head8.gguf> [offset] [--force]
//
// Steps (each prints before it starts: the last line printed is the step that killed it; then see dmesg):
//   1  registered range: the kernel's sum = the CPU's (works on any boot)
//   2  unregistered, pages in memory and mapped by the CPU
//   3  unregistered, pages not in memory (evicted): read from the file on the GPU's faults; MB/s
//   4  those pages evicted again: do they stay out (demand paging), or does the driver bring them back (as it does
//      for a registered range with noretry=1)? Then read once more
//   5  64 ranges of 4.25 MB (an expert tensor's size), evicted: as they are, and with MADV_WILLNEED first
//   6  1 GB evicted, one launch
// Without --force it stops after step 1 when /sys/module/amdgpu/parameters/noretry is not 0.

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The sum of n bytes at p, over 64 x 256 threads (n a multiple of 16384)
__global__ void sum_k(const unsigned char* p, size_t n, unsigned long long* out) {
    const size_t per = n / ((size_t)gridDim.x * blockDim.x);
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long s = 0;
    for (size_t i = t * per; i < (t + 1) * per; ++i) s += p[i];
    atomicAdd(out, s);
}

static int g_fd = -1;
static char* g_base = nullptr;
static unsigned long long* g_out = nullptr;
static int g_bad = 0;

static size_t resident(const char* p, size_t sz) {
    std::vector<unsigned char> v(sz / 4096);
    if (mincore((void*)p, sz, v.data()) != 0) return 0;
    size_t n = 0;
    for (unsigned char c : v) n += c & 1;
    return n;
}

// The sum the kernel must find, read with pread (not through the mapping)
static unsigned long long cpu_sum(const char* p, size_t sz) {
    std::vector<unsigned char> buf(1 << 20);
    unsigned long long s = 0;
    for (size_t o = 0; o < sz; o += buf.size()) {
        const ssize_t r = pread(g_fd, buf.data(), buf.size(), (off_t)(p - g_base + o));
        if (r != (ssize_t)buf.size()) { printf("  pread failed\n"); exit(2); }
        for (unsigned char c : buf) s += c;
    }
    return s;
}

// Out of the page tables and the page cache; returns the pages still resident
static size_t evict(const char* p, size_t sz) {
    (void)madvise((void*)p, sz, MADV_PAGEOUT);
    (void)posix_fadvise(g_fd, p - g_base, (off_t)sz, POSIX_FADV_DONTNEED);
    return resident(p, sz);
}

static double launch(const char* p, size_t sz, unsigned long long want, const char* what) {
    printf("  %s: launching ...", what);
    fflush(stdout);
    *g_out = 0;
    const double t0 = now();
    hipLaunchKernelGGL(sum_k, dim3(64), dim3(256), 0, 0, (const unsigned char*)p, sz, g_out);
    const hipError_t e = hipDeviceSynchronize();
    const double ms = (now() - t0) * 1e3;
    const bool ok = e == hipSuccess && *g_out == want;
    if (!ok) ++g_bad;
    printf(" %.2f ms (%.0f MB/s), %s%s\n", ms, (double)(sz >> 20) / ms * 1e3,
           e != hipSuccess ? hipGetErrorString(e) : *g_out == want ? "sum ok" : "SUM WRONG", ok ? "" : "  <-- FAILED");
    fflush(stdout);
    return ms;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file> [offset] [--force]\n", argv[0]); return 2; }
    bool force = false;
    size_t off = (size_t)64 << 20;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--force")) force = true;
        else off = strtoull(argv[i], nullptr, 0);
    }
    off = off / (2u << 20) * (2u << 20);
    g_fd = open(argv[1], O_RDONLY);
    struct stat st;
    if (g_fd < 0 || fstat(g_fd, &st) != 0) { perror(argv[1]); return 2; }
    const size_t MB = (size_t)1 << 20, need = off + 1800 * MB;
    if ((size_t)st.st_size < need) { fprintf(stderr, "%s: need %zu MB from offset %zu\n", argv[1], need >> 20, off); return 2; }
    g_base = (char*)mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, g_fd, 0);
    if (g_base == MAP_FAILED) { perror("mmap"); return 2; }

    int noretry = -1;
    if (FILE* f = fopen("/sys/module/amdgpu/parameters/noretry", "r")) {
        if (fscanf(f, "%d", &noretry) != 1) noretry = -1;
        fclose(f);
    }
    printf("amdgpu noretry = %d%s\n", noretry, noretry == 0 ? " (GPU faults are retried)" : "");
    if (hipInit(0) != hipSuccess || hipHostMalloc((void**)&g_out, 4096, 0) != hipSuccess) {
        fprintf(stderr, "no GPU\n");
        return 2;
    }
    volatile char sink = 0;
    auto touch = [&](const char* p, size_t sz) { for (size_t o = 0; o < sz; o += 4096) sink += p[o]; };
    char* p = g_base + off;

    printf("1  registered, 16 MB\n");
    {
        const size_t sz = 16 * MB;
        const unsigned long long want = cpu_sum(p, sz);
        touch(p, sz);
        if (hipHostRegister(p, sz, hipHostRegisterReadOnly) != hipSuccess) { printf("  hipHostRegister failed\n"); return 2; }
        launch(p, sz, want, "registered");
        (void)hipHostUnregister(p);
        // (unregistering leaves the range with the driver: the next steps use other ranges)
    }
    if (noretry != 0 && !force) {
        printf("stopping: noretry is not 0, a GPU fault would be fatal (--force to try anyway)\n");
        return g_bad ? 1 : 0;
    }
    p += 64 * MB;

    printf("2  unregistered, in memory, 16 MB\n");
    {
        const size_t sz = 16 * MB;
        const unsigned long long want = cpu_sum(p, sz);
        touch(p, sz);
        launch(p, sz, want, "first");
        launch(p, sz, want, "again");
    }
    p += 64 * MB;

    printf("3  unregistered, not in memory, 16 MB\n");
    const size_t sz3 = 16 * MB;
    const unsigned long long want3 = cpu_sum(p, sz3);
    {
        printf("  evicted: %zu of %zu pages still in memory\n", evict(p, sz3), sz3 / 4096);
        launch(p, sz3, want3, "from the file");
        printf("  after it: %zu of %zu pages in memory\n", resident(p, sz3), sz3 / 4096);
        launch(p, sz3, want3, "again");
    }

    printf("4  the same range evicted after the GPU read it\n");
    {
        const size_t left = evict(p, sz3);
        usleep(200000);
        const size_t l1 = resident(p, sz3);
        sleep(2);
        const size_t l2 = resident(p, sz3);
        printf("  pages in memory: %zu at once, %zu after 0.2 s, %zu after 2.2 s (of %zu): %s\n", left, l1, l2, sz3 / 4096,
               l2 * 2 < sz3 / 4096 ? "they stay out - demand paging" : "the driver holds / restores them - as registered");
        launch(p, sz3, want3, "read again");
    }
    p += 64 * MB;

    printf("5  64 ranges of 4.25 MB, 8 MB apart, not in memory\n");
    {
        const size_t usz = 4456448, n = 64;  // an expert of one tensor: 1088 pages
        std::vector<unsigned long long> want(n);
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t i = 0; i < n; ++i) {
                if (pass == 0) (void)cpu_sum(p + i * 8 * MB, 5 * MB);  // (as in the other steps: read once before)
                (void)evict(p + i * 8 * MB, usz);
            }
            double t0 = now();
            if (pass == 1)
                for (size_t i = 0; i < n; ++i) (void)madvise(p + i * 8 * MB, usz, MADV_WILLNEED);
            printf("  %s\n", pass == 0 ? "as they are:" : "MADV_WILLNEED on all first:");
            fflush(stdout);
            int bad = 0;
            for (size_t i = 0; i < n; ++i) {
                *g_out = 0;
                // 4456448 = 16384 * 272: the kernel's threads divide it evenly
                hipLaunchKernelGGL(sum_k, dim3(64), dim3(256), 0, 0, (const unsigned char*)(p + i * 8 * MB), usz, g_out);
                if (hipDeviceSynchronize() != hipSuccess) ++bad;
                if (pass == 0) want[i] = *g_out;
                else if (*g_out != want[i]) ++bad;
            }
            const double ms = (now() - t0) * 1e3;
            g_bad += bad;
            printf("    %.1f ms, %.2f ms a range (%.0f MB/s)%s\n", ms, ms / n, (double)(n * usz >> 20) / ms * 1e3,
                   bad ? "  <-- FAILED (errors or sums that differ between the passes)" : pass ? ", sums equal the first pass's" : "");
        }
    }
    p += 600 * MB;

    printf("6  1 GB not in memory, one launch\n");
    {
        const size_t sz = 1024 * MB;
        const unsigned long long want = cpu_sum(p, sz);
        printf("  evicted: %zu of %zu pages still in memory\n", evict(p, sz), sz / 4096);
        launch(p, sz, want, "from the file");
        launch(p, sz, want, "again");
    }
    printf(g_bad ? "FAILED: %d step(s)\n" : "all steps ran (%d failed)\n", g_bad);
    return g_bad ? 1 : 0;
}
