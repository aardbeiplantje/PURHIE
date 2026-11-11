// expert_pager.h -- the experts of a mixture-of-experts model that does not fit in memory, registered with the GPU
// on demand (inference.cpp pin_model_memory: INFERENCE_EXPERT_PIN_MB).
//
// A registered range of the GGUF mapping has to be resident: the driver keeps it so (a page the kernel evicts from
// it is read back at once). So not more than fits in memory may be registered at a time, and a model whose experts
// are most of a file larger than that (GLM-5.3-Flash: 163 of 171 GB) registers only the experts its tokens are
// routed to: a unit is one expert of one expert tensor (its whole pages; a page it shares with a neighbour is
// registered while either is - not for good: the page cache holds the file in 2 MB folios and a registered page
// keeps its folio, so 36,288 such pages kept 79 GB of a 171 GB file in memory), registered at its first use, and,
// once more than the budget is held (the folios with a registered page, counted exactly), the least recently used ones are released: their pages go back to being plain page cache, which the
// kernel may drop and read again from the file. Nothing is copied either way: the GPU reads the file's pages.
//
// The forward pass calls use() with a layer's routed expert ids after the router has run and before the expert
// kernels are queued, with nothing queued that reads an expert (it synchronizes first): a released unit is not
// read by the GPU.

#ifndef EXPERT_PAGER_H
#define EXPERT_PAGER_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "gguf_mmap.h"

struct ExpertPager {
    ExpertPager(gguf_model_t* g, size_t budget_bytes) : gguf(g), budget(budget_bytes) {}

    // The n experts of expert_bytes each at data: their units (-1: nothing of them is a whole page)
    int add_tensor(const void* data, size_t expert_bytes, int n);
    // The ranges of all units, sorted (what the static registration of the file leaves out)
    std::vector<std::pair<const char*, const char*>> ranges() const;

    // Every unit registered and none ever to be released: use() has nothing to do
    bool all_registered() const { return n_registered == (int)units.size() && total <= budget; }
    // Register experts ids[0 .. n) (ids outside 0 .. n_expert are skipped) of the tensors whose first units are
    // first[0 .. nt), releasing the least recently used other units over the budget. False: a registration failed
    bool use(const int* first, int nt, int n_expert, const int* ids, int n);
    // The same without releasing anything (evict false, the default): experts are registered while the budget
    // has room and stay; on_gpu[i] says whether ids[i]'s tensors all are - the caller computes the others on the
    // CPU from the mapping (expert_cpu.h), whose pages the kernel caches and drops as any file's
    bool acquire(const int* first, int nt, int n_expert, const int* ids, int n, unsigned char* on_gpu);
    bool evict = false;                                // INFERENCE_EXPERT_EVICT=1: use(), all experts on the GPU
    long long st_cpu = 0;                              // experts (of tokens) computed on the CPU, and the time
    double st_cpu_ms = 0.0;
    // Unregister everything (before the mapping goes)
    void release_all();

    size_t total = 0, registered = 0;                  // what all units / those registered keep in memory, bytes
    size_t span_bytes = 0;                             // bytes of the tensors' whole pages (ranges())
    size_t budget;
    int n_registered = 0;
    long long st_calls = 0, st_registered = 0, st_released = 0;  // use() calls with work, units registered / released
    double st_ms = 0.0, st_release_ms = 0.0;                      // ... and the time in them / of it releasing

  private:
    // p, len: its whole pages; f0 .. f1: the folios (2 MB of the page cache) its pages and edges are in, which
    // it keeps in memory when registered; edge: the pages it shares with the experts before / after it (-1: none)
    struct Unit { char* p; size_t len; uintptr_t f0, f1; int edge[2]; int prev, next; uint32_t stamp; bool reg; uint32_t uses; };
    void hold(const Unit& u);
    void drop(const Unit& u);
    struct Edge { char* p; int refs; };
    bool edge_ref(int e);
    void edge_unref(int e);
    void unlink(int u);
    void push_front(int u);
    bool release(int u);

    gguf_model_t* gguf;
    std::vector<Unit> units;
    std::vector<Edge> edges;
    // Statistics for the policy (INFERENCE_VERBOSE): how well a layer's experts of the last tokens predict the
    // next token's (what a prefetch could read ahead), by the layer's first unit; one-token calls only
    struct Hist { std::vector<int> ids[8]; int n = 0; };
    std::unordered_map<int, Hist> hist;
    long long pr_total = 0, pr_hit[4] = {0, 0, 0, 0};  // ids seen / of them among the last 1, 2, 4, 8 tokens'
    double st_unreg_ms = 0.0, st_remap_ms = 0.0;
    void report();
    std::unordered_map<uintptr_t, int> folios;         // registered units per folio (by address / folio size)
    size_t folio_bytes = 0;
    std::vector<std::pair<const char*, const char*>> spans;  // the tensors' whole pages
    int head = -1, tail = -1;                          // registered units, most to least recently used
    uint32_t clock = 0;
    bool warned = false, full = false;
    std::mutex mtx;
};

#endif
