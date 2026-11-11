// expert_pager.h -- the experts of a mixture-of-experts model that does not fit in memory, registered with the GPU
// on demand (inference.cpp pin_model_memory: INFERENCE_EXPERT_PIN_MB).
//
// A registered range of the GGUF mapping has to be resident: the driver keeps it so (a page the kernel evicts from
// it is read back at once). So not more than fits in memory may be registered at a time, and a model whose experts
// are most of a file larger than that (GLM-5.3-Flash: 163 of 171 GB) registers only the experts its tokens are
// routed to: a unit is one expert of one expert tensor (its whole pages; the pages it shares with its neighbours
// stay registered with the rest of the file), registered at its first use, and, once more than the budget is
// registered, the least recently used ones are released: their pages go back to being plain page cache, which the
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
    // Unregister everything (before the mapping goes)
    void release_all();

    size_t total = 0, registered = 0;                  // bytes of all units / of those registered
    size_t budget;
    int n_registered = 0;
    long long st_calls = 0, st_registered = 0, st_released = 0;  // use() calls with work, units registered / released
    double st_ms = 0.0, st_release_ms = 0.0;                      // ... and the time in them / of it releasing

  private:
    struct Unit { char* p; size_t len; int prev, next; uint32_t stamp; bool reg; };
    void unlink(int u);
    void push_front(int u);
    bool release(int u);

    gguf_model_t* gguf;
    std::vector<Unit> units;
    int head = -1, tail = -1;                          // registered units, most to least recently used
    uint32_t clock = 0;
    bool warned = false;
    std::mutex mtx;
};

#endif
