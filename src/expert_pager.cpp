// expert_pager.cpp -- see expert_pager.h.

#include "expert_pager.h"

#include <algorithm>
#include <chrono>
#include <sys/mman.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include "inference_log.h"

int ExpertPager::add_tensor(const void* data, size_t expert_bytes, int n) {
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    if (!data || n <= 0 || expert_bytes < 2 * page) return -1;
    const int first = (int)units.size();
    for (int e = 0; e < n; ++e) {
        const uintptr_t b0 = (uintptr_t)data + (size_t)e * expert_bytes;
        const uintptr_t b = (b0 + page - 1) / page * page, end = (b0 + expert_bytes) / page * page;
        units.push_back({(char*)b, (size_t)(end - b), -1, -1, 0, false});
        total += (size_t)(end - b);
    }
    return first;
}

std::vector<std::pair<const char*, const char*>> ExpertPager::ranges() const {
    std::vector<std::pair<const char*, const char*>> r;
    r.reserve(units.size());
    for (const Unit& u : units) r.push_back({u.p, u.p + u.len});
    std::sort(r.begin(), r.end());
    return r;
}

void ExpertPager::unlink(int i) {
    Unit& u = units[i];
    if (u.prev >= 0) units[u.prev].next = u.next; else head = u.next;
    if (u.next >= 0) units[u.next].prev = u.prev; else tail = u.prev;
    u.prev = u.next = -1;
}

void ExpertPager::push_front(int i) {
    Unit& u = units[i];
    u.prev = -1;
    u.next = head;
    if (head >= 0) units[head].prev = i; else tail = i;
    head = i;
}

// hipHostUnregister alone leaves the range with the driver, which goes on keeping it resident: mapping the file
// over it again is what lets go of it (gguf_remap)
bool ExpertPager::release(int i) {
    Unit& u = units[i];
    (void)hipHostUnregister(u.p);
    unlink(i);
    u.reg = false;
    registered -= u.len;
    --n_registered;
    ++st_released;
    if (gguf_remap(gguf, u.p, u.len) != 0) {
        INFER_LOG("[expert_pager] cannot map %zu bytes of the model again: they stay resident\n", u.len);
        return false;
    }
    return true;
}

bool ExpertPager::use(const int* first, int nt, int n_expert, const int* ids, int n) {
    std::lock_guard<std::mutex> lk(mtx);
    const auto t0 = std::chrono::steady_clock::now();
    if (++clock == 0) {  // wrapped: no stale stamp may equal a new one
        for (Unit& u : units) u.stamp = 0;
        clock = 1;
    }
    std::vector<int> add;
    size_t need = 0;
    for (int t = 0; t < nt; ++t) {
        if (first[t] < 0) continue;
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0 || ids[i] >= n_expert) continue;
            const int ui = first[t] + ids[i];
            Unit& u = units[ui];
            if (u.stamp == clock) continue;
            u.stamp = clock;
            if (u.reg) {
                unlink(ui);
                push_front(ui);
            } else {
                add.push_back(ui);
                need += u.len;
            }
        }
    }
    if (add.empty()) return true;
    while (registered + need > budget && tail >= 0 && units[tail].stamp != clock) release(tail);
    st_release_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (registered + need > budget && !warned) {
        warned = true;
        INFER_LOG("[expert_pager] one layer's experts of a batch (%zu MB) are more than the budget (%zu MB): "
                  "registered beyond it\n", (registered + need) >> 20, budget >> 20);
    }
    // read ahead what is not in memory, all ranges at once, before the registrations fault them in one by one
    for (int ui : add) (void)madvise(units[ui].p, units[ui].len, MADV_WILLNEED);
    bool ok = true;
    for (int ui : add) {
        Unit& u = units[ui];
        const hipError_t e = hipHostRegister(u.p, u.len, hipHostRegisterReadOnly);
        if (e != hipSuccess) {
            INFER_LOG("[expert_pager] hipHostRegister(%zu bytes) failed: %s\n", u.len, hipGetErrorString(e));
            ok = false;
            break;
        }
        u.reg = true;
        push_front(ui);
        registered += u.len;
        ++n_registered;
        ++st_registered;
    }
    ++st_calls;
    st_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

void ExpertPager::release_all() {
    std::lock_guard<std::mutex> lk(mtx);
    INFER_LOG("[expert_pager] %lld of %zu experts' tensors registered (%lld released again) in %lld calls, %.0f ms (%.0f of "
              "them releasing); %zu of %zu MB registered at the end (budget %zu MB)\n", st_registered, units.size(),
              st_released, st_calls, st_ms, st_release_ms, registered >> 20, total >> 20, budget >> 20);
    for (Unit& u : units)
        if (u.reg) {
            (void)hipHostUnregister(u.p);
            u.reg = false;
            u.prev = u.next = -1;
        }
    head = tail = -1;
    registered = 0;
    n_registered = 0;
}
