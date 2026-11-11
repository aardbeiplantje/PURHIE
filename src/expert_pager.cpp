// expert_pager.cpp -- see expert_pager.h.

#include "expert_pager.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <sys/mman.h>
#include <unistd.h>

#include <hip/hip_runtime.h>

#include "inference_log.h"

int ExpertPager::add_tensor(const void* data, size_t expert_bytes, int n) {
    const uintptr_t page = (uintptr_t)sysconf(_SC_PAGESIZE);
    if (!data || n <= 0 || expert_bytes < 2 * page) return -1;
    const int first = (int)units.size();
    // what a registered page keeps in memory is its folio of the page cache: 2 MB as the file is mapped
    const char* env = getenv("GGUF_HUGE");
    const uintptr_t folio = env && env[0] == '0' ? page : (uintptr_t)2 << 20;
    for (int e = 0; e < n; ++e) {
        const uintptr_t b0 = (uintptr_t)data + (size_t)e * expert_bytes;
        const uintptr_t b = (b0 + page - 1) / page * page, end = (b0 + expert_bytes) / page * page;
        Unit u{(char*)b, (size_t)(end - b), 0, 0, {-1, -1}, -1, -1, 0, false, 0};
        folio_bytes = folio;
        // the page it shares with the expert before it (the one after it: that expert's)
        if (e > 0 && b != b0) {
            u.edge[0] = (int)edges.size();
            units.back().edge[1] = (int)edges.size();
            edges.push_back({(char*)(b - page), 0});
        }
        u.f0 = (b - (u.edge[0] >= 0 ? page : 0)) / folio;
        u.f1 = (end + (e + 1 < n && end != b0 + expert_bytes ? page : 0) + folio - 1) / folio;
        for (uintptr_t f = u.f0; f < u.f1; ++f)
            if (folios.emplace(f, 0).second) total += folio;
        units.push_back(u);
    }
    const uintptr_t t0 = ((uintptr_t)data + page - 1) / page * page, t1 = ((uintptr_t)data + (size_t)n * expert_bytes) / page * page;
    spans.push_back({(const char*)t0, (const char*)t1});
    span_bytes += (size_t)(t1 - t0);
    return first;
}

// An edge page is registered while an expert on either side of it is
bool ExpertPager::edge_ref(int i) {
    if (i < 0 || edges[i].refs++ > 0) return true;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    (void)mlock(edges[i].p, page);
    if (hipHostRegister(edges[i].p, page, hipHostRegisterReadOnly) == hipSuccess) return true;
    --edges[i].refs;
    return false;
}

void ExpertPager::edge_unref(int i) {
    if (i < 0 || --edges[i].refs > 0) return;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    (void)hipHostUnregister(edges[i].p);
    (void)gguf_remap(gguf, edges[i].p, page);
}

std::vector<std::pair<const char*, const char*>> ExpertPager::ranges() const {
    std::vector<std::pair<const char*, const char*>> r = spans;
    std::sort(r.begin(), r.end());
    return r;
}

// What is held: the folios with a registered page
void ExpertPager::hold(const Unit& u) {
    for (uintptr_t f = u.f0; f < u.f1; ++f)
        if (folios[f]++ == 0) registered += folio_bytes;
}

void ExpertPager::drop(const Unit& u) {
    for (uintptr_t f = u.f0; f < u.f1; ++f)
        if (--folios[f] == 0) registered -= folio_bytes;
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
    const auto t0 = std::chrono::steady_clock::now();
    (void)hipHostUnregister(u.p);
    const auto t1 = std::chrono::steady_clock::now();
    st_unreg_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    unlink(i);
    u.reg = false;
    drop(u);
    --n_registered;
    ++st_released;
    edge_unref(u.edge[0]);
    edge_unref(u.edge[1]);
    const bool ok = gguf_remap(gguf, u.p, u.len) == 0;
    st_remap_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    if (!ok) INFER_LOG("[expert_pager] cannot map %zu bytes of the model again: they stay resident\n", u.len);
    return ok;
}

bool ExpertPager::use(const int* first, int nt, int n_expert, const int* ids, int n) {
    std::lock_guard<std::mutex> lk(mtx);
    const auto t0 = std::chrono::steady_clock::now();
    if (++clock == 0) {  // wrapped: no stale stamp may equal a new one
        for (Unit& u : units) u.stamp = 0;
        clock = 1;
    }
    std::vector<int> add;
    if (inference_verbose() && nt > 0 && first[0] >= 0 && n > 0 && n <= 64) {
        Hist& h = hist[first[0]];
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0 || ids[i] >= n_expert) continue;
            ++pr_total;
            static const int back[4] = {1, 2, 4, 8};
            for (int k = 0; k < 4; ++k) {
                bool hit = false;
                for (int j = 0; j < back[k] && j < h.n && !hit; ++j) {
                    const std::vector<int>& v = h.ids[(h.n - 1 - j) % 8];
                    hit = std::find(v.begin(), v.end(), ids[i]) != v.end();
                }
                pr_hit[k] += hit;
            }
        }
        h.ids[h.n++ % 8].assign(ids, ids + n);
    }
    for (int t = 0; t < nt; ++t) {
        if (first[t] < 0) continue;
        for (int i = 0; i < n; ++i) {
            if (ids[i] < 0 || ids[i] >= n_expert) continue;
            const int ui = first[t] + ids[i];
            Unit& u = units[ui];
            if (u.stamp == clock) continue;
            u.stamp = clock;
            ++u.uses;
            if (u.reg) {
                unlink(ui);
                push_front(ui);
            } else {
                add.push_back(ui);
                hold(u);
            }
        }
    }
    if (add.empty()) return true;
    while (registered > budget && tail >= 0 && units[tail].stamp != clock) release(tail);
    st_release_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (registered > budget && !warned) {
        warned = true;
        INFER_LOG("[expert_pager] one layer's experts of a batch (%zu MB) are more than the budget (%zu MB): "
                  "registered beyond it\n", registered >> 20, budget >> 20);
    }
    // read ahead what is not in memory, all ranges at once, before the registrations fault them in one by one
    for (int ui : add) (void)madvise(units[ui].p, units[ui].len, MADV_WILLNEED);
    bool ok = true;
    for (int ui : add) {
        Unit& u = units[ui];
        // locked: the GPU's reads do not mark the pages as used, so under memory pressure the kernel would take
        // them for cold, evict them, and the driver read them back (the 96 GB run: 45 tokens in 20 minutes).
        // Mapping the file over the range again (release) unlocks it
        (void)mlock(u.p, u.len);
        const hipError_t e = hipHostRegister(u.p, u.len, hipHostRegisterReadOnly);
        if (e != hipSuccess) {
            INFER_LOG("[expert_pager] hipHostRegister(%zu bytes) failed: %s\n", u.len, hipGetErrorString(e));
            ok = false;
            break;
        }
        if (!edge_ref(u.edge[0]) || !edge_ref(u.edge[1])) {
            INFER_LOG("[expert_pager] hipHostRegister(a page between two experts) failed\n");
            ok = false;  // (the unit stays unlisted but registered: release_all does not see it, the unmap frees it)
            break;
        }
        u.reg = true;
        push_front(ui);
        ++n_registered;
        ++st_registered;
    }
    ++st_calls;
    st_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return ok;
}

// How the reads are spread over the experts, and how predictable they are
void ExpertPager::report() {
    std::vector<uint32_t> v;
    unsigned long long all = 0;
    for (const Unit& u : units) {
        v.push_back(u.uses);
        all += u.uses;
    }
    if (!all) return;
    std::sort(v.begin(), v.end(), [](uint32_t a, uint32_t b) { return a > b; });
    const size_t used = (size_t)(std::find(v.begin(), v.end(), 0u) - v.begin());
    INFER_LOG("[expert_pager] %llu reads of %zu of %zu units; the most read units covering a share of them:", all,
              used, v.size());
    const double share[] = {0.5, 0.75, 0.9, 0.95, 0.99};
    unsigned long long acc = 0;
    size_t i = 0;
    for (double s : share) {
        while (i < v.size() && (double)acc < s * (double)all) acc += v[i++];
        INFER_LOG(" %.0f%%: %zu (%.1f GB)", s * 100, i, (double)i * (double)units[0].len / 1073741824.0);
    }
    INFER_LOG("\n[expert_pager] release: %.0f ms unregistering, %.0f ms mapping again\n", st_unreg_ms, st_remap_ms);
    if (pr_total)
        INFER_LOG("[expert_pager] a token's experts that the same layer used for one of the last 1 / 2 / 4 / 8 tokens: "
                  "%.1f%% / %.1f%% / %.1f%% / %.1f%% (%lld reads)\n", 100.0 * pr_hit[0] / pr_total,
                  100.0 * pr_hit[1] / pr_total, 100.0 * pr_hit[2] / pr_total, 100.0 * pr_hit[3] / pr_total, pr_total);
}

void ExpertPager::release_all() {
    std::lock_guard<std::mutex> lk(mtx);
    report();
    INFER_LOG("[expert_pager] %lld of %zu experts' tensors registered (%lld released again) in %lld calls, %.0f ms (%.0f of "
              "them releasing); %zu of %zu MB registered at the end (budget %zu MB)\n", st_registered, units.size(),
              st_released, st_calls, st_ms, st_release_ms, registered >> 20, total >> 20, budget >> 20);
    for (Unit& u : units)
        if (u.reg) {
            (void)hipHostUnregister(u.p);
            u.reg = false;
            u.prev = u.next = -1;
        }
    for (Edge& e : edges) {
        if (e.refs > 0) (void)hipHostUnregister(e.p);
        e.refs = 0;
    }
    head = tail = -1;
    for (auto& f : folios) f.second = 0;
    registered = 0;
    n_registered = 0;
}
