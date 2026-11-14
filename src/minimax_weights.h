/* minimax_weights.h -- the weights of one MiniMax-H3-Image checkpoint, from a GGUF (rocm-convert's output) or the original
 * safetensors (a directory or a file, LFS pointers resolved): a tensor by its checkpoint name, its bytes in place. */
#ifndef MINIMAX_WEIGHTS_H
#define MINIMAX_WEIGHTS_H

#include "gguf_mmap.h"
#include "safetensors_mmap.h"

#include <cstring>
#include <string>

struct WeightFile {
    gguf_model_t* g = nullptr;
    st_model_t* st = nullptr;
    WeightFile() = default;
    WeightFile(const WeightFile&) = delete;
    WeightFile& operator=(const WeightFile&) = delete;
    ~WeightFile() {
        if (g) gguf_mmap_close(g);
        if (st) st_close(st);
    }
    const void* tensor(const std::string& name, size_t* bytes) const {
        if (g) return gguf_get_tensor_ptr(g, name.c_str(), bytes);
        const st_tensor_t* t = st ? st_find(st, name.c_str()) : nullptr;
        if (!t) return nullptr;
        *bytes = t->bytes;
        return t->data;
    }
};

#endif /* MINIMAX_WEIGHTS_H */
