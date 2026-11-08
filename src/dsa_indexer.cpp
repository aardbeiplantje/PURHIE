// dsa_indexer.cpp -- host launchers for dsa_indexer.hip (GLM-5.3-Flash DSA indexer).

#include "inference_log.h"
#include "kernel_loader.h"
#include "dsa_indexer.h"

#include <stdio.h>

#ifndef DSA_MAX_K
#define DSA_MAX_K 64
#endif

static hipModule_t g_mod = nullptr;
static hipFunction_t g_k_norm = nullptr, g_matmul = nullptr, g_score = nullptr;
static hipFunction_t g_key_pool = nullptr, g_top_k = nullptr, g_query_proj = nullptr;

int dsa_indexer_init(const char* co_path) {
    if (g_mod) return 0;
    if (kernel_module_load(&g_mod, co_path) != hipSuccess) {
        INFER_LOG("[dsa_indexer] cannot load %s\n", co_path);
        g_mod = nullptr;
        return -1;
    }
    if (hipModuleGetFunction(&g_k_norm, g_mod, "dsa_k_norm") != hipSuccess ||
        hipModuleGetFunction(&g_matmul, g_mod, "dsa_matmul") != hipSuccess ||
        hipModuleGetFunction(&g_score, g_mod, "dsa_score") != hipSuccess ||
        hipModuleGetFunction(&g_key_pool, g_mod, "dsa_key_pool") != hipSuccess ||
        hipModuleGetFunction(&g_top_k, g_mod, "dsa_top_k_simple") != hipSuccess ||
        hipModuleGetFunction(&g_query_proj, g_mod, "dsa_query_proj") != hipSuccess) {
        INFER_LOG("[dsa_indexer] kernels missing in %s\n", co_path);
        dsa_indexer_free();
        return -1;
    }
    return 0;
}

void dsa_indexer_free(void) {
    if (g_mod) (void)hipModuleUnload(g_mod);
    g_mod = nullptr;
    g_k_norm = g_matmul = g_score = nullptr;
    g_key_pool = g_top_k = g_query_proj = nullptr;
}

int dsa_indexer_k_norm(const float* x, float* x_out, const float* k_norm, const float* k_norm_b,
                       int n_embd, int indexer_head_size, int T, float eps, hipStream_t stream) {
    if (!g_k_norm) return -1;
    const int n = indexer_head_size;
    const int blocks = (n + 255) / 256;
    void* args[] = {(void*)&x, (void*)&x_out, (void*)&k_norm, (void*)&k_norm_b, (void*)&n, (void*)&eps, (void*)&T};
    return hipModuleLaunchKernel(g_k_norm, blocks, (unsigned)T, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int dsa_indexer_proj(const float* x, float* x_out, const float* indexer_proj, int n_embd, int indexer_n_head,
                     int indexer_head_size, int T, hipStream_t stream) {
    if (!g_matmul) return -1;
    const int K = indexer_head_size;
    const int N = n_embd;
    const int blocks = (N + 255) / 256;
    void* args[] = {(void*)&x, (void*)&indexer_proj, (void*)&x_out, (void*)&K, (void*)&N, (void*)&T};
    return hipModuleLaunchKernel(g_matmul, blocks, (unsigned)T, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int dsa_indexer_attn_k(const float* x, float* x_out, const float* indexer_attn_k, int n_embd, int indexer_n_head,
                       int indexer_head_size, int T, hipStream_t stream) {
    if (!g_matmul) return -1;
    const int K = n_embd;
    const int N = indexer_n_head * indexer_head_size;
    const int blocks = (N + 256) / 256;
    void* args[] = {(void*)&x, (void*)&indexer_attn_k, (void*)&x_out, (void*)&K, (void*)&N, (void*)&T};
    return hipModuleLaunchKernel(g_matmul, blocks, (unsigned)T, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int dsa_indexer_attn_q(const float* x, float* x_out, const float* indexer_attn_q_b, int q_lora_rank,
                       int indexer_n_head, int indexer_head_size, int T, hipStream_t stream) {
    if (!g_query_proj) return -1;
    const int K = q_lora_rank;
    const int N = indexer_n_head * indexer_head_size;
    const int blocks = (N + 256) / 256;
    void* args[] = {(void*)&x, (void*)&indexer_attn_q_b, (void*)&x_out, (void*)&K, (void*)&N, (void*)&T};
    return hipModuleLaunchKernel(g_query_proj, blocks, (unsigned)T, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int dsa_indexer_score(const float* q, const float* k, float* scores, int n_ctx, int n_ctx_pooled,
                      int indexer_n_head, int indexer_head_size, int T, hipStream_t stream) {
    if (!g_score) return -1;
    const int D = indexer_n_head * indexer_head_size;
    const int blocks = (n_ctx_pooled + 255) / 256;
    void* args[] = {(void*)&q, (void*)&k, (void*)&scores, (void*)&n_ctx, (void*)&D, (void*)&T};
    return hipModuleLaunchKernel(g_score, blocks, (unsigned)T, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int dsa_indexer_key_pool(const float* scores, float* pooled, int n_ctx, int n_ctx_pooled, int kpool, int T,
                         hipStream_t stream) {
    if (!g_key_pool) return -1;
    const int blocks = (n_ctx_pooled + 255) / 256;
    void* args[] = {(void*)&scores, (void*)&pooled, (void*)&n_ctx, (void*)&n_ctx_pooled, (void*)&kpool, (void*)&T};
    return hipModuleLaunchKernel(g_key_pool, blocks, (unsigned)T, 1, 256, 1, 1, 0, stream, args, nullptr);
}

int dsa_indexer_top_k(const float* scores, int* ids, float* sel_w, int n_pooled, int k, int T,
                      hipStream_t stream) {
    if (!g_top_k) return -1;
    if (k <= 0 || k > DSA_MAX_K || n_pooled <= 0 || n_pooled > 512 || T <= 0) return -1;
    void* args[] = {(void*)&scores, (void*)&ids, (void*)&sel_w, (void*)&n_pooled, (void*)&k, (void*)&T};
    return hipModuleLaunchKernel(g_top_k, (unsigned)T, 1, 1, 256, 1, 1, 0, stream, args, nullptr);
}
