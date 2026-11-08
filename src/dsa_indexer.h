// dsa_indexer.h -- host launchers for dsa_indexer.hip (GLM-5.3-Flash DSA indexer on the GPU).
//
// The DSA indexer selects top-k positions from a pooled key index for sparse attention.
// The flow: normalize K â project to indexer space â compute scores â pool by k=4 â top-k selection.
// All pointers must be device-accessible (pinned host memory on UMA).

#ifndef DSA_INDEXER_H
#define DSA_INDEXER_H

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Load dsa_indexer.co. Returns 0 on success.
int dsa_indexer_init(const char* co_path);
void dsa_indexer_free(void);

// Indexer key normalization (RMSNorm): x_out = (x - mean(x)) / sqrt(var(x) + eps) * w.
// x [T][n_embd], k_norm [indexer_head_size], k_norm_b [indexer_head_size].
// x_out [T][indexer_head_size]. grid (indexer_head_size / 256, T), block 256.
int dsa_indexer_k_norm(const float* x, float* x_out, const float* k_norm, const float* k_norm_b,
                       int n_embd, int indexer_head_size, int T, float eps, hipStream_t stream);

// Indexer projection: x_out = x Ã indexer_proj. x [T][indexer_head_size], indexer_proj [n_embd Ã indexer_n_head Ã indexer_head_size].
// x_out [T][n_embd Ã indexer_n_head]. Uses int8 matmul for efficiency (q8_device.h quantization).
int dsa_indexer_proj(const float* x, float* x_out, const float* indexer_proj, int n_embd, int indexer_n_head,
                     int indexer_head_size, int T, hipStream_t stream);

// Indexer attention key: x_out = x Ã indexer_attn_k. x [T][n_embd Ã indexer_n_head], indexer_attn_k [n_embd Ã indexer_head_size].
// x_out [T][indexer_n_head Ã indexer_head_size].
int dsa_indexer_attn_k(const float* x, float* x_out, const float* indexer_attn_k, int n_embd, int indexer_n_head,
                       int indexer_head_size, int T, hipStream_t stream);

// Indexer attention query (low-rank): x_out = x Ã indexer_attn_q_b. x [T][q_lora_rank], indexer_attn_q_b [q_lora_rank Ã indexer_n_head Ã indexer_head_size].
// x_out [T][indexer_n_head Ã indexer_head_size].
int dsa_indexer_attn_q(const float* x, float* x_out, const float* indexer_attn_q_b, int q_lora_rank,
                       int indexer_n_head, int indexer_head_size, int T, hipStream_t stream);

// Score computation: scores [T][n_ctx_pooled] = q Ã k^T. q [T][indexer_n_head Ã indexer_head_size],
// k [n_ctx][indexer_n_head Ã indexer_head_size]. n_ctx_pooled = n_ctx / index_kpool.
// grid (n_ctx_pooled / 256, T), block 256.
int dsa_indexer_score(const float* q, const float* k, float* scores, int n_ctx, int n_ctx_pooled,
                      int indexer_n_head, int indexer_head_size, int T, hipStream_t stream);

// Key pooling: reduce scores by index_kpool=4. pooled [T][n_ctx_pooled], scores [T][n_ctx].
// pooled[t][i] = sum_j scores[t][i * index_kpool + j] for j=0..index_kpool-1.
// grid (n_ctx_pooled / 256, T), block 256.
int dsa_indexer_key_pool(const float* scores, float* pooled, int n_ctx, int n_ctx_pooled, int index_kpool,
                         int T, hipStream_t stream);

// Top-k selection over pooled scores. For each token t, select the top-k positions from the pooled index.
// scores [T][n_pooled], ids [T][k] (selected position indices), sel_w [T][k] (selection weights).
// Uses iterative max-finding with mask array. grid T, block 256.
int dsa_indexer_top_k(const float* scores, int* ids, float* sel_w, int n_pooled, int k, int T,
                      hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // DSA_INDEXER_H
