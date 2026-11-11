// glm5next_verify -- GLM-5.3-Flash (glm5-next) against the llama.cpp reference.
//
//   glm5next_verify <model.gguf> [ref_dir]
//
// The model is normally tools/gen_glm5next_ckpt.py's tiny checkpoint converted to FAST (rocm-convert); ref_dir a
// `ref_dump acts` directory of that file decoded to F32 (rocm-convert model.gguf model-f32.gguf f32) under
// upstream llama.cpp's glm5-next (it has no ROCmFP4 types): the same weight values on both sides. PLAN.md,
// GLM-5.3-Flash.
//
// Now: the loader. Each block's kind from the file (KDA where attention.head_count_kv is 0, else MLA with the DSA
// indexer), the indexer flags, the MLA cache slots, the FFN kinds (dense in the leading_dense_block_count layers,
// experts + the shared expert after), the hparams, inference_model_info's cache size; and that the forward pass
// refuses cleanly (not written yet).

#include "inference.h"
#include "inference_internal.h"
#include "gguf_mmap.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [ref_dir]\n", argv[0]);
        return 2;
    }
    inference_model_info_t info;
    check(inference_model_info(argv[1], &info) == 0 && info.kind == INFERENCE_MODEL_CHAT,
          "inference_model_info: a text model");
    printf("    arch %s, %.1f MB, cache %llu bytes a position%s%s\n", info.arch, info.bytes / 1e6,
           (unsigned long long)info.kv_bytes_per_token, info.why[0] ? ", " : "", info.why);

    inference_model_t* m = inference_load_model(argv[1]);
    check(m != nullptr, "inference_load_model");
    if (!m) {
        printf("    %s\n", inference_get_last_error());
        return 1;
    }
    const inference_model& e = *m;
    check(e.glm5 && e.n_hc == 0, "glm5-next: its own mHC (n_hc 0: not qwen4exp's)");
    printf("    %d layers + %d nextn, n_embd %d, heads %d, kda head %d, q_lora %d, kv_lora %d, k/v %d/%d, "
           "indexer %d x %d top %d kpool %d tail %d, mHC %d streams %d iterations, dense lead %d\n",
           e.n_layer, e.n_nextn, e.n_embd, e.n_head, e.kda_head_dim, e.q_lora, e.kv_lora, e.k_mla, e.v_mla,
           e.ix_heads, e.ix_dim, e.ix_topk, e.ix_kpool, e.ix_select_tail, e.hc_mult, e.hc_sinkhorn_iters,
           e.n_dense_lead);

    // the file's own description of each block: head_count_kv[i] (0: KDA) and the dense / expert FFN tensors
    std::vector<int> n_kv;
    {
        int32_t type = -1;
        const void* data = nullptr;
        size_t n = 0;
        if (gguf_kv_get_array(e.gguf, "glm5-next.attention.head_count_kv", &type, &data, &n) == 0)
            for (size_t i = 0; i < n; ++i) n_kv.push_back(type == 5 ? ((const int32_t*)data)[i] : ((const uint32_t*)data)[i]);
    }
    int kda = 0, mla = 0, slots = 0, dense = 0, moe = 0;
    bool kinds_ok = (int)n_kv.size() >= e.n_layer, ffn_ok = true, slot_ok = true;
    for (int i = 0; i < e.n_layer; ++i) {
        const LayerWeights& L = e.layers[i];
        const char* k = L.kind == LAYER_KDA ? "KDA" : L.kind == LAYER_MLA_DSA ? "MLA" : "?";
        printf("    blk.%d %s%s%s, %s FFN, clamp %g / %g\n", i, k, L.kind == LAYER_MLA_DSA && L.ix_full ? " + indexer" : "",
               L.kind == LAYER_MLA_DSA ? (" slot " + std::to_string(L.kv_slot)).c_str() : "", L.moe ? "MoE" : "dense",
               L.swiglu_limit, L.swiglu_limit_sh);
        if (i < (int)n_kv.size()) kinds_ok = kinds_ok && (L.kind == (n_kv[i] == 0 ? LAYER_KDA : LAYER_MLA_DSA));
        if (L.kind == LAYER_KDA) ++kda;
        if (L.kind == LAYER_MLA_DSA) slot_ok = slot_ok && L.kv_slot == slots++, ++mla;
        ffn_ok = ffn_ok && L.moe == (i >= e.n_dense_lead) && (L.moe ? L.We_gate.data && L.exp_probs_b && L.W_gate.data
                                                                    : L.W_gate.data && L.W_up.data && L.W_down.data);
        dense += !L.moe;
        moe += L.moe;
        ffn_ok = ffn_ok && L.hcs_attn.fn.data && L.hcs_ffn.fn.data && L.hcs_attn.scale && L.hcs_ffn.base;
    }
    check(kinds_ok, "block kinds as head_count_kv says (0: KDA, else MLA)");
    check(slot_ok && e.n_mla_layers == mla, "MLA cache slots 0 .. n - 1 in layer order");
    check(ffn_ok, "FFN: dense in the leading layers, experts + shared expert after; both mHC mixers bound");
    printf("    %d KDA + %d MLA layers, %d dense + %d MoE FFNs\n", kda, mla, dense, moe);
    // the cache: kv_lora F16 a position per MLA layer + the indexer's 3 x ix_dim
    int full = 0;
    for (int i = 0; i < e.n_layer + e.n_nextn && i < (int)n_kv.size(); ++i) full += n_kv[i] != 0;
    check(info.kv_bytes_per_token == (unsigned long long)full * 2 * (e.kv_lora + 3 * e.ix_dim),
          "inference_model_info: the latent + indexer cache per position (MLA layers incl. the MTP block)");

    // the forward pass is not written yet: kv_cache_init must refuse with that, not crash
    const int rc = inference_kv_cache_init(m, 256);
    check(rc != 0 && strstr(inference_get_last_error(), "not implemented"), "kv_cache_init refuses: forward pass not written");
    printf("    %s\n", inference_get_last_error());
    inference_free_model(m);
    printf(g_fail ? "FAIL: %d checks\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
