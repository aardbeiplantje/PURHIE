// glm5next_verify -- GLM-5.3-Flash (glm5-next) against the llama.cpp reference.
//
//   glm5next_verify <model.gguf> [ref_dir]
//   glm5next_verify --bench <model.gguf> [prompt tokens] [decode tokens]   (speed: prefill, then decode tok/s;
//                                                                           QWEN_PROFILE=1 adds the breakdown)
//
// The model is normally tools/gen_glm5next_ckpt.py's tiny checkpoint converted to FAST (rocm-convert); ref_dir a
// `ref_dump acts` directory of that file decoded to F32 (rocm-convert model.gguf model-f32.gguf f32) under
// upstream llama.cpp's glm5-next (it has no ROCmFP4 types): the same weight values on both sides. PLAN.md,
// GLM-5.3-Flash.
//
// The loader: each block's kind from the file (KDA where attention.head_count_kv is 0, else MLA with the DSA
// indexer), the indexer flags, the MLA cache slots, the FFN kinds (dense in the leading_dense_block_count layers,
// experts + the shared expert after), the hparams, inference_model_info's cache size; and that the GPU forward
// pass on it.
// With ref_dir, the CPU forward pass (src/glm5_ref.cpp, the GPU kernels' reference) against llama.cpp's:
//  - each layer alone from the reference's input streams at every position (teacher forcing, positions in order:
//    the KDA state, the latents and the indexer's pools build up as in the reference): the block input (mHC pre),
//    the token mixer's output (KDA / MLA over the indexer's picks), the streams after it, the FFN input and output
//    and the layer's output streams, cosine >= 0.999 at every position (the same weight values, float math; only
//    the summation order differs); the indexer's scores of every visible pool (max |difference| <= 1e-2 of the
//    largest: the reference keeps the indexer's keys, gates and pooled keys in its F16 cache, we in F32). At a position where equal scores straddle the top-k boundary the pick among them is a tie rule's,
//    and llama.cpp has none (std::partial_sort): such positions are counted, not compared after the indexer;
//  - the whole model run token by token from the embeddings: the last layer's streams and the last token's logits
//    (cosine >= 0.999, the same argmax).
// Then the GPU forward pass (qwen_model.cpp forward_glm5, glm5_gpu.hip), on the reference's tokens (or fixed ones):
//  - QWEN_MATVEC_FP32=1 (float matvec inputs): token by token against the CPU one (Glm5Ref), every step's logits
//    (cosine >= 0.9999, the same argmax): the kernels' arithmetic;
//  - else (int8 matvec inputs, the engine's path): with QMATVEC_NO_WMMA=1 a prefill of all tokens equals
//    token-by-token decoding bit for bit (the batched kernels: KDA over a run, the per-token selection, pools
//    completed within a batch); with the WMMA matmuls it is printed (they sum a batch in another order). Against
//    the CPU this path is printed, not checked: with the tiny model's random weights the int8 noise flips
//    near-ties of the indexer's pool scores and the routers, and a flipped pick changes that position's output.

#include "inference.h"
#include "inference_internal.h"
#include "gguf_mmap.h"
#include "glm5_ref.h"
#include "qwen_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char* what) {
    if (!ok) ++g_fail;
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
}

static std::string g_dir;

// A tensor of the reference's index.tsv by name (the last of that name), as floats; ne its shape
static bool ref(const std::string& name, std::vector<float>& out, std::vector<long long>* ne = nullptr) {
    std::ifstream f(g_dir + "/index.tsv");
    std::string line, file;
    long long n = 0;
    std::vector<long long> shape;
    std::getline(f, line);
    while (std::getline(f, line)) {
        std::vector<std::string> c;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) c.push_back(cell);
        if (c.size() < 8 || c[1] != name) continue;
        shape = {std::stoll(c[3]), std::stoll(c[4]), std::stoll(c[5]), std::stoll(c[6])};
        n = shape[0] * shape[1] * shape[2] * shape[3];
        file = c[7];
    }
    if (file.empty()) return false;
    std::ifstream d(g_dir + "/" + file, std::ios::binary);
    out.resize(n);
    d.read((char*)out.data(), n * 4);
    if (ne) *ne = shape;
    return (bool)d;
}

static double cosine(const float* a, const float* b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; ++i) ab += (double)a[i] * b[i], aa += (double)a[i] * a[i], bb += (double)b[i] * b[i];
    return aa > 0 && bb > 0 ? ab / sqrt(aa * bb) : (aa == bb ? 1.0 : 0.0);
}


struct StreamCapture {
    int SE = 0;
    std::vector<std::vector<float>> at;  // [layer * T + pos][SE]
    int T = 0;
};

static void capture(int layer, int pos, const float* x, int n, void* user) {
    StreamCapture* c = (StreamCapture*)user;
    if (n != c->SE || pos >= c->T) return;
    c->at[(size_t)layer * c->T + pos].assign(x, x + n);
}

// The GPU forward pass against the CPU one (see the top)
static void verify_gpu(inference_model_t* m, const std::vector<int>& toks) {
    const inference_model& e = *m;
    const int T = (int)toks.size(), SE = e.hc_mult * e.n_embd, V = e.vocab_size;
    const bool fp32 = getenv("QWEN_MATVEC_FP32") != nullptr;
    qwen_model_t* q = inference_get_qwen_model(m);
    check(q != nullptr, "GPU model");
    if (!q) return;
    // the CPU's streams after every layer and logits at every position
    Glm5Ref r(&e);
    std::vector<std::vector<float>> cpu_streams((size_t)e.n_layer * T), cpu_logits(T);
    for (int t = 0; t < T; ++t) {
        std::vector<float> s(SE);
        r.embed(toks[t], s.data());
        for (int l = 0; l < e.n_layer; ++l) {
            r.layer(l, t, s.data());
            cpu_streams[(size_t)l * T + t] = s;
        }
        cpu_logits[t].resize(V);
        r.logits(s.data(), cpu_logits[t].data());
    }
    auto argmax = [&](const float* v) { return (int)(std::max_element(v, v + V) - v); };
    std::vector<float> lg(V), pre_last;
    StreamCapture c, d;  // the streams after every layer: prefill, token by token
    if (!fp32) {  // a prefill of all tokens (the float path has no batched FAST expert kernel)
        c.SE = SE, c.T = T;
        c.at.resize((size_t)e.n_layer * T);
        qwen_model_set_layer_callback(q, capture, &c);
        const int rc = qwen_model_prefill(q, toks.data(), T, 0, lg.data());
        qwen_model_set_layer_callback(q, nullptr, nullptr);
        check(rc == 0, "GPU prefill");
        if (rc != 0) { printf("    %s\n", qwen_model_last_error()); return; }
        for (int l = 0; l < e.n_layer; ++l) {
            double worst = 1.0;
            int wt = -1;
            for (int t = 0; t < T; ++t) {
                const auto& g = c.at[(size_t)l * T + t];
                const double cs = g.size() == (size_t)SE ? cosine(g.data(), cpu_streams[(size_t)l * T + t].data(), SE) : 0.0;
                if (cs < worst) worst = cs, wt = t;
            }
            printf("    (int8 activations) GPU prefill blk.%d %s: streams worst cos vs CPU %.6f (position %d)\n", l,
                   e.layers[l].kind == LAYER_KDA ? "KDA" : "MLA", worst, wt);
        }
        pre_last = lg;
    }
    double worst = 1.0;
    int agree = 0;
    d.SE = SE, d.T = T;
    d.at.resize((size_t)e.n_layer * T);
    qwen_model_set_layer_callback(q, capture, &d);
    for (int t = 0; t < T; ++t) {  // token by token
        if (qwen_model_forward(q, toks[t], t, lg.data()) != 0) {
            qwen_model_set_layer_callback(q, nullptr, nullptr);
            check(false, "GPU decode");
            printf("    %s\n", qwen_model_last_error());
            return;
        }
        worst = std::min(worst, cosine(lg.data(), cpu_logits[t].data(), V));
        agree += argmax(lg.data()) == argmax(cpu_logits[t].data());
    }
    qwen_model_set_layer_callback(q, nullptr, nullptr);
    if (!fp32)  // the first layer and position where prefill and token-by-token streams part
        for (int l = 0; l < e.n_layer; ++l) {
            int first = -1;
            for (int t = 0; t < T && first < 0; ++t)
                if (c.at[(size_t)l * T + t] != d.at[(size_t)l * T + t]) first = t;
            if (first >= 0) {
                printf("    prefill vs token by token: first differ after blk.%d at position %d (cos %.9f)\n", l, first,
                       cosine(c.at[(size_t)l * T + first].data(), d.at[(size_t)l * T + first].data(), SE));
                break;
            }
        }
    printf("    GPU decode vs CPU%s: worst logits cos %.6f over %d steps, argmax agrees at %d\n",
           fp32 ? " (float path)" : " (int8 activations: near-tie picks of the indexer and routers flip)", worst, T,
           agree);
    if (fp32) {
        check(worst >= 0.9999 && agree == T, "GPU decode = CPU at every step (float path)");
    } else {
        size_t diff = 0;
        for (int i = 0; i < V; ++i) diff += pre_last[i] != lg[i];
        printf("    GPU prefill vs token by token: %zu of %d last logits differ (cos %.9f)\n", diff, V,
               cosine(pre_last.data(), lg.data(), V));
        if (getenv("QMATVEC_NO_WMMA"))  // (the WMMA matmuls of a batch sum in another order)
            check(diff == 0, "GPU prefill = token by token, bit for bit (QMATVEC_NO_WMMA=1)");
    }
}

// The CPU forward pass against the reference (see the top)
static void verify_ref(const inference_model& e) {
    std::vector<int> toks;
    {
        std::ifstream f(g_dir + "/tokens.txt");
        int t;
        while (f >> t) toks.push_back(t);
    }
    const int T = (int)toks.size(), E = e.n_embd, HC = e.hc_mult, SE = HC * E;
    printf("  reference %s: %d tokens\n", g_dir.c_str(), T);
    std::vector<float> init;
    check(T > 0 && ref("hc_init", init) && init.size() == (size_t)T * SE, "reference: tokens and hc_init");
    if (!T || init.size() != (size_t)T * SE) return;
    {  // the embeddings
        Glm5Ref r(&e);
        std::vector<float> s(SE);
        double worst = 1.0;
        for (int t = 0; t < T; ++t) r.embed(toks[t], s.data()), worst = std::min(worst, cosine(s.data(), &init[(size_t)t * SE], SE));
        printf("    embeddings: worst cos %.6f\n", worst);
        check(worst >= 0.9999, "embeddings = hc_init");
    }
    // each layer alone from the reference's input streams
    Glm5Ref r(&e);
    std::vector<float> in = init;
    bool all = true;
    for (int l = 0; l < e.n_layer; ++l) {
        const bool kda = e.layers[l].kind == LAYER_KDA;
        const std::string L = "-" + std::to_string(l);
        std::vector<float> a_in, mix, a_post, f_in, f_out, l_out;
        if (!ref("hc_attn_pre" + L, a_in) || !ref((kda ? "kda_out" : "attn_out") + L, mix) ||
            !ref("hc_attn_post" + L, a_post) || !ref("hc_ffn_pre" + L, f_in) || !ref("ffn_out" + L, f_out) ||
            !ref("l_out" + L, l_out)) {
            check(false, ("reference tensors of layer " + std::to_string(l)).c_str());
            all = false;
            continue;
        }
        std::vector<float> isc;
        std::vector<long long> ine;
        const bool have_sc = !kda && e.layers[l].ix_full && ref("indexer_score" + L, isc, &ine);
        double w[6] = {1, 1, 1, 1, 1, 1}, sc_err = 0;
        int nsel = 0, ties = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<float> s(in.begin() + (size_t)t * SE, in.begin() + (size_t)(t + 1) * SE);
            Glm5LayerTrace tr;
            r.layer(l, t, s.data(), &tr);
            if (have_sc && !tr.scores.empty()) {  // [n_pool (padded), T]: the visible pools first
                const float* row = &isc[(size_t)t * ine[0]];
                float big = 1e-30f, d = 0.0f;
                for (size_t j = 0; j < tr.scores.size(); ++j)
                    big = std::max(big, fabsf(row[j])), d = std::max(d, fabsf(row[j] - tr.scores[j]));
                sc_err = std::max(sc_err, (double)(d / big));
            }
            w[0] = std::min(w[0], cosine(tr.attn_in.data(), &a_in[(size_t)t * E], E));
            if (tr.tie) {  // the pick among equal scores is the tie rule's: nothing after the indexer compares
                ++ties;
                continue;
            }
            w[1] = std::min(w[1], cosine(tr.mixer_out.data(), &mix[(size_t)t * E], E));
            w[2] = std::min(w[2], cosine(tr.attn_post.data(), &a_post[(size_t)t * SE], SE));
            w[3] = std::min(w[3], cosine(tr.ffn_in.data(), &f_in[(size_t)t * E], E));
            w[4] = std::min(w[4], cosine(tr.ffn_out.data(), &f_out[(size_t)t * E], E));
            w[5] = std::min(w[5], cosine(s.data(), &l_out[(size_t)t * SE], SE));
            nsel = std::max(nsel, (int)tr.selected.size());
        }
        const double worst = *std::min_element(w, w + 6);
        char extra[160] = "";
        if (!kda)
            snprintf(extra, sizeof extra, ", up to %d positions attended, indexer scores off by %.1e, %d tied positions",
                     nsel, sc_err, ties);
        printf("    blk.%d %s: worst cos block in %.6f, mixer %.6f, after it %.6f, ffn in %.6f, ffn out %.6f, out %.6f%s\n",
               l, kda ? "KDA" : "MLA", w[0], w[1], w[2], w[3], w[4], w[5], extra);
        all = all && worst >= 0.999 && sc_err <= 1e-2 && (kda || !e.layers[l].ix_full || have_sc);
        in = l_out;
    }
    check(all, "every layer from the reference's input: cos >= 0.999 at every position");
    // the whole model, token by token
    Glm5Ref f(&e);
    std::vector<float> s(SE), logits(e.vocab_size), rl, last;
    for (int t = 0; t < T; ++t) {
        f.embed(toks[t], s.data());
        for (int l = 0; l < e.n_layer; ++l) f.layer(l, t, s.data());
    }
    f.logits(s.data(), logits.data());
    ref("l_out-" + std::to_string(e.n_layer - 1), last);
    const double cs = last.size() >= (size_t)T * SE ? cosine(s.data(), &last[(size_t)(T - 1) * SE], SE) : 0.0;
    const bool have = ref("result_output", rl) && rl.size() >= (size_t)e.vocab_size;
    const float* rlast = have ? &rl[rl.size() - e.vocab_size] : nullptr;
    const double cl = have ? cosine(logits.data(), rlast, e.vocab_size) : 0.0;
    const int am = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    const int rm = have ? (int)(std::max_element(rlast, rlast + e.vocab_size) - rlast) : -1;
    printf("    whole model: last streams cos %.6f, logits cos %.6f, argmax %d (reference %d)\n", cs, cl, am, rm);
    check(cs >= 0.999 && cl >= 0.999 && am == rm, "whole model from the embeddings: streams, logits, argmax");
}

// --bench: prefill a prompt of fixed tokens, then decode greedily, timed (tools/gen_glm5next_ckpt.py's GLM_BENCH model:
// the release's widths)
static int bench(const char* path, int np, int nd) {
    inference_model_t* m = inference_load_model(path);
    if (!m || inference_kv_cache_init(m, np + nd + 16) != 0) {
        printf("%s\n", inference_get_last_error());
        return 1;
    }
    qwen_model_t* q = inference_get_qwen_model(m);
    std::vector<int> toks;
    for (int i = 0; i < np; ++i) toks.push_back((int)((785u + 7919u * (unsigned)i) % (unsigned)m->vocab_size));
    std::vector<float> lg(m->vocab_size);
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    // a warm-up pass (kernel loads, first touches), then the timed ones
    if (qwen_model_prefill(q, toks.data(), np, 0, lg.data()) != 0) { printf("%s\n", qwen_model_last_error()); return 1; }
    qwen_model_profile_reset(q);
    double t0 = now();
    if (qwen_model_prefill(q, toks.data(), np, 0, lg.data()) != 0) { printf("%s\n", qwen_model_last_error()); return 1; }
    const double tp = now() - t0;
    qwen_model_profile_print(q, "prefill");
    qwen_model_profile_reset(q);
    int tok = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    t0 = now();
    for (int i = 0; i < nd; ++i) {
        if (qwen_model_forward(q, tok, np + i, lg.data()) != 0) { printf("%s\n", qwen_model_last_error()); return 1; }
        tok = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    }
    const double td = now() - t0;
    qwen_model_profile_print(q, "decode");
    printf("bench %s: %d layers, prefill %d tokens %.1f ms (%.1f tok/s), decode %d tokens %.2f ms/token (%.2f tok/s)\n",
           path, m->n_layer, np, tp * 1e3, np / tp, nd, td * 1e3 / nd, nd / td);
    inference_free_model(m);
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 3 && !strcmp(argv[1], "--bench"))
        return bench(argv[2], argc > 3 ? atoi(argv[3]) : 512, argc > 4 ? atoi(argv[4]) : 32);
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

    if (argc > 2) {
        g_dir = argv[2];
        verify_ref(e);
    }
    std::vector<int> toks;
    {
        std::ifstream f(g_dir.empty() ? std::string() : g_dir + "/tokens.txt");
        int t;
        while (f >> t) toks.push_back(t);
    }
    if (toks.empty())
        for (int i = 0; i < 50; ++i) toks.push_back((int)((785u + 7919u * (unsigned)i) % (unsigned)e.vocab_size));
    const int rc = inference_kv_cache_init(m, 256);
    check(rc == 0, "kv_cache_init");
    if (rc != 0) printf("    %s\n", inference_get_last_error());
    else verify_gpu(m, toks);
    inference_free_model(m);
    printf(g_fail ? "FAIL: %d checks\n" : "PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
