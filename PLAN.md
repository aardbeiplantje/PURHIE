# Project Plan: rocm-hip (Strix Halo, gfx1151)

> What is open: next steps, planned work, leads. How the engine is built and what it runs: [ARCHITECTURE.md](ARCHITECTURE.md).
> Finished work, with its measurements, is in the git history (this file was pruned of it on 2026-11-05 and 2026-11-06).

## ▶ Next (week of 2026-11-09; user, 2026-11-05: "we will work on it next week")

1. **Qwen3.8-Flash-Next from BF16** (section below): the own MTP block as draft head and the parallel PLE reads are
   in; both to be measured in the deployment (the user).
2. **Concurrent runs** in rocm-inference-server (section below).
3. **Server logging** (section below): implemented and tested (`t/62-server-log.t`); the Flash soak run is open.
4. **GLM-5.3-Flash (`glm5_next`) as an architecture** (section below): the gate is memory capacity, not kernels -
   95% of its 320B params are expert weights, and none of FP8 / FP4 / Q4_0 fits 128 GB, so it reads experts from disk:
   they are registered on demand, least recently used released (section below, done 2026-11-11 on the 8-block
   slice; the whole file not run yet) - measure that first. Start with the tiny GGUF generator, not the kernels. A first attempt
   (4fe7dae, another agent; reverted in d4a45fa) skipped the gate and the oracle: tensor names unlike `glm-dsa`'s, a
   tiny GGUF its own loader refused, and a DSA path that reweighted dense attention output instead of selecting keys.

5. **Independent of llama.cpp: a quantizer and a converter here** (user, 2026-11-09: "i want to be independent of
   llama.cpp"). One command, `bin/rocm-convert`, takes a checkpoint or a GGUF. Quantizing (`inference_quantize`; with `-r plain` byte for byte
   llama-quantize's on FAST / ROCmFP4 / Q8_0, `tests/quantize_verify.sh`; default recipe `strix`: Q8_0 lookup
   tables, LM head, indexer and mHC, measured in ARCHITECTURE.md s4). Open on it: a better quality measure than
   perplexity for small per-tensor choices (K / V moved perplexity by +-2% without following precision): the KL
   divergence of the logits against the Q8_0 / BF16 model; and the recipe on Flash in the deployment (its Q8_0 LM
   head and hc mixers are untested on the real model). The converter is in too (`bin/rocm-convert`; byte for byte
   convert_hf_to_gguf.py's for every architecture the engine runs - qwen2/3/3moe/35/35moe, qwen4exp, llama,
   nemotron_h_moe, gemma4, gemma-embedding - and glm5-next / deepseek4, `tests/convert_verify.sh`): `rocm-convert
   /data/hf/GLM-5.3-Flash GLM-5.3-Flash-FAST.gguf` reads the FP8 release and writes FAST, no intermediate file
   (done 2026-11-10: 171 GB, in /models). Not converted yet: a SentencePiece Llama 2, Gemma 4's dense / per-layer
   input variants (31B, E2B), NVFP4 / modelopt checkpoints (/data/hf/Qwen3.6-35B-A3B-NVFP4-Fast).
   DeepSeek-V4-Flash (the user's choice, 2026-11-09) converts too (`DeepSeek4`, MXFP4 experts kept: 152 GB FAST);
   its MTP layer is left out, as upstream does (a separate draft file: to do with the MTP work). DeepSeek-V4.1-Flash
   (`DeepseekV41ForCausalLM`, on /hf, 510 GB: Engram n-gram tables of 197 GB FP8, 289 GB of FP4 experts) is not
   converted: a new package (upstream `conversion/deepseek41.py`: Engram tables, token map and numpy-PCG64 hash
   multipliers, 32x32 FP8 blocks, shared KV / indexer layers) and ~450 GB free disk (/data has 180 GB); its experts
   alone exceed 128 GB. Open: the model-card `general.*` keys (license, base model, size label), more
   architectures as needed (each a package in `ROCm::Inference::Convert` checked against convert_hf_to_gguf.py),
   DeepSeek-V4 in the engine (`deepseek4`:
   KV compressors, hash routing, grouped output projection; like GLM-5.3 it does not fit 128 GB resident).

## Status in brief

The engine runs qwen2, qwen3, qwen3moe, qwen35 / qwen35moe, qwen4exp (Qwen3.8-Flash-Next), gemma4, gemma-embedding,
nemotron_h_moe and llama from zero-copy GGUF mappings on the UMA. Surfaces: the C API (`libinference.so`), the Perl
XS module ROCm::Inference (Minijinja templates), rocm-inference-server (OpenAI-compatible, `--models-dir`). Regression
suite: `make regress` / `make regress-full`. Deployed by the user's llama.cpp-rocm-gfx1151-docker (branch
rocm-inference), which builds rocm-hip from GitHub `main`. Design: ARCHITECTURE.md; measurements: git history.

## Qwen3.8-Flash-Next from BF16

Quantized: /models/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST-PLE_Q8_0.gguf (PLE table Q8_0 on disk). Open:
- **The model's own MTP block as its draft head: implemented, to be measured on Flash.** `set_draft_model` with the
  model's own file takes blk.48 from the model's mapping (pinned with it; a staged model pins the block's pages on
  their own), with the model's embeddings and LM head and the block's final mixer nextn.hc_head_*; the server
  drafts with it by default (`--parallel 1`, no `--draft`; `--draft none` turns it off). Attention there is dense
  (compress ratio 4: exact below top_k 2048 x 4 positions; above, only which drafts are accepted can differ). On
  tiny models (`tests/mtp_own_verify.sh`, regress row mtp_own_qwen4exp_tiny) it drafts exactly what the same block
  as a sidecar drafts. Open: acceptance and tok/s on Flash (`build/mtp_verify F F 128` with F the model; the sidecar
  measured ~38 -> ~50 tok/s). One difference to watch: the base file's nextn.hc_head_norm is the sidecar's
  output_hc_norm + 1.0 (all its other tensors are byte-identical); every other norm in both files has the
  converter's +1 (zero-centred gammas), so the base file is likely right and the sidecar off - if the own block's
  acceptance is worse than the sidecar's, that is the first suspect.
- **PLE rows in parallel: implemented, to be measured in the deployment.** It saw 400-480 tok/s on 30k-token
  opencode prompts (2026-11-09), the GPU mostly idle: cold rows of the 50.7 GiB table read one at a time.
  Measured on the real table (a scratch benchmark, only uncached pages): 995 us a cold row through the file's
  MADV_HUGEPAGE mapping, 232 us in 4 KB folios with readahead (what the engine had), 77 us with MADV_RANDOM, 6.5 us
  on 16 threads, 4.6 us with MADV_WILLNEED on the batch. Now: MADV_RANDOM on the CPU-only tensors, and ple_rows
  decodes a batch's rows on 32 threads (PLE_THREADS; WILLNEED cost 7% on a cached table, threads gain 5%). On a
  tiny model with a 3 GB table on the disk (gen_qwen4exp_tiny.py TINY_PLE_HEAD_ROWS=5500000), 64 wikitext chunks
  cold: 2,140 -> 6,890 tok/s, the same perplexity. In the deployment: a 512-token chunk's 8192 rows ~0.05 s instead
  of ~1.9 s cold.
- **Memory budget on 124 GB** (2026-11-09). Resident: ~65 GB (all the GPU reads, experts included, pinned); the
  rest of the 122.8 GB file (PLE table 35.8 GiB, token embeddings) is read by the CPU through the page cache, and
  slows prefill to a crawl when that cache is squeezed. What squeezed it: a 19 GB block store in /dev/shm (tmpfs,
  not evictable without swap), two 262k-token KV caches (`--parallel 2`), and the user's downloads (dirty pages
  waiting for a slow disk). The server now warns at a load when the models, their KV caches and a store in memory
  leave too little page cache for the CPU-read parts (`memory_check`), and a store on a disk works (the KV cache
  stays in /dev/shm; blocks are copied in, 2970612). Not the cause: split-invariant prefill
  (`QMATVEC_PREFILL_SPLIT_VARIANT=1`, set in the docker repo's run.sh on 2026-11-09, can go again).
- The fork's qwen4exp is behind upstream's (recurrent_layers as indices vs upstream's bool flags; llama-server
  rejects these files): worth porting in the fork (the user's).

## GLM-5.3-Flash (`glm5_next`): what has to change, and in what order (started 2026-11-10; planned 2026-11-06)

**Where it stands (2026-11-10).** The release is converted (`/models/GLM-5.3-Flash-FAST.gguf`, 171 GB FAST; its
`indexer_compressor_ape` / `_gate` came out 4-bit: the strix rule missed the underscore names, fixed since, tiny
tensors, not worth the 19 h to redo). The oracle: upstream llama.cpp has `src/models/glm5-next.cpp` (KDA, the
k-pool DSA indexer, NoPE MLA over the latent, DeepSeek-V4-style mHC with Sinkhorn, sigmoid MoE with the SwiGLU
clamp); its CPU build at /data/conv/upstream-build, `make ref_dump_stock REF_STOCK=/data/conv/upstream
REF_STOCK_B=/data/conv/upstream-build`. The test model is `tools/gen_glm5next_ckpt.py`'s tiny checkpoint
(8 layers: 6 KDA + 2 DSA, 3 dense + 5 MoE, MTP) as FAST, the engine's input, and that file decoded to F32
(`rocm-convert tiny-fast.gguf tiny-f32.gguf f32`), the oracle's: the same weight values on both sides, so the
comparison is of the forward pass alone (`ref_dump_stock acts`: 935 named activations, hc_* / kda_* / indexer_* /
ffn_moe_* per layer). User, 2026-11-10: the reference is for numbers only, the engine keeps its UMA way (the
GGUF mapped and pinned in place, GPU-first with the CPU alongside), and FP4-sized weights (ROCmFP4 / FAST) are the
target. Work order: (a) the loader - done 2026-11-11: `glm5-next` in `archs`, `load_glm5next` /
`load_layer_glm5next` bind every tensor into LayerWeights (kinds LAYER_KDA / LAYER_MLA_DSA by head_count_kv, the
mHC `HcSinkhorn` hcs_attn / hcs_ffn, MLA `W_q_a` .. `W_v_b`, the indexer `W_ix_*` with `ix_full` per layer, the
SwiGLU clamps), shape-checked against the hparams; `inference_model_info` counts the latent + indexer rows per
position for the cache; `inference_kv_cache_init` (and `qwen_model_create`) refuse with "the forward pass is not
implemented yet" (`tests/glm5next_verify.sh`, quick tier: the tiny model as FAST, every block's kind, slots, FFNs,
the refusal). The MTP block (blk.N) is not bound yet; (b) the CPU forward pass - done 2026-11-11:
`src/glm5_ref.cpp` (Glm5Ref: mHC with Sinkhorn, KDA with the per-channel decay, MLA over the latent with the
k-pool DSA indexer, sigmoid MoE + shared expert with the SwiGLU clamps; weights decoded row by row in place) equals
upstream's glm5-next on the tiny model to cosine 1.000000 in every layer, teacher-forced at all 50 positions, and
end to end (logits, argmax); `glm5next_verify` checks it whenever the oracle is built. Indexer ties: equal pool
scores across the top-k boundary (all heads' ReLU 0 gives exact zeros) are picked by the lower pool index here;
llama.cpp's top_k is std::partial_sort, unordered, so it has no rule to match - the GPU kernels follow ours, and
decode and prefill must pick alike; (c) the GPU forward pass - first version done 2026-11-11: `forward_glm5`
(qwen_model.cpp) with the kernels in `src/glm5_gpu.hip` (mHC pre / post / mean, KDA conv / L2 / delta rule with
the per-channel decay and the gated norm, SwiGLU with the clamp, the indexer's LayerNorm / pooling / scores /
top-k select (ties to the lower pool index), sparse MLA over the latents, head transposes) and the engine's own
matvecs, sigmoid router and MoE combine; GLM's caches and KDA state in its own pinned region (GlmState),
`inference_kv_cache_init` gives it a stand-in one-layer KV cache. Checked (`glm5next_verify.sh`): on the float
path decode = Glm5Ref at every step (cos 1.000000, all argmax); with QMATVEC_NO_WMMA=1 prefill = token by token
bit for bit. On the int8 path against the CPU it drifts on the tiny model's random weights (logits cos ~0.82 by the end:
near-tie indexer and router picks flip under the activation noise) - not so on real weights (2026-11-11): the
release's first 8 blocks as a model (`tools/gguf_head.pl /models/GLM-5.3-Flash-FAST.gguf
/data/conv/glm53-head8.gguf 8`: 3 dense KDA, 5 MoE, MLA + indexer at blk.3 and blk.7; 21.5 GB, kept there) give,
on 200 tokens of English text, streams cos against the CPU of mean 0.99989 / worst 0.9948 after blk.7, logits
worst 0.9970, the same argmax at 196 of 200 steps (other architectures' whole-model bar here: 0.99); on the float
path GPU = CPU at every step (cos 1.000000) and with QMATVEC_NO_WMMA=1 prefill = token by token bit for bit, at
the release's widths (`tests/glm5next_head_verify.sh`, full tier row glm5next_head8). The tokenizer: `glm4`
(llama.cpp's CHATGLM4 + ignore_merges) is llama3's regex and whole-word lookup; ids equal upstream's on 28 prompts
(tests/data/glm5_tokenizer.tsv, checked in glm5next_verify.sh) - before this the model loaded without a
tokenizer. Speed (2026-11-11), measured on `GLM_BENCH=4,16` (tools/gen_glm5next_ckpt.py:
the release's widths, 4 layers - KDA dense, 2 KDA + MoE, MLA + MoE - 16 experts; `glm5next_verify --bench`):
decode 8.47 -> 7.5 ms/token (7.9 at 8k context), prefill 2,145 -> 3,700 tok/s at 512 tokens and 379 -> 2,320 at
8,192. What did it: k_b / v_b as one expert-indexed matvec a layer (the heads as experts, int8, no transposes);
sparse MLA as split online-softmax partials (fixed 256-position splits, so a batch sums as one token does; 4 heads
a block, a wave a position, coalesced rows) + a combine; the KDA state column in registers across a run of tokens;
the mHC RMS folded into glm_hc_pre (fn on the raw streams, times r); the indexer's score a wave a pool with the
queries in LDS; select with block scans; and above all fully unrolled register arrays (runtime-bounded loops over
them had put them in scratch: 3x on prefill). Decode is now the engine's weight matvecs (~6.4 of 6.95 ms of kernel
time; GLM's own kernels ~0.4 ms for the 4 layers). Projected to the release (45 layers, all weights resident),
~45-50 ms/token before any expert paging. Open: prefill attention on WMMA (sparse MLA is ~40% of a long prefill:
scalar FP32 over ~2k positions x 64 heads x 512 a token), a chunked KDA for prefill, an F16 latent cache (half the
bytes), the MTP block, batched sequences, the stand-in KV cache's sequence features (store, checkpoints) for GLM's
state; the whole file end to end - text out through the chat template, never run yet (ask the user before loading
it on the shared host); (d) the routing trace of item 1 on the real file, once the deployment can spare the memory.

**Experts registered on demand (2026-11-11, user: "go with option 1"; `src/expert_pager.h`).** What registration
is on this stack (kernel 6.17 OEM, ROCm 7, measured with strace and mincore): `hipHostRegister` makes the range a
KFD SVM range, which the driver keeps resident (pages evicted from it are read back within 0.2 s, launch or not) -
so what is registered has to fit in memory, and registering all 171 GB would thrash. `hipHostUnregister` issues no
ioctl and does not let go of the range; mapping the file over it again (`gguf_remap`: MAP_FIXED) does, and the VMAs
merge back. So: the experts of glm5-next's MoE layers (one unit an expert of one tensor, 4.25 MB, its whole pages)
are left out of the load's registration when the file is more than 3/4 of the memory (or `INFERENCE_EXPERT_PIN_MB=N`:
at most N MB of them; 0: never); `glm_ffn` drains the queue after the router and hands the routed ids to the pager,
which registers the new ones (MADV_WILLNEED on all first) and releases the least recently used over the budget.
On the slice (18.3 GB of experts): results bit for bit the same with a 2 GB budget (`glm5next_head_verify.sh`'s
third run); with room for all of them decode costs nothing extra (11.9 ms/token either way: the drain per MoE
layer is free, and it stops once everything is registered); a miss costs ~0.57 ms a unit with the pages in the
page cache (0.28 registering, 0.29 unregistering + remapping), ~1.7 ms an expert - the bench's random tokens at a
2 GB budget: 25 ms/token. Open: (1) the whole file - the budget rule (MemAvailable less the rest of the model and
16 GB) is untested, as is a miss from the disk and whether the kernel really drops released pages under pressure
(shown only in a standalone test); (2) the miss rate of real text: the routing trace, which the pager's counters
now give (`INFERENCE_VERBOSE`: registered / released at the free); (3) a cheaper miss (the remap drops the page
tables: registering again faults 1088 pages in; try 2 MB-aligned units or MADV_POPULATE_READ on a CPU thread
while the GPU runs the attention); (4) the MTP block's experts are registered with the rest (3.6 GB); (5) a
second context on the same model forwarding at the same time could have its experts released under it (one
forward at a time is assumed everywhere; the pager would need use counts); (6) the same for deepseek4; (7) no registration at all: with `amdgpu.noretry=0` the GPU's page faults may be
retried and the driver page the file in on demand, the kernel's LRU instead of the pager's - never tried (noretry
is 1 here: a fault is fatal). `tools/gpu_demand_paging.cpp` tests it in six steps (build line in its header; it
stops after the registered step while noretry is 1): to run after the user reboots with noretry=0, deployment down.

Config read from huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json on 2026-11-06; the numbers below are
computed from it, not copied from its model card. 320B total / 18B active, 45 decoder layers, hidden 4096, vocab
154,880, 1M context. 34 linear-attention layers (KDA) + 11 sparse-attention layers (MLA with a top-2048 index over
a k=4-pooled compressed index) at layers 3, 7, 11 ... 43; 42 MoE layers (the first 3 layers are dense FFN), 288
routed experts + 1 shared, top-8, sigmoid routing with `mlp.gate.e_score_correction_bias` and
`routed_scaling_factor` 2.5; `mhc` with `hc_mult` 4 and `hc_sinkhorn_iters` 20; `swiglu_limit` 10.0;
`num_nextn_predict_layers` 1. Two readings are inference, not config: `kda_layers` as Kimi Delta Attention
(gated-deltanet family) and `index_kpool_compress` as key pooling by 4 with APE to restore the lost position
information. Both set buffer sizes (`conv_dim = (2*n_k + n_v) * 128`), so check them against the checkpoint's
tensor shapes before sizing anything.

1. **Gate: capacity, not kernels.** 42 layers x 289 experts x 3 x 4096 x 2048 = **305.5B of the 320B params
   (95%) are expert weights**; the non-expert part (attention, embed, LM head, norms) is 14.5B. As released (FP8
   e4m3) that is 320 GB; at ROCmFP4_FAST 4.25 bpw it is 170 GB; Q4_0 is 180 GB. None fit 128 GB, and ~2.6 bpw is
   the first quantization that does. Pinning decides the speed: the non-experts cost 7.7 GB pinned at FP4, leaving
   120 GB -> 8,996 of the 12,096 experts resident (**74%**), and the other 26% of expert reads are NVMe traffic at
   3-5 GB/s, ~50x slower than the 220-256 GB/s UMA. Per token, 8 experts x 42 layers = 8.46B params are touched:
   all of it off NVMe at 3-5 GB/s is 0.7-1.1 tok/s, 26% cold is 2.6-4.3 tok/s (the NVMe reads only, no compute), and the all-resident
   bandwidth ceiling is 26-30 tok/s at FP8 (8.46 GB/token) and 49-57 at FP4 (4.5 GB/token) at 220-256 GB/s. The earlier
   12-14 / 23-27 were estimates with no stated efficiency; measure them before quoting. So "can we run it" is a routing-trace question, not a kernel question:
   route ~10k tokens through the router alone (small enough to run on CPU from a quantized copy), count the
   distinct experts hit per layer, and settle the target bpw + the pinned fraction **before** writing a kernel.
   The cold part is read from disk through the mapping, registered on demand (above): the routing trace says how
   much of it a token touches.
2. **The converter and the loader, no kernels.** The converter is done (`rocm-convert`, the GGUF names of upstream
   llama.cpp's `glm5-next`: `ssm_*` / `attn_q_a` / `attn_k_b` / `indexer.*` / `hc_*` / `blk.45.nextn.*`; see ▶ Next
   item 5); the GGUF's tensor table is what the loader binds. The loader dispatches on `general.architecture` against a
   hardcoded list (`archs` in `inference_model_info`, `src/inference.cpp`), so an unknown arch fails with "architecture %s is not supported"
   before anything runs. Tensor name -> slot map from the checkpoint's own `modules_to_not_convert` (a literal
   tensor-name list): `self_attn.{q,k,v,qkv_proj,o,f_a,f_b,g_a,g_b,b}_proj*`, `{q,k,v}_conv1d`, `A_log`, `dt_bias`,
   `o_norm`, `fused_qkvbfg_a_proj`, `indexer.{wq_b,wk,k_norm,weights_proj,index_kpool_compress_ape,
   index_kpool_compress_gate}`, `hyper_connection`, `mapping_proj`, `mlp.gate`, `mlp.gate.e_score_correction_bias`.
   Keys to carry through the loader: `index_topk` (2048), `index_kpool` (4), `index_kpool_compress`,
   `index_share_for_mtp_iteration`, `hc_mult` (4), `hc_sinkhorn_iters` (20), `hc_eps`, `kv_lora_rank` (512),
   `q_lora_rank` (1536), `qk_head_dim` (256), `qk_rope_head_dim` (0), `mla_use_nope`, `swiglu_limit` (10.0),
   `first_k_dense_replace` (3), `num_nextn_predict_layers` (1). De-risk first: extend
   `tools/gen_qwen4exp_tiny.py` into a `gen_glm5next_tiny.py` that writes a 4-layer tiny GGUF (3 dense + 1 DSA,
   8 experts, tiny hidden) from these names, so phases 1-2 below can run on a CPU host. Hermes has ROCm 10.1 access (user, 2026-11-06), as the work host
   does (`/opt/rocm`, hipcc, the gfx1151 GPU); phases 3-4 need the GPU, so they run on the work host.
3. **CPU reference path against the oracle.** The vendored fork already has `llama.cpp/src/models/glm-dsa.cpp`
   implementing MLA + the DSA indexer (`ATTENTION_KEY_LENGTH_MLA`, `ATTENTION_VALUE_LENGTH_MLA`,
   `INDEXER_HEAD_COUNT`, `INDEXER_KEY_LENGTH`, `INDEXER_TOP_K`, `wq_b / wv_b / wo / indexer_k_norm /
   indexer_proj`), so `tools/ref_oracle/ref_dump` is a working oracle for the sparse path from day one. Check
   whether that file also has mHC and the KDA per-channel decay; if not, the reference for those does not exist
   and has to be written from HF's modeling file. CPU ops in the `qwen_ops.cpp` style the forward pass and
   `tests/forward_verify.cpp` already share: the KDA step (per-channel decay), indexer score -> kpool compress ->
   top-k -> gather, the mHC Sinkhorn mixer, the MoE router **with sigmoid scoring + `e_score_correction_bias`**
   (this router does softmax - a real difference, not a detail: sigmoid + bias, then top-8, then renormalize, x2.5)
   and the `swiglu_limit` clamp at 10.0.
   **The #1 correctness risk is here, not in the kernels:** the indexer's top-k is a *selection*, not a summation.
   Exactness by construction (batched kernels reproducing the single-token summation order bit for bit under
   `QMATVEC_NO_WMMA=1`) does not cover selection order - a prefill of N tokens must select exactly the entries N
   single-token steps would have, or the sparse attention reads different positions and the logits diverge
   silently. Pin the tie-break down (ties to the lower index, as `moe_router` already does) and test it as a
   property, not as a perplexity number.
4. **GPU kernels.** KDA's decay is per-**channel** diagonal, not this engine's per-head scalar gate: `g_out` grows
   from `[T][n_v]` to `[T][n_v*128]` and the state update becomes `S *= exp(g_c)` per channel, in
   `deltanet_pre`/`deltanet_rule` (`DELTANET_HEAD_SIZE` is already 128 and KDA's head_dim is 128, so the state
   shape carries over unchanged). The indexer's top-2048 selection over the k=4-pooled index is the same problem
   shape as the bitonic top-k in `moe_router` (32 index heads x 128 in, an index out) - reuse that machinery.
   Prefill stays dense FP16-WMMA flash attention (with the whole prefix there is nothing to select); decode gathers
   the selected 2048 positions, replacing the split-K-over-256-position-chunks decode for those 11 layers only.
5. **KV cache layout + quantization, both gates.** MLA stores one latent of `kv_lora_rank` 512 (+ a 64 rope slot;
   `qk_rope_head_dim` is 0, so NoPE) per position, not per-head K/V: `kv_bytes_per_token` is computed at load from
   the `attn_k` / `attn_v` tensor widths in `src/inference.cpp` and must become a latent-cache path or the KV budget
   is wrong by a large factor. It is a big win: 576 floats x 11 layers = **12 KB/position** in F16 against 704
   KB/position for a 64-head x 256-dim layout, and that is what makes 1M context addressable. The block store's
   SHA-hashed `<hash>.kvb` slots must also key on the **compressed index**, or a resumed sequence has no index for
   its reused prefix. On quantization: the release is FP8 with dynamic scaling and gfx1151 has no FP8/BF8/FP6/FP4
   instructions (ARCHITECTURE.md s4), so either an e4m3 -> F16 LUT decode feeding the F16 WMMA path (256-entry LUT
   = 512 B, fits LDS) or offline requantization to ROCmFP4_FAST; the `v_perm` codebook trick that the int8 path
   uses for 4-bit codes does not reach 8-bit codes (3-bit indices only), so there is no free version of the first.
   Decide before the kernels, because it sets the resident set and therefore the tok/s in item 1.

What breaks that the new-kernel list does not show: `index_share_for_mtp_iteration: true` means the draft head
shares the main model's index, so the verify path's `deltanet_mid_states` / `qwen_model_rollback(keep)` has to roll
the index back too or rebuild it per pass; the `n_hc == 0` dispatch (`src/inference_internal.h:95`) branches the
whole forward pass's residual handling, and mHC replaces *every* norm, so the decode profile's `hc_norm` 98 x 7.6 us
and `hc_mix` 97 x 2.9 us items stay and grow (a per-token 4x4 doubly-stochastic matrix, cheap per layer but on the
critical path of all 45); and layer kinds are classified from the tensors present, never by index, so the
classifier's predicate must learn KDA's `A_log`/`dt_bias`/`b_proj`/`f_a_proj` names or every KDA layer silently
falls through to the wrong kind and the model runs wrong-but-silently. Vision (`glm5_next_vision`: 24 layers,
hidden 1024, 16 heads, patch 14, 448 px, `spatial_merge_size` 2, `temporal_patch_size` 2, `projection_intermediate_size`
10240 -> `out_hidden_size` 4096) is an independent track that can go last: the engine has no encoder and the C API
is text-in/text-out, so it needs an encoder path, a patch + merger projection to `n_embd`, the image/video token
interleave (image_token_id 154854, image_start/end 154830/154831, video_start/end 154832/154833) and the Perl
template's thinking channel - the engine streams every emitted token, so `reasoning_effort` (low/high/max, default
max) and `clear_thinking` (default false) need the reply-kind channel flag currently listed as "only if a non-Perl
user needs it". A GLM tokenizer (vocab 154,880) is a GGUF-metadata problem for `tokenizer.cpp`, not a new tokenizer.

Ordering: the gate comes first because its answer decides whether the rest is worth doing on this hardware. The
phases after it are cheap, CPU-only and oracle-backed, and are where the indexer's selection-order risk resolves;
GPU time on KDA / indexer / sparse attention belongs after those. An optional cheaper first target: a smaller DSA
sibling, which the oracle already knows as `glm-dsa`, exercises KDA + indexer + MLA + MoE with none of the capacity
problem, then scale up to the full model.

## Concurrent runs (to investigate; user, 2026-11-05)

Found in the opencode config (opencode_fn.json, not in the repo): each agent is a slot (`X-LLamaCPP-Id-slot`: plan 6,
build 5, explore1 4, explore2 2, verifier 3), `max_concurrency` 2, and compaction summarizes with a separate request.
The server keeps ONE sequence, so every switch between slots re-prefills the conversation: that is the 12k-token
reuse in the logs (the shared system prompt and tools), and a 59k prefill at 263 tok/s (227 s).

Decision (user, 2026-11-07): KV persists, and one conversation's cache is not cleared by another's requests. Per-slot
sequences are dropped (the engine does not have them; tried for llama.cpp without success). Status, checked
2026-11-07:
- **Persistence: implemented, opt-in.** The block store (`--store DIR`) keeps the blocks and recurrent snapshots as
  files; a new process or a restart resumes from them. `tests/store_verify.sh` passes all four phases on
  Qwen3-0.6B (on /dev/shm). The server uses it only with `--store`; the Docker repo passes it only when ROCM_STORE
  is set (run.sh, llama.sh; empty by default).
- **Switching reads the prefix back** instead of prefilling it: a sequence resumes from any stored prefix.
- **Eviction: in.** The server trims the store to `--store-max` GB (default 16) after each request, least recently
  used first, and logs what each request mapped and published (`kv` and `evict` lines).
- **Measured (2026-11-08): a 27B switch.** Qwen3.8-27B, store on /dev/shm: A (12,628 tokens) cold in 35.8 s, B
  (5,892) takes the sequence, A resumes with 12,544 tokens from the store and 93 prefilled in 0.79 s; re-prefilled
  without the store 36.8 s (47x). 27B store files: 16 MB blocks (64 KB/token), 149.6 MB snapshots; A and B together
  3.9 GB, 2.8 GB of it snapshots. At 16 GB, one 130k-token 27B conversation (8 GB of blocks, ~40 snapshots = 6 GB)
  nearly fills the store: `--store-max` 32+ on the deployment (/dev/shm is RAM, shared with the models).
- **Not done:** a disk tier; the store on by default in run.sh (the deployment sets ROCM_STORE_DIR already).
- **Fixed (2026-11-08): a resumed agent lost its prefix.** opencode's log (505 requests, all `id_slot` 5) is one main
  agent, append-only, with compaction requests and a subagent between; after the subagent the main agent resumed a
  ~130k-token prefix from the store and found it evicted: blocks were touched only when mapped from the store, so
  a conversation going on in the sequence kept its first blocks at their write time, the oldest files, while each
  request writes a 62.8 MB snapshot (Qwen3.6-35B: 5 MB blocks; 458 requests ~28 GB > 16 GB). Now every lookup
  touches the prompt's blocks and its deepest snapshot (`t/45-store-lru.t`). And fewer snapshots (272f5e7): sparser
  with depth (8 per doubling past 64 blocks), and a sequence's superseded prompt-end snapshot is deleted. And (2026-11-08) lazy
  prompt ends, the server's default (`--store-snapshots lazy|eager`, `--store-snapshot-every N`): held in a state
  slot per sequence and written when the sequence leaves the chain, on reset / seq free / store close, and on the
  server's SIGTERM (the Docker repo's llama.sh now forwards docker stop's SIGTERM; before, nginx was PID 1 and the
  server was SIGKILLed). A conversation that goes on writes no prompt-end snapshot; a crash loses the held one. Open: the
  10 identical resends in the log (5 each before two compactions, at ~210k tokens) look like failed requests,
  whose replies the log does not have.
- **Fixed (2026-11-08): each new user message re-prefilled the previous turn.** Qwen3.6's template renders an
  assistant message's reasoning only after the last user message, so a new one changed the whole previous turn
  (requests5.log #5: 20,070 of the old prompt's tokens; the deployment's log: 20,424 prefilled). The server now sets
  `preserve_thinking` (`--no-preserve-thinking`; a request's `chat_template_kwargs` wins): 0 re-prefilled. opencode
  sends `reasoning_content` back (37,203 of 39,613 assistant messages in requests4.log).
Next: watch the resume on the deployment (the `kv` line's mapped blocks) and `--parallel 2` there.

The server serves one request at a time unless `--parallel N`, which batches N conversations of one model (not
with a draft head); with `--models-dir` each loaded model has its own sequences and the loop steps them in turn. A
request that arrives during another, or during a model load (2 s from the page cache, minutes from a cold disk),
waits: a test request waited 7 min 22 s, then ran at 38 tok/s. `--parallel 2` passes the server tests on
Qwen3.5-4B but has not run on the 27B / 35B deployments. To look at: `--parallel` by default, and with
speculative decoding (a draft per sequence, or verify batches); several models generating at once (one forward pass
each per round; GPU sharing); long prefills in chunks between other requests' tokens (prefill_chunk exists); loads
off the request path (between rounds, or in a second process); fairness and queue limits per user (with the
logging's user and conversation id); llama-server on the same GPU.

## Server logging (implemented; the Flash soak is open)

The engine's event hook (C), its Perl binding (`set_event_handler`), the formatter `ROCm::Inference::Log`, the server's
lines (start, done, prefill progress, generating every 5 s, loads, access, cancels) and the options `--log-level`,
`--log-format`, `--log-color`, `--debug-stream` are in, for sequential and `--parallel` requests alike (a chunked
stream prompt reports as one prefill). A client that leaves cancels its prefill also for a plain JSON request
(Web::Route's `client_gone`). Tests: `t/05-log.t`, `t/15-events.t`, `tests/events_test` (chunked streams too), and
`t/62-server-log.t` (Qwen3.5-4B: the lines in order, `--debug-stream`, cancels mid-prefill; both modes). Open:
- **Soak on the Flash model**: one request with a long prompt; the prefill ticks visible in real time, no per-token
  lines. Needs a window with the deployment stopped (see the draft-events item under Open leads).
- With `--parallel`, the done line's store counters (`kv` line) are the model's over the request, so another job's
  blocks can show in them.
Identity: the user (nginx X-Remote-User; a Keycloak JWT later), the client address (X-Real-IP), and the conversation
id (the X-LLamaCPP-Id-slot header of the opencode fixups). The docker repo's nginx needs `proxy_set_header
X-Remote-User` on /rocm/ (not done here).

## Later

- **Converter follow-ups**: an importance matrix (the weighted ROCmFP4 search; collected on the GPU by the
  engine), and a faster encoder (AVX-512 across 16 blocks keeps each block's summation order, so the bytes stay the
  reference's).
- **Qwen3.8-Flash-Next beyond 262,144 tokens (YaRN)** (HF card, 2026-11-06). The checkpoint is 262,144 natively
  (`max_position_embeddings`, `rope_type` default); the card reaches 1M by a YaRN override (`factor` 4.0, original
  262144, with an attention scale). Nothing is dropped by the converter: the GGUF has no rope-scaling keys, and the
  engine has no YaRN. To do: YaRN frequencies and the attention scale in the RoPE, a `--rope-factor` option, and a
  quality check at long context (not only that it runs). The 16 GB KV budget caps it at about 600k positions at F16
  even then. Keep in mind; not started.

## Open leads

Speed:
- Long-context prefill attention runs at ~25% of the FP16 WMMA peak (Qwen3.8-Flash 22% of a 32k prefill, more
  beyond; already ahead of llama.cpp's: 5.5 vs 7.7 s on Qwen3.6-35B at 32k): the largest lever for long prompts.
- Qwen3.8-27B (dense ROCmFP4_FAST) is compute-bound: matmul_wmma_q4_0_rocmfp4_fast ~26 of ~59 TOPS (70% of a 32k
  prefill); its epilogue (exact per-block scaling, ~19%) would need FP16 WMMA on pre-scaled weights (twice the
  LDS). Tabled.
- MTP verify passes: read experts shared by the verified tokens once (~3.8 of 10 shared by two tokens: ~19% of a
  2-token pass's expert bytes); draft cost 2.7 ms, mostly the head's LM head (a vocabulary cut hurts other
  scripts; a shortlist from the main model's top logits of the previous row, or an int8 copy).
- Decode leftovers (Flash): moe_router 48 x 14 us, hc_norm 98 x 7.6 us, deltanet_rule 36 x 19 us, hc_mix 97 x
  2.9 us, ~930 launches x ~1.7 us of gaps; each well under 5%.
- Prefill: deltanet_conv (~100 GB/s); the quantizing producers (swiglu, add_rmsnorm_rows) compute both Q8_0 and
  Q8_K though each consumer reads one; the CPU embedding row decode; attn_k / attn_v as two launches (~1%).
- Small F16 matrices (Qwen3.5-0.8B, 1,024-2,048 rows) latency-bound at 55-76 GB/s; MoE experts in F16 / BF16 run
  matmul_f_id (no model here has them); BF16 FFN down (K 17408) the slowest shape (23 TFLOPS).

Quality / correctness:
- Prefill is split-invariant now (ARCHITECTURE.md section 5). Open: the tests that allowed for the old WMMA
  differences between cuts (store_verify.sh and seq_verify compare first tokens, generate_verify a tolerance) can
  demand bit-exactness; the full tier's perplexity baselines of the models not run here (the Flash model, the 122B
  and 120B files) need `tests/regress.sh --update` on those rows (prefill attention rounds differently now).
- Draft events are not verified: `events_test MODEL DRAFT` is written but not run. The Flash model (122.8 GB file) shares
  the work host's 124 GB with the user's deployment: a load from /models paged in 48 GB in 11 min (~70 MB/s,
  2026-11-08) and was stopped so as not to evict the deployment's model. Flash work needs a window with the
  deployment stopped, or a generator for a tiny model with a draft head (tools/gen_qwen4exp_tiny.py writes none).
- K-quants dot Q8_K activations (one scale per 256), as ggml's CPU backend; ggml's GPU backend uses q8_1 (per 32):
  ~1% better perplexity at Q2_K. An option for the K-quant kernels.
- A quantized KV cache needs n_kv * D % 256 == 0 (Qwen2.5-0.5B, 2 x 64, cannot have one).
- Qwen3.6-35B Q4_0 forward_verify drifts to cos 0.989 at layer 21 (smooth, next token = llama.cpp's): its suite
  row uses min cos 0.985.
- One forward_verify run on the 35B once hit a GPU memory fault in matvec_q8_id_q5_K, never reproduced; likely
  the test's buffer overrun fixed on 2026-11-04 (5e6e9e6). Close after a few clean runs.

Tests and API:
- In `make regress` (2026-11-08): kv_verify, generate_verify, events_test, seq_verify (0.6B, 4B), store_verify.sh,
  and the Perl suites (Web::Route, Minijinja, ROCm::Inference with the server tests on Qwen3.5-4B). Still out:
  mtp_verify and `events_test MODEL DRAFT` (the Flash model and its draft head: a window with the deployment stopped).
- Untested: Mistral (llama architecture, SentencePiece tokenizer), Llama-3.1-8B (no files); Llama 3.1's built-in
  tools (<|python_tag|> brave_search.call(...)) not parsed; Nemotron's reply format (check against its template).
- The XS module lacks inference_prefill_logits, model_name and set_seed (add when a Perl user needs them); a C
  API channel flag for reply kinds only if a non-Perl user needs it.

## Tabled

- Older architectures: gpt-oss, deepseek2, glm4, phi3, gemma2 / gemma3, Mistral (user, 2026-11-04).
- The ROCmFP4_FAST WMMA kernel work (user, 2026-11-04: "move on, table this for now").
- A CPAN release of ROCm::Inference, Minijinja and Web::Route (user, 2026-11-05: the Docker image builds them
  from source).

## Tried and rejected (measured; do not repeat without a new idea)

- HIP graphs for decode launch gaps: ~0.3% (1.96 vs 1.88 us per dependent kernel; per-token parameter updates).
- Small-kernel merges (moe_combine into add_rmsnorm: slower, every workgroup recomputes the expert sum; others
  assessed not worth it, 2026-10-21).
- WMMA operand pipelining in the FAST kernel: 10% slower (173 VGPRs); bigger dense WMMA tiles, 2 tiles per wave,
  32 weight rows per wave: no gain.
- Decode attention on a Q8_0 cache: 16-byte K pieces, 4 V dims per thread, 128-position chunks: no gain.
- Flash prefill prefetch of the next K/V block at D 256: spills (13 VGPRs), -5%; kept for D <= 128 only.
- MTP vocabulary shortlist: +2-5% on English, -18% on Chinese (tokens spread over the vocabulary).
- moe_router: per-wave argmax top-k instead of the bitonic sort (15.3 vs 14.3 us); 8 waves per router row.
- Unrolling the FAST matvec quad loop by 2; fewer lanes per row for the Q8_0 matvec (32 is best).
- IQ WMMA staging prefetch; a K rotation per workgroup in the expert kernels.
- Loads inside branches or `c ? load : 0` in kernels: they serialize (keep loads unconditional, select after).
