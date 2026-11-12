# Project Plan: rocm-hip (Strix Halo, gfx1151)

> What is open: next steps, planned work, leads. How the engine is built and what it runs: [ARCHITECTURE.md](ARCHITECTURE.md).
> Finished work, with its measurements, is in the git history (this file was pruned of it on 2026-11-05, 2026-11-06
> and 2026-11-12).

## ▶ Next (2026-11-12)

1. **Qwen3.8-Flash-Next in the deployment** (section below): the model's own MTP block as draft head and the parallel
   PLE reads are in; both are to be measured there (the user).
2. **Concurrent runs** (section below): the deployment runs `--parallel` now; watch the store's resume (the `kv`
   line's mapped blocks) and the queue there.
3. **Server logging** (section below): in; the Flash soak run is open.
4. **GLM-5.3-Flash (`glm5_next`)**: runs whole (4.3-7.6 tok/s); its speed work is tabled (user, 2026-11-12). The open
   leads are in its section below.
5. **Independent of llama.cpp** (user, 2026-11-09): `bin/rocm-convert` converts (byte for byte convert_hf_to_gguf.py's,
   `tests/convert_verify.sh`) and quantizes (`inference_quantize`; `-r plain` byte for byte llama-quantize's,
   `tests/quantize_verify.sh`; default recipe `strix`; the quality measure for a recipe is `perplexity --kl`, the
   KL divergence from the reference model, ARCHITECTURE.md s4). Open: the recipe choices measured by KL on a real
   model (Qwen3-30B-A3B's K / V, the Q8_0 LM head; only Qwen3-0.6B was run: strix 0.188 against plain 0.221); the
   `strix` recipe on Flash in the deployment (its Q8_0 LM head and hc mixers are untested on the real model); the
   model-card `general.*` keys (license, base model, size label). Not converted:
   a SentencePiece Llama 2, Gemma 4's dense / per-layer input variants (31B, E2B), NVFP4 / modelopt checkpoints
   (/data/hf/Qwen3.6-35B-A3B-NVFP4-Fast), DeepSeek-V4.1-Flash (`DeepseekV41ForCausalLM`, on /hf, 510 GB: Engram
   n-gram tables of 197 GB FP8, 289 GB of FP4 experts; a new package after upstream `conversion/deepseek41.py` and
   ~450 GB free disk - /data has 180 GB; its experts alone exceed 128 GB). DeepSeek-V4-Flash converts (152 GB FAST,
   MXFP4 experts kept; its MTP layer left out as upstream does: a separate draft file, with the MTP work) but the
   engine has no `deepseek4` (KV compressors, hash routing, grouped output projection; like GLM-5.3 it does not fit
   128 GB resident, so it would use the expert pager).

## Status in brief

The engine runs qwen2, qwen3, qwen3moe, qwen35 / qwen35moe, qwen4exp (Qwen3.8-Flash-Next), gemma4, gemma-embedding,
nemotron_h_moe, llama and glm5-next (GLM-5.3-Flash, its experts paged from disk) from zero-copy GGUF mappings on the
UMA. Surfaces: the C API (`libinference.so`), the Perl XS module ROCm::Inference (Minijinja templates),
rocm-inference-server (OpenAI-compatible: chat, raw completions, embeddings, `--models-dir`, `--parallel`), and
`bin/rocm-convert`. Regression suite: `make regress` / `make regress-full`. Deployed by the user's
llama.cpp-rocm-gfx1151-docker (branch rocm-inference), which builds rocm-hip from GitHub `main` (a rebuild needs a
new `CACHEBUST` to take new commits). Design: ARCHITECTURE.md; measurements: git history.

## Qwen3.8-Flash-Next

Quantized: /models/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_FAST-PLE_Q8_0.gguf (PLE table Q8_0 on disk). Open:
- **The own MTP block as draft head: to be measured on Flash.** `set_draft_model` with the model's own file takes
  blk.48 from the model's mapping; the server drafts with it by default (`--parallel 1`, no `--draft`; `--draft none`
  turns it off). On tiny models it drafts exactly what the same block as a sidecar drafts. Open: acceptance and tok/s
  on Flash (`build/mtp_verify F F 128` with F the model; the sidecar measured ~38 -> ~50 tok/s). If the own block's
  acceptance is worse than the sidecar's, the first suspect: the base file's nextn.hc_head_norm is the sidecar's
  output_hc_norm + 1.0 (every other norm in both files has the converter's +1, so the base file is likely right and
  the sidecar off).
- **PLE rows in parallel: to be measured in the deployment.** Cold rows of the 50.7 GiB table were read one at a time
  (400-480 tok/s on 30k-token opencode prompts, 2026-11-09); now MADV_RANDOM on the CPU-only tensors and a batch's
  rows decoded on 32 threads (PLE_THREADS): a 512-token chunk's 8192 rows ~0.05 s instead of ~1.9 s cold.
- **Memory on 124 GB**: ~65 GB resident; the PLE table and token embeddings are read by the CPU through the page
  cache, and prefill crawls when that cache is squeezed (a block store in /dev/shm, two 262k-token KV caches,
  downloads with dirty pages). The server warns at a load (`memory_check`); a store on a disk works.
  `QMATVEC_PREFILL_SPLIT_VARIANT=1` in the docker repo's run.sh (2026-11-09) was not the cause and can go.
- The fork's qwen4exp is behind upstream's (recurrent_layers as indices vs upstream's bool flags; llama-server
  rejects these files): worth porting in the fork (the user's).

## GLM-5.3-Flash (`glm5_next`)

**Where it stands (2026-11-12).** `/models/GLM-5.3-Flash-FAST.gguf` (171 GB FAST; its `indexer_compressor_ape` /
`_gate` came out 4-bit - tiny tensors, not worth the 19 h to redo). Loader, CPU reference (`src/glm5_ref.cpp`, equal
to upstream's glm5-next on the tiny model, cos 1.000000 every layer) and GPU forward pass (`forward_glm5`,
`src/glm5_gpu.hip`) are in; the tokenizer (`glm4`) equals upstream's on 28 prompts. Tests: `tests/glm5next_verify.sh`
(the tiny model of `tools/gen_glm5next_ckpt.py` as FAST against its F32 decode in the oracle: upstream llama.cpp's
CPU build at /data/conv/upstream-build, `make ref_dump_stock REF_STOCK=/data/conv/upstream
REF_STOCK_B=/data/conv/upstream-build`), `tests/glm5next_head_verify.sh` (the release's first 8 blocks as a model,
/data/conv/glm53-head8.gguf, 21.5 GB; full tier row glm5next_head8), `glm5next_verify --bench` on `GLM_BENCH=4,16`.
Rules (user, 2026-11-10): the reference is for numbers only; the engine keeps its UMA way (the GGUF mapped and pinned
in place, GPU-first with the CPU alongside) and FP4-sized weights.

Memory: 95% of the 320B params are expert weights, and no 4-bit quantization fits 128 GB, so the experts are paged
(`src/expert_pager.h`): `hipHostRegister` makes a range a KFD SVM range the driver keeps resident (so what is
registered must fit, and is mlocked against reclaim); `hipHostUnregister` does not let go, a MAP_FIXED remap does. The
default: the pager registers experts until the budget is full (`INFERENCE_EXPERT_PIN_MB`; 0: never), promotes the hot
ones (`INFERENCE_EXPERT_PROMOTE=0`: first come only), and an expert not on the GPU is computed on the CPU from the plain
mapping (`src/expert_cpu.h`, AVX-512 VNNI, a batch's work grouped by expert). `INFERENCE_EXPERT_EVICT=1` is the LRU
mode (releases; 1.8 tok/s); `INFERENCE_CPU_ONLY=1` runs the whole forward pass on the CPU pool (2.9 tok/s, no
registration at all; no batched prefill, checkpoints, sequences or MTP draft there). `tools/glm_chat.pl` chats through
the model's template (needs Minijinja's `loop_controls`; scripts/setup-minijinja.sh passes it). Whole file: 4.3-7.6
tok/s (a conversation that stays on its subject promotes its experts: the higher figure; a new subject is a new set
and the disk decides), first text 0.2-0.7 s warm, 6 s cold; prompts 31-43 tok/s on random tokens. **Tabled (user,
2026-11-12: this speed is fine).**

Open leads, when it is picked up again:
- The MTP block (blk.45) as draft head: not bound yet; `index_share_for_mtp_iteration` means the draft shares the
  main index, so the verify path's rollback has to roll the index back too. Its experts are registered with the
  rest (3.6 GB).
- Reads of cold experts on several threads (readahead into the page cache, the mapping kept); layer L+1's router
  on the stream before layer L to read ahead; use counts kept in a file beside the model for a warm start; the GPU
  budget on varied prompts (the sweep was on one prompt); cheaper promotion swaps (the release on a thread behind
  the forward pass, fewer swaps under memory pressure).
- Profile the GPU's own ~100 ms/token on the whole file (3x the all-resident projection: the drains, first reads of
  new pages?); a real prompt's prefill rate in the deployment.
- Prefill: sparse MLA on WMMA (~40% of a long prefill: scalar FP32 over ~2k positions x 64 heads x 512 a token); a
  chunked KDA; an F16 latent cache (half the bytes).
- Batched sequences; the stand-in one-layer KV cache's sequence features (store, checkpoints) for GLM's state; a
  second context on the same model forwarding at the same time would have its experts swapped under it (one
  forward at a time is assumed; the pager would need use counts).
- The routing trace of real text (the pager's counters, `INFERENCE_VERBOSE`); the indexer's ties are picked by the
  lower pool index on both paths (llama.cpp's top_k is unordered: no rule to match).
- The NPU (amdxdna, no /dev/accel in the container) behind `cpu_expert_ffn`'s interface: judged far-fetched.
- Vision (`glm5_next_vision`) is an independent track: an encoder path, a patch + merger projection, the image
  token interleave, and a reply-kind channel in the C API for `reasoning_effort` / `clear_thinking`.

## Concurrent runs

Decision (user, 2026-11-07): KV persists, one conversation's cache is not cleared by another's; per-slot sequences
are dropped. In: the block store (`--store DIR`, `--store-max` GB, lazy prompt-end snapshots written on SIGTERM), a
switch between conversations reads the prefix back (a 27B switch: 0.79 s instead of 36.8), `preserve_thinking` (a
new user message no longer re-prefills the previous turn), `--parallel N` (N conversations of one model in one forward
pass, prompts prefilled in chunks between the others' tokens; not with a draft head). Open:
- Watch the resume on the deployment (the `kv` line's mapped blocks) and `--parallel` on the 27B / 35B there (the
  tests ran on Qwen3.5-4B). `--store-max` 32+ on the deployment (one 130k-token 27B conversation ~14 GB).
- The 10 identical resends in opencode's log (5 each before two compactions, at ~210k tokens) look like failed
  requests, whose replies the log does not have.
- Not done: a disk tier for the store; the store on by default in run.sh (the deployment sets ROCM_STORE_DIR).
- **Speculative decoding under `--parallel` (scoped 2026-11-12; not started).** Blocked in the engine, not the
  server: `forward_seqs` refuses a model with a draft head ("batched decoding: not for this architecture yet"); the
  head's state (`mtp_carry`, `mtp_pos`, its KV cache) is one per model, not per sequence; a batched pass takes one
  token per sequence (a second row of a sequence is refused); a verify needs k + 1 rows of one sequence, with its
  recurrent layers run row by row and a rollback to the snapshot after row keep - per sequence (`rs_mid` is one).
  Stages: (1) per-sequence head state (carry, pos, KV) and forward_seqs with the head run on single-token rows (drafts
  for every sequence, no verify: a correctness check against the sequential path, bit for bit); (2) verify rows
  (k + 1 per sequence in one pass, attention over the sequence's own KV, recurrent layers per row), tested against
  forward_verify; (3) per-sequence rollback, the server's loop with drafts per job, the `--parallel` flag without
  `--draft` refused lifted. Expected gain unmeasured: the sequential sidecar gave ~38 -> ~50 tok/s, the batch already
  amortizes the weight reads across sequences, so the k + 1 rows' cost is the thing to measure first (step_chats at
  2 sequences vs 1 on Qwen3.5-4B, then the verify rows' pass time).
- To look at: `--parallel` by default; several models generating at once (one forward pass each per round; GPU sharing); loads off the request path
  (between rounds, or in a second process: a request that arrives during a load waits, 2 s from the page cache,
  minutes from a cold disk); fairness and queue limits per user (the logging's user and conversation id);
  llama-server on the same GPU.

## Server logging

In: the engine's event hook, `set_event_handler`, `ROCm::Inference::Log`, the server's lines (start, done, prefill
progress, generating every 5 s, loads, access, cancels), `--log-level` / `--log-format` / `--log-color` /
`--debug-stream`, for sequential and `--parallel` requests alike; `t/05-log.t`, `t/15-events.t`, `tests/events_test`,
`t/62-server-log.t`. Open:
- **Soak on the Flash model**: one request with a long prompt; the prefill ticks visible in real time, no per-token
  lines. Needs a window with the deployment stopped (see the draft-events item under Open leads).
- With `--parallel`, the done line's store counters (`kv` line) are the model's over the request, so another job's
  blocks can show in them.
- Identity: the user (nginx X-Remote-User; a Keycloak JWT later), the client address (X-Real-IP), the conversation
  id (X-LLamaCPP-Id-slot). The docker repo's nginx needs `proxy_set_header X-Remote-User` on /rocm/ (not done).

## Later

- **Converter follow-ups**: an importance matrix (the weighted ROCmFP4 search; collected on the GPU by the
  engine), and a faster encoder (AVX-512 across 16 blocks keeps each block's summation order, so the bytes stay the
  reference's).
- **Qwen3.8-Flash-Next beyond 262,144 tokens (YaRN)** (HF card, 2026-11-06). The checkpoint is 262,144 natively;
  the card reaches 1M by a YaRN override (`factor` 4.0, original 262144, with an attention scale). The GGUF has no
  rope-scaling keys and the engine has no YaRN. To do: YaRN frequencies and the attention scale in the RoPE, a
  `--rope-factor` option, and a quality check at long context. The 16 GB KV budget caps it at about 600k positions
  at F16 even then. Not started.

## Open leads

Speed:
- Long-context prefill attention runs at ~25% of the FP16 WMMA peak (already ahead of llama.cpp's: 5.5 vs 7.7 s on
  Qwen3.6-35B at 32k): the largest lever for long prompts. `build/attn_bench` (the kernels alone, 1,024 tokens at
  position 31,744, best of 20, +-2%): D 256 17 TFLOPS (32 ms), D 128 12 (22 ms), D 64 13 (21 ms). Ablations: the
  WMMAs are ~58% of D 256 (~29 TFLOPS while they run, near the practical peak), the K / V staging ~18%, the per-block
  glue (4 barriers, sP through LDS, the softmax's moves) the rest. 32-position blocks lost (Tried and rejected), so
  the barriers are not what holds it back: the lever left is the registers (D 256 runs 4 waves per SIMD at 256
  VGPRs). Share of a 32k prefill in attention: Qwen3-0.6B 84%, Gemma-4-26B 67%, Llama-3.2-1B 63%, Qwen3.6-35B 35%,
  Qwen3.5-4B 28%, Nemotron-Nano 25%, Qwen3.8-27B 20%.
- Qwen3.8-27B (dense ROCmFP4_FAST) is compute-bound: matmul_wmma_q4_0_rocmfp4_fast ~26 of ~59 TOPS (70% of a 32k
  prefill); its epilogue (exact per-block scaling, ~19%) would need FP16 WMMA on pre-scaled weights (twice the
  LDS). Tabled.
- MTP verify passes: read experts shared by the verified tokens once (~3.8 of 10 shared by two tokens: ~19% of a
  2-token pass's expert bytes); draft cost 2.7 ms, mostly the head's LM head (a vocabulary cut hurts other
  scripts; a shortlist from the main model's top logits of the previous row, or an int8 copy).
- MoE decode (Qwen3.6-35B): the experts' matvecs stream at ~125 GB/s against ~200 for dense layers (5.5 of 13 ms a
  token), the DeltaNet matvecs of a 2,048-wide model at ~135: latency-bound launches.
- Decode leftovers (Flash): moe_router 48 x 14 us, hc_norm 98 x 7.6 us, deltanet_rule 36 x 19 us, hc_mix 97 x
  2.9 us, ~930 launches x ~1.7 us of gaps; each well under 5%.
- Prefill: deltanet_conv (~100 GB/s); the quantizing producers (swiglu, add_rmsnorm_rows) compute both Q8_0 and
  Q8_K though each consumer reads one; the CPU embedding row decode; attn_k / attn_v as two launches (~1%).
- Small F16 matrices (Qwen3.5-0.8B, 1,024-2,048 rows) latency-bound at 55-76 GB/s; MoE experts in F16 / BF16 run
  matmul_f_id (no model here has them); BF16 FFN down (K 17408) the slowest shape (23 TFLOPS).

Quality / correctness:
- Prefill is split-invariant (ARCHITECTURE.md section 5). Open: the tests that allowed for the old WMMA differences
  between cuts (store_verify.sh and seq_verify compare first tokens, generate_verify a tolerance) can demand
  bit-exactness; the full tier's perplexity baselines of the models not run here (the Flash model, the 122B and 120B
  files) need `tests/regress.sh --update` on those rows.
- Draft events are not verified: `events_test MODEL DRAFT` is written but not run. The Flash model (122.8 GB file)
  shares the work host's 124 GB with the deployment: Flash work needs a window with the deployment stopped, or a
  generator for a tiny model with a draft head (tools/gen_qwen4exp_tiny.py writes none).
- K-quants dot Q8_K activations (one scale per 256), as ggml's CPU backend; ggml's GPU backend uses q8_1 (per 32):
  ~1% better perplexity at Q2_K. An option for the K-quant kernels.
- A quantized KV cache needs n_kv * D % 256 == 0 (Qwen2.5-0.5B, 2 x 64, cannot have one).
- Qwen3.6-35B Q4_0 forward_verify drifts to cos 0.989 at layer 21 (smooth, next token = llama.cpp's): its suite
  row uses min cos 0.985.
- One forward_verify run on the 35B once hit a GPU memory fault in matvec_q8_id_q5_K, never reproduced; likely
  the test's buffer overrun fixed on 2026-11-04 (5e6e9e6). Close after a few clean runs.

Tests and API:
- In `make regress`: kv_verify, generate_verify, events_test, seq_verify (0.6B, 4B), store_verify.sh, and the Perl
  suites (Web::Route, Minijinja, ROCm::Inference with the server tests on Qwen3.5-4B). Still out: mtp_verify and
  `events_test MODEL DRAFT` (the Flash model and its draft head: a window with the deployment stopped).
- `t/60-server.t`'s "is 91 prime" content check fails on Qwen3-0.6B (the model says yes); the suite runs it on
  Qwen3.5-4B (tests/regress.tsv). Loosen it or keep the 4B.
- **Raw completions (`POST /v1/completions`, 2026-11-12: whole and streamed, sequential and `--parallel`) leave out
  two parts of OpenAI's; likely not needed** (coding agents use chat; both are what evaluation harnesses use), so do
  them when a client asks: `prompt` as an array (several prompts answered as choices[i], or token ids instead of
  text; 400 now - on the `--parallel` server one job per prompt) and `logprobs` (per-token log-probabilities with the
  top-N alternatives, with `echo` to score the prompt itself; always null now - needs the logits out per step:
  `get_logits` exists, a read a token).
- Untested: Mistral (llama architecture, SentencePiece tokenizer), Llama-3.1-8B (no files); Llama 3.1's built-in
  tools (<|python_tag|> brave_search.call(...)) not parsed; Nemotron's reply format (check against its template).
- The XS module lacks inference_prefill_logits, model_name and set_seed (add when a Perl user needs them); a C
  API channel flag for reply kinds only if a non-Perl user needs it.

## Tabled

- GLM-5.3-Flash speed work (user, 2026-11-12: 4.3-7.6 tok/s is fine); its leads are in the GLM section.
- Older architectures: gpt-oss, deepseek2, glm4, phi3, gemma2 / gemma3, Mistral (user, 2026-11-04).
- The ROCmFP4_FAST WMMA kernel work (user, 2026-11-04: "move on, table this for now").
- A CPAN release of ROCm::Inference, Minijinja and Web::Route (user, 2026-11-05: the Docker image builds them
  from source).

## Tried and rejected (measured; do not repeat without a new idea)

- Prefill attention with 32-position blocks (2026-11-12: each 16-column half its own S and P V product, one softmax
  rescale per 32 positions, D 256's Q aliased with the K / V staging): slower on every shape - D 256 41.7 / 42.7 ms
  against 31.7 / 32.6, D 128 28.3 against 22.8, D 64 22.7 against 21.0; the same code with 16-position blocks
  already loses 8-19% on D 256 from the aliasing; the Q row padding (D + 8 halves instead of D + 4) is neutral.
- GPU reads of an unregistered mapping (2026-11-11, after a boot with `amdgpu.noretry=0`, `tools/gpu_demand_paging.cpp`):
  a kernel reading unregistered pages that are in memory hangs the GPU ("HW Exception ... GPU Hang", a full reset
  after ~20 s, no page fault logged). Do not repeat on this kernel / firmware; registration is not optional.
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
