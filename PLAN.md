# Project Plan: rocm-hip (Strix Halo, gfx1151)

State as of 2026-11-23. This file holds open work, the current status of each feature, and the decisions and measured
dead ends that still matter. Finished work is in the git history (the commit messages say what changed and where its
checks are). Design: ARCHITECTURE.md.

How the work runs: each open item is a package with its check. The main session briefs it, reviews the commit, reruns the
check on `main`, commits, pushes, and (AGENTS.md) touches `.build-trigger` when the image needs a build. Timing packages
run alone on the machine.

## Status in brief

- **Engine:** runs qwen2, qwen3, qwen3moe, qwen35 / qwen35moe, qwen4exp (Qwen3.8-Flash-Next), gemma4, gemma-embedding,
  nemotron_h_moe, llama and glm5-next (GLM-5.3-Flash, experts paged from disk) from zero-copy GGUF mappings on the UMA.
- **Surfaces:** the C API (`libinference.so`), the Perl XS module ROCm::Inference (Minijinja templates),
  `rocm-inference-server` (OpenAI-compatible: chat, raw completions, embeddings, audio, images, `--models-dir`,
  `--parallel`), and `bin/rocm-convert`.
- **Deployment:** the user's llama.cpp-rocm-gfx1151-docker builds rocm-hip from GitHub `main` (its Dockerfile is this
  repository's). The image's watcher restarts the container from each new image; a build starts when `.build-trigger`
  is touched.
- **Tests:** `make regress` (quick tier) passes 47 of 47 on this host (2026-11-16). `make regress-full` has not been run.
- **Image:** `/version` carries the rocm-hip commit. Last image build with nginx's `-t` check: 2026-11-16 19:42.

## Status (saved before a reboot)

Z-Image-Turbo MVE (aim: 1024 x 1024, 9 steps, under 30 s on ROCm HIP, matching the ComfyUI workflow in comfy/workflows/zimage_mve.json):
- Done: the DiT forward, sampler, output and VAE take the image grid (defaults keep 256 px unchanged); the causal text attention is
  dynamic-shared and the text encoder takes any length (the example prompt is 101 tokens); the pipeline takes height, width and steps
  (`zi_generate_steps`, `zi_generate_noise_steps`).
- Timed (2026-11-22, `SD_TIME=1 SD_TIME_MS=50 zimage_1024_verify 1`, one DiT step at 1024 and the VAE decode): the DiT is
  about 3.8 s a step (the WMMA linears 2.7 s, flash attention 0.4 s, the bf16 and norm kernels the rest); the FP32 VAE decode
  is about 3.6 s. Nine steps plus the decode come to about 39 s against the 30 s target. The largest single launch is 50 ms;
  no ring timeouts.
- VAE: `sd_gemm_af128` (128 x 128 FP32 tiles) and 128 MB im2col chunks (the default, FP32). Opt-in `SD_VAE_BF16=1`: the
  convolutions with cout >= 128 and K % 64 == 0 run on the 256-row WMMA (`sd_gemm_wmma256b`) with BF16 activations and weights
  (FP32 im2col, then `sd_transpose_bf16`, then `sd_bias_cm`). The 256 check passes both ways: FP32 exact 22.54%, within one
  level 79.36%, mean abs 1.022; BF16 exact 23.28%, 78.53%, 1.024. One step: 8.7 s BF16, 9.5 s FP32 (run-to-run noise about 0.4 s).
  Not default until the lowering is fused (below). `sd_gemm_wmma128` faults on the conv shape (K 1152, cs_m 1, cs_n = HW): not
  used, not investigated.
- Not yet checked: the 1024 engine output against the diffusers reference (9 steps). The reference's VAE decode stalled the
  desktop's graphics ring at 1024 (dmesg: ring gfx timeouts); run it only when the desktop is idle.
- hipBLASLt (2026-11-22, dependency added by the user's decision): the DiT's large linears (out >= 256, T >= 1024) run
  `hipblasLtMatmul` (`src/sd_device.cpp`, namespace `blt`): BF16 operands, FP32 output and accumulation, the bias as the
  epilogue, one plan per (out, T, in, bias); each plan times the heuristic's candidates once on its first call and keeps the
  fastest (9 candidates on these shapes; the fastest was index 4). `SD_LINEAR_KERNEL=256b` keeps the kernel; `SD_BLT_LOG=1`
  prints the candidates. The BF16 copy of the input is device-only scratch (`device_scratch`). Measured: the 9-step 1024
  generation 36.2 s (43.0 s with the kernel); the bench (`sd_gemm_bench`, CPU-checked, bias included) 25 / 24 / 15 TFLOPS on
  the three shapes with pinned W, x, y (device W, x, y: 27 / 28 / 16). The 1024 output against the kernel's: 65% of bytes
  exact, 98% within 3 levels, mean abs 0.53, max 48 on a few pixels; both are fp32-accumulated GEMMs on BF16 operands, so the
  difference is the summation order carried through nine steps (not checked against a reference: none at 1024). The 256 check
  does not reach this path (T < 1024), so the bench is its gate.
- Dependency: `libinference.so` needs `libhipblaslt.so.1` (the Makefile's link lines carry `-lhipblaslt`). The Dockerfile's
  build stage takes all of ROCm from the tarball and the runtime stage copies `lib/*.so*` and `lib/hipblaslt/`: no change needed.
- DiT, the non-GEMM part (2026-11-22, bit-identical to before: the 1024 one-step output matches byte for byte): the norm and
  the AdaLN scale in one pass written as BF16 (`zimage_rms_mod_bf16`) feed the three projections and the two feed-forward
  projections without conversion copies; the SwiGLU writes BF16 (`zimage_swiglu_bf16`) for the down projection; the BF16
  inputs live in device-only memory (`sddev::dalloc`, pooled). The nine-step generation: 36.2 s -> 35.3 s.
- DiT step, measured: 3.76 s wall (`zimage_1024_verify` 1 vs 2 steps). Of it the hipBLASLt products are about 2.6 s (seven
  shapes: T = 4096 for the embedder and the noise refiner, 4224 for the main layers), flash attention 0.43 s, the RMS norms
  0.14, SwiGLU 0.07, the gated residuals 0.06, the BF16 conversions (flash's q, k) 0.05, the modulation 0.05, RoPE 0.04.
- The first image in a process pays the hipBLASLt tuning: about 2.3 s (7 plans; one warm-up run per candidate, and a timed
  run only for the candidates within 2.5x of the best).
- The ComfyUI reference, the workflow's own output (`comfy/zimage_mve_00001_.png`, 1024, 9 steps, seed 42, the same prompt as
  `/data/zimage-ref/z1024/prompt.txt`): the engine matches it in content and composition. Its initial latent is torch's CPU
  randn after `manual_seed(42)` (`noise.f32` in z1024 is not that: max abs diff 6.5). Write it with
  `torch.Generator('cpu').manual_seed(42)`, `torch.randn(1, 16, 128, 128)[0]`, and pass it as `ZI_NOISE=<file>` to
  `zimage_1024_verify`. Result: mean abs 3.18 levels, PSNR 27.7 dB, 39% of the bytes exact, 83% within 3 levels. ComfyUI's
  "simple" schedule gives the same timesteps as ours (t = 1 - k/9, then the shift 3: checked analytically). The rest is
  numerics (ComfyUI runs the DiT's linears in BF16; the VAE's precision there is not checked).
- BF16 outputs for the DiT's large linears (2026-11-22, the user's decision: yes): hipBLASLt's product with a BF16 output
  (the bias as its epilogue) runs at 38 / 36 / 21 TFLOPS against 23 / 23 / 15 with the FP32 output, and the library's rounding
  is round-to-nearest-even (256 of 256 outputs equal to it; the bench's `SD_BENCH_OUT16=1`). The DiT's chain now runs in BF16:
  the QKV outputs go to the per-head norm and RoPE (`zimage_head_norm_rope_bf16`, BF16 in and out) and V to the transpose
  (`zimage_transpose_pad_bf16_in`); the attention on BF16 q, k, v (`attention_bf16`); the o and f outputs to the fused RMS norm
  and gated residual (`zimage_rms_gate_add_bf16`, the residual x stays FP32); a and b to the SwiGLU in BF16
  (`zimage_swiglu_bf16_in`). Where hipBLASLt does not take the shape, the FP32 product is rounded to BF16
  (`linear_bf16_xb16`). One step: 6.2 s (7.15 before). Nine steps: 24.2 s (34.5 before). Against ComfyUI at nine steps: PSNR
  27.45 dB, mean abs 3.34 levels (FP32 chain: 27.74 dB, 3.17); the 256 check improves: exact 26.99%, within one level 80.06%,
  mean abs 0.972 (was 22.54 / 79.36 / 1.022). With `SD_LINEAR_KERNEL=256b` the same chain runs on the kernel's FP32
  product, rounded to BF16. The experiment that rounds only the linear outputs (`SD_ROUND_OUT=1`, FP32 consumers) gave 28.59 dB: the chain and
  that experiment should match, and they differ by 1.1 dB. The likely cause is the algorithm the tuner picks by timing (each
  plan's first call), not the rounding (checked); not yet proved.
- VAE convolutions, BF16 by default (2026-11-22; `SD_VAE_CONV=fp32` keeps the FP32 kernel): the im2col written as BF16 in
  one pass through shared memory (`sd_im2col_bf16_tile`: a block is 64 consecutive pixels of one row and 8 channels; stride
  1, k <= 3, Wo a multiple of 64, cin a multiple of 8), the weights as BF16, one hipBLASLt BF16 product per convolution with
  the FP32 output channel-major (the product's rows are the pixels, the output row stride 1), the bias as a pass. One step:
  7.15 s against 8.15 s (FP32). Against ComfyUI at nine steps: PSNR 27.74 dB, as before (mean abs 3.17). The 256 check:
  exact 23.89%, within one level 78.87%, mean abs 1.014 (FP32: 22.54 / 79.36 / 1.022). The column matrix matches the
  single-tap kernel's on all 48 convolutions of the 256 run. The gate moved by 0.8 points between two versions of the
  column kernel with identical columns: the product's algorithm is chosen by timing, so it may differ between runs; this is
  not yet proved, and it is the same kind of effect as the DiT's tuning.
- hipBLASLt's algorithm choice (2026-11-22): the candidates' numerics differ. Identical code in three processes gave outputs that
  differed on 42% of the bytes (mean 0.8 levels, max 65): the timing picked different algorithms. Each algorithm's own output is
  reproducible (the same hash in every process). The tuner now times only the candidates whose output equals the first
  candidate's bit for bit (`diff_words`, on the device; the generic product's span of rows the same way), and picks the fastest
  of those. Three processes now give identical outputs (md5 equal). The cost: for the VAE's convolutions the first candidate is
  about 2x slower than the fastest (for example 56 against 27 ms at 128 x 1048576 x 1152), once per image: about 1.3 s
  (9 steps: 25.5 s, was 24.2 s with the timing choice). A calibrated table (the best index per shape, measured once) would
  recover it; not done. The nine-step image: PSNR 27.45 dB (as before).
- The seed (2026-11-22): the engine's noise is torch's CPU randn for the seed (`src/torch_randn.h`: mt19937, the 24-bit
  uniforms, the 16-value Box-Muller blocks; `tests/torch_randn_check.cpp` against torch's file: 83% of the values bit-exact, the
  rest within 5e-7). The server route (`/v1/images/generations`, z-image-turbo) takes the steps (9 by default) and the size
  (multiples of 16 from 64 to 1024; 1024 x 1024 by default); the Perl XS `_zi_generate_png` passes the steps to
  `zi_generate_steps`. End to end through the Perl module, seed 42, 9 steps, 1024 x 1024: 27.44 dB against ComfyUI's PNG, 41 s
  with the model's load.
- The ComfyUI node (2026-11-22): `steps` (9) and the size (1024 x 1024) are inputs now; the request carries the steps.
- The calibrated algorithms (2026-11-23, `src/blt_calib.h`): the fastest candidate of each convolution and attention product
  shape at 1024 x 1024 (15 shapes), identified by hipBLASLt's serialized descriptor (the first two bytes: the solution index).
  The tuner takes the entry when the heuristic offers that algorithm; otherwise the same-bits rule. Measured: the convolution
  products 1.36 s against 3.42 s over the two generations of a one-step run; one step 6.07 s against 7.3 s; nine steps 24.25 s
  (25.5 s without). Reproducible: three processes give identical output. Against ComfyUI at nine steps: 27.45 dB, unchanged; the
  output moves 0.04 levels on average against the rule-based choice. The table names the shapes of 1024 x 1024 only (other sizes
  take the rule). To regenerate: `SD_BLT_CALIBRATE=1 zimage_1024_verify 1` (stderr: one `calib:` line per candidate), keep the
  fastest per shape, write the table. `SD_BLT_NOCALIB=1` turns the table off (for the comparison).
  Not yet: the linear plans (the DiT's BF16 outputs: the rule's first candidate is already the fastest class there).
- Next, in order: the chain's remaining cost
  (flash attention 0.43 s a step, the attention's FP32 output converted to BF16, the modulation), and the calibrated table
  for the convolutions' algorithms (about 1.3 s a generation).
- DiT linears, measured 2026-11-22 (`tests/sd_gemm_bench`, the three DiT shapes; `SD_BENCH_DEVICE=1` puts W, x, y in
  device memory): `sd_gemm_wmma256b` runs at about 19-20 TFLOPS on QKV and MLP-up and 9-13 on MLP-down (K 10240: a 20 KB row
  stride, half the rate of MLP-up at the same FLOPs). The DiT step's linears take about 2.7 s (the engine's own timing).
  The profiler (rocprofv3, `SQ_WAIT_CNT_VMVS`) puts 58-63% of the wave cycles in waits on vector memory; LDS waits are 4%
  (`ALUStalledByLDS` 0.1%, bank conflicts 3.9%); the L2 hit rate is 3%; the address translation is 2% busy.
  Vendor check (hipBLAS / hipBLASLt, the same shapes, BF16 in): the default algorithm for FP32 output runs at 4-6 TFLOPS; its
  algorithm 4 (of 8) at 32 / 32 / 18; BF16 output (`hipblasGemmEx`) at 40 / 40 / 22. Adopted 2026-11-22 (the user's
  decision): see the hipBLASLt bullet below. Tried and rejected (measured in the engine's kernels; do not repeat without a new idea): one LDS
  tile with two blocks per CU (slower: 15.8 / 14.9 / 8.8); three register sets two tiles ahead (spills, 8 TFLOPS); the 256 x 256
  tile with 16 waves (no gain; the engine's step 5.9 s against 5.4 s); 64-byte row pieces for the loads (no gain). Device-only
  scratch for the BF16 copy of the input raised the bench 15-20%, not the engine's step (reverted). Pinned host memory against
  device memory for the operands: +30% in the bench, not taken (the zero-copy rule).
- Build trap, fixed: `sd_gemm_bench` did not depend on `libqwen.a` (it holds `sd_device.o` too), so it linked a stale device
  library; its timings in that state were the previous build's. Benches and verifiers that link the device code depend on
  `$(QWEN_MODEL_LIB)` now.
- Open lead: the MLP-down shape. Its weights have a 20 KB row stride; the activation copy is ours to lay out (the weights are
  read in place, so theirs is fixed).
- Open: the 30 s target is unmeasured end to end; the DiT uses BF16-rounded GEMMs (linear_bf16), which cost accuracy elsewhere
  (see the FP32 fixes).

## Open items, in order

1. **Deployed service check (on the user's host).** `GET /rocm/v1/models`, one short chat through `/rocm/` (the fixup:
   `conv_id` and `id_slot` added), one short speech request. This session cannot reach the service (nothing listens on
   port 8000 here). The user runs the checks, or gives a reachable address.

2. **KV snapshots during prompt processing, for the large CPU-only models (GLM first, then DeepSeek).** Wanted: snapshots
   written during prefill, about every 500 tokens, so that a restart does not lose a 10k-token prompt (5 minutes or more on
   GLM CPU-only). Opt-in per model, default off.
   - **The gap:** the block store holds only a one-layer stand-in KV cache for GLM. The real per-position state (MLA
     latents, indexer keys, gates, pooled rows, in `GlmState`'s `LazyRows`) and the KDA recurrent state are never written.
     A GLM resume from the store cannot work at any snapshot frequency until they are saved.
   - **Possible wrong output (not confirmed):** `store_lookup` treats a stored block as valid whenever the model has no
     recurrent state in `rs.mem`, which holds for GLM. A GLM resume could then reuse stored blocks without their state.
     Needs a test that goes through the sequence path (my repro through `inference_generate` never reached the lookup,
     even on Qwen3-0.6B, so it proves nothing either way).
   - **Proposed design:** `--store-state-every 500` for the opted-in models. During prefill the prompt is cut at each
     500-token boundary (as the recurrent snapshots already are). At each boundary, write the new rows of the span (latent,
     ik, ig, and the pooled rows) and the KDA state. Resume restores both. Per-token size: fp32 rows of `kv_lora` plus
     two `ix_dim` values per MLA layer; the exact figure will be read from the GGUF before the format is fixed. Also
     possible: F16 rows (half the bytes, a quality check needed).
   - **Decision needed:** go ahead with the design, and the F32 or F16 row format.

3. **GLM-5.3 on the CPU in the deployment (user).** Measure tokens per second and `VmLck`, `RssAnon`, `RssFile`, `RssShmem`
   during a real conversation. Last measured: 2.9 tok/s CPU-only (expert pages in the page cache). Wait for item 2 first:
   a container restart currently resets the prefill.

4. **Request logs (done 2026-11-16, `f75975c`).** One file per hour: `requests_YYYYMMDD_HH.log`. The user deletes old files
   by hand; nothing rotates or deletes automatically.

5. **`tests/convert_verify.sh` in full.** Blocked on disk: 5.5 GB free on `/data` and `/models` (one filesystem). The run
   needs several GB more.

6. **The full regression tier (`make regress-full`)** on the models here. The quick tier passes; the full one has perplexity
   rows for models not run here (Flash, the 122B and 120B files) that need `tests/regress.sh --update` on those rows.

7. **ComfyUI backend, milestone 1: the component API** (verified on sd-turbo), then the node package
   (`comfyui-rocm-inference`: `EngineLoad`, `EngineTextEncode`, `EngineSample`, `EngineVAEDecode`, `EngineSaveImage`,
   calling the server; installed by the user into `/comfyui/custom_nodes`, which does not exist yet). See "ComfyUI" below.

8. **MiniMax-H3-Image:**
   - Check the images against the released pipeline (the timestep convention, t = 1 - sigma; the PDD heads' role). The
     engine's arithmetic is checked (the reference chain matches), but whether the released model is reproduced is not.
   - FP8 as a weight type: not measured (the engine has no FP8 weight type to write; needs an encoder and its operand path).
   - DiT drift from block 30 (the engine's error grows against the fp32 chain). Not chased.
   - The text encoder from the original checkpoint (the Qwen3 LLM through the qwen loader needs the same tensor source
     as the DiT; a larger change, after the image path). The user chose the VL encoder.
   - License: the user's confirmation covers the project; the community license terms are the upstream's.

9. **Alpamayo 2 Edge.** Licence confirmed (OpenMDW-1.1 weights, Apache-2.0 code). The modeling code is at
   `NVlabs/alpamayo2-edge`, outside this session's GitHub scope: blocked until the user attaches the repository. The weight
   headers are read (the language model: 28 layers, hidden 2048, 16 query / 8 KV heads of 128, `relu2` MLP, M-RoPE
   `[24, 20, 20]`, vocab 135168; the vision tower: 27 layers, hidden 1152, 16 heads, learned absolute positions; the action
   expert: 28 layers, hidden 512, non-causal attention over the VLM's keys and values).

10. **Concurrency (`--parallel`).**
    - Speculative decoding under `--parallel` (not started; three stages): per-sequence draft-head state and batched
      forward with single-token rows (a bit-for-bit check against the sequential path); verify rows (k+1 per sequence,
      tested against `forward_verify`); per-sequence rollback, and the server's loop with drafts per job. Blocked in the
      engine: `forward_seqs` refuses a model with a draft head, and the head's state is one per model. Measure first: the
      k+1 rows' cost (`step_chats` at 2 sequences vs 1 on Qwen3.5-4B, then the verify rows' pass time). The sequential
      sidecar gave about 38 -> 50 tok/s.
      Measured (2026-11-18, `step_chats` on Qwen3.5-4B ROCmFP4, one run, greedy, 128 new tokens, 2 prompts): one sequence
      15.8 ms a step (63.5 tok/s); two sequences 23.7 ms a step (84.2 tok/s together), a 1.33x gain from the batch. The verify rows'
      pass time is not measured yet (the MTP head the verify pass needs is not on this host for the 4B). A batched forward (qwen_model_prefill) of 1, 2, 3 and 4 rows, same position: 31.3, 33.4, 35.2, 35.9 ms; a plain decode step (qwen_model_forward) 15.6 ms. The batched path has about 15 ms of fixed cost at one row, so its rows cost 1.1 to 2.2 ms each on top of that; the verify path itself is not what this measured, so the break-even for speculation is still open.
    - Watch on the deployment: the store's resume (the `kv` line's mapped blocks), and `--parallel` on the 27B and 35B
      (the tests ran on Qwen3.5-4B). `--store-max` 32 or more (one 130k-token 27B conversation is about 14 GB).
    - The 10 identical resends in opencode's log (5 each before two compactions, at about 210k tokens) look like failed
      requests whose replies the log lacks.
    - Not done: a disk tier for the store; the store on by default in `run.sh` (the deployment sets `ROCM_STORE_DIR`).
    - Later: `--parallel` by default; several models generating at once; loads off the request path (a request during a
      cold load waits minutes); fairness and queue limits per user; llama-server on the same GPU.

11. **Server identity and logging.** With `--parallel`, the done line's store counters (the `kv` line) are the model's over
    the request, so another job's blocks can show in them. Identity: the user (nginx `X-Remote-User`; a Keycloak JWT later),
    the client address (`X-Real-IP`), the conversation id (`X-LLamaCPP-Id-slot`). The docker repository's nginx needs
    `proxy_set_header X-Remote-User` on `/rocm/`.

12. **Flash (Qwen3.8-Flash-Next) measurements, in a deployment window** (the model file is 122.8 GB and shares the host's
    124 GB with the deployment):
    - The model's own MTP block as draft head: acceptance and tok/s (`build/mtp_verify F F 128`; the sidecar measured
      about 38 -> 50 tok/s). If the own block's acceptance is worse, check the base file's `nextn.hc_head_norm` (the
      sidecar's `output_hc_norm` + 1.0; the base file is likely right and the sidecar off).
    - PLE rows in parallel, in the deployment (`PLE_THREADS`, `MADV_RANDOM`): 30k-token prompts were 400-480 tok/s with
      cold rows read one at a time.
    - `events_test MODEL DRAFT` with the Flash draft (not run; a tiny model with a draft head is the substitute, and
      `tools/gen_qwen4exp_tiny.py` writes none).
    - YaRN beyond 262,144 positions (the card reaches 1M by a `factor` 4.0 override): not started (frequencies, attention
      scale, a `--rope-factor` option, a quality check). The 16 GB KV budget caps it at about 600k positions at F16.
    - The fork's qwen4exp is behind upstream's (`recurrent_layers` as indices, not bool flags): the user's to port.

13. **Gemma-4 26B, chat and streaming checks (done).** `generate_verify` passes on Gemma-4 26B. The streamed greedy tokens
    differed from `inference_generate_ex`'s because the non-streaming generate stopped only at the tokenizer's EOS, not at every
    end-of-generation token (a turn ends at Gemma's end-of-turn token); it now uses the same check. The chat and template checks
    had assumed Qwen's markers; they now accept the engine's own end-of-generation and the GGUF's template. The quick regression
    tier passes (47 of 47).

14. **Clear by hand (the user):** `/data/iqtest` (336 KB of logs), and `nginx.conf~` in the repository root (the editor's
    backup, excluded from the image).

15. **Watch or table:** Qwen4 (not on Hugging Face yet); Qwen3.8-2.4T-A95B (tabled); the memory measurement of the
    deployment's GPU pins (tabled: GLM runs on the CPU).

16. **Read-only `/models` and voices (done).** A voice that cannot be written returns a 500 that names the path and the error
    ("voice 'clip2' not written: /models/.../voices/clip2: Read-only file system"), and the directory it started is removed. The
    deployment's voices directory must still be writable for voices to be created.

17. **Eviction with the 122B and the Perl suite on the 122B (the vision tower's memory: done).** The combined server (`--models-dir` with the
    122B, its sidecar, whisper, the TTS talker and codec, sd-turbo; `--parallel 2`) answered every endpoint. Under memory pressure (`--models-margin 60`), loading Qwen3-0.6B unloaded the 122B (load and unload lines in the log), and the 122B's image answer came through its sidecar before that. A model's vision tower is now counted in the memory check (`footprint` and the `make_room` subtraction: its file size, while loaded); the small-model image, models-dir and server tests pass with it (130 tests). The full Perl suite on the 122B has not finished: `t/10` and `t/15`
    fail with `hipHostRegister failed (-3)` on an in-process load of the 122B (the server's load of the same file works), `t/65-models-dir`
    hung once on a server (killed), and `t/70-stage` ran for hours (killed). The cause of the `hipHostRegister` failure is not known.

18. **`--parallel 2` and the second big model (122B).** With `--parallel 2`, a server that loads the 122B and then switches to
    another 122B-sized file (alpha, beta, alpha, all under `--models-max 1`) fails at beta's load: `hipHostRegister of 66353926592
    bytes (a GGUF map) failed: invalid argument (-3)`. The same switch with `--parallel 1` passes, in one process and across two
    processes; so does a single 122B load, and Flash (122.8 GB, pager path) loads twice in two processes. What was ruled out: the
    KV size (`--ctx 2048` at parallel 2 still fails; `--ctx 8192` at parallel 1 passes), the second sequence's logits
    registration, and its recurrent-state registration (each removed on its own: still fails). Not found: what `--parallel 2`
    changes at load time. Later, not now: find what `--parallel 2` changes at load time, and whether the other large
    models hit it. The host rule in item 19 is the workaround until then. **Until this is found, run the 122B with `--parallel 1`** when another 122B-sized model may be loaded
    in the same server. Per-model parallel settings are item 19.

19. **Per-model settings (done).** Each model's `parallel`, `ctx` and `kv_type` come from, in order: its sidecar `NAME.json` (beside the model,
    read at each load; an unknown key or a bad value fails the load and names the file), the host rules in `bin/rocm-inference-server`
    (`@HOST_RULES`: defaults for this host, gfx1151, 128 GB unified memory; now one rule: a Qwen3.5 MoE of 60 GB or more gets
    parallel 1, PLAN.md item 18), then the global flags. `GET /v1/models` shows each chat model's `parallel`. Measured: with the
    global `--parallel 2` and the 122B alpha, beta, alpha switch, all three return 200 (the rule keeps the 122B at 1); the
    small-model test `t/71-model-config.t` (sidecar override, bad key) and the server, audio, image and models-dir tests pass
    (8 files, 211 tests). To add a rule or a key: a line in `@HOST_RULES` or `%MODEL_KEY_OK`.

20. **TRELLIS.2 (image to 3D), in progress: a real image to a textured GLB on the engine.** Each stage is checked against its python
    oracle (`tests/*_verify.cpp`, `tools/*_oracle.py`, outputs in `/data/trellis2/oracle_*`): the sparse-structure flow and decoder,
    the shape-SLAT flow, the shape decoder, the mesh step, `fill_holes` (exact on the shape mesh), the 512 and 1024 cascades, the
    texture flow and decoder (512 and 1024), the DINOv3 ViT-L/16 image encoder (`src/dinov3.cpp`, 1.4e-6 against transformers; the
    weights are ComfyUI's local copy, the Hugging Face repository is gated) and the image conditioning (`src/trellis_image.cpp`,
    PIL-style Lanczos, 6.7e-4). The post-processing is on the device: `src/trellis_simplify.cpp` (CuMesh's simplify to 1,000,000
    faces; `tests/simplify_verify.cpp`), `src/trellis_charts.cpp` (CuMesh's chart clustering, `tests/charts_verify.cpp`: 29,360 charts
    on the 249k test mesh against CuMesh's 29,308 in 2.2 s against 0.6 s; 6.9% of faces lie in other charts, inside CuMesh's own
    run-to-run spread of about 4%, since CuMesh's atomics make its output vary between runs), and the unwrap per chart
    (`src/trellis_bake.cpp` `unwrap_uv_by_charts`: 4 s on the 249k mesh against 71 s for the whole-mesh xatlas call;
    `tests/unwrap_charts_verify.cpp`). The bake and the GLB are as before (`tests/bake_verify.cpp`, `tests/textured_glb_verify.cpp`).
    Last run, 512 on `comfy/example.png` (`tests/image_to_glb.cpp`): simplify 8.5 s (2.68M to 969k faces), charts 13.9 s (119,340
    charts), unwrap 17.1 s, bake 0.3 s, GLB 0.1 s. Before the charts were ported the unwrap took 570 s on the same mesh. The
    whole run at 512 (stage times printed by `image_to_glb`): conditioning 1.3 s, sparse structure 115 s, shape SLAT 167 s, shape
    decoder 65 s, mesh 1 s, texture flow 90 s, texture decoder 60 s, export 40 s: about 540 s in process, model loading not counted.
    The attention of every flow is now the fused fp32 kernel (`sd_gpu.hip` `attn_flash_d128` and `attn_flash_d64`, on by default;
    `SD_ATTN_FLASH=0` restores the GEMM path). One self-attention call at 23,660 tokens: 1.3 s against 10.0 s (`tests/flash_attn_bench.cpp`,
    measured with the 1024 run sharing the GPU); 2e-6 relative against a double-precision reference (`tests/flash_attn_verify.cpp`).
    The DINOv3, shape-SLAT and sparse-structure checks are unchanged with it (1.4e-6, PASS, PASS). The 1024 run in progress uses
    the build from before the kernel.
    The ComfyUI workflow the user supplied (`193b0901-VAZE_PROMPT_to_GLB_3d_TRELLIS.json`) runs the 1536 cascade, a RemeshMesh (udf,
    768) before a DecimateMesh to 700,000 faces, an UnwrapMesh ("pec", 4096), a 4096 texture, and normal-map and AO bakes. The engine
    matches the workflow's step counts and guidance for the structure (12 steps, CFG 7.5, rescale 0.7, shift 5), the high-resolution
    shape SLAT (12 steps, rescale 0.5) and the texture (12 steps, CFG 1), but at 1024, not 1536. `image_to_glb` now uses the
    workflow's 20-step 512 shape-SLAT stage (`ShapeConfig::slat_lr_steps`, used by the cascade only), the 700,000-face decimation and
    the 4096 bake. A 512 run with those export settings (`example_512_workflow.glb`, 150 MB): 2.38M faces to 667k (simplify 9 s),
    96k charts (9 s), unwrap 35 s, bake 2 s (coverage 0.37), GLB 1.5 s. Its stages ran alongside other GPU jobs (sparse structure
    341 s, shape SLAT 304 s, texture flow 91 s), so they are not comparable with the earlier runs. The user's ComfyUI run of the
    workflow took 1106.2 s.
    The 1024 cascade now runs end to end. Fixed on the way (2026-11-25): the shape decoder held every intermediate buffer until it
    returned (about 75 GB at 1024; now each level keeps only its features and coordinates, and the 1024 decode peaks at 1.8 GB of
    pinned memory); the allocator pools recorded a reused block under the request's size and so kept growing (they now keep each
    block's true size, and cache at most 16 GiB each); the attention launches in the flows were fixed at 256 queries (now sized from the
    key count, about 40 ms each); each decoder call ran twice after a too-small capacity guess (the decoder now returns its output
    sized to the result). The vase the user supplied, prepared as the workflow does (`tools/bg_crop_birefnet.py`: BiRefNet in python
    on the CPU, with the model code from Hugging Face and the ComfyUI weights, then the crop to the mask at 1024 with a 1.1 margin on
    black): `vase_1024.glb`, 4.35M faces to 670k, about 814 s of stages (cascade 245 s, texture flow 140 s, the two decoders 90 s
    each before the single-run fix, unwrap 74 s). The user found it about the same as the ComfyUI result, if anything sharper (the
    workflow remeshes at 768 and smooths the normals). The ComfyUI run (1106.2 s) also generates the image and runs the 1536 cascade.
    Served (2026-11-25): `rocm-inference-server --trellis2 DIR` (`trellis-2`, kind `3d`, a unit of the memory policy, loaded by its
    first request): `POST /v1/3d/generations` starts a job on a background thread (`src/trellis_image_to_3d.cpp`, `src/trellis2_api.h`)
    and answers its id at once; `GET /v1/3d/generations/{id}` polls it, `.../content` gives the GLB; one job at a time (429), the image
    and audio generators answer 503 meanwhile (the job holds the device layer); `t/72-trellis2.t` passes (18 checks, a 512 job in about
    3.5 minutes). Images with an alpha channel are cut out by it; the textures are deflated now (the 512 vase: 148 MB to 53 MB).
    BiRefNet on the engine (2026-11-25, `src/birefnet.{h,cpp}`, the `bir_*` kernels): the Swin-L backbone at 1024 and 512, the
    context skips, the squeeze, the decoder with the deformable ASPP (torchvision's modulated sampling) and the gradient gates; the
    F16 weights read in place, the batch norms as scales and shifts. `birefnet_verify` against the python model's dumps: all 25
    stages within 1e-5 of the norm, the 2000 x 2000 vase mask within 3 of 255 (mean 0.001); 2.8 s a forward. The server's jobs cut
    out photos with no alpha channel with it (`birefnet.safetensors` in DIR or DIR/background_removal), so python is no longer needed.
    The finished models are files in a scratch tmpfs (2026-11-26: `--files DIR`; the image's `/scratch`, mounted by `run.sh`), not
    in the server's memory, and nginx sends the downloads (`X-Sendfile-Type` / `X-Accel-Mapping` from nginx, `X-Accel-Redirect`
    from the server to the internal `/rocm-files/`): 55 MB in 0.02 s through nginx 1.24, the server's part 1 ms; direct clients
    get the bytes. A job is its user's (nginx's `X-Remote-User`): random 128-bit ids, 404 and no listing for another user, busy
    answers that name the job only to its user. The image: `TRELLIS2_DIR` in `run.sh` (mounted at /trellis2; DINOv3 and BiRefNet
    also found in the `--zimage` ComfyUI directory). The image decoder applies a JPEG's or PNG's EXIF orientation (a phone photo stored on its side), as PIL's
    `exif_transpose` does for ComfyUI (`src/image.cpp`; `image_orient_verify`: 8 orientations, both byte orders, against PIL).
    The mesh alone and STL (2026-11-26): `texture: false` skips the texture flow, the unwrap and the bake (a GLB of the mesh,
    `encode_glb_mesh`); `format: stl` writes a binary STL (`encode_stl`: z up, centred, on z = 0, the largest side `size_mm`,
    100 mm by default; faces turned when the signed volume is negative). `stl_verify` checks both on a cube; `t/72` runs an STL
    job (44 checks). The vase at 512: 131 s for the STL against about 200 s textured. The mesher's surface is not printable (the
    vase's 1.1M faces: 14,689 boundary edges, 22,858 non-manifold ones, 17% of the manifold edges join faces wound opposite ways,
    6,971 pieces; the flexible dual grid keeps which edges are crossed, not the inside; the textured GLB hides it, drawn
    double-sided), so without texture it is rebuilt as a closed solid (2026-11-26, `src/trellis_solid.{h,cpp}`, sd_gpu.hip
    `solid_*`, shared math in `trellis_solid_math.h`): on a grid of 512 cells across the object, the exact distance to the mesh
    near it, an exact distance transform to the wall (points within 0.6 cells), a fill from the border through the points at
    least `close` cells from the wall, the field min(distance - 0.6, close + 0.5 - distance to the outside), a second fill that
    makes every positive pocket the outside does not reach inside, marching tetrahedra (the Kuhn split: a closed 2-manifold wound
    outward by construction), small pieces dropped. `close` starts at 4 and doubles to 16 while the result is a thin shell (mean
    thickness 2 V / A under 3 cells: the fill went in through a wider hole; the seed-1 vase has a 10-cell gap in its cup's floor).
    Then the simplifier's manifold mode (edges taken from the faces, no sort; the link condition and the degrees in the cost
    kernel; no zero-area face). `solid_verify`: six synthetic cases (turned faces, a small and a large hole, an inner wall, two
    boxes on an edge, a speck) and real meshes (device = host reference on a 96 grid: the grids equal, the same surface; the
    result closed). The vase: one closed piece, Euler number 2 (trimesh: watertight, consistent, a volume), 0.7 to 2 s for the
    solid and 2 to 6 s for the simplification. Hollow prints (`hollow_mm`, an STL): the inside further than the wall from the
    outside is emptied (f = max(f, distance to the outside - (close - iso + wall + 0.25)), decided after the close is chosen on
    the solid), each hollow gets a drain hole (`drain_mm`, 3 by default) from the middle of its lowest layer straight down (-z)
    through the solid, hollows under min_piece filled. The vase at 2 mm: one surface (Euler 2), 31% of the solid's material;
    the hollow box within 0.6% of its exact volume. Openings wider than about 2 close (1.6% of the size) stay open (a
    whistle's mouth); the user: hollow is cheaper and sometimes needed. Limits: a thin sheet of the model stays about 1.2 cells
    thick (too thin for FDM at 100 mm: a minimum wall option is open); crevices narrower than about 2 close are filled; the
    1024 grid leaks (close is in cells) and its 26M-face simplification is slow, so the grid is 512 for both resolutions.
    Open: (b) the 1536 cascade
    (memory is no longer the limit; its time is); (c) the normal-map and AO bakes and their GLB inputs; (d) UnwrapMesh's "pec"
    segmenter (ours is CuMesh's charts); (e) a minimum wall thickness for the solid (thin sheets thickened for printing); (f) the simplify stops below its target (667k for 700k), where
    python's stops at it; (g) the unwrap (74 s on the vase: xatlas per chart, single-threaded) and the host sorts, to move to the
    device or threads; (h) RemeshMesh (udf, 768) only if wanted: the user prefers the sharper result without it; (i) access by
    the harness's login (waits on the user: when this project joins the AI harness, `./aicli.git`, or its token is accepted by
    purhie.aardbeiplantje.link). The harness's auth service (`aicli.git/auth.pl`) signs an RS256 JWT (cookie `aicli_jwt` or a
    Bearer token; claims `sub`, `roles`, `acct`: the account's uuid, `sid`, `exp`, `aud` for API tokens) and its nginx checks it
    with `auth_request` to `/auth/verify` (200 with `X-User`, `X-Roles`, `X-Account`). Then: this nginx checks the token the same
    way (or with the harness's public key), sets `X-Remote-User` from `X-User` and drops the client's; the models go to
    `/scratch/<acct>/<job>/model.glb`; `/rocm-files/<acct>/...` is served only to a token whose `acct` is that directory (the
    login token names the directory, not the job). Until then: random job ids, jobs per `X-Remote-User`, the internal files
    location (the identity is only as good as the proxy that sets the header).
21. **Qwen-Image 2.1 image edit (next, after TRELLIS.2).** The 2.1 DiT, text encoder and VAE run on the engine (items above
    in the Qwen-Image status), but there is no reference-image input and no edit route: `src/qwenimage21_*.cpp` has no edit
    path, and the server has no image-edit endpoint. Needs: the edit conditioning in the 2.1 DiT (how the reference image enters,
    to be confirmed against the reference edit pipeline), a stage check against its python oracle, and a server route. The
    ComfyUI edit workflow is in the uploads (`image_qwen_image_2_1_image_edit`), with the Edit Lightning LoRA.

22. **FLUX.2 dev (image generation), not started.** The weights are on disk (see the ComfyUI section): the DiT
    `diffusion_models/flux2_dev_fp8mixed` (BF16 and FP8 layers with scales and a quantization table; double and single blocks, dim
    6144), the text encoder `text_encoders/mistral_3_small_flux2_bf16`, the VAE `vae/flux2-vae` (32 channels) and the Turbo LoRA
    `loras/Flux_2-Turbo-LoRA_comfyui`. Needs: the DiT on the engine (the FP8 layers stay FP8 in the FP8 operand path, per the policy in
    the ComfyUI section), the text encoder, the 32-channel VAE decoder, a stage check against a python oracle, and a server route.
    The oracle needs a diffusers build with Flux.2 (0.35.2 has none) or ComfyUI on CPU. Open: the sampler's steps and the LoRA's role.

## Audio, vision and image: status

Verified against the references in the sections that follow. Each row says what works, what is open, and where it runs.

| Feature | Works now | Open |
|---|---|---|
| **Speech output** (`/v1/audio/speech`, Qwen3-TTS 0.6B) | `wav`, `pcm`, `flac` native; `stream` for `wav` and `pcm`; `mp3`, `opus`, `aac` through ffmpeg in the image (`t/69`: 37 of 37); voices from audio (`POST /v1/audio/voices`, a voice clone from new audio); long texts in pieces with the seed per piece; the BF16 talker is the default, the Q8_0 predictor an option | Pieces do not share prosody (decided: kept); a streamed WAV's header is not a finished file; the sampled frames are not torch's (its generator); one request at a time per talker |
| **Speech input** (`/v1/audio/transcriptions`, Whisper tiny) | `json` and `text`; any length, in 30 s windows with the timestamp seek rule and the temperature fallback; `task` transcribe or translate; WAV decode for any rate (band-limited resampler, soxr-like) | The cut between windows is a pause, not a timestamp; the timestamp rules are the decoder's own choices; languages other than English are checked only on the token table (plus one French clip); the GPU path (the host runs it: about 0.4 s for the encoder per window) |
| **Text to audio, AudioLDM 2** (`/v1/audio/generations`, `--audioldm2 DIR`) | the whole pipeline on the device (CLAP and T5 encoders, projection, GPT-2, the UNet with classifier-free guidance and DDIM, the VAE, the SpeechT5 HiFi-GAN vocoder); the RoBERTa and T5 tokenizers match the transformers ids; "ocean waves crashing on a beach" (5 s, 100 steps, guidance 3.5, seed 0) against diffusers: relative error 3.3e-4 through the server (2.7e-4 in the pipeline check), 160 044 bytes of WAV, 48 s | One clip at a time, about 48 s for 5 s at 100 steps; only that setting is checked against diffusers (other lengths, steps and guidance are not); the T5 tokenizer is the engine's own unigram Viterbi, not SentencePiece |
| **Vision** (`image_url` in chat, Qwen3.5-0.8B checked) | a vision tower per model: a sidecar `NAME.mmproj.gguf` beside `NAME.gguf` in `--models-dir` (looked for at each load), or `--mmproj` with `--model` (`data:` URLs; a URL is refused; an image without a tower is a 400); with `--parallel`, an image is a job's own prefill (its sequence reset, its image rows and RoPE offset per sequence; a text job beside it gives the same answers, exact); the image's chat matches transformers (cosine 0.999987, the reply tokens equal) | the tower runs on the host in float32 (a GPU path would save at most 0.5 s an image: not started); the 122B's tower matches transformers (the features of the test image: cosine min 0.999999, relative L2 3.3e-4, 140 x 3072; `ref_vision.py` and `verify_vision.py` in `/data/rocm-work/q35-122b/`); the 4B is not checked; t/63 on the 122B FP4 file: 22 of 24 (the two misses are the text chat's answer cut at 40 tokens, after a preamble; the image answers and the `--parallel 2` exactness pass), and the 122B's sidecar answers the test image in the combined server; t/63 on Qwen3.5-0.8B: 24 of 24; the reply's "black and green stripe at the bottom" describes the test image's black band and green-yellow strip, so it is not shown to be wrong; the store does not share image sequences |
| **Image generation, sd-turbo** (`/v1/images/generations`) | PNG (`b64_json` or `png`), sizes 64 to 1024 in multiples of 8, the GPU path by default (512 x 512 in 3.3 s); a flag (`--sd`) or a file in `--models-dir` (arch `sd`), a unit of the memory policy (loaded on its first request, unloaded when another model needs the room) | One image at a time; the seeded noise is not torch's (the reference is the proof of the arithmetic); only sd-turbo's one-step sampler; the VAE's convolutions (the GEMM) are next for speed |
| **Image generation, MiniMax-H3-Image** (`--minimax DIR`) | the pipeline end to end on the original checkpoints (the DiT, PDD heads and VAE read in place; the text encoder from its GGUF); 256 x 256 in about 166 s; the Q8_0 and ROCmFP4 DiT on the device; the last 32 prompts' embeddings cached; images queue, one at a time | Item 8 above (the released-pipeline check, FP8, the text encoder's source, block-30 drift) |
| **Qwen3.5 text** | qwen35 and qwen35moe run; the plain-text attention checks pass; plain Qwen3.5 text was not re-run after the M-RoPE change | None listed |

Not on this host, or not checked: the 4B vision tower; the Whisper GPU path (only whisper-tiny is here); other
languages' transcription beyond the French clip; ffmpeg's encoders on any host other than the image (t/69 uses the one
installed here).

## One server, all the models (the combined setup)

One `rocm-inference-server` can serve every model kind at once: `--models-dir DIR` (chat and embedding models, loaded on request,
least recently used unloaded first), a vision tower per chat model (the sidecar), and the speech and image models in the same
directory (whisper, the TTS talker and codec, sd-turbo), each loaded by its first request. The speech and image models are units
of the same memory policy as the chat models (no idle timer, as for the chat models). Tested with the 122B and its sidecar, whisper,
the talker and codec, and sd-turbo (`--parallel 2`, `--ctx 4096`): the image chat, transcription, speech (after a voice is created),
image generation, and the model list all answered. The `--voices` default is `DIR/voices`, set when the talker is first known.
Tests: `t/65-models-dir` (44 of 44), `t/64-audio` (11 of 11), `t/67-image`, and `t/63` on Qwen3.5-0.8B (24 of 24) pass with this code.
The model list (`GET /v1/models`, `GET /v1/models/{id}`) names every model with `kind` (`chat`, `embedding`, `transcription`, `speech`,
`image`), `endpoints` (the routes it is served at) and `capabilities` (`completion` and `tools` for a chat model, `vision` when it has a
tower, loaded or as its sidecar; `embedding`; `transcription`; `speech`; `image`). A chat request that names a speech or image model
gets a 400 that names the endpoint. The model names and `/models/load` / `/models/unload` follow llama.cpp's router; `kind` and
`capabilities` follow the layout Ollama and OpenRouter use (checked 2026-11-18: a model's capabilities, not one type label; no
standard endpoint field exists, so `endpoints` is this server's). Not done: the read-only voices directory (item 16), eviction under
the 122B's memory and the vision tower's memory (item 17). This code is committed to `main` with this note.


## Per-model gaps

From PLAN.md's notes, not re-tested for this summary. A blank means no open item is recorded.

- **GLM-5.3-Flash (glm5-next):** no store resume for GLM's state (item 2). CPU-only speed 2.9 tok/s (earlier); the
  deployment is unmeasured (item 3). Memory on the real model is unmeasured at a long conversation (the pinned rows and
  the page cache for the CPU experts). The MTP block (blk.45) is not bound as a draft head: the index is shared, so a
  verify's rollback must roll it back too. Sparse MLA prefill is about 40% of a long prefill and is not on WMMA. Batched
  sequences are not supported. Its vision track (`glm5_next_vision`) is not started. Speed work tabled (user: 4.3-7.6
  tok/s is fine).
- **Qwen3.8-Flash-Next (qwen4exp):** item 12 (MTP acceptance and tok/s, PLE parallel reads, draft events, YaRN, memory on
  124 GB). Its `--parallel` and store checks are on the deployment (item 10).
- **Qwen3.5 (0.8B, 4B, 122B):** vision on the host only; the 4B and 122B vision not checked (the 0.8B works with `--parallel`).
- **Qwen3.6-35B:** the full regression's perplexity baselines are not recorded (item 6). The MoE decode streams its experts at
  about 125 GB/s against about 200 for dense layers (latency-bound launches). Its Q4_0 forward_verify drifts to cos 0.989 at
  layer 21 (smooth; the next token equals llama.cpp's); the suite's row uses min cos 0.985.
- **Qwen3.8-27B (dense, ROCmFP4_FAST):** compute-bound on `matmul_wmma_q4_0_rocmfp4_fast` (about 26 of 59 TOPS, 70% of a 32k
  prefill). The epilogue fix (FP16 WMMA on pre-scaled weights, twice the LDS) is tabled.
- **Qwen3 (0.6B and the rest of the family):** prefill against token-by-token decoding is a bound, not exact (up to 0.4
  logits on Qwen3-0.6B; the top token the same). Exact only with `QMATVEC_NO_WMMA=1`.
- **Gemma-4 26B (gemma4):** prefill is fp32 for Gemma (about 3x slower than the WMMA prefill; see "Done in this round").
  Its streamed and chat checks are open (item 13).
- **Nemotron-3 (Nano, Super):** the reply format is not checked against its template.
- **Llama 3.x:** no Llama 3.1 files here; its built-in tools (`<|python_tag|> brave_search.call(...)`) are not parsed.
- **Mistral:** llama architecture with a SentencePiece tokenizer; untested (an older architecture, tabled).
- **EmbeddingGemma (gemma-embedding):** none open.
- **Qwen2.5-0.5B:** no quantized KV cache (2 x 64 cannot satisfy `n_kv * D % 256 == 0`).
- **DeepSeek-V4 (deepseek4):** not implemented: the engine has no KV compressors, hash routing or grouped output projection.
  Like GLM it does not fit 128 GB resident and would use the expert pager. Its MTP layer is a separate draft file (later).

## Done in this round (kept short; details in git)

- TRELLIS.2 image to textured GLB on the engine: the DINOv3 encoder and image conditioning, the device simplify to 1M faces,
  CuMesh's chart clustering on the device, and the per-chart unwrap (570 s to 17 s on the 512 run's mesh; item 20).
- Gemma-4 prefill keeps fp32: the WMMA matmul and flash attention are off for a Gemma prefill only. Prefill matches
  token-by-token decoding exactly (commit `08d968b`). The cost: 620 tok/s instead of 1840 on a 562-token prompt.
- Quick regression: 47 of 47 (`b4b5b97`: `tests/regress.sh` puts `build/` on `LD_LIBRARY_PATH`).
- Request logs: one file per hour (`f75975c`).
- Earlier: the lazy KV registration (the GLM state and every KV slot); the sliding-window ring with lazy KV (Gemma-4); the
  MTP draft under lazy KV (qwen4exp tiny pair); the `events_test` progress fix; the nginx configuration and fixup; the
  Dockerfile's nginx modules; the Q8_0 predictor's selection (BF16 default); the vision linears (2x); `HIP_CHECK`
  consolidation; the LRU stale-block test; the Q6_K encodes and the Q8_K decode-only check.

## Open leads (speed and quality; not scheduled)

Speed:
- Long-context prefill attention runs at about 25% of the FP16 WMMA peak. The attention kernel alone (`build/attn_bench`,
  1024 tokens at position 31,744): D 256 at 17 TFLOPS, D 128 at 12, D 64 at 13. The lever left is registers (D 256 runs 4
  waves per SIMD at 256 VGPRs). Share of a 32k prefill in attention: Qwen3-0.6B 84%, Gemma-4-26B 67%, Llama-3.2-1B 63%,
  Qwen3.6-35B 35%, Qwen3.5-4B 28%, Nemotron-Nano 25%, Qwen3.8-27B 20%.
- MTP verify: read the experts shared by the verified tokens once (about 19% of a 2-token pass's expert bytes). Draft
  cost 2.7 ms, mostly the head's LM head (a vocabulary cut hurts other scripts; a shortlist from the main model's top
  logits, or an int8 copy).
- Decode leftovers on Flash (each under 5%): `moe_router`, `hc_norm`, `deltanet_rule`, `hc_mix`, about 930 launches of
  about 1.7 us of gaps.
- Prefill: `deltanet_conv` at about 100 GB/s; the quantizing producers compute both Q8_0 and Q8_K though each consumer reads
  one; attn_k and attn_v as two launches (about 1%).
- Small F16 matrices (1024 to 2048 rows) are latency-bound at 55 to 76 GB/s; BF16 FFN down (K 17408) is the slowest shape
  (23 TFLOPS). MoE experts in F16 or BF16 run `matmul_f_id` (no model here has them).

Speech (the open speed items):
- Matvec kernels at 175 to 206 GB/s (the predictor stack: about 905 us against a floor of about 760 at 256 GB/s; fewer
  launches and the samplers' sort are the leads, a few percent each).
- Codec host stages: 36 ms of a 142 ms piece run on the host before the device; overlap them with the device's work or move
  the pre-convolution and the upsamplers to the kernels. The kernel-7 inner loop of the residual units is 63% of the codec.
- Whisper's encoder: the attention allocates a score vector per query and head (about 9000 allocations a call); the
  decoder's cross-attention runs on six threads. Low value: only whisper-tiny is here.
- A streamed codec decoder with the reference's state cached per voice is done (`U6`); the talker's prefill is not a
  share worth a cache (decided).

Quality and correctness:
- K-quants dot Q8_K activations (one scale per 256, as ggml's CPU backend); ggml's GPU backend uses Q8_1 (per 32): about
  1% better perplexity at Q2_K. An option for the K-quant kernels.
- The Q8_0 predictor's activations are read as `float4` from dynamic shared memory, which assumes 16-byte alignment (true
  here, not guaranteed by the language). A toolchain change would show in the predictor's timing and codes first.
- The predictor's Q8_0 draws differ from BF16's in about 1 case of 43 (total variation mean 2.3%). The BF16 default stays.
- One forward_verify run on the 35B hit a GPU memory fault in `matvec_q8_id_q5_K`, never reproduced; likely the test buffer
  overrun fixed on 2026-11-04. Close after a few clean runs.
- The XS module lacks `inference_prefill_logits`, `model_name` and `set_seed` (add when a Perl user needs them).
- Raw completions: `prompt` as an array and `logprobs` are not implemented (400, and always null). Likely not needed (coding
  agents use chat); do them when a client asks. `logprobs` needs the logits out per step (`get_logits` exists).
- GLM's chat template key order: Perl's hash order differs from Python's (`tool.items()` and `|tojson` sort keys), so GLM's
  tool definitions may render differently. Not measured. Fix: keep each object's key order from the JSON parser.

## ComfyUI

`/comfyui` holds the user's model library: 28 model files, about 167 GB (safetensors; all valid). Nothing is wired yet.

- **Z-Image-Turbo, status (2026-11-20).** Headers read (no weights loaded by us): the DiT is 453 BF16 tensors, `layers.N` (30)
  with `attention.qkv` (11520 x 3840: 30 heads of 128 for q, k, v), `attention.out`, per-head `q_norm`/`k_norm`, `adaLN_modulation`
  (256 -> 15360: four 3840 vectors), a SwiGLU feed-forward (`w1`, `w3` 10240 x 3840; `w2` 3840 x 10240), four RMSNorms; two
  `noise_refiner.N` and two `context_refiner.N` blocks of the same shape; `x_embedder` (64 -> 3840: 2 x 2 x 16 patches),
  `t_embedder` (1024), `cap_embedder` (2560 -> 3840), `final_layer`. The text encoder is `text_encoders/qwen_3_4b.safetensors`
  (398 BF16 tensors, HF names `model.layers.N...`, hidden 2560, MLP 9728, 32 q heads, 8 kv heads of 128). The VAE is
  `vae/ae.safetensors` (244 F32 tensors, Flux AE: `encoder.*`, `decoder.*`, 16 latent channels).
  Reference: `diffusers` 0.43.0 (the newest index entry with `ZImagePipeline`) installed in `/data/zimage-lib2` (use
  `PYTHONPATH=/data/zimage-lib2` with the system python: torch 2.9.1 ROCm, transformers 5.5.1); it loads the ComfyUI-format
  DiT with `ZImageTransformer2DModel.from_single_file` (config: patch 2, 16 channels, dim 3840, 30 layers, 2 refiners, 30
  heads, QK-norm, cap width 2560, rope theta 256, t_scale 1000). Not yet: the Qwen3-4B tokenizer (not on this host; the
  text encoder's config and tokenizer come from the Hugging Face repo), the reference image, and the engine's DiT forward.
  Reference run (2026-11-21): `/data/zimage-ref/ref_gen.py` runs the whole pipeline on the ComfyUI files, with the official Qwen3-4B
  config and tokenizer (`/data/zimage-ref/qwen3-4b`, downloaded from Hugging Face) and the Flux AE config written by hand
  (`/data/zimage-ref/flux-ae`: 16 latent channels, scaling 0.3611, shift 0.1159, no quant convs). 256 x 256, 8 steps, guidance 0,
  seed 0, prompt "a red square on a blue background": a red square on blue (`out/image.png`); `out/prompt_embeds.pt` and
  `out/latent.pt` are the checks for the engine. DiT stage oracle (2026-11-21, `/data/zimage-ref/ref_dit_dump.py`): one DiT call on a seeded latent [16, 1, 32, 32], t = 0.5, the
  prompt's embeddings: `dit/stages.pt` holds t_embedder [1, 256], x_embedder [256, 3840] (16 x 16 patches), noise_refiner.0/1
  [1, 256, 3840], cap_embedder [32, 3840], context_refiner.0/1 [1, 32, 3840], layers.0 and layers.29 [1, 288, 3840] (32 text + 256
  image rows), final_layer, and `dit/out.pt` holds the output [16, 1, 32, 32]. Engine port (2026-11-21): the host stages are REFERENCE-ONLY (AGENTS.md, Architecture): `src/zimage_dit.{h,cpp}` (host path: safetensors reader, timestep embedder, patchify, patch embedding,
  caption embedder) and `tests/zimage_dit_verify.cpp` pass against the oracle (relative 0.1-0.24%, cosine > 0.999998; build line in
  Device path (AGENTS.md rules: the mapping registered and read in place, activations in shared memory, no copies): the whole DiT
  forward runs on the GPU and matches the reference's output (`tests/zimage_dit_verify`, the device checks): the timestep, patch and
  caption embeddings, the two noise refiners, the two context refiners, the 30 main layers over the joint sequence (image rows,
  then caption rows, in one buffer), and the output (final layer, unpatchify). Relative errors: 0.1-0.24% for the embeddings,
  about 1% for the refiners, 0.6-0.7% for the layers, 0.3% for the final layer alone; the whole stack's output is 3.3% (cosine
  0.99989) from accumulated bf16 rounding (the reference rounds the same way). Kernels in sd_gpu.hip: sd_rms_norm,
  zimage_modulate, zimage_gate_add, zimage_swiglu, zimage_rope, zimage_layer_norm_plain. The sampler is done (`dev_sample`: 8 flow-matching Euler steps from the reference's noise for seed 0, sigmas linspace(1, 1/8, 8) with a
  static shift 3, the pipeline's negated velocity; 6.3% relative to the reference's final latent, cosine 0.99997, under 5 s on the GPU).
  The shift 3 is verified: the Hugging Face Z-Image-Turbo `scheduler/scheduler_config.json` (FlowMatchEulerDiscreteScheduler,
  `shift` 3.0, `use_dynamic_shifting` false) and diffusers 0.43.0's pipeline (sigmas `linspace(1, 1/steps, steps)`) both match
  `dev_sample`. The Flux AE decoder is done (`src/zimage_vae.cpp`, `dev_vae_decode`: F32 weights read in place, the
  decoder's convolutions via `sd_device`'s `conv2d_f32`, GroupNorm via `group_norm_f32`, the mid attention on tokens: 3e-6 relative
  to the reference decode of the final latent). The text encoder is on the device (`src/zimage_text.{h,cpp}`, `dev_text_encode`:
  Qwen3-4B read in place from the ComfyUI safetensors, 35 layers, the hidden state before the final norm (layer -2 of the
  reference), causal attention; 2.2% relative, cosine 0.999983 against the reference's ids and embeddings). The whole chain from the
  prompt's ids (text encoder, 8-step sampler, Flux AE decode) runs on the device: against `/data/zimage-ref/out/image.png` (8-bit)
  23% of the values match exactly, 79% within one level, mean absolute difference 1.03 levels. The scheduler shift is verified
  (see above). The pipeline is `src/zimage_pipe.{h,cpp}` (`zi_open`, `zi_generate`, `zi_generate_with_noise`): the Qwen3 chat
  template and the tokenizer (the vocabulary and merges from tokenizer.json, passed in by the caller), the three files read in
  place, 256 x 256 only (the DiT's grid is 32 x 32 latents), prompts of at most 32 tokens (the caption length). The verifier
  `tests/zimage_pipe_verify.cpp` (the Makefile's ZIMAGE_PIPE_VERIFY) runs the whole chain with the reference's noise: 22.5% of the
  8-bit values exact, 79.4% within one level, mean absolute difference 1.02 levels; a seeded run is not flat. The server serves it:
  `--zimage DIR` (the ComfyUI layout plus `text_encoders/tokenizer.json`, the Qwen3-4B tokenizer), model `z-image-turbo` on
  `/v1/images/generations` at 256 x 256, listed in `/v1/models` as kind image; `ROCm::Inference::ZImage` in the Perl module and
  `t/68-zimage.t` (env-gated). Not yet: the memory policy (the pipeline stays loaded for the server's life, as MiniMax does; the
  load-per-request and eviction policy from the decisions above is open), other sizes, the server's text chat for the prompt.
- **Z-Image-Turbo** (`diffusion_models/z_image_turbo_bf16`, S3-DiT, 30 blocks, dim 3840, text dim 2560 = Qwen3-4B, which the
  engine runs as `qwen3`); VAE `vae/ae.safetensors` (16 channels, flow matching).
- **SD1.5 family:** `checkpoints/DreamShaper_8_pruned` (LDM key names, 768-wide CLIP-L), `vae/vae-ft-mse-840000-ema-pruned`,
  `diffusion_models/lotus-depth-d-v1-1` (depth output), `loras/nlc_pink`. The engine's sd is SD2.x: a config-driven UNet and
  an LDM key converter are needed.
- **Qwen-Image** (`diffusion_models/qwen_image_fp8_e4m3fn`, MMDiT, 60 blocks, dim 3072, plain E4M3 without scales; the edit
  variant `qwen_image_edit_2509_fp8`); text `text_encoders/qwen_2.5_vl_7b_fp8_scaled`; VAE `vae/qwen_image_vae` (3D causal,
  single images); LoRAs Lightning 4 and 8 steps, Edit Lightning.
  Status (2026-11-22): the reference oracle is done (`/data/zimage-ref/qwenimage/ref_stages.py`, diffusers 0.43.0, the ComfyUI
  files; configs from Hugging Face Qwen/Qwen-Image and Qwen/Qwen2.5-VL-7B-Instruct, the tokenizer in `tokenizer/`). The DiT is
  F8_E4M3 with no scales (loaded as BF16 on the GPU, 1933 of 1933 tensors); the text encoder's FP8 linears are dequantised as
  weight x scale_weight (scale_input is not used); the VAE's Wan key names go through diffusers' `convert_wan_vae_to_diffusers`
  (194 of 194, strict). Stage files in `out/`: te_ids (46 ids; 12 kept after the 34-token drop), te_embeds [12][3584], dit_lat and
  dit_out [256][64] (seed 0, t 500/1000), vae_lat [16][32][32] and vae_dec [3][256][256]. Not yet: the engine's port of any stage.
  Recorded facts: true CFG 4 by default (two DiT passes); the sigmas linspace(1, 1/n, n) with the exponential dynamic shift
  (base 256 -> 0.5, max 8192 -> 0.9, shift_terminal 0.02); RoPE axes (16, 56, 56), theta 10000, scale_rope on (the text
  positions start at max(h/2, w/2)); the timestep is t/1000 with a 1000x sinusoid.
  Status (2026-11-22, all stages on the device; `tests/qwenimage_verify`, all checks pass in one process): the FP8 linears (img_in,
  q_proj: 2e-7); the text encoder (min row cosine 0.99967 against the BF16 oracle); the DiT forward (against the FP32 reference
  streamed block by block: 6e-5 relative, cosine 1.0; the engine\'s joint attention runs in FP32 — the BF16 flash path was 1 % off);
  the VAE decode (0.999999 cosine, 1.2 % relative: its convolutions are BF16 GEMMs that round the activations to BF16); the sampler
  with exact timesteps (against the FP32 sampler reference, `ref_sampler_fp32_stream.py`: 1.2e-4 relative). The BF16 pipeline itself
  Qwen-Image 2.1 (ComfyUI int8 ConvRot files in /comfyui; `src/qwenimage21_dit.cpp`): the DiT forward (32 blocks) against the exact
  dequantised FP32 oracle (`/data/zimage-ref/qwenimage2/ref_dit21_fp32.py`): min row cosine 0.99997, 0.5 % max error; the ConvRot linear
  itself matches the exact quantised layer to 1.7e-6. Found and fixed: the BF16 linears rounded their activations (2 % error in one
  block), so they now use an FP32-activation kernel (`qi21_linear_bf16`). Not yet: the Qwen3-VL 8B text encoder, the 2.1 VAE, the sampler
  (25 steps, `simple`, CFG 1 in the ComfyUI workflow), the server path.
  Qwen3-VL 8B text encoder for 2.1 (`src/qwenimage21_text.cpp`, ComfyUI int8 ConvRot file): the 36-layer language model, last layer
  output before the final norm, system tokens dropped; matches the exact dequantised FP32 oracle (`/data/zimage-ref/qwen3vl`,
  ref_te21_fp32.py) at min row cosine 1.000000, 1.6e-4 relative. Its embedding table is ConvRot too (rotated along the hidden dim).
  The 2.1 VAE decoder (`src/qwenimage21_vae.cpp`, the residual variant, 64 latent channels, 16 x 16 latent tokens for 256 px, 4 output
  channels): matches the diffusers oracle (`/data/zimage-ref/qwenimage2/ref_vae21_fp32.py`, the file's names mapped onto the class) at
  0.13 % (channel cosine 1.000000). The Qwen-Image 1 VAE now also uses the FP32-activation linears: 5e-6 (it was 1.2 %).
  Still to do for 2.1: the sampler (the ComfyUI workflow: 25 steps, `simple`, CFG 1, no shift node: the shift to use is open), the server path.
  (`ref_pipeline.py`) differs from these by 6 to 15 % over 8 steps: it is not the reference for the engine. Not yet: the server path
  (`/v1/images/generations`) and the CFG (the default pipeline run has no negative prompt; true CFG 4 is for the negative case).
- **Flux.2 dev** (`diffusion_models/flux2_dev_fp8mixed`: BF16 and FP8 layers with scales and a quantization table; double and
  single blocks, dim 6144); text `text_encoders/mistral_3_small_flux2_bf16`; VAE `vae/flux2-vae` (32 channels); LoRA
  `Flux_2-Turbo-LoRA`.
- **Utility models:** BiRefNet (`background_removal/birefnet`: done, item 20), MoGe-2 (`geometry_estimation/moge_2_vitl_normal`),
  DINOv3 (`clip_vision/dino_v3_L_naf_fp32`: done, item 20). Also Pixal3D in `diffusion_models` (3D: deferred; TRELLIS.2 is item 20).

Decisions (2026-11-20): the component API and the node package are on hold until the engine supports the models. The node API is
ComfyUI's, not ours: a node is a Python class (`INPUT_TYPES` classmethod, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`; registered
through `NODE_CLASS_MAPPINGS`, or `comfy_entrypoint()` in V3), and nodes pass typed values (MODEL, CLIP, VAE, CONDITIONING, LATENT,
IMAGE) between each other. What is ours is the step set those nodes call (encode text, sample, decode), which the component
API would define. Memory policy: load per request; a new request that needs room evicts the least recently used idle model
(the server's `make_room`). Engine support comes first, starting with Z-Image-Turbo.

Plan (before the decisions above): the component API first (handles for the text encoder, the diffusion model and the VAE that
stay mapped; verified on sd-turbo), then the node package.
 The image models stay BF16 (safetensors is BF16 or F32 for most); GGUF quants only as an
explicit choice; FP8 sources stay FP8 in the engine's FP8 operand path. Open: the UMA budget policy (load per request, or
keep the last model); ComfyUI's node JSON; diffusers 0.35.2 has no Z-Image or Flux.2 pipeline, so those references need a
newer diffusers or ComfyUI on CPU.

## Decisions by the user

- 2026-11-16: the speech pieces keep their own prosody. ffmpeg is in the image. The Q8_0 predictor is an option (BF16 default).
- 2026-11-16: Gemma-4's prefill keeps fp32 (the 3x prefill cost accepted).
- 2026-11-16: request logs roll over each hour; the user deletes old files.
- 2026-11-16: requested: KV snapshots during prefill, every 500 tokens, opt-in for the large CPU-only models. The design
  is open (item 2).
- 2026-11-15: audio closed as functionally complete; the talker's and codec's device paths are the default.
- 2026-11-14: the image models load from the original checkpoints, not GGUF conversions; LLM safetensors loading is left as is.
- 2026-11-14: the DiT's BF16 stays the default; the mixed Q8_0 recipe stays supported as an option.
- 2026-11-12: GLM-5.3's speed work is tabled (4.3-7.6 tok/s is fine).
- 2026-11-07: a conversation's KV persists; one conversation's cache is not cleared by another's.
- 2026-11-04: the older architectures (gpt-oss, deepseek2, glm4, phi3, gemma2 / gemma3, Mistral) are tabled.

## Tabled

- GLM-5.3-Flash speed work (see above).
- Older architectures: gpt-oss, deepseek2, glm4, phi3, gemma2 / gemma3, Mistral.
- The ROCmFP4_FAST WMMA kernel work ("move on, table this for now", 2026-11-04).
- A CPAN release of ROCm::Inference, Minijinja and Web::Route (the Docker image builds them from source).
- A disk tier for the KV store.

## Tried and rejected (measured; do not repeat without a new idea)

- Prefill attention with 32-position blocks (2026-11-12): slower on every shape (D 256 41.7 / 42.7 ms against 31.7 / 32.6, D 128
  28.3 against 22.8, D 64 22.7 against 21.0). The same code with 16-position blocks already loses 8 to 19% on D 256 from the
  aliasing; the Q row padding is neutral.
- GPU reads of an unregistered mapping (2026-11-11, with `amdgpu.noretry=0`): a kernel reading unregistered pages that are in
  memory hangs the GPU (a full reset after about 20 s, no page fault logged). Registration is not optional.
- HIP graphs for decode launch gaps: about 0.3% (1.96 vs 1.88 us per dependent kernel).
- Small-kernel merges (moe_combine into add_rmsnorm: slower, each workgroup recomputes the expert sum).
- WMMA operand pipelining in the FAST kernel: 10% slower (173 VGPRs); bigger dense WMMA tiles, 2 tiles per wave, 32 weight
  rows per wave: no gain.
- Decode attention on a Q8_0 cache: 16-byte K pieces, 4 V dims per thread, 128-position chunks: no gain.
- Flash prefill prefetch of the next K/V block at D 256: spills (13 VGPRs), -5%; kept for D <= 128 only.
- MTP vocabulary shortlist: +2 to 5% on English, -18% on Chinese (tokens spread over the vocabulary).
- moe_router: per-wave argmax top-k instead of the bitonic sort (15.3 vs 14.3 us); 8 waves per router row.
- Unrolling the FAST matvec quad loop by 2; fewer lanes per row for the Q8_0 matvec (32 is best).
- IQ WMMA staging prefetch; a K rotation per workgroup in the expert kernels.
- Loads inside branches or `c ? load : 0` in kernels: they serialize (keep loads unconditional, select after).
- Speech: 32 input channels per chunk for the codec's kernel 7 (302 ms against 282: 52 KB of shared memory a block);
  decoding only the last 25 reference frames as codec context (waveform changes, cosine 0.973).
- Vision: a 4x4 register block (the OpenMP reduction over array elements gave wrong results); a 4x2 blocked linear and a
  per-pair attention buffer on Whisper (no measurable gain; reverted).
