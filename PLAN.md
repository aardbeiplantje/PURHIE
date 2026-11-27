# Project Plan: rocm-hip (Strix Halo, gfx1151)

State as of 2026-11-26. This file holds open work only: the next steps, the open items with what is known about them, the
leads, the decisions that still matter, and the measured dead ends. Finished work is in git (the commit messages carry the
numbers and the checks; this file before the 2026-11-26 pruning is PLAN.md at `7b85ea5`). Design: ARCHITECTURE.md.

How the work runs: each open item is a package with its check. The main session briefs it, reviews the commit, reruns the
check on `main`, commits, pushes, and (AGENTS.md) touches `.build-trigger` when the image needs a build. Timing packages
run alone on the machine.

## ▶ Next

1. **Qwen-Image 2.1: the image edit, its speed** (item 16; the engine's edit is verified, 2026-11-27: `qi21_edit`, the
   stages against `tools/qwenimage21_edit_oracle.py`, on the BF16 file too; served as `POST /v1/images/edits` (2026-11-27,
   checked by `t/73-qwenimage21.t`), and aicli's `generate_image` sends a picture of the workspace as `image`). A client that
   leaves cancels its image (`t/75-qwenimage21-cancel.t`: the next image is accepted 4 s after the client left, where 40 steps take
   about 30 s): `qi21_cancel`, checked at each step, the decoder not interrupted.
2. **Waiting on the user:** the deployed service check (item 1); the design and the row format of the GLM prefill
   snapshots (item 2).

## Status in brief

- **Engine:** runs qwen2, qwen3, qwen3moe, qwen35 / qwen35moe, qwen4exp (Qwen3.8-Flash-Next), gemma4, gemma-embedding,
  nemotron_h_moe, llama and glm5-next (GLM-5.3-Flash, experts paged from disk) from zero-copy GGUF mappings on the UMA.
- **Images, audio, 3D on the engine:** sd-turbo; Z-Image-Turbo (1024 x 1024, nine steps in about 24 s of DiT and VAE;
  item 17); Qwen-Image 2.1 (RGBA, 1024 x 1024 at 40 steps in about 127 s; item 16); MiniMax-H3-Image; AudioLDM 2; Qwen3-TTS (speech) and Whisper tiny (transcription); vision towers for chat
  models; TRELLIS.2 image to 3D (textured GLB, the mesh alone, STL for printing, solid or hollow) with BiRefNet for
  background removal. On the engine but not served: Qwen-Image 1 (item 16).
- **Surfaces:** the C API (`libinference.so`), the Perl XS module ROCm::Inference (Minijinja templates),
  `rocm-inference-server` (OpenAI-compatible: chat, raw completions, embeddings, audio, images, 3D jobs, `--models-dir`,
  `--parallel`), and `bin/rocm-convert`. The AI harness (`./aicli.git`, its own repository) has tools for each generator.
- **Deployment:** the image is built from this repository (`docker-bake.hcl`) from GitHub `main`; the user's watcher
  restarts the container from each new image; a build starts when `.build-trigger` is touched. Last build: 2026-11-26
  from `7b85ea5`.
- **Tests:** `make regress` (quick tier) passed 47 of 47 on 2026-11-16 (not rerun since); `make regress-full` has not
  been run. The 3D server test `t/72-trellis2.t`: 53 of 53 (2026-11-26).

## Open items

1. **Deployed service check (on the user's host).** `GET /rocm/v1/models`, one short chat through `/rocm/` (the fixup:
   `conv_id` and `id_slot` added), one short speech request. The 3D route works through nginx and the harness (the user,
   2026-11-26). This session cannot reach the service.

2. **KV snapshots during prompt processing, for the large CPU-only models (GLM first, then DeepSeek).** Wanted: snapshots
   written during prefill, about every 500 tokens, so that a restart does not lose a 10k-token prompt (5 minutes or more on
   GLM CPU-only). Opt-in per model, default off.
   - **The gap:** the block store holds only a one-layer stand-in KV cache for GLM. The real per-position state (MLA
     latents, indexer keys, gates, pooled rows, in `GlmState`'s `LazyRows`) and the KDA recurrent state are never written.
     A GLM resume from the store cannot work at any snapshot frequency until they are saved.
   - **Possible wrong output (not confirmed):** `store_lookup` treats a stored block as valid whenever the model has no
     recurrent state in `rs.mem`, which holds for GLM. A GLM resume could then reuse stored blocks without their state.
     Needs a test that goes through the sequence path (a repro through `inference_generate` never reached the lookup,
     even on Qwen3-0.6B).
   - **Proposed design:** `--store-state-every 500` for the opted-in models. During prefill the prompt is cut at each
     500-token boundary (as the recurrent snapshots already are). At each boundary, write the new rows of the span (latent,
     ik, ig, and the pooled rows) and the KDA state. Resume restores both. Per-token size: fp32 rows of `kv_lora` plus
     two `ix_dim` values per MLA layer (the exact figure read from the GGUF before the format is fixed). Also possible:
     F16 rows (half the bytes, a quality check needed).
   - **Decision needed:** go ahead with the design, and the F32 or F16 row format.

3. **GLM-5.3 on the CPU in the deployment (user).** Measure tokens per second and `VmLck`, `RssAnon`, `RssFile`, `RssShmem`
   during a real conversation. Last measured: 2.9 tok/s CPU-only. Wait for item 2: a container restart resets the prefill.

4. **`tests/convert_verify.sh` in full.** Blocked on disk: `/data` has 20 GB free (2026-11-26); the run needs several GB.

5. **The full regression tier (`make regress-full`).** It has perplexity rows for models not run here (Flash, the 122B and
   120B files) that need `tests/regress.sh --update` on those rows.

6. **ComfyUI backend: the component API and the node package** (on hold until the engine supports the models; see
   "ComfyUI").

7. **MiniMax-H3-Image:**
   - Check the images against the released pipeline (the timestep convention, t = 1 - sigma; the PDD heads' role). The
     engine's arithmetic is checked (the reference chain matches), but whether the released model is reproduced is not.
   - FP8 as a weight type: not measured (the engine has no FP8 weight type to write; needs an encoder and its operand path).
   - DiT drift from block 30 (the engine's error grows against the fp32 chain). Not chased.
   - The text encoder from the original checkpoint (the Qwen3 LLM through the qwen loader needs the same tensor source
     as the DiT). The user chose the VL encoder.

8. **Alpamayo 2 Edge.** Licence confirmed (OpenMDW-1.1 weights, Apache-2.0 code). The modeling code is at
   `NVlabs/alpamayo2-edge`, outside this session's GitHub scope: blocked until the user attaches the repository. The weight
   headers are read (the language model: 28 layers, hidden 2048, 16 query / 8 KV heads of 128, `relu2` MLP, M-RoPE
   `[24, 20, 20]`, vocab 135168; the vision tower: 27 layers, hidden 1152, 16 heads, learned absolute positions; the action
   expert: 28 layers, hidden 512, non-causal attention over the VLM's keys and values).

9. **Concurrency (`--parallel`).**
   - Speculative decoding under `--parallel` (not started; three stages): per-sequence draft-head state and batched
     forward with single-token rows (a bit-for-bit check against the sequential path); verify rows (k+1 per sequence,
     tested against `forward_verify`); per-sequence rollback, and the server's loop with drafts per job. Blocked in the
     engine: `forward_seqs` refuses a model with a draft head, and the head's state is one per model. Measured on
     Qwen3.5-4B (2026-11-18): one sequence 15.8 ms a step, two 23.7 ms (1.33x together); a batched forward of 1 to 4 rows
     has about 15 ms of fixed cost, then 1.1 to 2.2 ms a row. The verify pass's cost is not measured (no MTP head for the 4B
     here), so the break-even is open. The sequential sidecar gave about 38 -> 50 tok/s.
   - Watch on the deployment: the store's resume (the `kv` line's mapped blocks), and `--parallel` on the 27B and 35B.
     `--store-max` 32 or more (one 130k-token 27B conversation is about 14 GB).
   - The 10 identical resends in opencode's log (5 each before two compactions, at about 210k tokens) look like failed
     requests whose replies the log lacks.
   - Not done: a disk tier for the store; the store on by default in `run.sh`. Later: `--parallel` by default; several
     models generating at once; loads off the request path (a request during a cold load waits minutes); fairness and
     queue limits per user; llama-server on the same GPU.

10. **Server identity and logging.** With `--parallel`, the done line's store counters (the `kv` line) are the model's over
    the request, so another job's blocks can show in them. Identity: the user (nginx `X-Remote-User`; the harness's login
    token later, item 15 i), the client address (`X-Real-IP`), the conversation id (`X-LLamaCPP-Id-slot`).

11. **Flash (Qwen3.8-Flash-Next) measurements, in a deployment window** (the model file is 122.8 GB and shares the host's
    124 GB with the deployment):
    - The model's own MTP block as draft head: acceptance and tok/s (`build/mtp_verify F F 128`). If the own block's
      acceptance is worse than the sidecar's, check the base file's `nextn.hc_head_norm` (the sidecar's `output_hc_norm` +
      1.0; the base file is likely right).
    - PLE rows in parallel, in the deployment (`PLE_THREADS`, `MADV_RANDOM`): 30k-token prompts were 400-480 tok/s with
      cold rows read one at a time.
    - `events_test MODEL DRAFT` with the Flash draft (not run).
    - YaRN beyond 262,144 positions (the card reaches 1M by a `factor` 4.0 override): not started (frequencies, attention
      scale, a `--rope-factor` option, a quality check). The 16 GB KV budget caps it at about 600k positions at F16.
    - The fork's qwen4exp is behind upstream's (`recurrent_layers` as indices, not bool flags): the user's to port.

12. **Watch or table:** Qwen4 (not on Hugging Face yet); Qwen3.8-2.4T-A95B (tabled); the memory measurement of the
    deployment's GPU pins (tabled: GLM runs on the CPU).

13. **The Perl suite on the 122B.** `t/10` and `t/15` fail with `hipHostRegister failed (-3)` on an in-process load of the
    122B (the server's load of the same file works); `t/65-models-dir` hung once on a server (killed); `t/70-stage` ran for
    hours (killed). Cause (2026-11-27, this container): a process can hold about 29 GiB registered for the device, and
    `hipHostRegister` answers "invalid argument (-3)" past it. The 122B file is 67 GB; with `--stage` the registered stage is
    66 GB (`/models/combined-test/Qwen3.5-122B-A10B-FAST.gguf`: the GPU reads nearly all of it), so the load fails there too.
    Without staging, `t/10` and `t/15` register the whole map at once. The server works on the user's host, so its limit is
    higher than this container's: the source of the limit was not found here (not the TTM page limit, which is 128 GiB; not
    the cgroup, which has no limit; not `ulimit -l`). To check on a host: `reg_limit`-style, one 1 GB chunk at a time until
    the first `invalid argument`. Tests now take `ROCM_INFERENCE_TEST_STAGE` (a tmpfs directory) for the stage option.
14. **`--parallel 2` and a second big model (122B).** A server under `--parallel 2` that loads one 122B-sized file and then
    another (alpha, beta, alpha under `--models-max 1`) fails at beta's load: `hipHostRegister of 66353926592 bytes (a GGUF
    map) failed: invalid argument (-3)`. `--parallel 1` passes, in one process and across two; so do a single 122B load and
    Flash's two loads. Ruled out: the KV size, the second sequence's logits registration and its recurrent-state
    registration. Not found: what `--parallel 2` changes at load time. The host rule (ARCHITECTURE.md section 11) keeps
    such models at `parallel` 1 until then.

15. **TRELLIS.2 (image to 3D): served; what is open.** The pipeline and its checks are in ARCHITECTURE.md section 12. Last
    numbers: the vase at 512, about 131 s for an STL and 200 s textured; the 1024 cascade about 814 s of stages; the
    solid rebuild 0.7 to 2 s and its simplification 2 to 6 s; the hollow vase at 2 mm walls 31% of the solid's material.
    The user found the engine's 1024 vase about the same as ComfyUI's, if anything sharper.
    - (b) the 1536 cascade the ComfyUI workflow runs (memory is no longer the limit; its time is).
    - (c) the normal-map and AO bakes and their GLB inputs.
    - (d) UnwrapMesh's "pec" segmenter (ours is CuMesh's charts).
    - (e) printing, as the user decided (2026-11-26): a solid request stays solid (the model's own thin sheets are not
      thickened: about 1.2 grid cells, 0.2 mm at 100 mm); a hollow request has walls of at least 2 mm; the model's own
      hollows are kept (open ones through openings wider than about 1.6% of the size, sealed ones by the even wall count;
      tiny ones under min_piece are filled). Crevices narrower than about 1.6% of the size are filled. Not checked on a
      real model with a designed cavity (a whistle): the example's 12 sealed hollows were all tiny.
    - (f) the simplification stops below its target (667k for 700k; python's stops at it).
    - (g) the unwrap (74 s on the vase: xatlas per chart, single-threaded) and the host sorts, to move to the device or
      threads.
    - (h) RemeshMesh (udf, 768) only if wanted: the user prefers the sharper result without it.
    - (i) access by the harness's login (waits on the user: when this project joins the AI harness, `./aicli.git`, or its
      token is accepted by purhie.aardbeiplantje.link). The harness's auth service (`aicli.git/auth.pl`) signs an RS256 JWT
      (cookie `aicli_jwt` or a Bearer token; claims `sub`, `roles`, `acct`: the account's uuid, `sid`, `exp`, `aud` for API
      tokens) and its nginx checks it with `auth_request` to `/auth/verify` (200 with `X-User`, `X-Roles`, `X-Account`).
      Then: this nginx checks the token the same way (or with the harness's public key), sets `X-Remote-User` from
      `X-User` and drops the client's; the models go to `/scratch/<acct>/<job>/model.glb`; `/rocm-files/<acct>/...` is served
      only to a token whose `acct` is that directory. Until then: random job ids, jobs per `X-Remote-User`, the internal
      files location (the identity is only as good as the proxy that sets the header).
    - The solid rebuild's grid stays at 512 for both resolutions: at 1024 the fill's `close` (in cells) let the vase leak,
      and the 26M-face simplification was slow.
    - Lost in the 2026-11-26 `git clean`: `comfy/example.png` (the TRELLIS example input of `tests/image_cond_verify.cpp`
      and `tools/dinov3_oracle.py`) and `comfy/zimage_mve_0000*_.png` (the ComfyUI reference images); the user may have
      them in ComfyUI's input and output folders. The oracle dumps in `/data` are intact.

16. **Qwen-Image (1 and 2.1) and the 2.1 image edit.** All stages of both run on the device and pass their oracles (see
    "ComfyUI"); neither is served. Qwen-Image 1: the server path and the CFG (true CFG 4 needs a negative prompt; the
    default pipeline run has none). Qwen-Image 2.1, the pipeline (2026-11-26, `src/qwenimage21_pipe.{h,cpp}`, the C API
    `qi21_*`): the t2i template with its system turn (14 tokens dropped), the Qwen tokenizer (identical to Qwen3's), the
    schedule of the model's scheduler_config.json (Qwen/Qwen-Image-2.1 on Hugging Face: the same as Qwen-Image 1's; no
    author sigmas, so linspace; the pipeline's 40 steps, no guidance), torch's noise for the seed, the DiT and the VAE at any
    size (multiples of 32 from 256 to 2048), RGBA out. `tests/qwenimage21_pipe_verify.cpp` against
    `tools/qwenimage21_oracle.py` (diffusers 0.43.0 in FP32 on the dequantised ComfyUI files; outputs in
    `/data/zimage-ref/qwenimage2/pipe`: 512 x 512, 8 steps, seed 0, "a red cube on a wooden table, soft studio light"): the
    ids exact, the embeddings 3e-5, the sigmas 6e-8, the noise 5e-7, the first velocity 7.5e-4, the final latent 1.6%, the
    decoder 2.3e-3 (its BF16 route, below; 6e-7 in FP32); the image against diffusers' PSNR 47.3 dB (mean 0.40 levels). The BF16 file (the same names) runs a BF16
    chain as Z-Image's DiT: hipBLASLt with BF16 outputs (`sddev::linear_bf16_via16`: the FP32-output plan's first candidate is
    a slow algorithm, about 5.5 TFLOPS, and the tuner keeps the first candidate's numerics; the BF16 output's is fast, 22-26
    TFLOPS with the conversions) and the BF16 flash attention; against its own FP32 reference (`pipe_bf16`) 40.1 dB. The int8
    file keeps the FP32 attention (47.6 dB; BF16 attention gave 33.7). The BF16 chain between the ops (2026-11-26, as Z-Image's:
    the residual FP32; `qi21_ln_mod_bf16` writes the normed, modulated input once in BF16 for q, k, v and for the MLP's two
    products; the products BF16 in and out, `linear_bf16_xb16`; `head_norm_rope_bf16`, `swiglu_bf16_in`, `qi21_gated_add_bf16`;
    the products' rows rounded up to 64, so prompt lengths within 64 tokens share hipBLASLt's plans: a new token count tunes
    about 100 candidates, about 2.5 s at 1024; 256 cost more in rows than it saved): against its FP32 reference 46.8 dB.
    Speed: 1024 x 1024, 40 steps, 130 s (was 287 with FP32 outputs and attention, 156 with BF16 outputs alone; another
    process held the GPU at about 16% in that run): about 3.0 s a step, the linears about 2.0 s at hipBLASLt's best (25 to
    30 TFLOPS: every BF16-output candidate has the same numerics, so the tuner picks the fastest), the attention 0.6; the
    text encoder 1.2 s once; 512 x 512, 8 steps, 6.3 s. The decoder's BF16 route (2026-11-26): the residual stream FP32, the
    convolutions' inputs BF16 (`qv_rms16`: the RMS norm and SiLU rounded once, bit-identical to the FP32 ops then the rounding),
    the 3 x 3 patches built in BF16 through shared memory (`qv_patch16`, the upsampler's nearest 2x read through, never written),
    hipBLASLt's BF16 products with BF16 outputs for any shape it takes (`sddev::linear_blt16`: the 144- and 288-channel layers
    too), widened with the bias and the residual (`qv_from16_bias`); the rows per product fixed by the patch width (64 M values,
    the last product overlapping its neighbour), so the larger layers keep one tuned shape at every image size. 1024 x 1024:
    1.36 s (was 7.1 s; the first decode tunes about 3 s more); 40 steps end to end 127 s. Against the reference 46.6 dB (BF16
    file) and 47.3 dB (int8). Served (2026-11-26):
    `qwen-image-2.1` on `/v1/images/generations` (`--qwen-image DIR`, else the `--zimage` directory when it has the files;
    a unit of the memory policy, about 24 GB, loaded by its first request), the image made on a background thread (`qi21_start`,
    `qi21_poll`, `qi21_take`) while the answer waits in a pull response; 429 for a second image, 503 for the other device
    generators meanwhile; `t/73-qwenimage21.t` (17 checks). The harness's `generate_image` takes the model (aicli `f5a6efa`).
    Open: the decoder's convolutions on hipBLASLt (about 7 s at 1024: `linear_bf16_f32`; Z-Image's VAE runs BF16 im2col and
    calibrated algorithms); the text tokens' keys and values could be computed once (`causal_condition`; a few tokens of
    thousands: small); the ComfyUI workflow runs 25 steps with `simple` and CFG 1 (its own sigmas: not checked).
    The image edit (2026-11-27): the condition image goes through the Qwen3-VL vision tower (`src/qwenimage21_vision.cpp`, all
    in FP32 with the BF16 weights read in place: a BF16 chain compounded to 10% over the 27 blocks), the text encoder with the
    image pads (`dev_text21_encode_vl`: the M-RoPE positions, the deepstack features after layers 0 to 2), the VAE encoder
    (`dev_vae21_encode`: conv_in on BF16 weights in place, the residual down blocks with the average-down shortcut), and the DiT
    with the condition's tokens (`Dit21`: the prefix of text and condition keys and values once, each step the target's rows).
    Image pixels: `condition_pixels` (composited over white), the resize of `resize_condition` in Pillow's fixed point
    (`trellis::resize_pillow8`: exact against Pillow 13). `qi21_edit`, `qi21_start_edit`, `qi21_edit_size`. The check
    `tests/qwenimage21_edit_verify.cpp` (512 x 512 from the red cube, 8 steps): the vision tower 2.5e-5, the text embeddings
    2.5e-5, the condition's latent 6.5e-3 (BF16 weights; cosine 0.9988), the first velocity 3e-4, the final latent 4e-4, the
    decoder 2.3e-3, the whole edit 54.8 dB against the diffusers' image (every pixel within 3 levels). The text-to-image check
    still passes (47.3 dB). The sizes from `calculate_dimensions` and the resize match Pillow exactly.
   On the BF16 file (what the server loads when it is present; oracle in `/data/zimage-ref/qwenimage2/edit_bf16`, the check
   `qwenimage21_edit_verify DIR IMAGE DIT`): every stage passes too: the first velocity 2.4e-3, the final latent 3.0e-3, the
   whole edit 54.6 dB; the sampler takes 7.3 s at 512 x 512 (the int8 file: 68 s).

17. **Z-Image-Turbo: what is open.** Served (`--zimage DIR`, `z-image-turbo` on `/v1/images/generations`, steps 9 by
    default, sizes in multiples of 16 from 64 to 1024, 1024 x 1024 by default). Nine steps at 1024: 24.25 s of DiT and VAE
    (the DiT's linears on hipBLASLt with BF16 outputs, the VAE's convolutions as BF16 hipBLASLt products, the algorithms of
    the 1024 shapes from the calibrated table `src/blt_calib.h`); about 41 s end to end through the Perl module with the
    model's load. Against ComfyUI's own output (seed 42, the noise is torch's CPU randn, `src/torch_randn.h`): PSNR 27.45
    dB. The user's ComfyUI run takes 26 s.
    - The chain's remaining cost: flash attention 0.43 s a step, the attention's FP32 output converted to BF16, the
      modulation; the linear plans are not calibrated (the rule's first candidate is already the fastest class there); the
      table covers 1024 x 1024 only (other sizes take the same-bits rule).
    - The warm end-to-end time in the server is not measured against the 30 s target (the model's load dominates the 41 s).
    - Not checked: the 1024 output against diffusers at nine steps (the reference's VAE decode stalled the desktop's
      graphics ring at 1024: run it only when the desktop is idle).
    - Unexplained: the BF16 chain and the `SD_ROUND_OUT=1` experiment should match and differ by 1.1 dB (likely the
      algorithm the tuner picked, not proved).
    - `sd_gemm_wmma128` faults on the VAE's conv shape (K 1152, cs_m 1, cs_n = HW): not used, not investigated.
    - Not yet: the memory policy for the pipeline (it stays loaded for the server's life).

18. **FLUX.2 dev (image generation), not started.** The weights are on disk (see "ComfyUI"). Needs: the DiT on the engine
    (the FP8 layers stay FP8 in the FP8 operand path), the text encoder, the 32-channel VAE decoder, a stage check against a
    python oracle, and a server route. The oracle needs a diffusers build with Flux.2 (0.35.2 has none) or ComfyUI on CPU.
    Open: the sampler's steps and the LoRA's role.

19. **CAD and PCB design (investigated 2026-11-26, not started; the user's question after the whistle case).** A working
    object (a whistle: a windway, a chamber, an edge) is CAD, not a generated mesh. The field's 2026 answer: a model writes
    CAD code (CadQuery or build123d Python, OpenSCAD) that a CAD kernel (OpenCascade) runs into an exact, editable B-rep, then
    STEP or STL; for boards, circuit code (SKiDL, atopile, tscircuit) into KiCad, with placement and routing by tools.
    - Open-weight CAD models: cadrille (ICLR 2026, Apache-2.0; Qwen2-VL-2B; a point cloud, views or text to CadQuery;
      `maksimko123/cadrille`, `-rl`), CAD-Recode (ICCV 2025; Qwen2-1.5B + a point projector; points to CadQuery; CC BY-NC),
      CAD-Coder (MIT; LLaVA-1.5 type; an image to CadQuery; GenCAD-Code, 163k pairs). Text to CadQuery fine-tunes up to 7B;
      CADFusion (Microsoft, research); STEP-LLM (text to STEP; code public, weights unclear). They know the datasets'
      mechanical parts (sketch and extrude, holes, fillets), not organic shapes (TRELLIS's).
    - Blender Python (`bpy`, meshes rather than CAD: free-form and organic shapes, materials, scenes): BlenderLLM
      (FreedomIntelligence; Qwen2.5-Coder-7B fine-tuned on BlendNet; weights' licence to check), MeshCoder (NeurIPS 2025;
      a point cloud through a triplane tokenizer to a Llama-3.2-1B LoRA writing part-based Blender code; release not
      confirmed), LL3M (no trained model: planner, Blender-doc retrieval, coder and a critic that renders and fixes, over
      stock LLMs), and blender-mcp (MIT: an LLM drives Blender over MCP; it runs the generated code unguarded, so a sandbox
      is needed). Fits the same harness tool shape as (a): Blender headless (`blender -b --python`) in the sandbox, the
      render back to the vision model.
    - PCB: no open model designs a board. PCBSchemaGen (a frozen LLM writes SKiDL, benchmarks on GitHub), SchGen (fine-tuned,
      82% valid circuits; weights not confirmed), PCBRouteAgent (routing by agents; large boards still fail). Placement and
      routing: KiCad, Freerouting (open), Quilter and DeepPCB (commercial). Konnect: KiCad tools for an LLM over MCP.
    - The user agrees with this analysis (2026-11-26).
    - Options: (a) tools in the harness (aicli, the user's to approve): a `cad` tool running OpenSCAD or CadQuery code in the
      sandbox to STL/STEP plus a rendered view for the vision model to check, a `blender` tool the same way for meshes, and
      SKiDL / kicad-cli / Freerouting for boards;
      the local chat models write the code (no new model on the engine). (b) cadrille on the engine (Qwen2-VL-2B and a point
      encoder): an image, or TRELLIS's mesh as points, to editable CadQuery. (c) CAD-Recode (non-commercial licence).

## Audio, vision and image: what is open per feature

| Feature | Open |
|---|---|
| **Speech output** (`/v1/audio/speech`, Qwen3-TTS 0.6B) | Pieces do not share prosody (decided: kept); a streamed WAV's header is not a finished file; the sampled frames are not torch's (its generator); one request at a time per talker |
| **Speech input** (`/v1/audio/transcriptions`, Whisper tiny) | The cut between windows is a pause, not a timestamp; languages other than English are checked only on the token table (plus one French clip); the GPU path (the host runs it: about 0.4 s for the encoder per window) |
| **Text to audio, AudioLDM 2** | One clip at a time (about 48 s for 5 s at 100 steps); only that setting is checked against diffusers; the T5 tokenizer is the engine's own unigram Viterbi, not SentencePiece |
| **Vision** (`image_url` in chat) | The tower runs on the host in float32 (a GPU path would save at most 0.5 s an image: not started); the 4B is not checked; the store does not share image sequences |
| **sd-turbo** | One image at a time; the seeded noise is not torch's; only the one-step sampler |
| **MiniMax-H3-Image** | Item 7 |
| **Z-Image-Turbo, Qwen-Image, TRELLIS.2** | Items 17, 16, 15 |

Not on this host, or not checked: the 4B vision tower; the Whisper GPU path (only whisper-tiny is here); ffmpeg's encoders
on any host other than the image.

## Per-model gaps

Not re-tested for this summary. A blank means no open item is recorded.

- **GLM-5.3-Flash (glm5-next):** no store resume for GLM's state (item 2). CPU-only speed 2.9 tok/s (earlier); the
  deployment is unmeasured (item 3); memory at a long conversation is unmeasured. The MTP block (blk.45) is not bound as a
  draft head (the index is shared, so a verify's rollback must roll it back too). Sparse MLA prefill is about 40% of a long
  prefill and is not on WMMA. Batched sequences are not supported. Its vision track (`glm5_next_vision`) is not started.
  Speed work tabled (user: 4.3-7.6 tok/s is fine).
- **Qwen3.8-Flash-Next (qwen4exp):** item 11. Its `--parallel` and store checks are on the deployment (item 9).
- **Qwen3.5 (0.8B, 4B, 122B):** vision on the host only; the 4B vision not checked.
- **Qwen3.6-35B:** the full regression's perplexity baselines are not recorded (item 5). The MoE decode streams its experts at
  about 125 GB/s against about 200 for dense layers (latency-bound launches). Its Q4_0 forward_verify drifts to cos 0.989 at
  layer 21 (smooth; the next token equals llama.cpp's); the suite's row uses min cos 0.985.
- **Qwen3.8-27B (dense, ROCmFP4_FAST):** compute-bound on `matmul_wmma_q4_0_rocmfp4_fast` (about 26 of 59 TOPS, 70% of a 32k
  prefill). The epilogue fix (FP16 WMMA on pre-scaled weights, twice the LDS) is tabled.
- **Qwen3 (0.6B and the rest of the family):** prefill against token-by-token decoding is a bound, not exact (up to 0.4
  logits on Qwen3-0.6B; the top token the same). Exact only with `QMATVEC_NO_WMMA=1`.
- **Gemma-4 26B (gemma4):** prefill is fp32 for Gemma (about 3x slower than the WMMA prefill; the user's decision).
- **Nemotron-3 (Nano, Super):** the reply format is not checked against its template.
- **Llama 3.x:** no Llama 3.1 files here; its built-in tools (`<|python_tag|> brave_search.call(...)`) are not parsed.
- **Mistral:** llama architecture with a SentencePiece tokenizer; untested (an older architecture, tabled).
- **EmbeddingGemma (gemma-embedding):** none open.
- **Qwen2.5-0.5B:** no quantized KV cache (2 x 64 cannot satisfy `n_kv * D % 256 == 0`).
- **DeepSeek-V4 (deepseek4):** not implemented: the engine has no KV compressors, hash routing or grouped output projection.
  Like GLM it does not fit 128 GB resident and would use the expert pager. Its MTP layer is a separate draft file (later).

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
- hipBLASLt's FP32-output plans (2026-11-26): the first candidate (the tuner's numerics) is a slow algorithm, about 5.5 TFLOPS on
  the DiT shapes, where the faster ones have other numerics; `sd_gemm_bench` measures this path (5.5 TFLOPS now, 25 when the
  timing chose). Z-Image's DiT and Qwen-Image 2.1's BF16 file use BF16 outputs; any other FP32-output large product (the VAEs'
  convolutions are calibrated, `blt_calib.h`) is worth checking.
- Qwen-Image 2.1's decoder (1.36 s at 1024): the products about 0.6 s (about 25 TFLOPS), the BF16 patches 0.38 s (about 49 GB
  written and read again at 1024), the norms 0.16, the widening 0.13. An implicit-GEMM convolution (the taps read from the
  activations, no patch matrix) would save most of the patches' share. Qwen-Image 1's VAE (`src/qwenimage_vae.cpp`) still
  takes the FP32 route (`qi21_linear_bf16`, about 3 TFLOPS); the same BF16 route applies when it is served.
- Z-Image's MLP-down shape: its weights have a 20 KB row stride (read in place, fixed); the activation copy is ours to lay
  out. The engine's own `sd_gemm_wmma256b` reached 19-20 TFLOPS on QKV and MLP-up and 9-13 on MLP-down (58-63% of the wave
  cycles waiting on vector memory, L2 hit rate 3%); hipBLASLt runs these now.

Speech:
- Matvec kernels at 175 to 206 GB/s (the predictor stack: about 905 us against a floor of about 760 at 256 GB/s; fewer
  launches and the samplers' sort are the leads, a few percent each).
- Codec host stages: 36 ms of a 142 ms piece run on the host before the device; overlap them with the device's work or move
  the pre-convolution and the upsamplers to the kernels. The kernel-7 inner loop of the residual units is 63% of the codec.
- Whisper's encoder: the attention allocates a score vector per query and head (about 9000 allocations a call); the
  decoder's cross-attention runs on six threads. Low value: only whisper-tiny is here.

Quality and correctness:
- K-quants dot Q8_K activations (one scale per 256, as ggml's CPU backend); ggml's GPU backend uses Q8_1 (per 32): about
  1% better perplexity at Q2_K. An option for the K-quant kernels.
- The Q8_0 predictor's activations are read as `float4` from dynamic shared memory, which assumes 16-byte alignment (true
  here, not guaranteed by the language). A toolchain change would show in the predictor's timing and codes first.
- The predictor's Q8_0 draws differ from BF16's in about 1 case of 43 (total variation mean 2.3%). The BF16 default stays.
- One forward_verify run on the 35B hit a GPU memory fault in `matvec_q8_id_q5_K`, never reproduced; likely the test buffer
  overrun fixed on 2026-11-04. Close after a few clean runs.
- The XS module lacks `inference_prefill_logits`, `model_name` and `set_seed` (add when a Perl user needs them).
- Raw completions: `prompt` as an array and `logprobs` are not implemented (400, and always null). Do them when a client
  asks. `logprobs` needs the logits out per step (`get_logits` exists).
- GLM's chat template key order: Perl's hash order differs from Python's (`tool.items()` and `|tojson` sort keys), so GLM's
  tool definitions may render differently. Not measured. Fix: keep each object's key order from the JSON parser.

## ComfyUI

`/comfyui` holds the user's model library (safetensors). What the engine has of it, and where the references are:

- **Z-Image-Turbo** (`diffusion_models/z_image_turbo_bf16`, `text_encoders/qwen_3_4b`, `vae/ae`): served (item 17).
  References: diffusers 0.43.0 in `/data/zimage-lib2` (`PYTHONPATH=/data/zimage-lib2` with the system python; torch 2.9.1
  ROCm, transformers 5.5.1); `/data/zimage-ref` (`ref_gen.py`, the stage dumps, `z1024/`, the Qwen3-4B config and tokenizer).
  The ComfyUI reference image for 1024 / seed 42 is `comfy/zimage_mve_00001_.png` (lost in the git clean, item 15); its noise
  is `torch.Generator('cpu').manual_seed(42)`, `torch.randn(1, 16, 128, 128)[0]` (`ZI_NOISE=<file>` for
  `zimage_1024_verify`).
- **Qwen-Image 1** (`diffusion_models/qwen_image_fp8_e4m3fn`, MMDiT, 60 blocks, dim 3072, plain E4M3 without scales; the
  edit variant `qwen_image_edit_2509_fp8`; text `text_encoders/qwen_2.5_vl_7b_fp8_scaled`; VAE `vae/qwen_image_vae`;
  Lightning LoRAs): every stage on the device passes `tests/qwenimage_verify` (the oracle `/data/zimage-ref/qwenimage/`).
  Facts for the server path: true CFG 4 by default (two DiT passes); the sigmas linspace(1, 1/n, n) with the exponential
  dynamic shift (base 256 -> 0.5, max 8192 -> 0.9, shift_terminal 0.02); RoPE axes (16, 56, 56), theta 10000, scale_rope on
  (the text positions start at max(h/2, w/2)); the timestep is t/1000 with a 1000x sinusoid. `ref_pipeline.py` differs
  from the stage checks by 6 to 15% over 8 steps: it is not the reference for the engine.
- **Qwen-Image 2.1** (ComfyUI int8 ConvRot files): the DiT (32 blocks), the Qwen3-VL 8B text encoder and the residual VAE
  (64 latent channels) on the device match their FP32 oracles (`/data/zimage-ref/qwenimage2`, `/data/zimage-ref/qwen3vl`).
  Open: item 16.
- **FLUX.2 dev** (`diffusion_models/flux2_dev_fp8mixed`: BF16 and FP8 layers with scales and a quantization table; double
  and single blocks, dim 6144; text `text_encoders/mistral_3_small_flux2_bf16`; VAE `vae/flux2-vae`, 32 channels; LoRA
  `Flux_2-Turbo-LoRA`): item 18.
- **SD1.5 family:** `checkpoints/DreamShaper_8_pruned` (LDM key names, 768-wide CLIP-L), `vae/vae-ft-mse-840000-ema-pruned`,
  `diffusion_models/lotus-depth-d-v1-1` (depth output), `loras/nlc_pink`. The engine's sd is SD2.x: a config-driven UNet and
  an LDM key converter are needed. Not planned.
- **Utility models:** BiRefNet and DINOv3 are on the engine (item 15). MoGe-2 (`geometry_estimation/moge_2_vitl_normal`)
  and Pixal3D (`diffusion_models`) are not planned.

Decisions (2026-11-20): the component API and the node package are on hold until the engine supports the models. The node
API is ComfyUI's (a node is a Python class: `INPUT_TYPES`, `RETURN_TYPES`, `FUNCTION`, `CATEGORY`, registered through
`NODE_CLASS_MAPPINGS` or `comfy_entrypoint()` in V3; nodes pass MODEL, CLIP, VAE, CONDITIONING, LATENT, IMAGE). Ours would be
the step set those nodes call (encode text, sample, decode). Memory policy: load per request; a request that needs room
evicts the least recently used idle model (`make_room`). The image models stay BF16 (or F32); GGUF quants only as an explicit
choice; FP8 sources stay FP8 in the engine's FP8 operand path.

## Decisions by the user

- 2026-11-26: 3D printing: hollow prints are wanted (cheaper, and some objects only work hollow), with walls of at least
  2 mm; a solid request stays solid; a model's own hollows (a whistle's) are kept. The STL is a closed solid by default.
  Access to the 3D files by the harness's login waits for the projects to join (item 15 i).
- 2026-11-25: TRELLIS.2 without RemeshMesh (the sharper result).
- 2026-11-22: hipBLASLt as a dependency; BF16 outputs for the DiT's large linears.
- 2026-11-16: the speech pieces keep their own prosody. ffmpeg is in the image. The Q8_0 predictor is an option (BF16 default).
- 2026-11-16: Gemma-4's prefill keeps fp32 (the 3x prefill cost accepted).
- 2026-11-16: request logs roll over each hour; the user deletes old files.
- 2026-11-16: requested: KV snapshots during prefill, every 500 tokens, opt-in for the large CPU-only models (item 2).
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
- Z-Image DiT linears in the engine's own kernel (2026-11-22): one LDS tile with two blocks per CU (15.8 / 14.9 / 8.8 TFLOPS);
  three register sets two tiles ahead (spills, 8 TFLOPS); the 256 x 256 tile with 16 waves (no gain); 64-byte row pieces
  (no gain); device-only scratch for the BF16 input (+15-20% in the bench, not the engine's step). Pinned host memory against
  device memory for the operands: +30% in the bench, not taken (the zero-copy rule).
- hipBLASLt's algorithm chosen by timing (2026-11-22): the candidates' numerics differ (42% of the bytes between processes);
  the tuner now takes only candidates bit-identical to the first, or the calibrated table.
- The solid rebuild on a 1024 grid (2026-11-26): the vase leaked to a thin shell (close is in cells) and the 26M-face
  simplification took over 10 minutes.
