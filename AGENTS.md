# Agents

This is a standalone inference engine for AMD ROCm (gfx1151, Strix Halo, zero-copy UMA). It began as a proof of concept
for HIP/ROCm mmap model loading and ROCmFP4 quantization. An earlier forked-llama.cpp snapshot is archived on the
`q4_0_rocmfp4_q8_1` branch. No llama.cpp code is linked. A local `llama.cpp/` directory holds reference oracle builds
only: it is git-ignored and not part of the repository.

See `ARCHITECTURE.md` for how it is built. See `PLAN.md` for current status and roadmap. **On any request involving progress, next steps, or work planning, read `PLAN.md` first** to understand what is done and what remains. Do not read it for git maintenance or one-off code changes that don't affect the project trajectory.

## Branches

All work happens on **`main`**.

- **`main`** — everything: POC demos (Mandelbrot, WMMA matmul, ROCmFP4 benchmarks), Phase 1–2 infrastructure (`gguf_mmap` zero-copy GGUF loader, verified FP4 matmul kernels + `fp4_verify` harness), and the inference-engine roadmap (Phases 3–7). Standalone zero-copy kernels proving the core concepts; no llama.cpp submodule.
- **`q4_0_rocmfp4_q8_1`** — archived reference only: an earlier forked-llama.cpp snapshot (ROCmFP4 ftype 107/108, zero-copy GGUF loading, INTEGRATION_STATUS.md analysis incl. the Qwen3.8-Flash-DGX comparison). No new work goes here; cherry-pick anything still useful into `main`.

## Core Pattern

All binaries use the same zero-copy approach:

1. Create buffer in `/dev/shm/*.bin` -> `ftruncate` -> `mmap(MAP_SHARED)`
2. `hipHostRegister(ptr, size, hipHostRegisterMapped/Default)` to pin pages
3. Launch kernel with **no `hipMemcpy`** — GPU reads/writes CPU-visible file-backed RAM directly
4. Kernels pre-compiled to `.co` objects (`hipcc --genco`) loaded at runtime via `hipModuleLoadData`

## Architecture (the rules every model path follows)

- **The target is gfx1151 (Strix Halo) only**: one GPU sharing 128 GB of unified memory with the CPU. There is no
  separate VRAM budget; `rocm-smi`'s VRAM and GTT figures are not a memory limit here.
- **Zero copy:** weights stay in their mapped file (mmap, registered with `hipHostRegister` via `register_weights`) and
  the GPU reads them in place; activations and KV live in `/dev/shm` buffers (`alloc`). No `hipMemcpy`, no
  host-to-device staging, no conversion copies of weights.
- **GPU first, CPU too:** the GPU runs the model; the CPU runs what it does better (the tokenizer, small per-step vectors
  such as timestep embeddings written into shared memory) and is never a second copy of the model.
- **Host (CPU) code is reference-only.** A CPU implementation exists to check the device path against an oracle (for
  example `src/zimage_dit.cpp` with `tests/zimage_dit_verify.cpp`). It is not a serving path. A new model's device path
  must pass the same stage checks before it is served.
- **One service:** the server (`bin/rocm-inference-server`) serves every model kind (chat, embeddings, speech, images)
  from one process, with per-model settings (`NAME.json`, host rules) and least-recently-used eviction under the memory
  check (`make_room`). New model kinds join that policy; they do not get their own process.

## GPU use

The desktop's display runs on the same GPU as the engine, so a long or hung kernel can freeze the screen.

- Keep each kernel launch short: under about 50 ms, as Z-Image's largest single launch is. Split a long grid into several
  launches, and synchronize between stages, so the desktop's queues get the GPU between them.
- Pass kernels only device memory: `sddev` buffers (`alloc`, `Floats`) or registered weights. A host pointer hangs the GPU
  and resets the driver.

Other GPU jobs may run alongside a test.

## Where Files Go (read this first)

The agent runs in a Docker container. `/home/agent` is inside the container: the user cannot reach it from the host,
and it is not safe to keep anything there. Put every deliverable, script, and note the user should use in the
repository (`/work/rocm-hip`, the mounted checkout), e.g. `tools/` for host helpers. Scratch files go in `/tmp`, as below.

## Temp Snippets and Debugging

For temporary code snippets, test scripts, and debugging runs, use files in `/tmp/`. On error or syntax issues, update the temp file using the **Edit tool** rather than rewriting it via `cat`, `echo`, or `bash` heredocs. File edit tools are preferred over inline command-line scripts because:

- Syntax errors are easier to spot and fix when editing a named file
- Long bash/python one-liners tend to accumulate escape-quote issues
- The Edit tool preserves surrounding context and indentation correctly
- Iterative fixes (e.g., wrapping blocks in conditionals, adding braces) are safer with file-based editing

Pattern: write initial snippet to `/tmp/snippet_name.cpp` or `/tmp/debug_script.py`, run it, then use Edit to fix any errors before re-running.

## File Cleanliness

Always keep source files clean — no trailing whitespace on any line. Run this after edits:

```bash
sed -i 's/[[:space:]]*$//' <file>
```

Or for all source files at once:

```bash
find src tests -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.hip' | xargs sed -i 's/[[:space:]]*$//'
```
## Container image

The image is built from this repository's `Dockerfile` (`docker-bake.hcl`, the image `local/ai/purhie-gfx1151:latest`).
Its ROCm build clones rocm-hip from GitHub `main` with `--recurse-submodules` (the `perl-minijinja-xs` submodule
included), so push before a build. The image's own files are `app.sh`, `nginx.conf`, `lib/` (the Perl and Lua request
code, copied to `/app/lib`) and the runtime packages; `.dockerignore` keeps the build context small. The image carries `/version`
(the rocm-hip commit, and this project's commit when `build-watch.sh` passes it).

To trigger an image build, touch the trigger from the repository root: `touch .build-trigger`. The user runs
`build-watch.sh` here, in their shell: it waits for the trigger, runs `docker buildx bake --pull` with the project's
HEAD commit, and appends the output to `build-watch.log`. The build is done when that log has `done, exit N`.

The user runs `./serve.sh` to serve the container (`run.sh`). `image-watch.sh` stops the container and exits when the
image changes; `serve.sh` then runs its own file again, so an edit to `serve.sh` takes effect with the next build.
`serve.sh` runs no docker commands itself: the container is `run.sh`'s and the watch is `image-watch.sh`'s.

Do not run the watchers or `serve.sh` yourself, and do not check whether they are running: you cannot see the user's
shell. Touch the trigger and report. The image runs nothing by itself.
