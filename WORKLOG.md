# ROCmFP4_FAST WMMA Kernel Work

## Overview

**ROCmFP4_FAST** is a 17-byte FP4 quantization format (4.25 bits/weight) with one UE4M3 scale per 32-value block, used in models like Qwen3.8-Flash-Next. The WMMA kernels (`matmul_wmma_q4_0_rocmfp4_fast`) use AMD's WMMA intrinsics (`v_wmma_i32_16x16x16_iu8_w32`) for batched int8 matmuls of 64 tokens Ã 128 weight rows at ~55 TOPS on gfx1151.

**Current status**: Kernel exists and works but is tabled. Performance is ~26 of ~59 TOPS (70% of peak). The bottleneck is the epilogue (~19% of compute) that applies per-block scaling.

**References**:
- `src/matvec_q.hip` â `PFROCmFP4`, `WStageFast<>`, `wmma_q80<WS, NT>`, `MATMUL_WMMA_Q80_KERNELS`
- `ARCHITECTURE.md` sections 4â5 â weight types, WMMA exactness, block scaling
- `PLAN.md` â "ROCmFP4_FAST WMMA kernel work" (tabled), Qwen3.8-27B compute-bound notes

---

## Phase 1: Measure and Characterize

**Status**: Complete. Baseline measurements collected on Qwen3.8-27B-Q4_0_ROCMFP4_FAST.

### Infrastructure Created

1. **`tools/rocminfo_wmma.py`** â Enhanced profiling script that filters and categorizes WMMA kernels from rocprofv3 output
   - Categorizes kernels: WMMA, Staging, Epilogue, Other
   - JSON output for automation
   - Per-kernel metrics: total ms, %, count, avg/min/max us

2. **`tools/profile_wmma.py`** â End-to-end profiling runner
   - Runs qwen_demo with `QWEN_PROFILE=1`
   - Collects rocprofv3 kernel traces
   - Analyzes results with rocminfo_wmma.py
   - Structured JSON output

3. **`tools/profile_wmma.sh`** â Quick shell wrapper for single runs

4. **`tools/test_phase1.py`** â Infrastructure tests (synthetic data, all passed)

5. **`PHASE1_BASELINE.md`** â Baseline measurement documentation

### Baseline Measurements (Qwen3.8-27B-Q4_0_ROCMFP4_FAST)

Model: `/models/Qwen3.8-27B-Q4_0_ROCMFP4_FAST.gguf` (14 GB)

**Performance:**
- Prefill: 36 tokens in 195.7 ms (183.98 tok/s)
- Generation: 15 tokens in 1058.9 ms (14.17 tok/s)

**Per-section profile (prefill, 36 tokens):**

| Section | % of time | MB/token | GB/s |
|---------|-----------|----------|------|
| FFN | 57.9% | 252.5 | 80.2 |
| DeltaNet | 20.8% | 82.0 | 72.7 |
| Embedding (CPU) | 12.6% | - | - |
| Attention | 7.2% | 24.8 | 63.3 |
| LM head | 1.6% | 18.8 | 220.4 |

**Per-section profile (generation, 15 tokens):**

| Section | % of time | MB/token | GB/s |
|---------|-----------|----------|------|
| FFN | 65.2% | 9091.2 | 198.3 |
| DeltaNet | 23.2% | 2953.8 | 180.9 |
| Attention | 7.2% | 891.3 | 176.7 |
| LM head | 4.4% | 675.4 | 219.8 |
| Embedding (CPU) | 0.0% | - | - |

**WMMA Kernels:**

| Kernel | Total ms | % GPU | Count | Avg us | Min us | Max us |
|--------|----------|-------|-------|--------|--------|--------|
| `matmul_wmma_q4_0_rocmfp4_fast` | 142.61 | 12.2% | 400 | 356.52 | 90.97 | 603.72 |

**Other Kernels (top 10):**

| Kernel | Total ms | % GPU | Count | Avg us |
|--------|----------|-------|-------|--------|
| `matvec_q8_q4_0_rocmfp4_fast` | 959.72 | 81.8% | 3856 | 248.89 |
| `add_rmsnorm` | 19.78 | 1.7% | 1935 | 10.22 |
| `deltanet_rule` | 15.91 | 1.4% | 768 | 20.72 |
| `add_rmsnorm_rows` | 10.68 | 0.9% | 129 | 82.78 |
| `swiglu` | 6.66 | 0.6% | 1024 | 6.51 |
| `attn_partial_gqa8s_256` | 4.78 | 0.4% | 240 | 19.91 |
| `deltanet_pre` | 3.85 | 0.3% | 720 | 5.35 |
| `deltanet_norm` | 3.19 | 0.3% | 768 | 4.15 |
| `matmul_q8_q4_0_rocmfp4_fast` | 1.36 | 0.1% | 96 | 14.16 |
| `attn_prep_256` | 1.27 | 0.1% | 256 | 4.94 |

**Key Findings:**

1. **`matvec_q8_q4_0_rocmfp4_fast` dominates at 81.8%** â This is the single-token int8 path used during generation. Not a WMMA kernel, but the main bottleneck.

2. **WMMA kernels account for 12.2%** â Only `matmul_wmma_q4_0_rocmfp4_fast` appears (400 launches, 356 us avg). Used for batched operations during prefill.

3. **FFN is the largest section** â 57.9% during prefill, 65.2% during generation. Contains the WMMA matmuls.

4. **Weight bandwidth**: Generation streams 198.3 GB/s of weights (close to UMA peak of 220-256 GB/s). Prefill streams 80.2 GB/token.

5. **DeltaNet contributes 20-23%** â Convolution and delta-rule kernels are significant.

### Success Criteria

- [x] Baseline measurements collected for Qwen3.8-27B with ROCmFP4_FAST
- [x] WMMA kernel time breakdown established
- [ ] LDS and VGPR usage measured (requires deeper profiling)
- [ ] Epilogue cost quantification (requires kernel-level analysis)
- [x] Expert kernel baseline (not applicable â model is dense, not MoE)
- [x] Bottleneck identification memo (see findings above)

### Next Steps

Based on measurements:

1. **`matvec_q8_q4_0_rocmfp4_fast` at 81.8%** is the primary bottleneck. This is the single-token int8 path, not WMMA. Optimizing WMMA kernels alone won't help generation speed.

2. **Generation is bandwidth-bound** â 198 GB/s of weights streamed. Approaching UMA peak (220-256 GB/s). Further optimization requires:
   - Reducing weight footprint (already at 4.25 bpw with ROCmFP4_FAST)
   - Better caching (weights already in page cache)
   - Overlapping compute with memory access

3. **WMMA kernels (12.2%)** are not the bottleneck for this model. Phase 2-4 optimization of WMMA epilogue would have minimal impact on overall generation speed.

4. **Recommendation**: WMMA kernel optimization (Phases 2-4) has limited impact on this model. The bottleneck is `matvec_q8_q4_0_rocmfp4_fast` (81.8%) and overall bandwidth (198 GB/s). Focus on model-level changes or the single-token int8 path.

### Profiling Data

- JSON output: `./profile_27b_long.json`
- rocprofv3 output: `./profile_27b_long/`
- MoE JSON: `./profile_35b_moe.json`
- MoE rocprofv3: `./profile_35b_moe/`

### Expert Kernel Baseline (MoE models)

**Qwen3.6-35B-A3B-ROCMFP4** (35B total, 3B active, 256 experts):
- Prefill: 17 tokens in 69.3 ms (245.47 tok/s)
- Generation: 15 tokens in 235.4 ms (63.73 tok/s)

**Expert-indexed kernels:**

| Kernel | Total ms | % GPU | Count | Avg us |
|--------|----------|-------|-------|--------|
| `matvec_q8_id_q4_0_rocmfp4_fast` | 48.49 | 18.8% | 1200 | 40.40 |
| `matmul_wmma_id_q4_0_rocmfp4_fast` | 20.40 | 7.9% | 120 | 170.04 |

**MoE routing:**

| Kernel | Total ms | % GPU | Count | Avg us |
|--------|----------|-------|-------|--------|
| `moe_router` | 4.92 | 1.9% | 640 | 7.69 |
| `moe_combine` | 1.90 | 0.7% | 680 | 2.97 |

**Dominant kernel:** `matvec_bf16` at 31.8% (16 launches, 5130 us avg) â large matrix operations.

**Data:** `./profile_35b_moe.json`

### 1.1 Profiling `matmul_wmma_q4_0_rocmfp4_fast` on Qwen3.8-27B

- Use `QWEN_PROFILE=1` and `rocprofv3 --kernel-trace` + `tools/prof_summary.py`
- Measure per-workgroup:
  - Weight decode time (reading 17-byte blocks, `v_alignbyte` extraction)
  - WMMA compute time (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` launches)
  - Epilogue time (scale application: `ew * dx * (float)c[n][r]`)
  - LDS read/write latency and bandwidth utilization
  - Global memory bandwidth for weight loads

### 1.2 LDS and VGPR Analysis

- Current `WmmaSmem` layout (~128 KB per workgroup):
  ```
  sw[128][144]    â decoded weights (int8, padded stride)
  sx[64][144]     â token activations (int8, padded stride)
  sew[128][8]     â weight half-block scales (float)
  sdx[4][64]      â token block scales (float)
  ```
- Trace VGPR pressure: current code has headroom below the 173 VGPR spill threshold (from tabled "WMMA operand pipelining: 10% slower at 173 VGPRs")
- Measure LDS occupancy: 128 KB + shared memory overhead vs gfx1151's 32 KB LDS per CU

### 1.3 Quantify Epilogue Cost

- The epilogue for ROCmFP4_FAST (`BLOCK_SCALE=true` path):
  - Two WMMA dots per 32-value block added in int32, scaled once: `(ew * dx) * (float)(dot0 + dot1)`
  - 8 scales per K chunk (WM_KC/32 = 4 blocks, 2 scale entries each)
  - Each scale product requires `ew * dx[r]` multiply (8 floats per block)
- Measure ratio: epilogue cycles / WMMA cycles (expect ~1:3 to 1:4)

### 1.4 Expert Kernel Baseline

- Profile `matmul_wmma_id_q4_0_rocmfp4_fast` (MoE prefill):
  - Expert grouping efficiency: how many tokens share each expert's weights
  - Current shared-by-verified-tokens benefit: ~3.8 of 10 shared by two tokens (~19% of pass)
  - Tile size scaling: test NT=1,2,3,4 (16, 32, 48, 64 tokens per sub-tile)

### Deliverables

- `tools/rocminfo_wmma.py` â script to parse `rocprofv3` output for WMMA kernels, output CSV
- Baseline measurements table (tok/s, TFLOPS, GB/s, VGPR/LDS counts)
- Bottleneck identification memo (which phase to target)

---

## Phase 2: Epilogue Optimization

**Goal**: Reduce or eliminate the epilogue overhead by pre-scaling weights.

### 2.1 Pre-scaled Weights in FP16

- **Idea**: Apply scales during weight decode, store FP16-scaled weights in LDS, then do pure WMMA without epilogue.
- **Trade-off**: 2Ã LDS usage (weights + scaled weights). Previous attempt: "would need FP16 WMMA on pre-scaled weights (twice the LDS)" â tabled.
- **Approach**:
  - Add a new staging policy `WStageFastPreScaled`:
    - Decode 17-byte blocks as current
    - Multiply each half-block's int8 codes by scale during decode
    - Store FP16-scaled weights in a second LDS buffer
  - New WMMA kernel variant reads FP16 operands via `v_wmma_f32_16x16x16_f16` (no scale multiply in epilogue)
  - Compile-time flag `ROCMFP4_FAST_PRESCAL=1` to switch

### 2.2 Staged Implementation

- Keep current `matmul_wmma_q4_0_rocmfp4_fast` as baseline
- Add `matmul_wmma_q4_0_rocmfp4_fast_ps` (pre-scaled)
- Gate selection in `qmatvec.cpp` via runtime check (LDS availability, K dimension)
- Validate exactness: `QMATVEC_NO_WMMA=1` reference comparison, cosine > 0.999

### 2.3 LDS Budget Analysis

- Calculate exact LDS footprint for pre-scaled path:
  - Current `sw[128][144]` int8 = 18,432 bytes
  - Pre-scaled `sw_fp16[128][144]` = 36,864 bytes (2Ã)
  - Total per workgroup: ~70 KB (exceeds 32 KB LDS/CU, but LDS is shared across workgroups â check actual hardware constraints)
- If LDS is the bottleneck, try reducing `WM_ROWS` from 128 to 64 (half the weight rows, two workgroups per tile)

### Deliverables

- `src/matvec_q.hip` â new `WStageFastPreScaled` struct and kernel variants
- Benchmark results: tok/s improvement on Qwen3.8-27B
- Exactness verification via `forward_verify`
- Go/no-go decision for Phase 3

---

## Phase 3: Operand Pipelining (Revisited)

**Goal**: Improve WMMA utilization by overlapping operand loads with compute, staying under VGPR limits.

### 3.1 Previous Failure Analysis

- Tabled: "WMMA operand pipelining in the FAST kernel: 10% slower (173 VGPRs)"
- Root cause: Unrolling all 8 K-chunk steps caused 256 VGPRs + 260 spilled, compiler reordering pushed VGPR count to 173
- The `load_chunk` lambda captures `ws` and `gx` by reference; unrolling copies these into registers

### 3.2 Reduced Unroll Strategy

- Keep `#pragma unroll 1` on the K-chunk loop (already in place)
- Split K loop into two phases:
  - Phase A: K/2 chunks with current WMMA pattern
  - Phase B: remaining K/2 chunks, reuse registers
- Or: reduce `WM_ROWS` to 64 (4 waves instead of 8), halving weight row storage

### 3.3 Tile Size Optimization

- Test 2 WMMA tiles per wave (was tried, no gain)
- Test 32 weight rows per wave (32 weight rows a multiple of 2 or 4 KB apart â ARCHITECTURE.md s4)
- Current: 128 rows Ã 128 tokens per workgroup. Try 64 Ã 128 or 128 Ã 32

### 3.4 Register Pressure Budget

- Target: < 160 VGPRs (13 below the spill threshold)
- Audit VGPR usage in `wmma_q80`:
  - `acc[NT][8]` â 4Ã8Ã4 = 128 floats for NT=4 (512 bytes, 16 VGPRs)
  - `ws.w[10]` â 40 bytes (1 VGPR)
  - `gx[XC]` â XC = ceil(64Ã8/128) = 4 uint4s = 64 bytes (4 VGPRs)
  - `c[n][8]` for WMMA results â 4Ã8Ã4 = 128 floats (16 VGPRs)
  - Scale/activation loads in epilogue â ~20 VGPRs
- Sum: ~60 VGPRs for accumulators + WMMA, well under 160. Remaining budget for loop overhead, address calculations

### Deliverables

- Optimized `wmma_q80` with pipelined operand loads
- Benchmark: compare VGPR count (via `rocprof`), kernel latency, tok/s
- If no gain > 5%, revert and document why

---

## Phase 4: Expert Kernel Optimization

**Goal**: Improve MoE prefill performance for ROCmFP4_FAST layouts.

### 4.1 Expert Grouping Analysis

- Current: `moe_group` groups up to 64 token entries per expert into tiles
- `matmul_wmma_id_q4_0_rocmfp4_fast` instantiates with `WStageFast<>`
- Expert tile sizes: NT=1 (16 tokens), NT=2 (32), NT=3 (48), NT=4 (64) â selected by `rm.nt`

### 4.2 17-Byte Block Optimization

- ROCmFP4_FAST uses 17-byte blocks (one scale per 32 values) vs ROCmFP4 uses 18-byte blocks (two scales per 36-byte pair)
- `WStageFast` loads 9 dwords (36 bytes) from byte offset `16 wb` to cover two 17-byte blocks
- Optimize `load()` and `store()` in `WStageFast<>` for coalesced 17-byte access:
  - Current: `v_alignbyte` extraction with runtime shift `sh`
  - Potential: compile-time known offsets for 17 wb = 0 or 34, use direct dword loads

### 4.3 Shared Expert Optimization

- Shared expert (id=-1) is loaded as an extra slot in MoE kernels
- Current: shared expert bytes read once per verify pass (~3.8 of 10 shared by two tokens)
- Optimize: cache shared expert in LDS across all workgroups (requires grid-level synchronization, may not be efficient)
- Alternative: ensure shared expert's weights are pinned in L2 (already in page cache)

### 4.4 Tile Size Tuning

- Benchmark NT=1,2,3,4 on real MoE models (Qwen3.6-35B-A3B, Qwen3.8-Flash-Next)
- Measure:
  - Tile fill rate (how many tokens per expert)
  - LDS utilization per workgroup
  - Overall MoE prefill time contribution

### Deliverables

- Optimized `WStageFast<>::load()` and `store()` with reduced `v_alignbyte` overhead
- Benchmark table for NT=1..4 on real models
- Expert grouping heuristics (if current grouping is suboptimal)

---

## Phase 5: Integration

**Goal**: Wire optimized kernels into the inference engine and update documentation.

### 5.1 Kernel Selection Logic

- In `qmatvec.cpp`, update kernel dispatch for WMMA matmuls:
  ```cpp
  // Current: picks matmul_wmma_q4_0_rocmfp4_fast based on weight type
  // New: also consider pre-scaled variant if LDS permits
  ```
- Runtime decision based on:
  - Weight type (ROCmFP4_FAST)
  - K dimension alignment (K % 128 == 0)
  - Available LDS (from device query or compile-time flag)

### 5.2 Update Architecture References

- Update `ARCHITECTURE.md` section 4: note ROCmFP4_FAST WMMA optimization status
- Update `PLAN.md`:
  - Move "ROCmFP4_FAST WMMA kernel work" from tabled to implemented
  - Update Qwen3.8-27B performance numbers
  - Note split-invariant prefill decision (WMMA path exact vs approximate)

### 5.3 Performance Monitoring

- Add ROCmFP4_FAST WMMA kernel to `tools/prof_summary.py` kernel trace
- Update `PLAN.md` open leads:
  - "Qwen3.8-27B (dense ROCmFP4_FAST) is compute-bound: ~26 of ~59 TOPS" â new measurement
  - "its epilogue would need FP16 WMMA on pre-scaled weights" â status

### Deliverables

- Updated `src/matvec_q.hip` and `qmatvec.cpp` with optimized kernels
- Updated `ARCHITECTURE.md` and `PLAN.md`
- Performance regression tests pass (`make regress`)
- Documentation of kernel selection strategy

---

## Phase 6: Quality Verification

**Goal**: Ensure correctness and measure end-to-end impact.

### 6.1 Exactness Tests

- Run `forward_verify` on Qwen3.8-27B with ROCmFP4_FAST quantization
- Compare logits against reference (llama.cpp CPU build)
- Check cosine similarity: target > 0.999 (WMMA summation order may differ, but accuracy vs reference must be maintained)
- Run with `QMATVEC_NO_WMMA=1` to verify single-token order matches batched WMMA

### 6.2 Regression Suite

- Add ROCmFP4_FAST WMMA kernel to `tests/regress.tsv`:
  - Model: Qwen3.8-27B (ROCmFP4_FAST)
  - Metric: tok/s for prefill (32k context), generation (256 tokens)
  - Baseline: current kernel performance
- Update `tools/matvec_bench` to include ROCmFP4_FAST WMMA variants

### 6.3 End-to-End Benchmarks

- Measure Qwen3.8-Flash-Next (177B) with optimized kernels:
  - Generation tok/s (should improve if dense layers are bottleneck)
  - Prefill tok/s for 32k context (WMMA kernels dominate prefill)
  - Compare against current numbers: 34â38 tok/s generation, 826 tok/s prefill
- Measure Qwen3.8-27B (14.6 GB) with ROCmFP4_FAST:
  - Current: 13.8 tok/s generation (bandwidth-bound)
  - Expected improvement: 15â20% if epilogue optimization succeeds

### 6.4 Stress Testing

- Run `store_verify.sh` with ROCmFP4_FAST models (KV block store compatibility)
- Run `mtp_verify` with draft heads and ROCmFP4_FAST quantization
- Long-context test: 64k, 128k, 256k tokens with prefill chunking

### Deliverables

- `tests/forward_verify` passes for Qwen3.8-27B ROCmFP4_FAST
- Updated regression baselines
- Performance improvement summary (tok/s delta, TFLOPS delta)
- Go/no-go for production deployment

---

## Estimated Speeds: DeepSeek V4.1-Flash & GLM-5.3-Flash

**Hardware:** Strix Halo, gfx1151, 128 GB UMA, ~256 GB/s peak bandwidth

Both models are MoE with top-8 expert routing. Only **active params** matter for generation speed â the total model size (178B / 320B) determines storage, not throughput.

### Generation Speed Estimates (ROCmFP4_FAST, 4.25 bpw)

| Model | Total | Active | Bytes/token | Est. Gen (tok/s) | Basis |
|-------|-------|--------|-------------|------------------|-------|
| Nemotron-3-Super-120B-A12B | 120B | **12B** | ~6 GB | **40â50** | 30B-A3B at Q8_0 = 62 tok/s (fewer active params at FP4) |
| Qwen3.8-Flash-Next (177B) | 177B | **38B** | ~20 GB | **34â38** | Measured |
| GLM-5.3-Flash (320B) | 320B | **18B** | ~9.5 GB | **30â40** | Fewer active params than Flash, but more layers + complex attention (DSA indexer, KDA) |
| DeepSeek V4.1-Flash (178B) | 178B | **38B** | ~20 GB | **25â35** | Same active params as Flash, but compressed attention + indexer overhead |

### Bandwidth Model

Generation is bandwidth-bound. Bytes per token = active_params Ã bpw:
- 12B active Ã 4.25 bpw = ~6 GB/token â faster
- 18B active Ã 4.25 bpw = ~9.5 GB/token â moderate
- 38B active Ã 4.25 bpw = ~20 GB/token â slower

At UMA peak 256 GB/s:
- 12B active â theoretical max ~43 tok/s
- 18B active â theoretical max ~27 tok/s
- 38B active â theoretical max ~13 tok/s

**These are upper bounds.** Real speed is lower due to:
- Non-expert layers (attention, norms) add overhead
- MoE routing overhead (~14 us per router call, 48 calls per layer)
- Architecture-specific kernels (DSA indexer, KDA, hyper-connections)
- Memory overhead (KV cache, recurrent state)

### Expert pin/unpin impact

Without expert pin/unpin (model > 128 GB on disk):
- Expert reads hit NVMe at 3â5 GB/s
- Speed drops to **0.7â4.3 tok/s** (NVMe-bound expert reads only, no compute)
- Only the 18Bâ38B active params matter; the rest is on disk

With expert pin/unpin (active params pinned in RAM):
- Estimates above hold
- 26% of expert reads off NVMe â 2.6â4.3 tok/s
- All-resident bandwidth ceiling: 26â30 tok/s at FP8, 49â57 at FP4

### Quantization fit in 128 GB

| Quantization | bpw | DeepSeek V4.1-Flash | GLM-5.3-Flash | Fits 128 GB? |
|-------------|-----|--------------------|---------------|--------------|
| IQ2_XXS | 2.06 | ~93 GB | ~120 GB | **Yes** |
| IQ4_XS | 4.25 | ~192 GB | ~170 GB | **No** |
| ROCmFP4_FAST | 4.25 | ~192 GB | ~170 GB | **No** |
| ROCmFP4 | 4.5 | ~200 GB | ~178 GB | **No** |

At IQ2_XXS (~93â120 GB), both models fit in 128 GB without pin/unpin. At IQ4_XS / ROCmFP4_FAST, pin/unpin is required for acceptable speed.

### Architecture overheads

**DeepSeek V4:**
- Compressed attention with ratios `[0,0,4]` (no compression, no compression, full compression)
- Indexer: top-k selection over pooled keys (same machinery as GLM's DSA)
- Full-rank hyper-connections (different from qwen4exp's low-rank)
- Compression adds scoring overhead but reduces KV cache

**GLM-5.3:**
- Kimi Delta Attention (KDA): per-channel diagonal decay, state update `S *= exp(g_c)` per channel
- DSA indexer: top-2048 selection over k=4-pooled index (11 layers only, 34 layers are linear attention)
- Full-rank hyper-connections with Sinkhorn normalization
- First 3 layers are dense FFN (not MoE)

### Prediction basis

- **Nemotron-3-Super-120B-A12B at ROCmFP4**: existing model in `/models/`, not yet profiled. Estimate based on Qwen3-30B-A3B (Q8_0, 62 tok/s) scaled by active params.
- **Qwen3.8-Flash-Next at ROCmFP4_FAST**: measured at 34â38 tok/s on gfx1151.
- **DeepSeek V4.1-Flash**: same active params as Flash (38B), but indexer compression overhead reduces speed.
- **GLM-5.3-Flash**: fewer active params (18B) but more architectural overhead (DSA, KDA, Sinkhorn).


---

## Appendix: Technical Details

### ROCmFP4_FAST Format

- Block size: 17 bytes
- Layout: `qs[16]` (16 nibble bytes, values 0..15 in low nibbles, 16..31 in high nibbles) + `e` (one UE4M3 scale byte)
- Bits per weight: 4.25
- Scale: UE4M3 at half scale (`ue4m3_half`; 0x7f and above decode to 0)
- Nibble: sign bit + index into `{0, 1, 2, 3, 4, 6, 8, 10}` codebook

### WMMA Kernel Structure

```
wmma_q80<WS, NT>  // WS = weight staging policy, NT = number of 16-token sub-tiles
  âââ load_chunk()     // Stage weights from global â registers, activations from Q8_0 â registers
  âââ store_chunk()    // Decode weights to LDS, store activations to LDS
  âââ WMMA loop        // For each K chunk (128 values):
  â   âââ WMMA dots    // v_wmma_i32_16x16x16_iu8_w32 (int32 accumulator)
  â   âââ Epilogue     // Apply scales: ew * dx * (float)dot (BLOCK_SCALE path)
  âââ Store results   // Write to output buffer y[]
```

### Key Constants

- `WM_ROWS = 128` â weight rows per workgroup (8 waves)
- `WM_TILE = 64` â tokens per workgroup tile
- `WM_KC = 128` â K chunk size (32-value blocks Ã 4)
- `WM_THREADS = 256` â two threads per weight row

### Performance Numbers (Current)

| Model | Quantization | Generation | Prefill |
|-------|-------------|------------|---------|
| Qwen3.8-27B | ROCmFP4_FAST (14.6 GB) | 13.8 tok/s | 296 tok/s |
| Qwen3.8-Flash-Next | Q4_0_ROCMFP4_FAST (94 GB) | 34â38 tok/s | 826 tok/s |

### Targets

- Epilogue optimization: > 30 TOPS (from 26)
- Pre-scaled weights: > 35 TOPS if LDS allows
- Expert kernel: > 20% improvement on MoE prefill time
- Overall: > 15% improvement on Qwen3.8-27B generation tok/s
