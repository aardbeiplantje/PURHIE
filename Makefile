# --- Configuration ---
ROCM_PATH ?= /opt/rocm
HIPCC     := $(ROCM_PATH)/bin/hipcc
CLANG     := $(wildcard $(ROCM_PATH)/core-*/lib/llvm/bin/clang++) $(ROCM_PATH)/lib/llvm/bin/clang++
CLANG     := $(firstword $(CLANG))

ARCH      := gfx1151

# Directories
SRCDIR     := src
BUILDDIR   := build
OBJDIR     := $(BUILDDIR)/obj
BINDIR     := $(BUILDDIR)
CODIR      := $(BUILDDIR)/co
SRCDIR     := src
TESTSDIR   := tests

# Host C++ flags (no offloading)
CXXFLAGS  := -O3 -Wall -I$(ROCM_PATH)/include -I$(SRCDIR) -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__
# the host float engines (qwen3tts, the codec, whisper, mel, audio): AVX-512 for the reductions and the convolutions
HOST_SIMD := -mavx2 -mfma -mf16c -mavx512f -mavx512vl -mavx512bw -mavx512dq

# Device/HIP kernel flags (with offloading)
HIPFLAGS  := -Xarch_device -mno-wavefrontsize64 -isystem $(ROCM_PATH)/include --rocm-path=$(ROCM_PATH) -O3 --offload-arch=$(ARCH) --genco -D__HIP_PLATFORM_AMD__ -D__HIP_ENABLE_AMD_HSA__

LDFLAGS   := -L$(ROCM_PATH)/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,$(ROCM_PATH)/lib

# `all` is the default goal; its rule is at the end of the file, after every variable it names is defined
.DEFAULT_GOAL := all

# Ensure build directories exist
$(shell mkdir -p $(OBJDIR) $(BINDIR) $(CODIR))

# --- Demos ---

# WMMA matmul (FP16): uses hipModuleLoadData + mmap zero-copy
WMMA_SRC  = $(SRCDIR)/wmma_kernel.hip
WMMA_CO   = $(CODIR)/wmma_kernel.co
WMMA_OBJ  = $(OBJDIR)/wmma_main.o
WMMA_TARGET = $(BINDIR)/wmma_matmul

# WMMA matmul (FP4 ROCmFP4): FP4 dequant via DP4A perm + WMMA FP16
WMMA_FP4_SRC  = $(SRCDIR)/wmma_kernel_fp4.hip
WMMA_FP4_CO   = $(CODIR)/wmma_kernel_fp4.co
WMMA_FP4_OBJ  = $(OBJDIR)/wmma_main_fp4.o
WMMA_FP4_TARGET = $(BINDIR)/wmma_matmul_fp4

# WMMA matmul (FP4 native): uses __builtin_amdgcn_wmma_f32_32x16x128_f4
WMMA_FP4_NATIVE_SRC  = $(SRCDIR)/wmma_kernel_fp4_native.hip
WMMA_FP4_NATIVE_CO   = $(CODIR)/wmma_kernel_fp4_native.co
WMMA_FP4_NATIVE_OBJ  = $(OBJDIR)/wmma_main_fp4_native.o
WMMA_FP4_NATIVE_TARGET = $(BINDIR)/wmma_matmul_fp4_native

# FP4 Matmul Benchmark
BENCH_TARGET = $(BINDIR)/wmma_bench
BENCH_HOST = $(OBJDIR)/wmma_bench_main.o
BENCH_LUT_CO = $(CODIR)/wmma_bench_fp4_lut.co

# Phase 1: GGUF mmap reader — no standalone demo binary

# Phase 3: KV cache manager + zero-copy self-test on hardware
KV_VERIFY    = $(BINDIR)/kv_verify

# Phase 3: FP16 WMMA GEMM verification harness (CPU reference vs GPU kernel)
GEMM_VERIFY  = $(BINDIR)/gemm_verify

# Phase 3: RMSNorm / RoPE / masked-softmax verification harnesses
LAYERNORM_VERIFY = $(BINDIR)/layernorm_verify
ROPE_VERIFY      = $(BINDIR)/rope_verify
SOFTMAX_VERIFY   = $(BINDIR)/softmax_verify

# Phase 3: FP16 activation x Codebook10-FP4 weight GEMM verification (the projection path)
ACTFP4_VERIFY    = $(BINDIR)/actfp4_verify

# Phase 9: Tokenizer verification
TOKENIZER_VERIFY = $(BINDIR)/tokenizer_verify
# GGUF tokenizer vs llama.cpp token ids (tests/data/*_tokenizer.tsv from tools/gen_tokenizer_ref.py)
TOKENIZER_GGUF_VERIFY = $(BINDIR)/tokenizer_gguf_verify
SAMPLER_VERIFY = $(BINDIR)/sampler_verify
# Engine generation API on a real model; run from build/co
GENERATE_VERIFY = $(BINDIR)/generate_verify
EVENTS_TEST = $(BINDIR)/events_test
QWEN4EXP_VERIFY = $(BINDIR)/qwen4exp_verify
GLM5NEXT_VERIFY = $(BINDIR)/glm5next_verify
MTP_VERIFY = $(BINDIR)/mtp_verify
MTP_OWN_VERIFY = $(BINDIR)/mtp_own_verify
# Older end-to-end smoke test with per-step top-5 logits; run from build/co
E2E_TEST = $(BINDIR)/inference_e2e_test
# WMMA operand layout on the hardware (Phase 20 W1); self-contained
WMMA_LAYOUT_VERIFY = $(BINDIR)/wmma_layout_verify
# Cold-cache matvec bandwidth per tensor role; run from build/co
MATVEC_BENCH = $(BINDIR)/matvec_bench
DEQUANT_VERIFY   = $(BINDIR)/dequant_verify
EXPERT_CPU_BENCH = $(BINDIR)/expert_cpu_bench
IMAGE_VERIFY   = $(BINDIR)/image_verify  # an image file through the processor and the vision tower vs transformers (tests/image_verify.cpp)
MM_VERIFY      = $(BINDIR)/mm_verify  # an image chat through the engine vs transformers (tests/mm_verify.cpp; run from build/co)
MEL_VERIFY     = $(BINDIR)/mel_verify  # whisper's log-mel features vs transformers' WhisperFeatureExtractor (tests/mel_verify.cpp)
WHISPER_VERIFY = $(BINDIR)/whisper_verify  # whisper's encoder and greedy decoder vs transformers (tests/whisper_verify.cpp)
CODEC_ENCODE_VERIFY = $(BINDIR)/codec_encode_verify  # the speech tokenizer encoder vs qwen-tts (tests/codec_encode_verify.cpp)
SD_IMAGE_VERIFY = $(BINDIR)/sd_image_verify  # the image engine end to end vs the reference image (tests/sd_image_verify.cpp)
SD_UNET_VERIFY = $(BINDIR)/sd_unet_verify  # the image engine's UNet, one prediction, vs the pipeline (tests/sd_unet_verify.cpp)
SD_VAE_VERIFY = $(BINDIR)/sd_vae_verify  # the image engine's VAE decoder vs diffusers (tests/sd_vae_verify.cpp)
SD_TEXT_VERIFY = $(BINDIR)/sd_text_verify  # the image engine's CLIP tokenizer and text encoder vs transformers (tests/sd_text_verify.cpp)
SPEAKER_VERIFY = $(BINDIR)/speaker_verify  # the TTS speaker encoder and the speaker mel vs qwen-tts (tests/speaker_verify.cpp)
AUDIO_VERIFY = $(BINDIR)/audio_verify  # the band-limited resampler and the WAV decode vs librosa's soxr_hq (tests/audio_verify.cpp)
QWEN3TTS_VERIFY = $(BINDIR)/qwen3tts_verify  # the TTS talker and code predictor vs qwen-tts (tests/qwen3tts_verify.cpp)
QWEN3TTS_CODEC_VERIFY = $(BINDIR)/qwen3tts_codec_verify  # the TTS codec's decoder vs qwen-tts (tests/qwen3tts_codec_verify.cpp)
QWEN3TTS_SPEECH_VERIFY = $(BINDIR)/qwen3tts_speech_verify  # the TTS speech (sampling, codec, the cut) vs its checks (tests/qwen3tts_speech_verify.cpp)
MROPE_VERIFY    = $(BINDIR)/mrope_verify  # Qwen3.5's M-RoPE positions vs transformers (tests/mrope_verify.cpp)
VISION_VERIFY   = $(BINDIR)/vision_verify  # the vision tower vs transformers (tests/vision_verify.cpp; reference: /data/vision-ref)
MATVEC_VERIFY    = $(BINDIR)/matvec_verify
# Phase 20 Step 3c: engine pieces vs the llama.cpp reference activations (ref_dump acts). Run from build/co.
FORWARD_VERIFY   = $(BINDIR)/forward_verify
# Prefill split invariance: a prompt cut anywhere gives the at-once bits (tests/regress.sh); run from build/co
SPLIT_VERIFY     = $(BINDIR)/split_verify
# Perplexity on a text as llama.cpp's llama-perplexity computes it (tests/regress.sh); run from build/co
PERPLEXITY       = $(BINDIR)/perplexity
# Long-context retrieval: a pass key hidden in a long text at several depths (tests/regress.sh); run from build/co
PASSKEY          = $(BINDIR)/passkey
GEMMA4_VERIFY    = $(BINDIR)/gemma4_verify
SEQ_VERIFY       = $(BINDIR)/seq_verify
EMBED_VERIFY     = $(BINDIR)/embed_verify

# Phase 3: end-to-end attention block integration test (chains every Phase-3 kernel)
ATTN_VERIFY      = $(BINDIR)/attn_verify
# Decode attention (F16 / Q8_0 cache, per query head / per KV head) vs a CPU reference; no model. Run from build/co
ATTN_DECODE_VERIFY = $(BINDIR)/attn_decode_verify
ATTN_SPLIT_VERIFY = $(BINDIR)/attn_split_verify

# Phase 4: Model architecture kernels

# Phase 4: Inference engine library and demo
INFERENCE_LIB    = $(BUILDDIR)/libinference.a
INFERENCE_DEMO   = $(BINDIR)/inference_demo

# Phase 4: Qwen 3 model library and demo
QWEN_MODEL_LIB   = $(BUILDDIR)/libqwen.a
QWEN_DEMO        = $(BINDIR)/qwen_demo

# --- Build Rules ---

$(WMMA_TARGET): $(WMMA_OBJ) $(WMMA_CO)
	$(CLANG) $(WMMA_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_TARGET): $(WMMA_FP4_OBJ) $(WMMA_FP4_CO)
	$(CLANG) $(WMMA_FP4_OBJ) $(LDFLAGS) -o $@

$(WMMA_FP4_NATIVE_TARGET): $(WMMA_FP4_NATIVE_OBJ) $(WMMA_FP4_NATIVE_CO)
	$(CLANG) $(WMMA_FP4_NATIVE_OBJ) $(LDFLAGS) -o $@

# Pattern rule: .hip -> .co (precompiled device code)
$(CODIR)/%.co: $(SRCDIR)/%.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Kernels that quantize matvec inputs share the rounding rules in q8_device.h
$(CODIR)/matvec_q.co $(CODIR)/gpu_ops.co $(CODIR)/attention.co $(CODIR)/deltanet.co: $(SRCDIR)/q8_device.h
$(CODIR)/matvec_q.co: $(SRCDIR)/iq_grids.h

# Verify minimal test kernel
verify_minimal.co: $(SRCDIR)/verify_minimal.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@
	cp $@ $(BINDIR)/

# Host C++ files (.cpp -> .o) -- compiled without device offloading
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CLANG) $(CXXFLAGS) -c $< -o $@

# Benchmark kernels -> .co files (same approach as wmma_matmul)
$(CODIR)/wmma_bench_fp4_dp4a.co: $(SRCDIR)/wmma_bench_fp4_dp4a.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

$(CODIR)/wmma_bench_fp4_native.co: $(SRCDIR)/wmma_bench_fp4_native.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# DP4A hardware intrinsics kernel (uses __builtin_amdgcn_perm + sudot4)
$(CODIR)/wmma_bench_fp4_dp4a_hardware.co: $(SRCDIR)/wmma_bench_fp4_dp4a_hardware.hip
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Build the benchmark binary (links host .o, loads .co at runtime)
$(BENCH_TARGET): $(BENCH_HOST) $(CODIR)/wmma_bench_fp4_dp4a.co $(CODIR)/wmma_bench_fp4_dp4a_hardware.co $(CODIR)/wmma_bench_fp4_native.co $(BENCH_LUT_CO)
	$(CLANG) $(BENCH_HOST) $(LDFLAGS) -o $@

# Phase 3 verify: KV cache manager + zero-copy self-test (loads kv_cache_kernel.co at runtime)
$(KV_VERIFY): $(SRCDIR)/kv_cache.cpp $(TESTSDIR)/kv_verify.cpp $(SRCDIR)/kv_cache.h $(CODIR)/kv_cache_kernel.co
	$(CLANG) $(CXXFLAGS) -I. $(SRCDIR)/kv_cache.cpp $(TESTSDIR)/kv_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: FP16 WMMA GEMM vs CPU reference over /dev/shm (loads fp16_gemm.co at runtime)
$(GEMM_VERIFY): $(TESTSDIR)/gemm_verify.cpp $(CODIR)/fp16_gemm.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/gemm_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: RMSNorm kernel vs CPU reference (loads layernorm.co at runtime)
$(LAYERNORM_VERIFY): $(TESTSDIR)/layernorm_verify.cpp $(CODIR)/layernorm.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/layernorm_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: RoPE rotate-half kernel vs CPU reference (loads rope_embedding.co at runtime)
$(ROPE_VERIFY): $(TESTSDIR)/rope_verify.cpp $(CODIR)/rope_embedding.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/rope_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: masked row-softmax vs CPU reference (loads softmax_masked.co at runtime)
$(SOFTMAX_VERIFY): $(TESTSDIR)/softmax_verify.cpp $(CODIR)/softmax_masked.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/softmax_verify.cpp $(LDFLAGS) -o $@

# Phase 3 verify: Act x FP4 GEMM vs CPU reference over /dev/shm (loads gemm_act_fp4.co at runtime)
$(ACTFP4_VERIFY): $(TESTSDIR)/actfp4_verify.cpp $(CODIR)/gemm_act_fp4.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/actfp4_verify.cpp $(LDFLAGS) -o $@

# Phase 9: Tokenizer verification (no .co files needed)
$(EXPERT_CPU_BENCH): $(TESTSDIR)/expert_cpu_bench.cpp $(SRCDIR)/expert_cpu.cpp $(SRCDIR)/expert_cpu.h $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/expert_cpu_bench.cpp $(SRCDIR)/expert_cpu.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp -lpthread -o $@

$(MEL_VERIFY): $(TESTSDIR)/mel_verify.cpp $(SRCDIR)/mel.cpp $(SRCDIR)/mel.h
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/mel_verify.cpp $(SRCDIR)/mel.cpp -o $@

$(QWEN3TTS_SPEECH_VERIFY): $(TESTSDIR)/qwen3tts_speech_verify.cpp $(SRCDIR)/qwen3tts.cpp $(SRCDIR)/mel.cpp $(SRCDIR)/mel.h $(SRCDIR)/qwen3tts.h $(SRCDIR)/qwen3tts_speech.cpp $(SRCDIR)/qwen3tts_speech.h $(SRCDIR)/qwen3tts_codec.cpp $(SRCDIR)/qwen3tts_codec.h $(SRCDIR)/tokenizer.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/qwen3tts_speech_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib -fopenmp -o $@

$(SD_IMAGE_VERIFY): $(TESTSDIR)/sd_image_verify.cpp $(SRCDIR)/sd.cpp $(SRCDIR)/sd.h $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/sd_image_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -lz -Wl,-rpath,/opt/rocm/lib -fopenmp -o $@

$(SD_UNET_VERIFY): $(TESTSDIR)/sd_unet_verify.cpp $(SRCDIR)/sd.cpp $(SRCDIR)/sd.h $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/sd_unet_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -lz -Wl,-rpath,/opt/rocm/lib -fopenmp -o $@

$(SD_VAE_VERIFY): $(TESTSDIR)/sd_vae_verify.cpp $(SRCDIR)/sd.cpp $(SRCDIR)/sd.h $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/sd_vae_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -lz -Wl,-rpath,/opt/rocm/lib -fopenmp -o $@

$(SD_TEXT_VERIFY): $(TESTSDIR)/sd_text_verify.cpp $(SRCDIR)/sd.cpp $(SRCDIR)/sd.h $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/sd_text_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -lz -Wl,-rpath,/opt/rocm/lib -fopenmp -o $@

$(SPEAKER_VERIFY): $(TESTSDIR)/speaker_verify.cpp $(SRCDIR)/qwen3tts.cpp $(SRCDIR)/qwen3tts.h $(SRCDIR)/mel.cpp $(SRCDIR)/mel.h $(SRCDIR)/tokenizer.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/speaker_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib -fopenmp -o $@

$(CODEC_ENCODE_VERIFY): $(TESTSDIR)/codec_encode_verify.cpp $(SRCDIR)/qwen3tts_codec.cpp $(SRCDIR)/qwen3tts_codec.h $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/codec_encode_verify.cpp $(SRCDIR)/qwen3tts_codec.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp -o $@

$(QWEN3TTS_CODEC_VERIFY): $(TESTSDIR)/qwen3tts_codec_verify.cpp $(SRCDIR)/qwen3tts_codec.cpp $(SRCDIR)/qwen3tts_codec.h $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/qwen3tts_codec_verify.cpp $(SRCDIR)/qwen3tts_codec.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp -o $@

$(QWEN3TTS_VERIFY): $(TESTSDIR)/qwen3tts_verify.cpp $(SRCDIR)/qwen3tts.cpp $(SRCDIR)/mel.cpp $(SRCDIR)/mel.h $(SRCDIR)/qwen3tts.h $(SRCDIR)/tokenizer.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/qwen3tts_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -Wl,-rpath,/opt/rocm/lib -fopenmp -o $@

$(AUDIO_VERIFY): $(TESTSDIR)/audio_verify.cpp $(SRCDIR)/audio.cpp $(SRCDIR)/audio.h
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/audio_verify.cpp $(SRCDIR)/audio.cpp -o $@

$(WHISPER_VERIFY): $(TESTSDIR)/whisper_verify.cpp $(SRCDIR)/whisper.cpp $(SRCDIR)/whisper.h $(SRCDIR)/audio.cpp $(SRCDIR)/audio.h $(SRCDIR)/mel.cpp $(SRCDIR)/mel.h $(SRCDIR)/tokenizer.cpp $(SRCDIR)/tokenizer.h $(SRCDIR)/unicode_flags.inc $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. $(TESTSDIR)/whisper_verify.cpp $(SRCDIR)/whisper.cpp $(SRCDIR)/audio.cpp $(SRCDIR)/mel.cpp $(SRCDIR)/tokenizer.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp -o $@

$(IMAGE_VERIFY): $(TESTSDIR)/image_verify.cpp $(SRCDIR)/image.cpp $(SRCDIR)/image.h $(SRCDIR)/vision.cpp $(SRCDIR)/vision.h $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/third_party/stb_image.h
	$(CLANG) $(CXXFLAGS) -fopenmp -I. $(TESTSDIR)/image_verify.cpp $(SRCDIR)/image.cpp $(SRCDIR)/vision.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp -o $@

$(MM_VERIFY): $(TESTSDIR)/mm_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -fopenmp -I. $(TESTSDIR)/mm_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -fopenmp -o $@

$(MROPE_VERIFY): $(TESTSDIR)/mrope_verify.cpp $(SRCDIR)/mrope.cpp $(SRCDIR)/mrope.h
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/mrope_verify.cpp $(SRCDIR)/mrope.cpp -o $@

$(VISION_VERIFY): $(TESTSDIR)/vision_verify.cpp $(SRCDIR)/vision.cpp $(SRCDIR)/vision.h $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp
	$(CLANG) $(CXXFLAGS) -fopenmp -I. $(TESTSDIR)/vision_verify.cpp $(SRCDIR)/vision.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp -o $@

$(DEQUANT_VERIFY): $(TESTSDIR)/dequant_verify.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/quant_types.h $(SRCDIR)/gguf_mmap.cpp
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/dequant_verify.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp -o $@

$(MATVEC_VERIFY): $(TESTSDIR)/matvec_verify.cpp $(SRCDIR)/qmatvec.cpp $(SRCDIR)/qmatvec.h $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/kernel_loader.cpp $(OBJDIR)/kernels_embed.o
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/matvec_verify.cpp $(SRCDIR)/qmatvec.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/kernel_loader.cpp $(OBJDIR)/kernels_embed.o $(LDFLAGS) -o $@

$(TOKENIZER_VERIFY): $(TESTSDIR)/tokenizer_verify.cpp $(SRCDIR)/tokenizer.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/tokenizer.h $(SRCDIR)/unicode_flags.inc
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/tokenizer_verify.cpp $(SRCDIR)/tokenizer.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/quant_types.cpp $(LDFLAGS) -o $@

$(TOKENIZER_GGUF_VERIFY): $(TESTSDIR)/tokenizer_gguf_verify.cpp $(SRCDIR)/tokenizer.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/tokenizer.h $(SRCDIR)/unicode_flags.inc
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/tokenizer_gguf_verify.cpp $(SRCDIR)/tokenizer.cpp $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/quant_types.cpp $(LDFLAGS) -o $@

$(SAMPLER_VERIFY): $(TESTSDIR)/sampler_verify.cpp $(SRCDIR)/sampler.cpp $(SRCDIR)/sampler.h
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/sampler_verify.cpp $(SRCDIR)/sampler.cpp -o $@

# Phase 3 verify: full attention block chain; loads layernorm/gemm_act_fp4/rope_embedding/kv_cache_store/fp16_gemm/softmax_masked .co files
$(ATTN_VERIFY): $(TESTSDIR)/attn_verify.cpp $(SRCDIR)/kv_cache.cpp $(SRCDIR)/kv_cache.h $(CODIR)/kv_cache_store.co $(CODIR)/gemm_act_fp4.co $(CODIR)/fp16_gemm.co $(CODIR)/layernorm.co $(CODIR)/rope_embedding.co $(CODIR)/softmax_masked.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/attn_verify.cpp $(SRCDIR)/kv_cache.cpp $(LDFLAGS) -o $@

# Phase 4: Inference engine static library
$(INFERENCE_LIB): $(SRCDIR)/inference.cpp $(SRCDIR)/inference.h $(SRCDIR)/gguf_mmap.cpp $(SRCDIR)/gguf_mmap.h $(SRCDIR)/quant_types.cpp $(SRCDIR)/quant_types.h $(SRCDIR)/quantize.cpp $(SRCDIR)/quantize.h $(SRCDIR)/kv_cache.cpp $(SRCDIR)/kv_cache.h $(SRCDIR)/tokenizer.cpp $(SRCDIR)/tokenizer.h $(SRCDIR)/unicode_flags.inc $(SRCDIR)/sampler.cpp $(SRCDIR)/sampler.h $(SRCDIR)/qwen_model.h $(SRCDIR)/sha256.cpp $(SRCDIR)/sha256.h $(SRCDIR)/inference_internal.h $(SRCDIR)/model_stage.cpp $(SRCDIR)/model_stage.h $(SRCDIR)/expert_pager.cpp $(SRCDIR)/expert_pager.h
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/inference.cpp -o $(OBJDIR)/inference.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/expert_pager.cpp -o $(OBJDIR)/expert_pager.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/gguf_mmap.cpp -o $(OBJDIR)/gguf_mmap.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/quant_types.cpp -o $(OBJDIR)/quant_types.o
	$(CLANG) $(CXXFLAGS) -ffp-contract=off -I. -fPIC -c $(SRCDIR)/quantize.cpp -o $(OBJDIR)/quantize.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/kv_cache.cpp -o $(OBJDIR)/kv_cache.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/tokenizer.cpp -o $(OBJDIR)/tokenizer.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/sampler.cpp -o $(OBJDIR)/sampler.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/sha256.cpp -o $(OBJDIR)/sha256.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/model_stage.cpp -o $(OBJDIR)/model_stage.o
	rm -f $@
	ar rcs $@ $(OBJDIR)/inference.o $(OBJDIR)/gguf_mmap.o $(OBJDIR)/quant_types.o $(OBJDIR)/quantize.o $(OBJDIR)/kv_cache.o $(OBJDIR)/tokenizer.o $(OBJDIR)/sampler.o $(OBJDIR)/sha256.o $(OBJDIR)/model_stage.o $(OBJDIR)/expert_pager.o

# Inference engine shared library (versioned): the whole engine (both archives, kernels embedded), exporting only
# the public C API (src/libinference.map: inference_*, tokenizer_*, sampler_*)
INFERENCE_SO    = $(BUILDDIR)/libinference.so.1
INFERENCE_SO_1  = $(BUILDDIR)/libinference.so.1.0
$(INFERENCE_SO_1): $(INFERENCE_LIB) $(QWEN_MODEL_LIB) $(SRCDIR)/libinference.map
	$(CLANG) -shared -Wl,-soname,libinference.so.1 -Wl,--version-script=$(SRCDIR)/libinference.map -o $@ \
		-Wl,--whole-archive $(INFERENCE_LIB) $(QWEN_MODEL_LIB) -Wl,--no-whole-archive $(LDFLAGS) -fopenmp -lz
$(INFERENCE_SO): $(INFERENCE_SO_1)
	ln -sf libinference.so.1.0 $@
	ln -sf libinference.so.1 $(BUILDDIR)/libinference.so

# KV block store across processes (run tests/store_verify.sh <model>)
STORE_VERIFY = $(BINDIR)/store_verify
$(STORE_VERIFY): $(TESTSDIR)/store_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $< -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

# C smoke test of the shared library: plain C, linked against libinference.so only (rpath to build/)
CAPI_SMOKE = $(BINDIR)/capi_smoke
$(CAPI_SMOKE): $(TESTSDIR)/capi_smoke.c $(SRCDIR)/inference.h $(SRCDIR)/sampler.h $(SRCDIR)/tokenizer.h $(INFERENCE_SO)
	cc -std=c99 -Wall -Wextra -O2 -I$(SRCDIR) $< -L$(BUILDDIR) -linference -Wl,-rpath,$(abspath $(BUILDDIR)) -o $@

HEAD_Q8_VERIFY = $(BINDIR)/head_q8_verify
$(HEAD_Q8_VERIFY): $(TESTSDIR)/head_q8_verify.cpp $(SRCDIR)/inference.h $(INFERENCE_SO)
	$(CLANG) $(CXXFLAGS) $< -L$(BUILDDIR) -linference -Wl,-rpath,$(abspath $(BUILDDIR)) -o $@

# pkg-config file
PKGCONFIG = $(BUILDDIR)/rocm-inference.pc
$(PKGCONFIG): $(INFERENCE_SO)
	@echo "prefix=/usr/local" > $@
	@echo "exec_prefix=\$${prefix}" >> $@
	@echo "libdir=\$${exec_prefix}/lib" >> $@
	@echo "includedir=\$${prefix}/include/rocm_inference" >> $@
	@echo "" >> $@
	@echo "Name: rocm-inference" >> $@
	@echo "Description: Zero-copy ROCmFP4 inference engine for Strix Halo" >> $@
	@echo "Version: 1.0" >> $@
	@echo "Libs: -L\$${libdir} -linference" >> $@
	@echo "Cflags: -I\$${includedir}" >> $@

# Installation
PREFIX ?= /usr/local
INSTALL_LIBDIR = $(PREFIX)/lib
INSTALL_INCDIR = $(PREFIX)/include/rocm_inference
INSTALL_PKGCONFIGDIR = $(PREFIX)/lib/pkgconfig

install: $(INFERENCE_SO) $(INFERENCE_LIB) $(PKGCONFIG)
	install -d $(DESTDIR)$(INSTALL_LIBDIR) $(DESTDIR)$(INSTALL_INCDIR) $(DESTDIR)$(INSTALL_PKGCONFIGDIR)
	cp -f $(INFERENCE_SO_1) $(DESTDIR)$(INSTALL_LIBDIR)/
	ln -sf libinference.so.1.0 $(DESTDIR)$(INSTALL_LIBDIR)/libinference.so.1
	ln -sf libinference.so.1 $(DESTDIR)$(INSTALL_LIBDIR)/libinference.so
	cp -f $(SRCDIR)/inference.h $(SRCDIR)/sampler.h $(SRCDIR)/tokenizer.h $(SRCDIR)/vision.h $(SRCDIR)/image.h $(SRCDIR)/mrope.h $(SRCDIR)/whisper.h $(SRCDIR)/mel.h $(SRCDIR)/audio.h $(DESTDIR)$(INSTALL_INCDIR)/
	sed 's|^prefix=.*|prefix=$(PREFIX)|' $(PKGCONFIG) > $(DESTDIR)$(INSTALL_PKGCONFIGDIR)/rocm-inference.pc

uninstall:
	rm -f $(INSTALL_LIBDIR)/libinference.so $(INSTALL_LIBDIR)/libinference.so.1 $(INSTALL_LIBDIR)/libinference.so.1.0
	rm -f $(INSTALL_INCDIR)/inference.h $(INSTALL_INCDIR)/sampler.h $(INSTALL_INCDIR)/tokenizer.h $(INSTALL_INCDIR)/vision.h $(INSTALL_INCDIR)/image.h $(INSTALL_INCDIR)/mrope.h $(INSTALL_INCDIR)/whisper.h $(INSTALL_INCDIR)/mel.h $(INSTALL_INCDIR)/audio.h
	rm -f $(INSTALL_PKGCONFIGDIR)/rocm-inference.pc

# Phase 4: Qwen 3 model static library
QWEN_SRCS = $(SRCDIR)/qwen_model.cpp $(SRCDIR)/qwen_ops.cpp $(SRCDIR)/qmatvec.cpp $(SRCDIR)/deltanet.cpp $(SRCDIR)/attention.cpp $(SRCDIR)/gpu_ops.cpp $(SRCDIR)/kernel_loader.cpp $(SRCDIR)/glm5_ref.cpp $(SRCDIR)/glm5_gpu.cpp $(SRCDIR)/expert_cpu.cpp $(SRCDIR)/vision.cpp $(SRCDIR)/image.cpp $(SRCDIR)/mrope.cpp $(SRCDIR)/whisper.cpp $(SRCDIR)/mel.cpp $(SRCDIR)/audio.cpp $(SRCDIR)/sd.cpp $(SRCDIR)/sd_device.cpp $(SRCDIR)/minimax_vae.cpp $(SRCDIR)/minimax_dit.cpp $(SRCDIR)/minimax_pipe.cpp $(SRCDIR)/safetensors_mmap.cpp $(SRCDIR)/qwen3tts.cpp $(SRCDIR)/qwen3tts_codec.cpp $(SRCDIR)/qwen3tts_speech.cpp
QWEN_HDRS = $(SRCDIR)/whisper.h $(SRCDIR)/mel.h $(SRCDIR)/qwen3tts.h $(SRCDIR)/qwen3tts_codec.h $(SRCDIR)/qwen3tts_speech.h $(SRCDIR)/vision.h $(SRCDIR)/image.h $(SRCDIR)/mrope.h $(SRCDIR)/third_party/stb_image.h $(SRCDIR)/expert_pager.h $(SRCDIR)/expert_cpu.h $(SRCDIR)/qwen_model.h $(SRCDIR)/qwen_ops.h $(SRCDIR)/qmatvec.h $(SRCDIR)/deltanet.h $(SRCDIR)/attention.h $(SRCDIR)/gpu_ops.h $(SRCDIR)/inference_internal.h $(SRCDIR)/kernel_loader.h $(SRCDIR)/inference_log.h $(SRCDIR)/glm5_ref.h $(SRCDIR)/glm5_gpu.h
# The engine's code objects, embedded in the library (kernels_embed.S, loaded by kernel_loader.cpp)
ENGINE_COS = $(CODIR)/matvec_q.co $(CODIR)/gpu_ops.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/glm5_gpu.co $(CODIR)/talker_gpu.co $(CODIR)/sd_gpu.co $(CODIR)/minimax_vae.co

$(OBJDIR)/kernels_embed.o: $(SRCDIR)/kernels_embed.S $(ENGINE_COS)
	$(CLANG) -c -DKERNEL_DIR=$(abspath $(CODIR)) $< -o $@

$(QWEN_MODEL_LIB): $(QWEN_SRCS) $(QWEN_HDRS) $(SRCDIR)/inference.h $(SRCDIR)/gguf_mmap.h $(SRCDIR)/kv_cache.h $(OBJDIR)/kernels_embed.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/qwen_model.cpp -o $(OBJDIR)/qwen_model.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/qwen_ops.cpp -o $(OBJDIR)/qwen_ops.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/qmatvec.cpp -o $(OBJDIR)/qmatvec.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/deltanet.cpp -o $(OBJDIR)/deltanet.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/attention.cpp -o $(OBJDIR)/attention.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/gpu_ops.cpp -o $(OBJDIR)/gpu_ops.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/kernel_loader.cpp -o $(OBJDIR)/kernel_loader.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/glm5_ref.cpp -o $(OBJDIR)/glm5_ref.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/glm5_gpu.cpp -o $(OBJDIR)/glm5_gpu.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/expert_cpu.cpp -o $(OBJDIR)/expert_cpu.o
	$(CLANG) $(CXXFLAGS) -fopenmp -I. -fPIC -c $(SRCDIR)/vision.cpp -o $(OBJDIR)/vision.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/image.cpp -o $(OBJDIR)/image.o
	$(CLANG) $(CXXFLAGS) -I. -fPIC -c $(SRCDIR)/mrope.cpp -o $(OBJDIR)/mrope.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/whisper.cpp -o $(OBJDIR)/whisper.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/mel.cpp -o $(OBJDIR)/mel.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/audio.cpp -o $(OBJDIR)/audio.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/qwen3tts.cpp -o $(OBJDIR)/qwen3tts.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/sd.cpp -o $(OBJDIR)/sd.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/sd_device.cpp -o $(OBJDIR)/sd_device.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/minimax_vae.cpp -o $(OBJDIR)/minimax_vae.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/minimax_dit.cpp -o $(OBJDIR)/minimax_dit.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/minimax_pipe.cpp -o $(OBJDIR)/minimax_pipe.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/safetensors_mmap.cpp -o $(OBJDIR)/safetensors_mmap.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/qwen3tts_codec.cpp -o $(OBJDIR)/qwen3tts_codec.o
	$(CLANG) $(CXXFLAGS) $(HOST_SIMD) -fopenmp -I. -fPIC -c $(SRCDIR)/qwen3tts_speech.cpp -o $(OBJDIR)/qwen3tts_speech.o
	rm -f $@
	ar rcs $@ $(OBJDIR)/qwen_model.o $(OBJDIR)/qwen_ops.o $(OBJDIR)/qmatvec.o $(OBJDIR)/deltanet.o $(OBJDIR)/attention.o $(OBJDIR)/gpu_ops.o $(OBJDIR)/kernel_loader.o $(OBJDIR)/glm5_ref.o $(OBJDIR)/glm5_gpu.o $(OBJDIR)/expert_cpu.o $(OBJDIR)/vision.o $(OBJDIR)/image.o $(OBJDIR)/mrope.o $(OBJDIR)/whisper.o $(OBJDIR)/mel.o $(OBJDIR)/audio.o $(OBJDIR)/sd.o $(OBJDIR)/sd_device.o $(OBJDIR)/minimax_vae.o $(OBJDIR)/minimax_dit.o $(OBJDIR)/minimax_pipe.o $(OBJDIR)/safetensors_mmap.o $(OBJDIR)/qwen3tts.o $(OBJDIR)/qwen3tts_codec.o $(OBJDIR)/qwen3tts_speech.o $(OBJDIR)/kernels_embed.o

# Phase 4: Qwen 3 demo binary
$(QWEN_DEMO): $(SRCDIR)/demo_qwen.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(SRCDIR)/demo_qwen.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(SPLIT_VERIFY): $(TESTSDIR)/split_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/split_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

MM_TE_VERIFY = $(BINDIR)/mm_te_verify  # MiniMax-H3-Image's text conditioning: the Qwen3-VL text encoder vs transformers (tests/mm_te_verify.cpp)
$(MM_TE_VERIFY): $(TESTSDIR)/mm_te_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/mm_te_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

MM_VAE_VERIFY = $(BINDIR)/mm_vae_verify  # MiniMax-H3-Image's VAE decoder vs diffusers (tests/mm_vae_verify.cpp)
$(MM_VAE_VERIFY): $(TESTSDIR)/mm_vae_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/mm_vae_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

MM_DIT_VERIFY = $(BINDIR)/mm_dit_verify  # MiniMax-H3-Image's DiT vs diffusers in bf16 (tests/mm_dit_verify.cpp)
$(MM_DIT_VERIFY): $(TESTSDIR)/mm_dit_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/mm_dit_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

MM_PIPE_VERIFY = $(BINDIR)/mm_pipe_verify  # MiniMax-H3-Image end to end: one PNG from a prompt (tests/mm_pipe_verify.cpp)
$(MM_PIPE_VERIFY): $(TESTSDIR)/mm_pipe_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(SRCDIR)/minimax_pipe.h
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/mm_pipe_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -fopenmp -lz -o $@

MM_CANVAS_VERIFY = $(BINDIR)/mm_canvas_verify  # the DiT sampler on the released pipeline's inputs (tests/mm_canvas_verify.cpp)
$(MM_CANVAS_VERIFY): $(TESTSDIR)/mm_canvas_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/mm_canvas_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -fopenmp -lz -o $@

$(FORWARD_VERIFY): $(TESTSDIR)/forward_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/forward_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

ATTN_BENCH = $(BINDIR)/attn_bench
$(ATTN_BENCH): $(TESTSDIR)/attn_bench.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/attention.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/attn_bench.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(ATTN_SPLIT_VERIFY): $(TESTSDIR)/attn_split_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/attention.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/attn_split_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(ATTN_DECODE_VERIFY): $(TESTSDIR)/attn_decode_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/attention.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/attn_decode_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(PERPLEXITY): $(TESTSDIR)/perplexity.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -fopenmp -I. $(TESTSDIR)/perplexity.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -fopenmp -o $@

$(PASSKEY): $(TESTSDIR)/passkey.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/passkey.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(GEMMA4_VERIFY): $(TESTSDIR)/gemma4_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/gemma4_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(EMBED_VERIFY): $(TESTSDIR)/embed_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/embed_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(SEQ_VERIFY): $(TESTSDIR)/seq_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/seq_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(EVENTS_TEST): $(TESTSDIR)/events_test.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB)
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/events_test.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(GENERATE_VERIFY): $(TESTSDIR)/generate_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/generate_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

# Phase 21: the MTP draft head and speculative decoding (qwen4exp; needs the real model and its head)
# the model's own MTP block as its draft head vs the same block as a sidecar (tools/gen_qwen4exp_tiny.py TINY_MTP=)
$(MTP_OWN_VERIFY): $(TESTSDIR)/mtp_own_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/mtp_own_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(MTP_VERIFY): $(TESTSDIR)/mtp_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/mtp_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

# Phase 21: qwen4exp (hyper-connections, PLE) vs the reference, on tools/gen_qwen4exp_tiny.py's model
$(QWEN4EXP_VERIFY): $(TESTSDIR)/qwen4exp_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/qwen4exp_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

# GLM-5.3-Flash (glm5-next) vs upstream llama.cpp, on tools/gen_glm5next_ckpt.py's model as FAST
$(GLM5NEXT_VERIFY): $(TESTSDIR)/glm5next_verify.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(TESTSDIR)/glm5next_verify.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

# End-to-end inference test (real model testing)

$(E2E_TEST): $(SRCDIR)/inference_e2e_test.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co $(CODIR)/deltanet.co $(CODIR)/attention.co $(CODIR)/gpu_ops.co
	$(CLANG) $(CXXFLAGS) -I. $(SRCDIR)/inference_e2e_test.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

$(MATVEC_BENCH): tools/matvec_bench.cpp $(QWEN_MODEL_LIB) $(INFERENCE_LIB) $(CODIR)/matvec_q.co
	$(CLANG) $(CXXFLAGS) -I. tools/matvec_bench.cpp -Wl,--start-group $(QWEN_MODEL_LIB) $(INFERENCE_LIB) -Wl,--end-group $(LDFLAGS) -o $@

# Same ROCm include path and --rocm-path as HIPFLAGS: without them hipcc can pick up a distro's /usr/include/hip,
# whose headers do not match ROCm's clang (undeclared __AMDGCN_WAVEFRONT_SIZE, __ocml_*)
$(WMMA_LAYOUT_VERIFY): $(TESTSDIR)/wmma_layout_verify.hip
	$(HIPCC) -isystem $(ROCM_PATH)/include --rocm-path=$(ROCM_PATH) -O3 --offload-arch=$(ARCH) -mno-wavefrontsize64 \
		$< -o $@

# Build all demos and benchmark
both: $(TARGET) $(WMMA_TARGET) $(WMMA_FP4_TARGET) $(WMMA_FP4_NATIVE_TARGET) $(BENCH_TARGET)

# Phase 20 reference oracle (test tooling only, never linked into libinference / the XS module).
# Needs the ROCmFPX llama.cpp CPU build: cmake -S llama.cpp -B llama.cpp/build-cpu && cmake --build llama.cpp/build-cpu
REF_LLAMA    ?= llama.cpp
REF_LLAMA_B  ?= $(REF_LLAMA)/build-cpu
REF_DUMP     = $(BINDIR)/ref_dump

ref_dump: $(REF_DUMP)

# The weight encoders (src/quantize.cpp) against the reference's, byte for byte (tests/quantize_verify.cpp)
QUANTIZE_VERIFY = $(BINDIR)/quantize_verify
quantize_verify: $(QUANTIZE_VERIFY)
$(QUANTIZE_VERIFY): $(TESTSDIR)/quantize_verify.cpp $(SRCDIR)/quantize.cpp $(SRCDIR)/quantize.h $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp
	c++ -O2 -std=c++17 -ffp-contract=off -I$(SRCDIR) -I$(REF_LLAMA)/ggml/include $(TESTSDIR)/quantize_verify.cpp \
		$(SRCDIR)/quantize.cpp $(SRCDIR)/quant_types.cpp $(SRCDIR)/gguf_mmap.cpp \
		$(REF_LLAMA_B)/ggml/src/libggml-base.a -lpthread -lm -o $@

$(REF_DUMP): tools/ref_oracle/ref_dump.cpp
	@mkdir -p $(BINDIR)
	c++ -O2 -std=c++17 -I$(REF_LLAMA)/include -I$(REF_LLAMA)/ggml/include $< \
		$(REF_LLAMA_B)/src/libllama.a $(REF_LLAMA_B)/ggml/src/libggml.a \
		$(REF_LLAMA_B)/ggml/src/libggml-cpu.a $(REF_LLAMA_B)/ggml/src/libggml-base.a \
		-fopenmp -lpthread -ldl -o $@

# The same oracle on stock upstream llama.cpp (a worktree of ggml-org master at llama.cpp-stock, CPU build in
# build-cpu): formats the fork predates or handles differently (NVFP4 / MXFP4). Its libraries are shared ones.
REF_STOCK    ?= llama.cpp-stock
REF_STOCK_B  ?= $(REF_STOCK)/build-cpu
REF_DUMP_STOCK = $(BINDIR)/ref_dump_stock

ref_dump_stock: $(REF_DUMP_STOCK)

$(REF_DUMP_STOCK): tools/ref_oracle/ref_dump.cpp
	@mkdir -p $(BINDIR)
	c++ -O2 -std=c++17 -I$(REF_STOCK)/include -I$(REF_STOCK)/ggml/include $< -L$(REF_STOCK_B)/bin \
		-Wl,-rpath,$(abspath $(REF_STOCK_B)/bin) -lllama -lggml -lggml-base -lggml-cpu -fopenmp -lpthread -ldl \
		-o $@

# The regression suite (tests/regress.sh, tests/regress.tsv): `make regress` the quick tier (small models, minutes),
# `make regress-full` all of it (Qwen3.6 / 3.8 and the other big models, KV cache types, long context)
regress: all
	tests/regress.sh --quick
regress-full: all
	tests/regress.sh

.PHONY: regress regress-full

clean:
	rm -rf $(BUILDDIR) *.co

# Copy .co files to root for test execution (tests expect .co files in cwd)
copy_co_files:
	cp -f $(CODIR)/*.co . 2>/dev/null || true

.PHONY: all clean both ref_dump ref_dump_stock tokenizer_verify copy_co_files

tokenizer_verify: $(TOKENIZER_VERIFY) copy_co_files

# Verify minimal test target
TEST_TARGET = $(BINDIR)/verify_minimal_test
TEST_OBJ = $(OBJDIR)/verify_minimal_test.o

$(TEST_TARGET): $(TEST_OBJ) verify_minimal.co
	$(CLANG) $(CXXFLAGS) -I. $(SRCDIR)/verify_minimal_test.cpp -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -o $@

$(OBJDIR)/verify_minimal_test.o: $(SRCDIR)/verify_minimal_test.cpp
	$(CLANG) $(CXXFLAGS) -I. -c $< -o $@

.PHONY: verify_minimal
verify_minimal: $(TEST_TARGET)
	@echo "=== Running verify_minimal_test ===" && \
	export HSA_OVERRIDE_GFX_VERSION=11.5.1 HSA_ENABLE_SDMA=0 && \
	./$(TEST_TARGET) 2>&1 || echo "Test exited with code $?"

# Last, so that every variable above is defined when its prerequisites are expanded
all: $(WMMA_TARGET) $(KV_VERIFY) $(GEMM_VERIFY) $(LAYERNORM_VERIFY) $(ROPE_VERIFY) $(SOFTMAX_VERIFY) $(ACTFP4_VERIFY) $(TOKENIZER_VERIFY) $(TOKENIZER_GGUF_VERIFY) $(SAMPLER_VERIFY) $(DEQUANT_VERIFY) $(EXPERT_CPU_BENCH) $(VISION_VERIFY) $(MROPE_VERIFY) $(MM_VERIFY) $(IMAGE_VERIFY) $(MEL_VERIFY) $(WHISPER_VERIFY) $(AUDIO_VERIFY) $(QWEN3TTS_VERIFY) $(QWEN3TTS_CODEC_VERIFY) $(QWEN3TTS_SPEECH_VERIFY) $(SPEAKER_VERIFY) $(SD_TEXT_VERIFY) $(SD_VAE_VERIFY) $(SD_UNET_VERIFY) $(SD_IMAGE_VERIFY) $(MM_TE_VERIFY) $(MM_VAE_VERIFY) $(MM_DIT_VERIFY) $(CODEC_ENCODE_VERIFY) $(HEAD_Q8_VERIFY) $(MATVEC_VERIFY) $(ATTN_VERIFY) $(ATTN_DECODE_VERIFY) $(INFERENCE_LIB) $(INFERENCE_SO) $(PKGCONFIG) $(QWEN_MODEL_LIB) $(QWEN_DEMO) $(FORWARD_VERIFY) $(PERPLEXITY) $(PASSKEY) $(GENERATE_VERIFY) $(EVENTS_TEST) $(GEMMA4_VERIFY) $(SEQ_VERIFY) $(EMBED_VERIFY) $(QWEN4EXP_VERIFY) $(GLM5NEXT_VERIFY) $(MTP_VERIFY) $(MTP_OWN_VERIFY) $(E2E_TEST) $(MATVEC_BENCH) $(WMMA_LAYOUT_VERIFY) $(CAPI_SMOKE) $(STORE_VERIFY) $(SPLIT_VERIFY) $(ATTN_SPLIT_VERIFY) $(ATTN_BENCH)
