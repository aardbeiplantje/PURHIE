#!/bin/sh
# convert_verify.sh [dir]: bin/rocm-convert against llama.cpp's convert_hf_to_gguf.py, byte for byte, for every
# architecture it converts.
# 1. Converted as BF16 by both, the same tensors (names, types, shapes, bytes) and metadata (model-card keys aside;
#    ssm_a = -exp(A_log) and llama3's rope_freqs may differ by 1 ulp an element: torch's exp and pow are MKL's):
#    - releases: Qwen3-0.6B, Qwen2.5-0.5B-Instruct, Qwen3.5-0.8B (Gated DeltaNet, its MTP layer), embeddinggemma-300m
#      (F32, SentencePiece);
#    - tiny random checkpoints of a release's architecture, from its config and tokenizer files
#      (tools/gen_tiny_ckpt.py: the model transformers builds from the shrunk config, saved with the release's
#      tensor names): Qwen3-30B-A3B (MoE), Qwen3.5-122B-A10B (DeltaNet with V heads != K heads, fused experts, an
#      MTP layer), Qwen3.8-Flash-Next (hyper-connections, QSA indexer, PLE n-gram table, MTP), Llama-3.2-1B (Q/K
#      permuted, llama3 rope_freqs), NVIDIA-Nemotron-3-Nano-30B-A3B (Mamba-2 + attention + MoE), Nemotron-3-Super
#      (latent MoE, an MTP head), gemma-4-26B-A4B (sliding + full attention, K = V, MoE beside a dense MLP);
#    - hand-made ones: GLM-5.3-Flash (tools/gen_glm5next_ckpt.py: FP8 weights with 128x128 block scales, KDA +
#      sparse MLA + MoE + MTP) and DeepSeek-V4-Flash (tools/gen_dsv4_ckpt.py: DeepSeek's layout, FP8 with E8M0
#      scales, packed FP4 experts, hash routing, KV compressors).
# 2. Each to FAST directly = convert_hf_to_gguf.py's BF16 GGUF re-encoded as FAST (the release in F32: = our F32
#    GGUF re-encoded; the GLM: FP8 decoded and encoded in one step = F32, then re-encoded), and the GLM from a git
#    clone whose files are LFS pointers = from the files.
# Needs: CONVERT_UPSTREAM (a llama.cpp checkout with convert_hf_to_gguf.py and gguf-py; default
# /data/conv/upstream), CONVERT_PYTHON (a Python with torch, transformers >= 5 and safetensors; default
# /data/hf/venv/bin/python), CONVERT_PYTHONPATH (modules ahead of it: a transformers that knows glm5_next,
# deepseek_v4 and qwen4_exp, 5.22 here; default /data/conv/pylib), CONVERT_DATA (default /data/conv) holding the
# releases (Qwen3-0.6B as qwen3-0.6b/, Qwen2.5-0.5B-Instruct/, Qwen3.5-0.8B/, embeddinggemma-300m/ with its
# modules.json and 1_Pooling/) and the small files of the others (tools/fetch_meta.sh REPO DIR; DeepSeek's and
# GLM's from the release's index and headers): qwen3moe-meta (Qwen/Qwen3-30B-A3B), qwen35moe-meta
# (Qwen/Qwen3.5-122B-A10B), qwen4exp-meta (Qwen/Qwen3.8-Flash-Next), llama-meta (unsloth/Llama-3.2-1B-Instruct),
# nemotronh-meta (nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), nemotronsuper-meta
# (nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16), gemma4-meta (google/gemma-4-26B-A4B-it), glm53-meta, dsv4-meta;
# the Perl module built.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$HERE/..
UP=${CONVERT_UPSTREAM:-/data/conv/upstream}
PY=${CONVERT_PYTHON:-/data/hf/venv/bin/python}
DATA=${CONVERT_DATA:-/data/conv}
PYP=${CONVERT_PYTHONPATH-/data/conv/pylib}
PL="perl -I$ROOT/perl/ROCm-Inference/blib/lib -I$ROOT/perl/ROCm-Inference/blib/arch"
CONV="$PL $ROOT/perl/ROCm-Inference/bin/rocm-convert -q -f"
CMP="$PL $HERE/gguf_compare.pl"
ULP='ssm_a$|^rope_freqs'
DIR=${1:-$DATA/convert_verify.$$}
mkdir -p "$DIR"
rc=0
up() { PYTHONPATH="$PYP:$UP:$UP/gguf-py" "$PY" "$UP/convert_hf_to_gguf.py" "$1" --outtype bf16 --outfile "$2" > "$2.log" 2>&1 || { tail -5 "$2.log"; rc=1; }; }
tiny() { PYTHONPATH="$PYP" "$PY" -P "$ROOT/tools/gen_tiny_ckpt.py" "$@" 2>&1 | grep -v 'Writing\|fast path\|install' || true; }
"$PY" -I "$ROOT/tools/gen_glm5next_ckpt.py" "$DATA/glm53-meta" "$DIR/glm-tiny"
"$PY" -I "$ROOT/tools/gen_dsv4_ckpt.py" "$DATA/dsv4-meta" "$DIR/ds-tiny"
T=$DIR/tiny
tiny "$DATA/qwen3moe-meta" "$T/Qwen3-30B-A3B" hidden_size=64 intermediate_size=128 moe_intermediate_size=32 \
    num_hidden_layers=2 num_attention_heads=4 num_key_value_heads=2 head_dim=32 num_experts=8 num_experts_per_tok=2
DELTANET='num_attention_heads=4 num_key_value_heads=2 head_dim=32 linear_key_head_dim=16 linear_value_head_dim=8
    linear_num_key_heads=2 linear_num_value_heads=6 moe_intermediate_size=32 shared_expert_intermediate_size=32
    num_experts=4 num_experts_per_tok=2 vision_config.depth=1 vision_config.hidden_size=32
    vision_config.intermediate_size=64 vision_config.num_heads=2 vision_config.out_hidden_size=64'
LT='text_config.layer_types=["linear_attention","linear_attention","linear_attention","full_attention"]'
# shellcheck disable=SC2086
tiny "$DATA/qwen35moe-meta" "$T/Qwen3.5-122B-A10B" hidden_size=64 num_hidden_layers=4 "$LT" $DELTANET +mtp=3
# shellcheck disable=SC2086
tiny "$DATA/qwen4exp-meta" "$T/Qwen3.8-Flash-Next" hidden_size=64 num_hidden_layers=4 "$LT" $DELTANET hc_lowrank=8 \
    indexer_n_heads=2 indexer_head_dim=16 indexer_budget=16 ple_embed_dim=64 ngram_vocab_size_base=1000 \
    split_ngram_parts=4 heads_per_ngram=2 make_ngram_vocab_size_divisible_by=8 +mtp=3
tiny "$DATA/llama-meta" "$T/Llama-3.2-1B-Instruct" hidden_size=64 intermediate_size=128 num_hidden_layers=2 \
    num_attention_heads=4 num_key_value_heads=2 head_dim=16
NEMO='hidden_size=64 num_hidden_layers=4 hybrid_override_pattern="ME*E" mamba_num_heads=4 mamba_head_dim=16 n_groups=2
    ssm_state_size=16 num_attention_heads=4 num_key_value_heads=2 head_dim=16 n_routed_experts=4 num_experts_per_tok=2
    moe_intermediate_size=32 moe_shared_expert_intermediate_size=64 intermediate_size=32 chunk_size=16'
# shellcheck disable=SC2086
tiny "$DATA/nemotronh-meta" "$T/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" $NEMO
# shellcheck disable=SC2086
tiny "$DATA/nemotronsuper-meta" "$T/NVIDIA-Nemotron-3-Super-120B-A12B-BF16" $NEMO moe_latent_size=16 \
    +copy=backbone.layers.2.:mtp.layers.0. +copy=backbone.layers.3.:mtp.layers.1. \
    +rand=mtp.layers.0.eh_proj.weight=64x128 +rand=mtp.layers.0.enorm.weight=64 +rand=mtp.layers.0.hnorm.weight=64 \
    +rand=mtp.layers.1.final_layernorm.weight=64
tiny "$DATA/gemma4-meta" "$T/gemma-4-26B-A4B-it" hidden_size=64 intermediate_size=64 moe_intermediate_size=32 \
    num_experts=4 top_k_experts=2 num_hidden_layers=6 \
    'text_config.layer_types=["sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention"]' \
    num_attention_heads=2 num_key_value_heads=2 num_global_key_value_heads=1 head_dim=32 global_head_dim=64 \
    sliding_window=16 vision_config.hidden_size=32 vision_config.intermediate_size=64 vision_config.num_hidden_layers=1 \
    vision_config.num_attention_heads=2 vision_config.num_key_value_heads=2 vision_config.head_dim=16 \
    vision_config.global_head_dim=16
for m in "$DATA/qwen3-0.6b:qwen3" "$DATA/Qwen2.5-0.5B-Instruct:qwen2" "$DATA/Qwen3.5-0.8B:qwen35" \
         "$DATA/embeddinggemma-300m:embgemma:f32" "$T/Qwen3-30B-A3B:qwen3moe" "$T/Qwen3.5-122B-A10B:qwen35moe" \
         "$T/Qwen3.8-Flash-Next:qwen4exp" "$T/Llama-3.2-1B-Instruct:llama" \
         "$T/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16:nemotronh" "$T/NVIDIA-Nemotron-3-Super-120B-A12B-BF16:nemotronsuper" \
         "$T/gemma-4-26B-A4B-it:gemma4" "$DIR/glm-tiny:glm" "$DIR/ds-tiny:deepseek4:f32"; do
    src=${m%%:*}; rest=${m#*:}; tag=${rest%%:*}; f32=${rest#"$tag"}
    up "$src" "$DIR/$tag-ref.gguf"
    $CONV "$src" "$DIR/$tag-ours.gguf" bf16-ref || rc=1
    printf '%s, BF16: ' "$tag"
    $CMP --ignore-card --ulp "$ULP" "$DIR/$tag-ours.gguf" "$DIR/$tag-ref.gguf" | tail -1 | grep 'no differences' || rc=1
    [ "$tag" = glm ] && continue  # FP8: below
    $CONV "$src" "$DIR/$tag-fast.gguf" fast || rc=1
    if [ -n "$f32" ]; then  # F32 or FP8: their BF16 rounds it (DeepSeek: Q8_0), so against our F32 GGUF
        $CONV "$src" "$DIR/$tag-f32.gguf" f32 || rc=1
        $CONV "$DIR/$tag-f32.gguf" "$DIR/$tag-ref-fast.gguf" fast || rc=1
    else
        $CONV "$DIR/$tag-ref.gguf" "$DIR/$tag-ref-fast.gguf" fast || rc=1
    fi
    printf '%s, FAST directly vs from the GGUF: ' "$tag"
    $CMP --ignore-card --ulp "$ULP" "$DIR/$tag-fast.gguf" "$DIR/$tag-ref-fast.gguf" | tail -1 | grep 'no differences' || rc=1
    find "$DIR" -name "$tag-*.gguf*" -delete
done
# A git clone with LFS whose working tree has only pointer files (no checkout): read from .git/lfs/objects
$CONV "$DIR/glm-tiny" "$DIR/glm-ours.gguf" bf16-ref || rc=1
( cd "$DIR" && git init -q lfs-src && cd lfs-src && git lfs install --local > /dev/null &&
  git lfs track '*.safetensors' '*.json' > /dev/null && cp "$DIR"/glm-tiny/* . && git add -A &&
  git -c user.name=t -c user.email=t@t commit -qm tiny && cd .. &&
  GIT_LFS_SKIP_SMUDGE=1 git clone -q lfs-src lfs-clone && cd lfs-clone && git lfs fetch > /dev/null ) || rc=1
$CONV --name 'Glm Tiny' "$DIR/lfs-clone" "$DIR/glm-lfs.gguf" bf16-ref || rc=1
printf 'glm, from LFS pointers vs from the files: '
$CMP "$DIR/glm-lfs.gguf" "$DIR/glm-ours.gguf" | tail -1 | grep 'no differences' || rc=1
$CONV "$DIR/glm-tiny" "$DIR/glm-f32.gguf" f32 || rc=1
$CONV "$DIR/glm-f32.gguf" "$DIR/glm-f32-fast.gguf" fast || rc=1
$CONV "$DIR/glm-tiny" "$DIR/glm-fast.gguf" fast || rc=1
printf 'glm, FP8 -> FAST directly vs F32 -> GGUF re-encoded: '
$CMP "$DIR/glm-fast.gguf" "$DIR/glm-f32-fast.gguf" | tail -1 | grep 'no differences' || rc=1
find "$DIR" -delete
[ $rc = 0 ] && echo "PASS: rocm-convert = convert_hf_to_gguf.py, byte for byte" || echo "FAIL: rocm-convert"
exit $rc
