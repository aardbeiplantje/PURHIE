"""gen_glm5next_ckpt.py <meta-dir> <out-dir>: a tiny random GLM-5.3-Flash (glm5_next) Hugging Face checkpoint.

<meta-dir> holds the release's small files (config.json, tokenizer.json, tokenizer_config.json, generation_config.json,
chat_template.jinja; huggingface.co/zai-org/GLM-5.3-Flash). The checkpoint has the release's tensor names, dtypes
(FP8 e4m3 weights with float32 128x128 block scales where the release has them, BF16 / F32 elsewhere), layer
pattern (dense FFN first, KDA linear attention with sparse MLA every fourth layer, the MTP layer after the last) and
tokenizer, at small widths, in two shards with an index: the input of tests/convert_verify.sh (rocm-convert against
llama.cpp's convert_hf_to_gguf.py). Needs torch and safetensors.
"""
import json
import math
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

meta, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(meta, "config.json")))
t = cfg["text_config"]

# The tiny widths: the dimensions FP8 blocks of 128 cover, and the release's head sizes where kernels assume them
NL = 8          # decoder layers; the MTP layer is layer NL
H = 256         # hidden
t.update(hidden_size=H, intermediate_size=512, moe_intermediate_size=128, n_routed_experts=8, num_experts_per_tok=2,
         num_hidden_layers=NL, num_attention_heads=2, num_key_value_heads=2, q_lora_rank=128, kv_lora_rank=128,
         qk_nope_head_dim=64, qk_head_dim=64, v_head_dim=64, index_n_heads=4, index_head_dim=64, index_topk=16,
         first_k_dense_replace=3)
lin = t["linear_attn_config"]
lin["num_heads"] = 2  # head_dim stays 128
kinds = ["deepseek_sparse_attention" if i % 4 == 3 else "linear_attention" for i in range(NL)]
t["layer_types"] = kinds
t["mlp_layer_types"] = ["dense" if i < t["first_k_dense_replace"] else "sparse" for i in range(NL)]
t["indexer_types"] = ["full"] * NL
lin["kda_layers"] = [i for i in range(NL) if kinds[i] == "linear_attention"]
lin["full_attn_layers"] = [i for i in range(NL) if kinds[i] != "linear_attention"]
cfg.pop("vision_config", None)
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
    if os.path.exists(os.path.join(meta, f)):
        shutil.copy(os.path.join(meta, f), out)

g = torch.Generator().manual_seed(1234)
V, I, IM, E = t["vocab_size"], t["intermediate_size"], t["moe_intermediate_size"], t["n_routed_experts"]
KH, KD = lin["num_heads"], lin["head_dim"]
NH, QL, KVL, QK, VH = t["num_attention_heads"], t["q_lora_rank"], t["kv_lora_rank"], t["qk_nope_head_dim"], t["v_head_dim"]
IH, ID, HC = t["index_n_heads"], t["index_head_dim"], t["hc_mult"]
tensors = {}


def bf16(name, *shape, std=0.02):
    tensors[name] = (torch.randn(*shape, generator=g) * std).to(torch.bfloat16)


def f32(name, *shape, std=0.1):
    tensors[name] = torch.randn(*shape, generator=g) * std


def ones(name, n):
    tensors[name] = (1.0 + torch.randn(n, generator=g) * 0.05).to(torch.bfloat16)


def fp8(name, rows, cols):
    """FP8 e4m3 weights in [-448, 448] and a positive float32 scale per 128 x 128 block"""
    tensors[name] = (torch.randn(rows, cols, generator=g) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    sr, sc = math.ceil(rows / 128), math.ceil(cols / 128)
    tensors[name + "_scale_inv"] = (torch.rand(sr, sc, generator=g) * 1.5 + 0.25) * 3e-4


p = "model.language_model"
bf16("lm_head.weight", V, H)
bf16(f"{p}.embed_tokens.weight", V, H)
ones(f"{p}.norm.weight", H)
for L in range(NL + 1):
    q = f"{p}.layers.{L}"
    mtp = L == NL
    ones(f"{q}.input_layernorm.weight", H)
    ones(f"{q}.post_attention_layernorm.weight", H)
    if not mtp:
        for k in ("attn", "ffn"):
            f32(f"{q}.hc_{k}_base", 2 * HC + HC * HC)
            bf16(f"{q}.hc_{k}_fn", 2 * HC + HC * HC, HC * H)
            f32(f"{q}.hc_{k}_scale", 3)
    a = f"{q}.self_attn"
    if mtp or kinds[L] != "linear_attention":  # sparse MLA with the DSA indexer
        fp8(f"{a}.q_a_proj.weight", QL, H)
        ones(f"{a}.q_a_layernorm.weight", QL)
        fp8(f"{a}.q_b_proj.weight", NH * QK, QL)
        fp8(f"{a}.kv_a_proj_with_mqa.weight", KVL, H)
        ones(f"{a}.kv_a_layernorm.weight", KVL)
        bf16(f"{a}.kv_b_proj.weight", NH * (QK + VH), KVL)
        fp8(f"{a}.o_proj.weight", H, NH * VH)
        x = f"{a}.indexer"
        bf16(f"{x}.wq_b.weight", IH * ID, QL)
        bf16(f"{x}.wk.weight", ID, H)
        ones(f"{x}.k_norm.weight", ID)
        bf16(f"{x}.k_norm.bias", ID)
        bf16(f"{x}.weights_proj.weight", IH, H)
        bf16(f"{x}.index_kpool_compress_ape", t["index_kpool"], ID)
        bf16(f"{x}.index_kpool_compress_gate", ID, H)
    else:  # KDA
        for k in ("q", "k", "v"):
            bf16(f"{a}.{k}_proj.weight", KH * KD, H)
            bf16(f"{a}.{k}_conv1d.weight", KH * KD, 1, lin["short_conv_kernel_size"], std=0.3)
        bf16(f"{a}.f_a_proj.weight", KD, H)
        bf16(f"{a}.f_b_proj.weight", KH * KD, KD)
        bf16(f"{a}.g_a_proj.weight", KD, H)
        bf16(f"{a}.g_b_proj.weight", KH * KD, KD)
        bf16(f"{a}.b_proj.weight", KH, H)
        f32(f"{a}.A_log", KH)
        f32(f"{a}.dt_bias", KH * KD)
        ones(f"{a}.o_norm.weight", KD)
        bf16(f"{a}.o_proj.weight", H, KH * KD)
    m = f"{q}.mlp"
    if not mtp and L < t["first_k_dense_replace"]:
        fp8(f"{m}.gate_proj.weight", I, H)
        fp8(f"{m}.up_proj.weight", I, H)
        fp8(f"{m}.down_proj.weight", H, I)
    else:
        bf16(f"{m}.gate.weight", E, H)
        f32(f"{m}.gate.e_score_correction_bias", E, std=0.01)
        for e in range(E):
            fp8(f"{m}.experts.{e}.gate_proj.weight", IM, H)
            fp8(f"{m}.experts.{e}.up_proj.weight", IM, H)
            fp8(f"{m}.experts.{e}.down_proj.weight", H, IM)
        fp8(f"{m}.shared_experts.gate_proj.weight", IM, H)
        fp8(f"{m}.shared_experts.up_proj.weight", IM, H)
        fp8(f"{m}.shared_experts.down_proj.weight", H, IM)
    if mtp:
        bf16(f"{q}.eh_proj.weight", H, 2 * H)
        ones(f"{q}.enorm.weight", H)
        ones(f"{q}.hnorm.weight", H)
        ones(f"{q}.shared_head.norm.weight", H)

names = sorted(tensors)
half = len(names) // 2
shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
wmap, total = {}, 0
for fn, ns in shards.items():
    save_file({n: tensors[n].contiguous() for n in ns}, os.path.join(out, fn), metadata={"format": "pt"})
    for n in ns:
        wmap[n] = fn
        total += tensors[n].numel() * tensors[n].element_size()
json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
          open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
print(f"{out}: {len(tensors)} tensors, {total / 1e6:.1f} MB, {NL} layers + MTP")
