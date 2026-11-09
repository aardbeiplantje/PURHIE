"""gen_dsv4_ckpt.py <meta-dir> <out-dir>: a tiny random DeepSeek-V4-Flash checkpoint (DeepSeek's own layout).

<meta-dir> holds the release's small files (config.json, tokenizer.json, tokenizer_config.json, generation_config.json;
huggingface.co/deepseek-ai/DeepSeek-V4-Flash). The checkpoint has the release's tensor names and dtypes: FP8 e4m3
weights with E8M0 (power-of-two) 128x128 block scales, routed experts as packed FP4 (two values a byte, I8) with an
E8M0 scale per 32 values, hash-routing tables (I64) in the first layers, KV compressors by compression ratio (4:
with the sparse indexer, 128: without), the MTP layer; small widths, the real tokenizer; two shards and an index.
The input of tests/convert_verify.sh (rocm-convert against llama.cpp's convert_hf_to_gguf.py). numpy only: the
shards are written directly (torch has no E8M0 dtype before 2.7).
"""
import json
import math
import os
import shutil
import struct
import sys

import numpy as np

meta, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
c = json.load(open(os.path.join(meta, "config.json")))
H, NL = 256, 6
c.update(hidden_size=H, num_hidden_layers=NL, num_attention_heads=4, head_dim=128, qk_rope_head_dim=64,
         q_lora_rank=128, o_groups=2, o_lora_rank=128, index_n_heads=4, index_head_dim=64, index_topk=16,
         n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=128, num_hash_layers=3,
         compress_ratios=[0, 0, 4, 128, 4, 128, 0])
json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    if os.path.exists(os.path.join(meta, f)):
        shutil.copy(os.path.join(meta, f), out)

rng = np.random.default_rng(1234)
V, NH, HD, QL = c["vocab_size"], c["num_attention_heads"], c["head_dim"], c["q_lora_rank"]
OG, OL, IH, ID = c["o_groups"], c["o_lora_rank"], c["index_n_heads"], c["index_head_dim"]
E, K, IM, HC = c["n_routed_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"], c["hc_mult"]
T = {}  # name -> (dtype string, shape, bytes)


def put(name, dtype, shape, data):
    T[name] = (dtype, list(shape), data.tobytes())


def f32(name, *shape, std=0.1):
    put(name, "F32", shape, (rng.standard_normal(shape) * std).astype("<f4"))


def bf16(name, *shape, std=0.02, base=0.0):
    x = (base + rng.standard_normal(shape) * std).astype("<f4").view("<u4")
    put(name, "BF16", shape, ((x + 0x7FFF + ((x >> 16) & 1)) >> 16).astype("<u2"))


def fp8(name, rows, cols):
    """e4m3 bytes (no NaN codes) and an E8M0 scale per 128 x 128 block"""
    w = rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x3F
    put(name + ".weight", "F8_E4M3", (rows, cols), w)
    put(name + ".scale", "F8_E8M0", (math.ceil(rows / 128), math.ceil(cols / 128)),
        rng.integers(112, 124, size=(math.ceil(rows / 128), math.ceil(cols / 128)), dtype=np.uint8))


def fp4(name, rows, cols):
    """FP4 pairs a byte and an E8M0 scale per 32 values"""
    put(name + ".weight", "I8", (rows, cols // 2), rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8))
    put(name + ".scale", "F8_E8M0", (rows, cols // 32), rng.integers(118, 128, size=(rows, cols // 32), dtype=np.uint8))


bf16("embed.weight", V, H)
bf16("head.weight", V, H)
bf16("norm.weight", H, base=1.0)
f32("hc_head_fn", HC, HC * H)
f32("hc_head_base", HC)
f32("hc_head_scale", 1)


def layer(p, ratio, hashed, mtp=False):
    bf16(f"{p}.attn_norm.weight", H, base=1.0)
    bf16(f"{p}.ffn_norm.weight", H, base=1.0)
    for k in ("attn", "ffn"):
        f32(f"{p}.hc_{k}_base", 2 * HC + HC * HC)
        f32(f"{p}.hc_{k}_fn", 2 * HC + HC * HC, HC * H)
        f32(f"{p}.hc_{k}_scale", 3)
    a = f"{p}.attn"
    f32(f"{a}.attn_sink", NH)
    bf16(f"{a}.q_norm.weight", QL, base=1.0)
    bf16(f"{a}.kv_norm.weight", HD, base=1.0)
    fp8(f"{a}.wq_a", QL, H)
    fp8(f"{a}.wq_b", NH * HD, QL)
    fp8(f"{a}.wkv", HD, H)
    fp8(f"{a}.wo_a", OG * OL, NH * HD // OG)
    fp8(f"{a}.wo_b", H, OG * OL)
    if ratio:
        w = 2 * HD if ratio == 4 else HD
        f32(f"{a}.compressor.ape", ratio, w)
        bf16(f"{a}.compressor.wkv.weight", w, H)
        bf16(f"{a}.compressor.wgate.weight", w, H)
        bf16(f"{a}.compressor.norm.weight", HD, base=1.0)
        if ratio == 4:
            x = f"{a}.indexer"
            f32(f"{x}.compressor.ape", ratio, 2 * ID)
            bf16(f"{x}.compressor.wkv.weight", 2 * ID, H)
            bf16(f"{x}.compressor.wgate.weight", 2 * ID, H)
            bf16(f"{x}.compressor.norm.weight", ID, base=1.0)
            bf16(f"{x}.weights_proj.weight", IH, H)
            fp8(f"{x}.wq_b", IH * ID, QL)
    f_ = f"{p}.ffn"
    bf16(f"{f_}.gate.weight", E, H)
    if hashed:
        put(f"{f_}.gate.tid2eid", "I64", (V, K), rng.integers(0, E, size=(V, K)).astype("<i8"))
    else:
        f32(f"{f_}.gate.bias", E, std=0.01)
    for e in range(E):
        fp4(f"{f_}.experts.{e}.w1", IM, H)
        fp4(f"{f_}.experts.{e}.w2", H, IM)
        fp4(f"{f_}.experts.{e}.w3", IM, H)
    fp8(f"{f_}.shared_experts.w1", IM, H)
    fp8(f"{f_}.shared_experts.w2", H, IM)
    fp8(f"{f_}.shared_experts.w3", IM, H)
    if mtp:
        fp8(f"{p}.e_proj", H, H)
        fp8(f"{p}.h_proj", H, H)
        for n in ("enorm", "hnorm", "norm"):
            bf16(f"{p}.{n}.weight", H, base=1.0)
        f32(f"{p}.hc_head_fn", HC, HC * H)
        f32(f"{p}.hc_head_base", HC)
        f32(f"{p}.hc_head_scale", 1)


for L in range(NL):
    layer(f"layers.{L}", c["compress_ratios"][L], L < c["num_hash_layers"])
layer("mtp.0", c["compress_ratios"][NL], False, mtp=True)


def write(path, names):
    header, off = {}, 0
    for n in names:
        dt, shape, data = T[n]
        header[n] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
    header["__metadata__"] = {"format": "pt"}
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for n in names:
            f.write(T[n][2])


names = sorted(T)
shards = {"model-00001-of-00002.safetensors": names[: len(names) // 2],
          "model-00002-of-00002.safetensors": names[len(names) // 2:]}
wmap, total = {}, 0
for fn, ns in shards.items():
    write(os.path.join(out, fn), ns)
    for n in ns:
        wmap[n] = fn
        total += len(T[n][2])
json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
          open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
print(f"{out}: {len(T)} tensors, {total / 1e6:.1f} MB, {NL} layers + MTP")
