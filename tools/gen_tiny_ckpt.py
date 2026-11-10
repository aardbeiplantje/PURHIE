"""gen_tiny_ckpt.py <meta-dir> <out-dir> [key=value ...]: a tiny random checkpoint of a release's architecture.

<meta-dir> holds the release's small files (config.json, the tokenizer files, the chat template; no weights). The
config is shrunk by the key=value overrides (JSON values; a dotted key reaches into a nested dict, e.g.
text_config.hidden_size=64; a key without a dot is set at the top and in text_config when that has it), the model
built by transformers from it (the config's architectures[0], random weights, seed 1234) and saved as BF16
safetensors in two shards with an index. With the release's model.safetensors.index.json in <meta-dir> the names are
made the release's where transformers' save differs (a wrapper prefix written more than once, experts one tensor
each where the release fuses them) and any name the release does not have stops it. Tensors transformers does not
build (an MTP block): +mtp=L (Qwen: mtp.layers.0 from decoder layer L), +copy=FROM:TO (the tensors named FROM...
copied as TO...), +rand=NAME=AxB. The input of tests/convert_verify.sh (rocm-convert against llama.cpp's
convert_hf_to_gguf.py) for architectures transformers implements.
"""
import json
import re
import os
import shutil
import sys

import torch
import transformers

meta, out, sets = sys.argv[1], sys.argv[2], [a for a in sys.argv[3:] if not a.startswith("+")]
# +mtp=L: the release's mtp.layers.0.* the tiny model lacks (transformers does not build Qwen's MTP block) as copies
# of decoder layer L's tensors (experts one tensor each), mtp.fc / mtp.*norm random
mtp_from = next((int(a.split("=")[1]) for a in sys.argv[3:] if a.startswith("+mtp=")), None)
# +copy=FROM:TO: the tensors named FROM... copied as TO... (an MTP block made of trunk layers);
# +rand=NAME=AxB: one more random tensor of that shape
copies = [a[6:].split(":") for a in sys.argv[3:] if a.startswith("+copy=")]
rands = [a[6:].rsplit("=", 1) for a in sys.argv[3:] if a.startswith("+rand=")]
os.makedirs(out, exist_ok=True)
cfg = json.load(open(os.path.join(meta, "config.json")))
cfg.pop("quantization_config", None)
for kv in sets:
    k, v = kv.split("=", 1)
    v = json.loads(v)
    if "." in k:
        d, *path = k.split(".")
        node = cfg[d]
        for p in path[:-1]:
            node = node[p]
        node[path[-1]] = v
    else:
        cfg[k] = v
        if isinstance(cfg.get("text_config"), dict) and k in cfg["text_config"]:
            cfg["text_config"][k] = v
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
for f in os.listdir(meta):
    if f not in ("config.json", "model.safetensors.index.json", "headers.json") and os.path.isfile(os.path.join(meta, f)):
        shutil.copy(os.path.join(meta, f), out)

torch.manual_seed(1234)
config = transformers.AutoConfig.from_pretrained(out)
cls = getattr(transformers, cfg["architectures"][0])
model = cls._from_config(config, dtype=torch.bfloat16)
with torch.no_grad():  # norms and biases away from their init constants, so a wrong mapping shows
    for n, p in model.named_parameters():
        if p.dtype.is_floating_point:
            p.copy_(torch.randn_like(p, dtype=torch.float32).mul(0.02).add(1.0 if p.dim() == 1 else 0.0).to(p.dtype))
total = sum(p.numel() * p.element_size() for p in model.state_dict().values())
model.save_pretrained(out, safe_serialization=True, max_shard_size=max(total // 2 + 1, 1 << 20))
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)  # the release's keys, not save_pretrained's


def pattern(name):
    return re.sub(r"shard_\d+", "shard_K", re.sub(r"\.\d+\.", ".N.", name))


# The names as the release has them (its index in <meta-dir>): save_pretrained's can differ (a repeated wrapper
# prefix, experts one tensor each where the release fuses them as gate_up_proj [E, 2 ff, n] / down_proj [E, n, ff])
index = os.path.join(meta, "model.safetensors.index.json")
if os.path.exists(index):
    from safetensors.torch import load_file, save_file
    want = {pattern(n) for n in json.load(open(index))["weight_map"]}
    oidx = json.load(open(os.path.join(out, "model.safetensors.index.json")))
    files = sorted(set(oidx["weight_map"].values()))
    T = {}
    for f in files:
        T.update(load_file(os.path.join(out, f)))
        os.remove(os.path.join(out, f))
    os.remove(os.path.join(out, "model.safetensors.index.json"))
    for n in sorted(T):
        m = re.match(r"^(.*\.mlp\.experts)\.0\.gate_proj\.weight$", n)
        if not m or not any(r.endswith(".mlp.experts.gate_up_proj") for r in want):
            continue
        p = m.group(1)
        ne = sum(1 for k in T if re.match(re.escape(p) + r"\.\d+\.gate_proj\.weight$", k))
        g = [T.pop(f"{p}.{e}.gate_proj.weight") for e in range(ne)]
        u = [T.pop(f"{p}.{e}.up_proj.weight") for e in range(ne)]
        d = [T.pop(f"{p}.{e}.down_proj.weight") for e in range(ne)]
        T[f"{p}.gate_up_proj"] = torch.stack([torch.cat([a, b], 0) for a, b in zip(g, u)]).contiguous()
        T[f"{p}.down_proj"] = torch.stack(d).contiguous()
    fixed = {}
    for n, t in T.items():
        # the release's wrapper prefix for the longest tail of the name a release name ends with
        segs = n.split(".")
        for k in range(len(segs)):
            tail = ".".join(segs[k:])
            hits = {r[: len(r) - len(pattern(tail))] for r in want if r == pattern(tail) or r.endswith("." + pattern(tail))}
            if len(hits) == 1:
                n = hits.pop() + tail
                break
            if hits:
                sys.exit(f"{n}: the release has {sorted(hits)} as prefixes of {tail}")
        fixed[n] = t
    T = fixed
    if mtp_from is not None:
        src = next(n for n in T if re.search(r"\.layers\.%d\." % mtp_from, n)).split(f".layers.{mtp_from}.")[0]
        H = cfg.get("text_config", cfg)["hidden_size"]
        fused = any(r.startswith("mtp.") and r.endswith(".mlp.experts.gate_up_proj") for r in want)
        for n in [n for n in T if n.startswith(f"{src}.layers.{mtp_from}.")]:
            tail = n[len(f"{src}.layers.{mtp_from}."):]
            if pattern("mtp.layers.0." + tail) not in want and not tail.startswith("mlp.experts."):
                continue
            t = T[n]
            if tail == "mlp.experts.gate_up_proj" and not fused:
                ff = t.shape[1] // 2
                for e in range(t.shape[0]):
                    T[f"mtp.layers.0.mlp.experts.{e}.gate_proj.weight"] = t[e, :ff].clone()
                    T[f"mtp.layers.0.mlp.experts.{e}.up_proj.weight"] = t[e, ff:].clone()
            elif tail == "mlp.experts.down_proj" and not fused:
                for e in range(t.shape[0]):
                    T[f"mtp.layers.0.mlp.experts.{e}.down_proj.weight"] = t[e].clone()
            else:
                T[f"mtp.layers.0.{tail}"] = (t.float() + 0.01).to(t.dtype)
        # the MTP block's own tensors: a trunk tensor of the same name's shape, [H, 2 H] for fc, [H, H] for fc_*,
        # [H] for norms
        for r in sorted(w for w in want if w.startswith("mtp.") and not w.startswith("mtp.layers.")):
            tail = r[4:]
            like = T.get(f"{src}.{tail}")
            shape = like.shape if like is not None else (H, 2 * H) if tail == "fc.weight" else (H,) if "norm" in tail else (H, H)
            T[r] = (torch.randn(*shape) * 0.02).to(torch.bfloat16)
    for a, b in copies:
        for n in [n for n in T if n.startswith(a)]:
            T[b + n[len(a):]] = (T[n].float() + 0.01).to(T[n].dtype)
    for n, shape in rands:
        T[n] = (torch.randn(*[int(x) for x in shape.split("x")]) * 0.02).to(torch.bfloat16)
    extra = sorted({pattern(n) for n in T} - want)
    if extra:
        sys.exit(f"names not in the release: {extra}")
    names = sorted(T)
    wmap = {}
    for i, part in enumerate((names[: len(names) // 2], names[len(names) // 2:])):
        fn = f"model-{i + 1:05d}-of-00002.safetensors"
        save_file({n: T[n].contiguous() for n in part}, os.path.join(out, fn), metadata={"format": "pt"})
        wmap.update({n: fn for n in part})
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    missing = sorted(want - {pattern(n) for n in T})
    if missing:
        print(f"release names the tiny model lacks: {missing}")
print(f"{out}: {cfg['architectures'][0]}, {len(model.state_dict())} tensors, {total / 1e6:.1f} MB")
