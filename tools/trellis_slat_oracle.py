"""Oracle dumps for the TRELLIS.2 shape-SLAT flow (SLatFlowModel, slat_flow_img2shape_dit_1_3B_512, bf16 checkpoint), fp32 on the GPU.

The input is a random set of N occupied voxels at 32^3 (batch 0), features [N][32], the timestep 0.5 and the image features
[1][1024][1024] (the same seed as the sparse-structure oracle). The python model runs with dtype float32 (the bf16 weights
upcast). Writes slat_*.f32 (token-major [N][C] in the model's coordinate order) and slat_coords.i32 ([N][4]) into ORACLE_DIR:
slat_h0 (input layer), slat_temb (timestep embedding), slat_mod (adaLN), slat_block00/01/14/28/29, slat_out (output layer).
"""
import importlib
import json
import os
import sys

import numpy as np
import torch

REPO = "/work/rocm-hip/TRELLIS.2"
CKPT = "/data/trellis2/TRELLIS.2-4B/ckpts"
ORACLE_DIR = "/data/trellis2/oracle_slat"
N = 6000
sys.path.insert(0, REPO)

from safetensors.torch import load_file  # noqa: E402
from trellis2.modules import sparse as sp  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import trellis2.modules.sparse.attention.modules as sparse_attn_modules  # noqa: E402


def _sdpa_one_sequence(*args):
    """The sparse attention for one sequence (batch 1): plain SDPA over all tokens. The sparse backends here are flash_attn
    (not installed); with one sequence the packed attention is the dense one, so this is the same computation. The inputs
    are VarLenTensors (their .feats) or dense tensors (keys and values of the cross attention: [1, L, 2, H, C])."""
    def feats(x):
        return x.feats if hasattr(x, "feats") else x

    def seq(x):  # a dense [1, L, H, C] (the batch of one) is the sequence [L, H, C]
        x = feats(x)
        return x[0] if x.dim() == 4 else x
    if len(args) == 1:
        qkv = feats(args[0])
        q, k, v = qkv.unbind(1)
    elif len(args) == 2:
        q = feats(args[0])
        kv = feats(args[1])
        if kv.dim() == 5:  # dense [1, L, 2, H, C]
            kv = kv[0]
        k, v = kv.unbind(1)
    else:
        q, k, v = feats(args[0]), seq(args[1]), seq(args[2])
    o = F.scaled_dot_product_attention(q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2))
    o = o.transpose(1, 2)[0]
    return args[0].replace(o) if hasattr(args[0], "replace") else o


sparse_attn_modules.sparse_scaled_dot_product_attention = _sdpa_one_sequence

os.makedirs(ORACLE_DIR, exist_ok=True)
cfg = json.load(open(os.path.join(CKPT, "slat_flow_img2shape_dit_1_3B_512_bf16.json")))
spec = cfg["models"]["denoiser"] if "models" in cfg else cfg
args = dict(spec["args"])
args["dtype"] = "float32"
model = getattr(importlib.import_module("trellis2.models"), spec["name"])(**args)
state = load_file(os.path.join(CKPT, "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"))
res = model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
model = model.float().eval().cuda()
print("load:", res, flush=True)


def save(name, t):
    t.detach().float().contiguous().cpu().numpy().tofile(os.path.join(ORACLE_DIR, name + ".f32"))
    print(name, tuple(t.shape), flush=True)


g = torch.Generator().manual_seed(1)
flat = torch.randperm(32 * 32 * 32, generator=g)[:N]
coords = torch.stack([torch.zeros(N, dtype=torch.long), flat // (32 * 32), (flat // 32) % 32, flat % 32], dim=1).int()
coords.cpu().numpy().tofile(os.path.join(ORACLE_DIR, "slat_coords.i32"))
torch.manual_seed(2)
feats = torch.randn(N, args["in_channels"])
cond = torch.randn(1, 1024, args["cond_channels"])
t = torch.tensor([0.5])
save("slat_feats", feats)
save("slat_cond", cond)
save("slat_t", t)

hooks = []
hooks.append(model.input_layer.register_forward_hook(lambda m, i, o: save("slat_h0", o.feats if hasattr(o, "feats") else o)))
hooks.append(model.t_embedder.register_forward_hook(lambda m, i, o: save("slat_temb", o)))
hooks.append(model.adaLN_modulation.register_forward_hook(lambda m, i, o: save("slat_mod", o)))
for k in (0, 1, 14, 28, 29):
    hooks.append(model.blocks[k].register_forward_hook(
        lambda m, i, o, k=k: save(f"slat_block{k:02d}", o.feats if hasattr(o, "feats") else o)))

with torch.no_grad():
    x = sp.SparseTensor(feats.cuda(), coords.cuda())
    out = model(x, t.cuda(), cond.cuda())
save("slat_out", out.feats if hasattr(out, "feats") else out)
for h in hooks:
    h.remove()
print("slat oracle written to", ORACLE_DIR, flush=True)
