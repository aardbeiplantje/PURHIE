"""Oracle stage dumps for the TRELLIS.2 sparse-structure flow (the python model, fp32, on the GPU; GPU matched CPU to 1e-5).

Writes raw little-endian float32 files to ORACLE_DIR, with the inputs (same seed as ss_check.py) and the intermediate
tensors the engine port is checked against. shapes.json lists every file's shape.
"""
import importlib
import json
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "sdpa")
REPO = "/work/rocm-hip/TRELLIS.2"
CKPT = "/data/trellis2/TRELLIS.2-4B/ckpts"
ORACLE_DIR = "/data/trellis2/oracle_ss"
sys.path.insert(0, REPO)

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

os.makedirs(ORACLE_DIR, exist_ok=True)
cfg = json.load(open(os.path.join(CKPT, "ss_flow_img_dit_1_3B_64_bf16.json")))
spec = cfg["models"]["denoiser"] if "models" in cfg else cfg
args = dict(spec["args"])
args["dtype"] = "float32"
model = getattr(importlib.import_module("trellis2.models"), spec["name"])(**args)
state = load_file(os.path.join(CKPT, "ss_flow_img_dit_1_3B_64_bf16.safetensors"))
model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
model = model.float().eval().cuda()

shapes = {}


def save(name, tensor):
    t = tensor.detach().float().contiguous().cpu()
    t.numpy().tofile(os.path.join(ORACLE_DIR, name + ".f32"))
    shapes[name] = list(t.shape)


torch.manual_seed(0)
R, C = args["resolution"], args["in_channels"]
x = torch.randn(1, C, R, R, R)
t = torch.tensor([0.5])
cond = torch.randn(1, 1024, args["cond_channels"])
save("x", x)
save("t", t)
save("cond", cond)
save("rope_phases", torch.view_as_real(model.rope_phases))  # complex [N, D/2] stored as [N, D/2, 2]

hooks = []
hooks.append(model.input_layer.register_forward_hook(lambda m, i, o: save("h0_input_layer", o)))
hooks.append(model.adaLN_modulation.register_forward_hook(lambda m, i, o: save("mod_adaLN", o)))
hooks.append(model.t_embedder.register_forward_hook(lambda m, i, o: save("t_embedder", o)))
for k in (0, 1, 14, 28, 29):
    hooks.append(model.blocks[k].register_forward_hook(
        lambda m, i, o, k=k: save(f"block{k:02d}", o)))
hooks.append(model.out_layer.register_forward_hook(lambda m, i, o: save("out_pre_layer", o)))

with torch.no_grad():
    out = model(x.cuda(), t.cuda(), cond.cuda())
save("out", out)
for h in hooks:
    h.remove()
json.dump(shapes, open(os.path.join(ORACLE_DIR, "shapes.json"), "w"), indent=1)
print("wrote", len(shapes), "tensors to", ORACLE_DIR, "; out", tuple(out.shape), flush=True)
