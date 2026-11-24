"""Debug oracle for block 0 of the TRELLIS.2 sparse-structure flow: the python block's intermediates, fp32 on the GPU,
from the oracle's own inputs (h0_input_layer, mod_adaLN, cond, rope_phases). Writes dbg_*.f32 next to the dumps."""
import importlib
import json
import os
import sys

os.environ.setdefault("ATTN_BACKEND", "sdpa")
sys.path.insert(0, "/work/rocm-hip/TRELLIS.2")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

ORACLE = "/data/trellis2/oracle_ss"
CKPT = "/data/trellis2/TRELLIS.2-4B/ckpts"
cfg = json.load(open(os.path.join(CKPT, "ss_flow_img_dit_1_3B_64_bf16.json")))
spec = cfg["models"]["denoiser"] if "models" in cfg else cfg
args = dict(spec["args"])
args["dtype"] = "float32"
model = getattr(importlib.import_module("trellis2.models"), spec["name"])(**args)
state = load_file(os.path.join(CKPT, "ss_flow_img_dit_1_3B_64_bf16.safetensors"))
model.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
model = model.float().eval().cuda()


def rd(name, shape):
    return torch.from_numpy(np.fromfile(os.path.join(ORACLE, name + ".f32"), dtype=np.float32).reshape(shape))


def save(name, t):
    t.detach().float().contiguous().cpu().numpy().tofile(os.path.join(ORACLE, name + ".f32"))


N, C, H, D = 4096, 1536, 12, 128
blk = model.blocks[0]
x = rd("h0_input_layer", (1, N, C)).cuda()
mod = rd("mod_adaLN", (1, 9216)).cuda()
cond = rd("cond", (1, 1024, 1024)).cuda()
phases = torch.view_as_complex(rd("rope_phases", (N, 64, 2)).cuda().contiguous())  # complex [N, 64]

with torch.no_grad():
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (blk.modulation + mod).chunk(6, dim=1)
    h = blk.norm1(x)
    h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
    save("dbg_ln1", h)
    sa = blk.self_attn
    qkv = sa.to_qkv(h).reshape(1, N, 3, H, D)
    q, k, v = qkv.unbind(dim=2)
    q = sa.q_rms_norm(q)
    k = sa.k_rms_norm(k)
    q = blk.self_attn.apply_rotary_embedding(q, phases) if hasattr(sa, "apply_rotary_embedding") else \
        __import__("trellis2.modules.attention.rope", fromlist=["x"]).RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
    k = __import__("trellis2.modules.attention.rope", fromlist=["x"]).RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
    save("dbg_qk", torch.cat([q.reshape(1, N, C), k.reshape(1, N, C), v.reshape(1, N, C)], dim=-1))
    o = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
    o = o.permute(0, 2, 1, 3).reshape(1, N, C)
    save("dbg_attn", o)
    hs = sa.to_out(o)
    x = x + hs * gate_msa.unsqueeze(1)
    save("dbg_self", x)
    ca = blk.cross_attn
    hq = blk.norm2(x)
    save("dbg_cq_ln", hq)
    save("dbg_cq_lin", ca.to_q(hq))
    qc = ca.to_q(hq).reshape(1, N, H, D)
    kv = ca.to_kv(cond).reshape(1, 1024, 2, H, D)
    kc, vc = kv.unbind(dim=2)
    qc = ca.q_rms_norm(qc)
    kc = ca.k_rms_norm(kc)
    save("dbg_cq", qc.reshape(1, N, C)); save("dbg_ck", kc.reshape(1, 1024, C)); save("dbg_cv", vc.reshape(1, 1024, C))
    oc = F.scaled_dot_product_attention(qc.permute(0, 2, 1, 3), kc.permute(0, 2, 1, 3), vc.permute(0, 2, 1, 3))
    oc = oc.permute(0, 2, 1, 3).reshape(1, N, C)
    save("dbg_cross_attn", oc)
    x = x + ca.to_out(oc)
    save("dbg_cross", x)
    hm = blk.norm3(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
    x = x + blk.mlp(hm) * gate_mlp.unsqueeze(1)
    ref = rd("block00", (1, N, C)).cuda()
    print("block00 self-check (python block vs oracle): rel", ((x - ref).norm() / ref.norm()).item(), flush=True)
print("debug dumps written", flush=True)
