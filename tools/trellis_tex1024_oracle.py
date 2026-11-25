"""Oracle for the TRELLIS.2 texture pass of the 1024 cascade (the pipeline's '1024_cascade': sample_tex_slat with the 1024 texture flow,
decode_tex_slat guided by the 1024 shape decoder's subdivisions), fp32 on the GPU with the shipped weights upcast. It continues the
cascade oracle (tools/trellis_cascade_oracle.py): the 64^3 voxels and the 1024 shape latent (de-normalised) are taken from
ORACLE_CASCADE, and the 1024 image features are its seeded cond_hr [4096][1024].

The shape decoder's subdivisions on the 64^3 voxels (return_subs, the 1024 decode), then the texture flow on [noise | normalised shape
latent] with the conditional prediction alone (guidance 1), de-normalised with tex_slat_normalization, then the texture decoder guided
by those subdivisions, x * 0.5 + 0.5 for the pipeline's output.

Writes ORACLE_DIR (oracle_tex1024): fwd_x.f32 [K][64] and fwd_out.f32 [K][32] (the flow's forward at t = 0.5); subs{i}_coords.i32 and
subs{i}_feats.f32 (the 1024 decode's subdivisions, i = 0..3); tex_noise.f32 [K][32]; tex_slat.f32 [K][32]; tex_dec_raw.f32 [D][6];
tex_out.f32 [D][6]; tex_out_coords.i32 [D][4]; tex_norm.f32 (the texture normalisation, 32 means then 32 stds).
"""
import importlib
import inspect
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("ATTN_BACKEND", "sdpa")
REPO = "/work/rocm-hip/TRELLIS.2"
CKPT = "/data/trellis2/TRELLIS.2-4B/ckpts"
ORACLE_CASCADE = "/data/trellis2/oracle_cascade"
ORACLE_DIR = "/data/trellis2/oracle_tex1024"
sys.path.insert(0, REPO)
for name in ["nvdiffrast", "nvdiffrast.torch"]:
    sys.modules[name] = types.ModuleType(name)

from safetensors.torch import load_file  # noqa: E402
from trellis2.modules import sparse as sp  # noqa: E402
from trellis2.pipelines import samplers  # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402
import trellis2.modules.sparse.attention.modules as sparse_attn_modules  # noqa: E402


def _sdpa_one_sequence(*args):
    """Sparse attention for one sequence as plain SDPA (flash_attn is not installed); see tools/trellis_shape_oracle.py."""
    def feats(x):
        return x.feats if hasattr(x, "feats") else x

    def seq(x):
        x = feats(x)
        return x[0] if x.dim() == 4 else x
    if len(args) == 1:
        q, k, v = feats(args[0]).unbind(1)
    elif len(args) == 2:
        q = feats(args[0])
        kv = feats(args[1])
        if kv.dim() == 5:
            kv = kv[0]
        k, v = kv.unbind(1)
    else:
        q, k, v = feats(args[0]), seq(args[1]), seq(args[2])
    o = F.scaled_dot_product_attention(q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2))
    o = o.transpose(1, 2)[0]
    return args[0].replace(o) if hasattr(args[0], "replace") else o


sparse_attn_modules.sparse_scaled_dot_product_attention = _sdpa_one_sequence
os.makedirs(ORACLE_DIR, exist_ok=True)

pipe = json.load(open(os.path.join(CKPT, "..", "pipeline.json")))["args"]
tex_params = pipe["tex_slat_sampler"]["params"]
tex_norm = pipe["tex_slat_normalization"]
shape_norm = pipe["shape_slat_normalization"]
tex_sampler = samplers.FlowEulerGuidanceIntervalSampler(**pipe["tex_slat_sampler"]["args"])

# the 1024 texture flow (64 inputs)
cfg = json.load(open(os.path.join(CKPT, "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json")))
spec = cfg["models"]["denoiser"] if "models" in cfg else cfg
args = dict(spec["args"])
args["dtype"] = "float32"
tex_flow = getattr(importlib.import_module("trellis2.models"), spec["name"])(**args)
res = tex_flow.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors")).items()},
                               strict=False)
print("1024 texture flow load:", res, "in_channels", tex_flow.in_channels, flush=True)
tex_flow = tex_flow.float().eval().cuda()

# the shape decoder (fp32): the 1024 decode's subdivisions
cfg = json.load(open(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.json")))
sdec_args = dict(cfg["args"])
sdec_args["use_fp16"] = False
sdec = importlib.import_module("trellis2.models.sc_vaes.fdg_vae").FlexiDualGridVaeDecoder(**sdec_args)
sdec.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.safetensors")).items()},
                     strict=True)
sdec = sdec.float().eval().cuda()

# the texture decoder (fp32)
cfg = json.load(open(os.path.join(CKPT, "tex_dec_next_dc_f16c32_fp16.json")))
tex_args = dict(cfg["args"])
tex_args["use_fp16"] = False
accepted = set(inspect.signature(SparseUnetVaeDecoder.__init__).parameters)
tex_args = {k: v for k, v in tex_args.items() if k in accepted}
tdec = SparseUnetVaeDecoder(**tex_args)
tdec.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, "tex_dec_next_dc_f16c32_fp16.safetensors")).items()}, strict=True)
tdec = tdec.float().eval().cuda()


def save(name, arr, dtype):
    np.asarray(arr).astype(dtype).tofile(os.path.join(ORACLE_DIR, name))


hr_coords = torch.from_numpy(np.fromfile(os.path.join(ORACLE_CASCADE, "hr_coords.i32"), dtype=np.int32).reshape(-1, 4)).cuda()
hr_slat = torch.from_numpy(np.fromfile(os.path.join(ORACLE_CASCADE, "hr_slat.f32"), dtype=np.float32).reshape(-1, 32)).cuda()
cond_hr = torch.from_numpy(np.fromfile(os.path.join(ORACLE_CASCADE, "cond_hr.f32"), dtype=np.float32).reshape(1, 4096, 1024)).cuda()
zero = torch.zeros_like(cond_hr)
K = hr_coords.shape[0]
mean = torch.tensor(shape_norm["mean"]).cuda()[None]
std = torch.tensor(shape_norm["std"]).cuda()[None]
shape_n = (hr_slat - mean) / std
save("tex_norm.f32", np.array(tex_norm["mean"] + tex_norm["std"]), np.float32)

with torch.no_grad():
    # the 1024 decode's subdivisions (the shape decoder's return_subs), the guide of the texture decoder
    h_sh, subs = SparseUnetVaeDecoder.forward(sdec, sp.SparseTensor(hr_slat, hr_coords), return_subs=True)
    for i, sub in enumerate(subs):
        save(f"subs{i}_coords.i32", sub.coords.cpu().numpy(), np.int32)
        save(f"subs{i}_feats.f32", sub.feats.cpu().numpy(), np.float32)
    print("1024 subdivision levels:", [int(s.coords.shape[0]) for s in subs], "decoder voxels", int(h_sh.coords.shape[0]), flush=True)

    # the flow's forward at t = 0.5 on [noise | normalised shape latent]
    g = torch.Generator().manual_seed(11)
    fwd_noise = torch.randn(K, 32, generator=g).cuda()
    fwd_x = torch.cat([fwd_noise, shape_n], dim=1)
    save("fwd_x.f32", fwd_x.cpu().numpy(), np.float32)
    fwd_out = tex_flow(sp.SparseTensor(fwd_x, hr_coords), torch.tensor([0.5]).cuda(), cond_hr)
    fwd_out = fwd_out.feats if hasattr(fwd_out, "feats") else fwd_out
    save("fwd_out.f32", fwd_out.cpu().numpy(), np.float32)

    # the sampler: the noise, the normalised shape latent as concat_cond, the conditional prediction alone
    torch.manual_seed(12)
    tex_noise = torch.randn(K, 32).cuda()
    save("tex_noise.f32", tex_noise.cpu().numpy(), np.float32)
    samp = tex_sampler.sample(tex_flow, sp.SparseTensor(tex_noise, hr_coords), concat_cond=sp.SparseTensor(shape_n, hr_coords),
                              cond=cond_hr, neg_cond=zero, steps=tex_params["steps"], rescale_t=tex_params["rescale_t"],
                              guidance_strength=tex_params["guidance_strength"],
                              guidance_rescale=tex_params["guidance_rescale"],
                              guidance_interval=tuple(tex_params["guidance_interval"])).samples
    tmean = torch.tensor(tex_norm["mean"]).cuda()[None]
    tstd = torch.tensor(tex_norm["std"]).cuda()[None]
    tex_slat = samp.feats * tstd + tmean
    save("tex_slat.f32", tex_slat.cpu().numpy(), np.float32)

    # the texture decoder, guided by the 1024 decode's subdivisions
    out = SparseUnetVaeDecoder.forward(tdec, sp.SparseTensor(tex_slat, hr_coords), guide_subs=subs)
    assert torch.equal(out.coords, h_sh.coords), "texture and shape decoders disagree on the voxels"
    save("tex_dec_raw.f32", out.feats.cpu().numpy(), np.float32)
    save("tex_out.f32", (out.feats * 0.5 + 0.5).cpu().numpy(), np.float32)
    save("tex_out_coords.i32", out.coords.cpu().numpy(), np.int32)
    print("1024 texture oracle:", K, "voxels at 64^3,", tuple(out.feats.shape), "texture voxels", flush=True)
