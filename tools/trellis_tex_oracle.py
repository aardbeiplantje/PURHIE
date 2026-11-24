"""Oracle for the TRELLIS.2 texture path of the 512 pipeline (sample_tex_slat and decode_tex_slat), fp32 on the GPU with the shipped
weights upcast. It continues the 512 shape oracle (tools/trellis_shape_oracle.py): the occupied voxels at 32^3 and the de-normalised
shape latent are taken from ORACLE_SHAPE. The shape decoder returns its per-level subdivisions (return_subs), which guide the texture
decoder (the pipeline's decode_shape_slat -> decode_tex_slat). The image features are the shape oracle's seeded [1][1024][1024].

Writes ORACLE_DIR (oracle_tex):
  fwd_x.f32 [N][64]: the texture flow's input at t = 0.5 (noise | normalised shape latent); fwd_out.f32 [N][32]: its output.
  tex_noise.f32 [N][32] and tex_slat.f32 [N][32] (de-normalised): the sampler's noise and result.
  subs{i}_coords.i32 and subs{i}_feats.f32 [N_i][8]: the shape decoder's subdivision logits at level i (i = 0..3).
  tex_slat (below) is the texture decoder's input, on the shape decoder's input voxels.
  tex_dec_raw.f32 [M][6] and tex_out.f32 [M][6] (raw, and *0.5 + 0.5 as the pipeline returns it); tex_out_coords.i32 [M][4].
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
ORACLE_SHAPE = "/data/trellis2/oracle_shape"
ORACLE_DIR = "/data/trellis2/oracle_tex"
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


def load_spec(json_name, safetensors_name):
    cfg = json.load(open(os.path.join(CKPT, json_name)))
    spec = cfg["models"]["denoiser"] if "models" in cfg else cfg
    args = dict(spec["args"])
    args["dtype"] = "float32"
    model = getattr(importlib.import_module("trellis2.models"), spec["name"])(**args)
    res = model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, safetensors_name)).items()}, strict=False)
    print(json_name, "load:", res, flush=True)
    return model.float().eval().cuda()


tex_flow = load_spec("slat_flow_imgshape2tex_dit_1_3B_512_bf16.json", "slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors")
print("tex flow in_channels", tex_flow.in_channels, flush=True)

# the shape decoder (fp32): its return_subs gives the subdivisions that guide the texture decoder
cfg = json.load(open(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.json")))
sdec_args = dict(cfg["args"])
sdec_args["use_fp16"] = False
sdec = importlib.import_module("trellis2.models.sc_vaes.fdg_vae").FlexiDualGridVaeDecoder(**sdec_args)
sdec.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.safetensors")).items()},
                     strict=True)
sdec = sdec.float().eval().cuda()

# the texture decoder (fp32): a sparse UNet without predicted subdivisions
cfg = json.load(open(os.path.join(CKPT, "tex_dec_next_dc_f16c32_fp16.json")))
tex_args = dict(cfg["args"])
tex_args["use_fp16"] = False
accepted = set(inspect.signature(SparseUnetVaeDecoder.__init__).parameters)
dropped = sorted(k for k in tex_args if k not in accepted)
print("texture decoder args not taken by the class:", dropped, flush=True)
tex_args = {k: v for k, v in tex_args.items() if k in accepted}
tdec = SparseUnetVaeDecoder(**tex_args)
res = tdec.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, "tex_dec_next_dc_f16c32_fp16.safetensors")).items()},
                           strict=True)
print("texture decoder load:", res, flush=True)
tdec = tdec.float().eval().cuda()


def save(name, arr, dtype):
    np.asarray(arr).astype(dtype).tofile(os.path.join(ORACLE_DIR, name))


coords32 = torch.from_numpy(np.fromfile(os.path.join(ORACLE_SHAPE, "coords.i32"), dtype=np.int32).reshape(-1, 4)).cuda()
slat = torch.from_numpy(np.fromfile(os.path.join(ORACLE_SHAPE, "slat.f32"), dtype=np.float32).reshape(-1, 32)).cuda()
cond = torch.from_numpy(np.fromfile(os.path.join(ORACLE_SHAPE, "cond.f32"), dtype=np.float32).reshape(1, 1024, 1024)).cuda()
zero = torch.zeros_like(cond)
N = coords32.shape[0]
mean = torch.tensor(shape_norm["mean"]).cuda()[None]
std = torch.tensor(shape_norm["std"]).cuda()[None]
shape_n = (slat - mean) / std                     # the normalised shape latent (the texture flow's concat_cond)

with torch.no_grad():
    # the shape decoder's subdivisions, and its output (the same voxels the texture decoder works on)
    h_sh, subs = SparseUnetVaeDecoder.forward(sdec, sp.SparseTensor(slat, coords32), return_subs=True)
    for i, sub in enumerate(subs):
        save(f"subs{i}_coords.i32", sub.coords.cpu().numpy(), np.int32)
        save(f"subs{i}_feats.f32", sub.feats.cpu().numpy(), np.float32)
    print("subdivision levels:", [int(s.coords.shape[0]) for s in subs], "shape output voxels", int(h_sh.coords.shape[0]), flush=True)

    # the texture flow's single forward at t = 0.5 with the 64-channel input (noise | shape latent)
    g = torch.Generator().manual_seed(10)
    fwd_noise = torch.randn(N, 32, generator=g).cuda()
    fwd_x = torch.cat([fwd_noise, shape_n], dim=1)
    save("fwd_x.f32", fwd_x.cpu().numpy(), np.float32)
    fwd_out = tex_flow(sp.SparseTensor(fwd_x, coords32), torch.tensor([0.5]).cuda(), cond)
    fwd_out = fwd_out.feats if hasattr(fwd_out, "feats") else fwd_out
    save("fwd_out.f32", fwd_out.cpu().numpy(), np.float32)

    # the texture sampler: noise, the normalised shape latent as concat_cond, cond only (guidance strength 1)
    torch.manual_seed(9)
    tex_noise = torch.randn(N, 32).cuda()
    save("tex_noise.f32", tex_noise.cpu().numpy(), np.float32)
    samp = tex_sampler.sample(tex_flow, sp.SparseTensor(tex_noise, coords32), concat_cond=sp.SparseTensor(shape_n, coords32),
                              cond=cond, neg_cond=zero, steps=tex_params["steps"], rescale_t=tex_params["rescale_t"],
                              guidance_strength=tex_params["guidance_strength"],
                              guidance_rescale=tex_params["guidance_rescale"],
                              guidance_interval=tuple(tex_params["guidance_interval"])).samples
    tex_mean = torch.tensor(tex_norm["mean"]).cuda()[None]
    tex_std = torch.tensor(tex_norm["std"]).cuda()[None]
    tex_slat = samp.feats * tex_std + tex_mean
    save("tex_slat.f32", tex_slat.cpu().numpy(), np.float32)

    # the texture decoder, guided by the shape decoder's subdivisions
    out = SparseUnetVaeDecoder.forward(tdec, sp.SparseTensor(tex_slat, coords32), guide_subs=subs)
    assert torch.equal(out.coords, h_sh.coords), "texture and shape decoders disagree on the voxels"
    save("tex_dec_raw.f32", out.feats.cpu().numpy(), np.float32)
    save("tex_out.f32", (out.feats * 0.5 + 0.5).cpu().numpy(), np.float32)
    save("tex_out_coords.i32", out.coords.cpu().numpy(), np.int32)
    print("texture oracle:", N, "voxels at 32^3,", tuple(out.feats.shape), "texture voxels", flush=True)
