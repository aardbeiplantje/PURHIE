"""Oracle for the TRELLIS.2 1024 cascade (the pipeline's '1024_cascade': sample_shape_slat_cascade and decode_latent at 1024), fp32
on the GPU with the shipped weights upcast. It continues the 512 shape oracle (tools/trellis_shape_oracle.py): the occupied voxels
at 32^3 and the 512 shape latent (the cascade's low-resolution pass) are taken from ORACLE_SHAPE, as the 512 pipeline produced them.

The decoder's first four levels give the voxels at 512 (SparseUnetVaeDecoder.upsample), quantised to 64^3 and made unique; the
1024 shape-SLAT flow samples on them from seeded noise (de-normalised with shape_slat_normalization); the shape decoder gives the
dual-grid features at 1024; the mesh is o-voxel's flexible_dual_grid_to_mesh at grid 1024. The 1024 image features are a seeded
random [1][4096][1024] (the DINOv3 encoder at 1024 px is not on the engine).

Writes ORACLE_DIR (oracle_cascade): coords512.i32 [M][4], hr_coords.i32 [K][4], hr_noise.f32 [K][32], cond_hr.f32 [4096][1024],
hr_slat.f32 [K][32] (de-normalised), dec_feats.f32 [D][7], dec_coords.i32 [D][4], mesh_vertices.f32, mesh_faces.i32.
"""
import importlib
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
CKPT_SS_DEC = "/data/trellis2/TRELLIS-image-large/ckpts"
ORACLE_SHAPE = "/data/trellis2/oracle_shape"
ORACLE_DIR = "/data/trellis2/oracle_cascade"
GRID = 1024
COND_LEN_HR = 4096
sys.path.insert(0, REPO)
for name in ["nvdiffrast", "nvdiffrast.torch"]:
    sys.modules[name] = types.ModuleType(name)

from safetensors.torch import load_file  # noqa: E402
from o_voxel.convert import flexible_dual_grid_to_mesh  # noqa: E402
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
slat_params = pipe["shape_slat_sampler"]["params"]
norm = pipe["shape_slat_normalization"]
slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(**pipe["shape_slat_sampler"]["args"])

# the shape decoder (fp32), the 1024 shape-SLAT flow
cfg = json.load(open(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.json")))
sdec_args = dict(cfg["args"])
sdec_args["use_fp16"] = False
sdec = importlib.import_module("trellis2.models.sc_vaes.fdg_vae").FlexiDualGridVaeDecoder(**sdec_args)
sdec.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.safetensors")).items()},
                     strict=True)
sdec = sdec.float().eval().cuda()

cfg = json.load(open(os.path.join(CKPT, "slat_flow_img2shape_dit_1_3B_1024_bf16.json")))
spec = cfg["models"]["denoiser"] if "models" in cfg else cfg
args = dict(spec["args"])
args["dtype"] = "float32"
slat_hr = getattr(importlib.import_module("trellis2.models"), spec["name"])(**args)
res = slat_hr.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors")).items()},
                              strict=False)
print("1024 flow load:", res, flush=True)
slat_hr = slat_hr.float().eval().cuda()


def save(name, arr, dtype):
    np.asarray(arr).astype(dtype).tofile(os.path.join(ORACLE_DIR, name))


# the 512 stages (the shape oracle's)
coords32 = torch.from_numpy(np.fromfile(os.path.join(ORACLE_SHAPE, "coords.i32"), dtype=np.int32).reshape(-1, 4)).cuda()
slat_lr = torch.from_numpy(np.fromfile(os.path.join(ORACLE_SHAPE, "slat.f32"), dtype=np.float32).reshape(-1, 32)).cuda()

with torch.no_grad():
    # the cascade's upsampling: the decoder's first four levels (python's upsample), then the 64^3 voxels
    up = SparseUnetVaeDecoder.upsample(sdec, sp.SparseTensor(slat_lr, coords32), upsample_times=4)
    coords512 = up.int()
    save("coords512.i32", coords512.cpu().numpy(), np.int32)
    quant = torch.cat([coords512[:, :1], ((coords512[:, 1:] + 0.5) / 512 * (1024 // 16)).int()], dim=1)
    hr_coords = quant.unique(dim=0)
    save("hr_coords.i32", hr_coords.cpu().numpy(), np.int32)
    K = hr_coords.shape[0]
    print("upsampled voxels at 512:", coords512.shape[0], "at 64^3:", K, flush=True)

    cond_hr = torch.randn(1, COND_LEN_HR, 1024, generator=torch.Generator().manual_seed(7)).cuda()
    save("cond_hr.f32", cond_hr.cpu().numpy(), np.float32)
    zero_hr = torch.zeros_like(cond_hr)

    torch.manual_seed(8)
    hr_noise = torch.randn(K, 32).cuda()
    save("hr_noise.f32", hr_noise.cpu().numpy(), np.float32)
    hr_sp = slat_sampler.sample(slat_hr, sp.SparseTensor(feats=hr_noise, coords=hr_coords.cuda()), cond=cond_hr, neg_cond=zero_hr,
                                steps=slat_params["steps"], rescale_t=slat_params["rescale_t"],
                                guidance_strength=slat_params["guidance_strength"], guidance_rescale=slat_params["guidance_rescale"],
                                guidance_interval=tuple(slat_params["guidance_interval"])).samples
    std = torch.tensor(norm["std"]).cuda()[None]
    mean = torch.tensor(norm["mean"]).cuda()[None]
    hr_slat = hr_sp.feats * std + mean
    save("hr_slat.f32", hr_slat.cpu().numpy(), np.float32)

    final = SparseUnetVaeDecoder.forward(sdec, sp.SparseTensor(hr_slat, hr_coords.cuda()))
    save("dec_feats.f32", final.feats.cpu().numpy(), np.float32)
    save("dec_coords.i32", final.coords.cpu().numpy(), np.int32)
    feats, dcoords = final.feats, final.coords
    dual = (1 + 2 * 0.5) * F.sigmoid(feats[:, 0:3]) - 0.5
    intersected = feats[:, 3:6] > 0
    quad_lerp = F.softplus(feats[:, 6:7])
    verts, faces = flexible_dual_grid_to_mesh(dcoords[:, 1:], dual, intersected, quad_lerp,
                                              aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], grid_size=GRID, train=False)
    save("mesh_vertices.f32", verts.cpu().numpy(), np.float32)
    save("mesh_faces.i32", faces.cpu().numpy(), np.int32)
    print("cascade oracle:", K, "voxels at 64^3,", feats.shape[0], "decoder voxels,", tuple(verts.shape), "vertices,",
          tuple(faces.shape), "faces", flush=True)
