"""Oracle for the TRELLIS.2 image-to-shape path: the shape half of trellis2/pipelines/trellis2_image_to_3d.py's run() for the
512 pipeline, fp32 on the GPU with the shipped bf16 weights upcast.

The sparse-structure flow is sampled with the guidance interval sampler (trellis2/pipelines/samplers) from seeded noise, decoded
(the sparse-structure decoder, then the occupied voxels: any of each 2x2x2 block at 64^3 is active) to the voxels at 32^3. The
shape-SLAT flow is sampled on those voxels, de-normalised with pipeline.json's shape_slat_normalization, and decoded by the
shape decoder (the sparse UNet's output: 7 features per voxel at grid 512). The mesh is o-voxel's flexible_dual_grid_to_mesh at
grid 512 (as the pipeline's decode_latent does with resolution 512). The image features are a seeded random [1][1024][1024]
(the DINOv3 encoder is not on the engine), and the negative condition is zeros, as in get_cond.

Writes ORACLE_DIR (oracle_shape): cond.f32 [1024][1024], ss_noise.f32 [8][16][16][16], ss_latent.f32 [8][16][16][16],
coords.i32 [N][4], slat_noise.f32 [N][32], slat.f32 [N][32] (de-normalised), dec_feats.f32 [M][7], dec_coords.i32 [M][4],
mesh_vertices.f32 [V][3], mesh_faces.i32 [T][3]. Run with the o-voxel and FlexGEMM scratch builds on PYTHONPATH.
"""
import importlib
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("ATTN_BACKEND", "sdpa")  # the dense attention: no flash_attn here (set before trellis2 is imported)
REPO = "/work/rocm-hip/TRELLIS.2"
CKPT = "/data/trellis2/TRELLIS.2-4B/ckpts"
CKPT_SS_DEC = "/data/trellis2/TRELLIS-image-large/ckpts"
ORACLE_DIR = "/data/trellis2/oracle_shape"
GRID = 512
COND_LEN = 1024
sys.path.insert(0, REPO)
for name in ["nvdiffrast", "nvdiffrast.torch"]:  # o-voxel and the decoder import their rendering code; the mesh does not use it
    sys.modules[name] = types.ModuleType(name)

from safetensors.torch import load_file  # noqa: E402
from o_voxel.convert import flexible_dual_grid_to_mesh  # noqa: E402
from trellis2.modules import sparse as sp  # noqa: E402
from trellis2.pipelines import samplers  # noqa: E402
from trellis2.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402
from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder  # noqa: E402
import trellis2.modules.sparse.attention.modules as sparse_attn_modules  # noqa: E402


def _sdpa_one_sequence(*args):
    """The sparse attention for one sequence (as in tools/trellis_slat_oracle.py): flash_attn is not installed; with one sequence
    the packed attention is plain SDPA over all tokens."""
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


def load_denoiser(json_name, safetensors_name):
    cfg = json.load(open(os.path.join(CKPT, json_name)))
    spec = cfg["models"]["denoiser"] if "models" in cfg else cfg
    args = dict(spec["args"])
    args["dtype"] = "float32"
    model = getattr(importlib.import_module("trellis2.models"), spec["name"])(**args)
    res = model.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, safetensors_name)).items()}, strict=False)
    print(json_name, "load:", res, flush=True)
    return model.float().eval().cuda()


ss_flow = load_denoiser("ss_flow_img_dit_1_3B_64_bf16.json", "ss_flow_img_dit_1_3B_64_bf16.safetensors")
slat_flow = load_denoiser("slat_flow_img2shape_dit_1_3B_512_bf16.json", "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors")

cfg = json.load(open(os.path.join(CKPT_SS_DEC, "ss_dec_conv3d_16l8_fp16.json")))
ss_dec_args = dict(cfg["args"])
ss_dec_args["use_fp16"] = False
ss_dec = SparseStructureDecoder(**ss_dec_args)
ss_dec.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT_SS_DEC, "ss_dec_conv3d_16l8_fp16.safetensors")).items()},
                       strict=True)
ss_dec = ss_dec.float().eval().cuda()

cfg = json.load(open(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.json")))
sdec_args = dict(cfg["args"])
sdec_args["use_fp16"] = False
sdec = importlib.import_module("trellis2.models.sc_vaes.fdg_vae").FlexiDualGridVaeDecoder(**sdec_args)
sdec.load_state_dict({k: v.float() for k, v in load_file(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.safetensors")).items()},
                     strict=True)
sdec = sdec.float().eval().cuda()

pipe = json.load(open(os.path.join(CKPT, "..", "pipeline.json")))["args"]
ss_params = pipe["sparse_structure_sampler"]["params"]
slat_params = pipe["shape_slat_sampler"]["params"]
norm = pipe["shape_slat_normalization"]
save("shape_norm.f32", np.array(norm["mean"] + norm["std"]), np.float32)
ss_sampler = samplers.FlowEulerGuidanceIntervalSampler(**pipe["sparse_structure_sampler"]["args"])
slat_sampler = samplers.FlowEulerGuidanceIntervalSampler(**pipe["shape_slat_sampler"]["args"])
print("ss params", ss_params, "slat params", slat_params, flush=True)


def save(name, t, dtype):
    np.asarray(t).astype(dtype).tofile(os.path.join(ORACLE_DIR, name))


torch.manual_seed(4)
cond = torch.randn(1, COND_LEN, 1024).cuda()
neg_cond = torch.zeros_like(cond)
save("cond.f32", cond.cpu().numpy(), np.float32)

torch.manual_seed(3)
ss_noise = torch.randn(1, 8, 16, 16, 16).cuda()
save("ss_noise.f32", ss_noise.cpu().numpy(), np.float32)
with torch.no_grad():
    z_s = ss_sampler.sample(ss_flow, ss_noise, cond=cond, neg_cond=neg_cond, steps=ss_params["steps"],
                            rescale_t=ss_params["rescale_t"], guidance_strength=ss_params["guidance_strength"],
                            guidance_rescale=ss_params["guidance_rescale"],
                            guidance_interval=tuple(ss_params["guidance_interval"])).samples
    save("ss_latent.f32", z_s.cpu().numpy(), np.float32)
    decoded = ss_dec(z_s) > 0                                  # [1][1][64][64][64]
    occupied = F.max_pool3d(decoded.float(), 2, 2, 0) > 0.5    # [1][1][32][32][32]
    coords = torch.argwhere(occupied)[:, [0, 2, 3, 4]].int()
    N = coords.shape[0]
    print("occupied voxels at 32^3:", N, flush=True)
    save("coords.i32", coords.cpu().numpy(), np.int32)

    torch.manual_seed(5)
    slat_noise = torch.randn(N, 32).cuda()
    save("slat_noise.f32", slat_noise.cpu().numpy(), np.float32)
    slat_noise_sp = sp.SparseTensor(feats=slat_noise, coords=coords.cuda())
    slat_s = slat_sampler.sample(slat_flow, slat_noise_sp, cond=cond, neg_cond=neg_cond, steps=slat_params["steps"],
                                 rescale_t=slat_params["rescale_t"], guidance_strength=slat_params["guidance_strength"],
                                 guidance_rescale=slat_params["guidance_rescale"],
                                 guidance_interval=tuple(slat_params["guidance_interval"])).samples
    std = torch.tensor(norm["std"]).cuda()[None]
    mean = torch.tensor(norm["mean"]).cuda()[None]
    slat = slat_s.feats * std + mean
    save("slat.f32", slat.cpu().numpy(), np.float32)

    final = SparseUnetVaeDecoder.forward(sdec, sp.SparseTensor(slat, coords.cuda()))
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
    print("shape oracle:", N, "voxels at 32^3,", tuple(final.feats.shape), "decoder voxels,",
          tuple(verts.shape), "vertices,", tuple(faces.shape), "faces", flush=True)
