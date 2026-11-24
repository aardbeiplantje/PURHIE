"""Oracle for the TRELLIS.2 shape decoder's sparse UNet (SparseUnetVaeDecoder.forward of FlexiDualGridVaeDecoder, the mesh
step left out: it needs nvdiffrast), fp32 on the GPU (the fp16 weights upcast), FlexGEMM's sparse convolutions.
Random latent voxels at 16^3 (N = 100, batch 0), feats [N][32]. Writes into ORACLE_DIR (sdec_*.f32 / .i32):
sdec_in_feats [N][32], sdec_in_coords [N][4]; sdec_l0_feats (after level 0's blocks, the C2S output) and sdec_l0_coords;
sdec_out_feats [M][7] and sdec_out_coords [M][4] (the final output layer's features and their voxels).
Run with the FlexGEMM build, the cumesh and o-voxel scratch builds on the path (PYTHONPATH)."""
import importlib
import json
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, "/work/rocm-hip/TRELLIS.2")
for name in ["nvdiffrast", "nvdiffrast.torch"]:
    sys.modules[name] = types.ModuleType(name)
from safetensors.torch import load_file  # noqa: E402
from trellis2.modules import sparse as sp  # noqa: E402

ORACLE_DIR = "/data/trellis2/oracle_sparse"
CKPT = "/data/trellis2/TRELLIS.2-4B/ckpts"
N, G = 100, 16
os.makedirs(ORACLE_DIR, exist_ok=True)
cfg = json.load(open(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.json")))
args = dict(cfg["args"])
args["use_fp16"] = False
m = importlib.import_module("trellis2.models.sc_vaes.fdg_vae").FlexiDualGridVaeDecoder(**args)
state = load_file(os.path.join(CKPT, "shape_dec_next_dc_f16c32_fp16.safetensors"))
m.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
m = m.float().eval().cuda()

torch.manual_seed(5)
flat = torch.randperm(G * G * G)[:N]
coords = torch.stack([torch.zeros(N, dtype=torch.long), flat // (G * G), (flat // G) % G, flat % G], dim=1).int()
feats = torch.randn(N, 32)


def save(name, arr, dtype):
    np.asarray(arr).astype(dtype).tofile(os.path.join(ORACLE_DIR, name))


save("sdec_in_feats.f32", feats.numpy(), np.float32)
save("sdec_in_coords.i32", coords.numpy(), np.int32)

with torch.no_grad():
    x = sp.SparseTensor(feats.cuda(), coords.cuda())
    h = m.from_latent(x)
    # level 0: its blocks (the ConvNeXt ones, then the channel-to-spatial block that returns (h, subdiv))
    for blk in m.blocks[0][:-1]:
        h = blk(h)
    h_c2s, _ = m.blocks[0][-1](h)
    save("sdec_l0_feats.f32", h_c2s.feats.cpu().numpy(), np.float32)
    save("sdec_l0_coords.i32", h_c2s.coords.cpu().numpy(), np.int32)
    # the whole sparse UNet from the input (its own forward: the output layer included)
    from trellis2.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    final = SparseUnetVaeDecoder.forward(m, x)
    save("sdec_out_feats.f32", final.feats.cpu().numpy(), np.float32)
    save("sdec_out_coords.i32", final.coords.cpu().numpy(), np.int32)
    print("level0", tuple(h_c2s.feats.shape), "out", tuple(final.feats.shape), flush=True)
