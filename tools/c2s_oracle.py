"""Oracle for the engine's sparse channel-to-spatial upsampling (SparseChannel2Spatial, factor 2) on its own: random voxels
with feats [N][8 C], a subdivision logit [N][8] (the block's to_subdiv output; the binary mask is logit > 0), and the python
op's output (the order is deterministic: each voxel's children in increasing child index). Writes into ORACLE_DIR:
c2s_feats.f32 [N][8C], c2s_coords.i32 [N][4], c2s_logit.f32 [N][8], c2s_out_coords.i32 [M][4], c2s_out_feats.f32 [M][C]."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/rocm-hip/TRELLIS.2")
from trellis2.modules import sparse as sp  # noqa: E402

ORACLE_DIR = "/data/trellis2/oracle_sparse"
N, C, G = 500, 16, 8
os.makedirs(ORACLE_DIR, exist_ok=True)
torch.manual_seed(4)
flat = torch.randperm(G * G * G)[:N]
coords = torch.stack([torch.zeros(N, dtype=torch.long), flat // (G * G), (flat // G) % G, flat % G], dim=1).int()
feats = torch.randn(N, 8 * C)
logit = torch.randn(N, 8) * 2.0
x = sp.SparseTensor(feats.cuda(), coords.cuda())
sub = sp.SparseTensor((logit > 0).cuda(), coords.cuda())
out = sp.SparseChannel2Spatial(2)(x, sub)
feats.numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "c2s_feats.f32"))
coords.numpy().astype(np.int32).tofile(os.path.join(ORACLE_DIR, "c2s_coords.i32"))
logit.numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "c2s_logit.f32"))
out.coords.cpu().numpy().astype(np.int32).tofile(os.path.join(ORACLE_DIR, "c2s_out_coords.i32"))
out.feats.detach().cpu().numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "c2s_out_feats.f32"))
print("c2s oracle:", N, "voxels ->", out.coords.shape[0], "children;", "feats", tuple(out.feats.shape), flush=True)
