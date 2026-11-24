"""Oracle for the TRELLIS.2 shape decoder's mesh step (FlexiDualGridVaeDecoder's eval path after the sparse UNet): the decoded
features [M][7] (sdec_out_feats.f32, from tools/sdec_oracle.py) are the vertices (2 sigmoid(f0..2) - 0.5 with the voxel margin
0.5), the intersection flags (f3..5 > 0) and the split weight softplus(f6); flexible_dual_grid_to_mesh (o-voxel, grid 256,
aabb [-0.5, 0.5]) gives the mesh. Writes ORACLE_DIR: mesh_vertices.f32 [V][3], mesh_faces.i32 [T][3].
Run with the o-voxel scratch build on the path (PYTHONPATH); the _C of o-voxel is the HIP build."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/work/rocm-hip/TRELLIS.2")
import types  # noqa: E402
for name in ["nvdiffrast", "nvdiffrast.torch"]:  # o-voxel imports its rendering code; the mesh function does not use it
    sys.modules[name] = types.ModuleType(name)
from o_voxel.convert import flexible_dual_grid_to_mesh  # noqa: E402

ORACLE_DIR = "/data/trellis2/oracle_sparse"
M_MARGIN = 0.5
feats = torch.from_numpy(np.fromfile(os.path.join(ORACLE_DIR, "sdec_out_feats.f32"), dtype=np.float32).reshape(-1, 7)).cuda()
coords = torch.from_numpy(np.fromfile(os.path.join(ORACLE_DIR, "sdec_out_coords.i32"), dtype=np.int32).reshape(-1, 4)).cuda()
dual = (1 + 2 * M_MARGIN) * F.sigmoid(feats[:, 0:3]) - M_MARGIN
intersected = feats[:, 3:6] > 0
quad_lerp = F.softplus(feats[:, 6:7])
aabb = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
verts, faces = flexible_dual_grid_to_mesh(coords[:, 1:], dual, intersected, quad_lerp, aabb=aabb, grid_size=256, train=False)
verts.cpu().numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "mesh_vertices.f32"))
faces.cpu().numpy().astype(np.int32).tofile(os.path.join(ORACLE_DIR, "mesh_faces.i32"))
print("mesh oracle:", tuple(verts.shape), "vertices,", tuple(faces.shape), "faces", flush=True)
