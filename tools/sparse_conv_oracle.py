"""Oracle for the engine's submanifold sparse convolution (src/sparse.c kernels): FlexGEMM's sparse_submanifold_conv3d on
random voxels, with fp16 weights in the checkpoint layout [Cout][Cin][3][3][3]. Writes into ORACLE_DIR: sc_x.f32 [N][Cin],
sc_coords.i32 [N][4], sc_w.u16 (the fp16 bits, [Cout][Cin][27]), sc_b.f32 [Cout], sc_y.f32 [N][Cout] (FlexGEMM's output).
Run with the FlexGEMM build on the path (PYTHONPATH: the flex_gemm package with its CUDA extension)."""
import os

import numpy as np
import torch
from flex_gemm.ops.spconv import sparse_submanifold_conv3d

ORACLE_DIR = "/data/trellis2/oracle_sparse"
N, CI, CO, G = 3000, 16, 32, 16
os.makedirs(ORACLE_DIR, exist_ok=True)
torch.manual_seed(3)
flat = torch.randperm(G * G * G)[:N]
coords = torch.stack([torch.zeros(N, dtype=torch.long), flat // (G * G), (flat // G) % G, flat % G], dim=1).int()
x = torch.randn(N, CI)
w16 = (torch.randn(CO, CI, 3, 3, 3) * 0.2).half()  # the checkpoint layout
b = torch.randn(CO) * 0.1

y, _ = sparse_submanifold_conv3d(
    x.cuda(), coords.cuda(), torch.Size([1, CI, G, G, G]),
    w16.float().permute(0, 2, 3, 4, 1).contiguous().cuda(), b.cuda(), None, (1, 1, 1))
x.numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "sc_x.f32"))
coords.numpy().astype(np.int32).tofile(os.path.join(ORACLE_DIR, "sc_coords.i32"))
w16.contiguous().view(torch.int16).numpy().astype(np.uint16).tofile(os.path.join(ORACLE_DIR, "sc_w.u16"))
b.numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "sc_b.f32"))
y.cpu().numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "sc_y.f32"))
print("sparse conv oracle:", N, "voxels,", CI, "->", CO, "channels; |y| max", float(y.abs().max()))
