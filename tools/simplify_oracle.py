"""Oracle for the mesh simplification of the TRELLIS.2 export: CuMesh's simplify (cumesh's Python driver: thresh 1e-8, lambdas 1e-2 and 1e-3)
on the filled shape mesh (oracle_fill: the mesh the python path has before simplifying), to TARGET faces. Writes ORACLE_DIR
(oracle_simplify): simplified_vertices.f32 and simplified_faces.i32. Run with the CuMesh scratch build on PYTHONPATH."""
import os
import sys
import numpy as np
import torch
import cumesh

SRC = "/data/trellis2/oracle_fill"
ORACLE_DIR = "/data/trellis2/oracle_simplify"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
os.makedirs(ORACLE_DIR, exist_ok=True)

verts = torch.from_numpy(np.fromfile(os.path.join(SRC, "filled_vertices.f32"), dtype=np.float32).reshape(-1, 3)).cuda()
faces = torch.from_numpy(np.fromfile(os.path.join(SRC, "filled_faces.i32"), dtype=np.int32).reshape(-1, 3)).cuda()
print("input:", tuple(verts.shape), tuple(faces.shape), "target", TARGET, flush=True)
mesh = cumesh.CuMesh()
mesh.init(verts, faces)
mesh.simplify(TARGET, verbose=False)
new_v, new_f = mesh.read()
new_v = new_v.cpu().numpy().astype(np.float32)
new_f = new_f.cpu().numpy().astype(np.int32)
new_v.tofile(os.path.join(ORACLE_DIR, "simplified_vertices.f32"))
new_f.tofile(os.path.join(ORACLE_DIR, "simplified_faces.i32"))
print("simplified:", new_v.shape, new_f.shape, flush=True)
