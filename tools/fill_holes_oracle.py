"""Oracle for the TRELLIS.2 mesh post-process that the image-to-3D pipeline applies before baking (mesh/base.py: fill_holes):
CuMesh's fill_holes at the default max_hole_perimeter (3e-2) on the 512 shape oracle's mesh (tools/trellis_shape_oracle.py).
Writes ORACLE_DIR (oracle_fill): filled_vertices.f32 [V'][3], filled_faces.i32 [T'][3] and the loop counts in stdout.
Run with the CuMesh scratch build on PYTHONPATH (the cumesh package with its _C extension)."""
import os
import numpy as np
import torch
import cumesh

SHAPE = "/data/trellis2/oracle_shape"
ORACLE_DIR = "/data/trellis2/oracle_fill"
MAX_PERIMETER = 3e-2
os.makedirs(ORACLE_DIR, exist_ok=True)

verts = torch.from_numpy(np.fromfile(os.path.join(SHAPE, "mesh_vertices.f32"), dtype=np.float32).reshape(-1, 3)).cuda()
faces = torch.from_numpy(np.fromfile(os.path.join(SHAPE, "mesh_faces.i32"), dtype=np.int32).reshape(-1, 3)).cuda()
print("input:", tuple(verts.shape), tuple(faces.shape), flush=True)

mesh = cumesh.CuMesh()
mesh.init(verts, faces)
mesh.get_edges()
mesh.get_boundary_info()
print("boundary edges:", mesh.num_boundaries, flush=True)
mesh.get_vertex_edge_adjacency()
mesh.get_vertex_boundary_adjacency()
mesh.get_manifold_boundary_adjacency()
mesh.read_manifold_boundary_adjacency()
mesh.get_boundary_connected_components()
mesh.get_boundary_loops()
print("boundary loops:", mesh.num_boundary_loops, flush=True)
mesh.fill_holes(max_hole_perimeter=MAX_PERIMETER)
new_v, new_f = mesh.read()
new_v = new_v.cpu().numpy().astype(np.float32)
new_f = new_f.cpu().numpy().astype(np.int32)
new_v.tofile(os.path.join(ORACLE_DIR, "filled_vertices.f32"))
new_f.tofile(os.path.join(ORACLE_DIR, "filled_faces.i32"))
print("filled:", new_v.shape, new_f.shape, flush=True)
