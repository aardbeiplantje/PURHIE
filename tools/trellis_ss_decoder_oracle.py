"""Oracle dumps for the TRELLIS.2 sparse-structure decoder (SparseStructureDecoder, ss_dec_conv3d_16l8_fp16), fp32 on the GPU.

The input is the flow's output (the oracle's out.f32 from tools/trellis_ss_oracle.py: [8][16][16][16]). The python model
runs with use_fp16=False (the fp16 weights upcast to fp32), so the check of the engine is against an fp32 oracle.
Writes dec_*.f32 into ORACLE_DIR: dec_in (after the input conv), dec_mid (after the middle blocks), dec_b2 (after the first
upsampling and its two blocks' input), dec_b5 (after the second upsampling), dec_out ([1][64][64][64] after the output layer).
"""
import os
import sys

import numpy as np
import torch

REPO = "/work/rocm-hip/TRELLIS.2"
CKPT = "/data/trellis2/TRELLIS-image-large/ckpts"
ORACLE_DIR = "/data/trellis2/oracle_ss"
sys.path.insert(0, REPO)

from safetensors.torch import load_file  # noqa: E402
from trellis2.models.sparse_structure_vae import SparseStructureDecoder  # noqa: E402

import json  # noqa: E402

cfg = json.load(open(os.path.join(CKPT, "ss_dec_conv3d_16l8_fp16.json")))
args = dict(cfg["args"])
args["use_fp16"] = False
dec = SparseStructureDecoder(**args)
state = load_file(os.path.join(CKPT, "ss_dec_conv3d_16l8_fp16.safetensors"))
res = dec.load_state_dict({k: v.float() for k, v in state.items()}, strict=True)
dec = dec.float().eval().cuda()


def save(name, t):
    t.detach().float().contiguous().cpu().numpy().tofile(os.path.join(ORACLE_DIR, name + ".f32"))
    print(name, tuple(t.shape), flush=True)


z = torch.from_numpy(np.fromfile(os.path.join(ORACLE_DIR, "out.f32"), dtype=np.float32).reshape(1, 8, 16, 16, 16)).cuda()
with torch.no_grad():
    h = dec.input_layer(z)
    save("dec_in", h)
    for blk in dec.middle_block:
        h = blk(h)
    save("dec_mid", h)
    # blocks: 0, 1 (ResBlock 512), 2 (upsample 512 -> 128), 3, 4 (ResBlock 128), 5 (upsample 128 -> 32), 6, 7 (ResBlock 32)
    for i, blk in enumerate(dec.blocks):
        h = blk(h)
        if i == 2:
            save("dec_b2", h)
        if i == 5:
            save("dec_b5", h)
    h = dec.out_layer(h)
    save("dec_out", h)
print("decoder oracle written to", ORACLE_DIR, "; load errors:", res, flush=True)
