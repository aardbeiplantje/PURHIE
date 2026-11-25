"""Oracle for the TRELLIS.2 image encoder: DINOv3 ViT-L/16 as TRELLIS's DinoV3FeatureExtractor runs it (512 px: PIL Lanczos resize,
ImageNet normalisation; embeddings, rotary position embeddings, the 24 blocks, then a layer norm without affine), fp32 on the GPU.

The weights are the local copy ComfyUI uses (/comfyui/clip_vision/dino_v3_L_naf_fp32.safetensors, the same tensor names as transformers'
DINOv3ViT with the rotary periods under naf.image_encoder.rope.periods = rope_theta ** (k / 16)), so the gated Hugging Face download is
not needed. The image is comfy/example.png (or IMAGE): the oracle writes its normalised input and the 1029 output tokens (CLS, 4 registers,
1024 patches).

Writes ORACLE_DIR (oracle_dino): input.f32 [3][512][512] and tokens.f32 [1029][1024].
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

WEIGHTS = "/comfyui/clip_vision/dino_v3_L_naf_fp32.safetensors"
IMAGE = os.environ.get("IMAGE", "/work/rocm-hip/comfy/example.png")
ORACLE_DIR = "/data/trellis2/oracle_dino"
SIZE = 512
os.makedirs(ORACLE_DIR, exist_ok=True)

config = DINOv3ViTConfig(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, patch_size=16,
    num_channels=3, num_register_tokens=4, rope_theta=100.0, layer_norm_eps=1e-5, hidden_act="gelu",
    query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True, layerscale_value=1.0,
    use_gated_mlp=False, drop_path_rate=0.0, image_size=SIZE,
)
model = DINOv3ViTModel(config)
state = load_file(WEIGHTS)
# the file names the blocks layer.N; transformers' DINOv3ViTModel holds them under model.layer.N
state = {("model." + k if k.startswith("layer.") else k): v for k, v in state.items() if not k.startswith("naf.")}
res = model.load_state_dict(state, strict=False)
print("missing:", res.missing_keys, "unexpected:", res.unexpected_keys, flush=True)
assert not res.missing_keys and not res.unexpected_keys, "the local weights do not match the model"
model = model.float().eval().cuda()

# TRELLIS's preprocessing: resize (PIL Lanczos) to 512, to [0, 1], ImageNet normalisation
img = Image.open(IMAGE).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
x = torch.from_numpy(np.array(img).astype(np.float32) / 255).permute(2, 0, 1)[None]
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
x = ((x - mean) / std).cuda()
x[0].cpu().numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "input.f32"))

with torch.no_grad():
    h = model.embeddings(x, bool_masked_pos=None)
    pos = model.rope_embeddings(x)
    for layer in model.model.layer:
        h = layer(h, position_embeddings=pos)
    tokens = F.layer_norm(h, h.shape[-1:])[0]
tokens.cpu().numpy().astype(np.float32).tofile(os.path.join(ORACLE_DIR, "tokens.f32"))
print("tokens", tuple(tokens.shape), "patches", SIZE // 16, flush=True)
