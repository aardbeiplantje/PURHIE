# Reference for MiniMax-H3-Image's DiT (M3): diffusers' MiniMaxH3ImageTransformer2DModel in bf16 on the GPU, loaded from the
# RELEASED diffusers-format weights (transformer_diffusers: the same key names as the class, the PDD heads inside), no key
# mapping. Text-to-image on a 16 x 16 latent with the 9-token prompt of the text encoder's reference (/data/minimax-ref/te).
# Writes the velocities of steps 0 and 5, the six-step sampler's final latent (raw float32): the engine's mmdit_* are checked
# against them. Run with PYTHONPATH=/data/diff43b:/data/diff-site (diffusers 0.43.0).
import json, os, re, numpy as np, torch
from safetensors import safe_open
from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel as M

ROOT = '/hf/MiniMax-H3-Image'
REL = ROOT + '/transformer_diffusers'
os.makedirs('/data/minimax-ref/dit_dbg', exist_ok=True)
OUT = '/data/minimax-ref/dit_dbg'  # the step-0 activations, named as the engine's MMDIT_DEBUG dumps (dbg_<name>.f32)
cfg = json.load(open(ROOT + '/transformer/config.json'))
kw = dict(num_attention_heads=cfg['num_attention_heads'], attention_head_dim=cfg['attention_head_dim'],
          hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], num_refiner_layers=cfg['token_refiner_num_layers'],
          ffn_dim=cfg['ffn_hidden_size'], in_channels=cfg['latents_dim'], patch_size=tuple(cfg['patch_size']),
          text_dim=cfg['text_dim'], freq_dim=cfg['timestep_input_dim'], time_embed_hidden_dim=cfg['time_embed_hidden_size'],
          time_embed_dim=cfg['time_embed_dim'], norm_eps=cfg['norm_eps'], qk_norm_eps=cfg['qk_norm_eps'],
          final_norm_eps=cfg['final_norm_eps'], rope_freq_dim=16, rope_theta=10000.0, num_pdd_heads=6)

def real(p):
    head = open(p, 'rb').read(200)
    if head.startswith(b'version https://git-lfs'):
        oid = re.search(rb'oid sha256:([0-9a-f]+)', head).group(1).decode()
        return os.path.join(ROOT, '.git', 'lfs', 'objects', oid[:2], oid[2:4], oid)
    return p

torch.nn.Linear.reset_parameters = lambda self: None
torch.set_default_dtype(torch.bfloat16)
with torch.device('cuda'):
    model = M(**kw)
torch.set_default_dtype(torch.float32)
model.eval()
sd = model.state_dict()
index = json.load(open(REL + '/diffusion_pytorch_model.safetensors.index.json'))['weight_map']
seen = set()
for f in sorted(set(index.values())):
    with safe_open(real(os.path.join(REL, f)), 'pt') as fh:
        for k in fh.keys():
            with torch.no_grad():
                sd[k].copy_(fh.get_tensor(k).to('cuda', torch.bfloat16))
            seen.add(k)
missing = [k for k in sd if k not in seen and 'inv_freq' not in k]
assert not missing, missing[:5]
print('loaded', len(seen), 'released tensors, all', len(sd), 'state entries', flush=True)

hs = np.load('/data/minimax-ref/te/hs50_0.npy')  # [9, 5120]
T = hs.shape[0]; h = w = 16; hp, wp = h // 2, w // 2; N = hp * wp; S = T + N
g = torch.Generator().manual_seed(1)
x = torch.randn((24, h, w), generator=g)
tok = x.reshape(24, hp, 2, wp, 2).permute(1, 3, 0, 2, 4).reshape(N, 96)
sq = np.sqrt(h * w)
def grid(dim, n):
    ratio = dim / sq; left = (1 - ratio) / 2
    return 32.0 * (left + np.arange(n) * ratio / n)
gh, gw = grid(h, hp), grid(w, wp)
pos = np.zeros((S, 3)); pos[:T, 0] = np.arange(T)
for i in range(hp):
    for j in range(wp):
        pos[T + i * wp + j] = [T, gh[i], gw[j]]
tags = torch.tensor([1] * T + [0] * N, dtype=torch.long, device='cuda')
text_idx = torch.arange(T, device='cuda'); image_idx = torch.arange(T, S, device='cuda')
enc = torch.tensor(hs, dtype=torch.bfloat16, device='cuda')[None]
sigmas = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
np.save(f'{OUT}/sample.f32.npy', x.numpy()); x.numpy().astype(np.float32).tofile(f'{OUT}/sample.f32')
hs.astype(np.float32).tofile(f'{OUT}/text.f32')


# the step-0 activations at the engine's dump points (src/minimax_dit.cpp): the token refiner's output (text rows), the packed
# sequence before block 0 and after blocks 0 and 49, and the output head's normalized rows (all S rows)
os.makedirs(OUT, exist_ok=True)
store = {}
def save(name, t):
    arr = t.detach().float().cpu().reshape(-1).numpy().astype(np.float32)
    arr.tofile(f'{OUT}/dbg_{name}.f32')
    store[name] = arr.size
model.token_refiner.register_forward_hook(lambda m, i, o: save('tn', o[0] if isinstance(o, tuple) else o))
for li, blk in enumerate(model.transformer_blocks):
    blk.register_forward_hook(lambda m, i, o, li=li: save(f'blk{li:02d}', o[0] if isinstance(o, tuple) else o))
model.transformer_blocks[0].register_forward_pre_hook(lambda m, a: save('b0_in', a[0]))
model.transformer_blocks[0].register_forward_hook(lambda m, i, o: save('b0_out', o[0] if isinstance(o, tuple) else o))
model.transformer_blocks[49].register_forward_hook(lambda m, i, o: save('b49_out', o[0] if isinstance(o, tuple) else o))
model.norm_out.register_forward_hook(lambda m, i, o: save('norm_out', o))

# the engine's step-0 velocity on the same inputs: one forward (the hooks write the dumps)
sigmas_l = sigmas
x0 = x.clone().float()
tok_s = x0.reshape(24, hp, 2, wp, 2).permute(1, 3, 0, 2, 4).reshape(N, 96)
t = torch.tensor([1.0 - sigmas_l[0]], dtype=torch.float32, device='cuda')
with torch.inference_mode():
    out = model(hidden_states=tok_s.to('cuda', torch.bfloat16)[None], encoder_hidden_states=enc, timestep=t,
                timestep_indices=torch.zeros(S, dtype=torch.long, device='cuda'), token_tags=tags,
                position_ids=torch.tensor(pos, dtype=torch.float64, device='cuda'), image_indices=image_idx,
                text_indices=text_idx, pdd_step_index=0, return_dict=False)[0]
print('dumped', store, flush=True)
