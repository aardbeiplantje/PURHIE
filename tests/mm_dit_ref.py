# Reference for MiniMax-H3-Image's DiT (M3): diffusers 0.43.0's MiniMaxH3ImageTransformer2DModel in bf16 on the GPU (the
# checkpoint's own BF16 values; the full model: 66 GB as a device model), text-to-image on a 16 x 16 latent with the 9-token
# prompt of the text encoder's reference (/data/minimax-ref/te/hs50_0.npy). Velocities for steps 0 and 5 and the
# sample used as input are saved (raw float32): the engine's mmdit_velocity is compared with them.
import json, re, os, struct, numpy as np, torch
from safetensors import safe_open
import diffusers.models.transformers.transformer_minimax_h3_image as tm
from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel as M

ROOT = '/hf/MiniMax-H3-Image'
OUT = '/data/minimax-ref/dit'
cfg = json.load(open(ROOT + '/transformer/config.json'))
kw = dict(num_attention_heads=cfg['num_attention_heads'], attention_head_dim=cfg['attention_head_dim'],
          hidden_size=cfg['hidden_size'], num_layers=cfg['num_layers'], num_refiner_layers=cfg['token_refiner_num_layers'],
          ffn_dim=cfg['ffn_hidden_size'], in_channels=cfg['latents_dim'], patch_size=tuple(cfg['patch_size']),
          text_dim=cfg['text_dim'], freq_dim=cfg['timestep_input_dim'], time_embed_hidden_dim=cfg['time_embed_hidden_size'],
          time_embed_dim=cfg['time_embed_dim'], norm_eps=cfg['norm_eps'], qk_norm_eps=cfg['qk_norm_eps'],
          final_norm_eps=cfg['final_norm_eps'], rope_freq_dim=16, rope_theta=10000.0, num_pdd_heads=6)
H = kw['hidden_size']; INNER = 56 * 128

def real(p):
    head = open(p, 'rb').read(200)
    if head.startswith(b'version https://git-lfs'):
        oid = re.search(rb'oid sha256:([0-9a-f]+)', head).group(1).decode()
        return os.path.join(ROOT, '.git', 'lfs', 'objects', oid[:2], oid[2:4], oid)
    return p

# the module, in bf16 on the GPU (no random init: the weights come from the checkpoint)
torch.nn.Linear.reset_parameters = lambda self: None
torch.set_default_dtype(torch.bfloat16)
with torch.device('cuda'):
    model = M(**kw)
torch.set_default_dtype(torch.float32)
model.eval()
sd = model.state_dict()
loaded = set()

def put(dst, t):
    with torch.no_grad():
        sd[dst].copy_(t.to('cuda', torch.bfloat16))
    loaded.add(dst)

index = json.load(open(ROOT + '/transformer/model.safetensors.index.json'))['weight_map']
files = sorted(set(index.values()))
for f in files:
    with safe_open(real(os.path.join(ROOT, 'transformer', f)), 'pt') as fh:
        for k in fh.keys():
            t = fh.get_tensor(k)
            if k.startswith('blocks.') or k.startswith('token_refiner.blocks.'):
                pass
            if k in ('video_patch_proj.weight', 'video_patch_proj.bias'):
                put('proj_in.' + k.split('.')[1], t)
            elif k in ('condition_proj.weight', 'condition_proj.bias'):
                put('context_embedder.' + k.split('.')[1], t)
            elif k.startswith('time_embedder.'):
                n = {'proj_in': 'linear_1', 'proj_out': 'linear_2'}[k.split('.')[1]]
                put('time_embedder.' + n + '.' + k.split('.')[2], t)
            elif k.startswith('final_layer.norm.'):
                put('norm_out.norm.' + k.split('.')[2], t)
            elif k.startswith('final_layer.adaln_proj.linear.'):
                put('norm_out.linear.' + k.split('.')[3], t)
            elif k.startswith('token_refiner.final_norm.'):
                put('token_refiner.final_norm.' + k.split('.')[2], t)
            elif k.startswith('blocks.') or k.startswith('token_refiner.blocks.'):
                parts = k.split('.')
                pre = 'token_refiner.refiner_blocks.' if parts[0] == 'token_refiner' else 'transformer_blocks.'
                idx = parts[2] if parts[0] == 'token_refiner' else parts[1]
                rest = '.'.join(parts[3:]) if parts[0] == 'token_refiner' else '.'.join(parts[2:])
                base = f'{pre}{idx}.'
                if rest == 'attn.qkv_proj.weight':
                    q, kk, v = t.split(INNER, dim=0)
                    put(base + 'attn.to_q.weight', q); put(base + 'attn.to_k.weight', kk); put(base + 'attn.to_v.weight', v)
                elif rest == 'attn.out_proj.weight': put(base + 'attn.to_out.0.weight', t)
                elif rest == 'attn.q_norm.weight': put(base + 'attn.norm_q.weight', t)
                elif rest == 'attn.k_norm.weight': put(base + 'attn.norm_k.weight', t)
                elif rest == 'mlp.fc1.weight': put(base + 'ff.net.0.proj.weight', t)
                elif rest == 'mlp.fc2.weight': put(base + 'ff.net.2.weight', t)
                elif rest.startswith('norm1.') or rest.startswith('norm2.'): put(base + rest, t)
                elif rest.startswith('adaln_proj.linear.'): put(base + rest, t)
# the PDD heads: the pipeline's pdd_proj_out[i] (fused in the file)
with safe_open(real(ROOT + '/pdd_fused_heads.safetensors'), 'pt') as fh:
    W = fh.get_tensor('video_out.weight'); B = fh.get_tensor('video_out.bias')
for i in range(6):
    put(f'pdd_proj_out.{i}.weight', W[i]); put(f'pdd_proj_out.{i}.bias', B[i])
missing = [k for k in sd if k not in loaded and 'inv_freq' not in k]
print('unloaded state entries:', missing[:10], len(missing), flush=True)
assert not missing, missing[:10]

# the inputs: the prompt rows, a fixed latent, the rotary positions (text (index,0,0); the image on the frame clock after the text)
hs = np.load('/data/minimax-ref/te/hs50_0.npy')  # [9, 5120]
T = hs.shape[0]
h = w = 16
hp, wp = h // 2, w // 2
N = hp * wp
S = T + N
g = torch.Generator().manual_seed(1)
x = torch.randn((24, h, w), generator=g)            # the sampler's latent
tok = x.reshape(24, hp, 2, wp, 2).permute(1, 3, 0, 2, 4).reshape(N, 96)   # patchify: (c, ph, pw) per token, row-major
sq = np.sqrt(h * w)
def grid(dim, n):
    ratio = dim / sq; left = (1 - ratio) / 2
    return 32.0 * (left + np.arange(n) * ratio / n)
gh, gw = grid(h, hp), grid(w, wp)
pos = np.zeros((S, 3))
pos[:T, 0] = np.arange(T)
for i in range(hp):
    for j in range(wp):
        pos[T + i * wp + j] = [T, gh[i], gw[j]]
tags = torch.tensor([1] * T + [0] * N, dtype=torch.long, device='cuda')
text_idx = torch.arange(T, device='cuda'); image_idx = torch.arange(T, S, device='cuda')
enc = torch.tensor(hs, dtype=torch.bfloat16, device='cuda')[None]
sigmas = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]
np.save(f'{OUT}/sample.f32', x.numpy().astype(np.float32))
np.save(f'{OUT}/text.f32', hs.astype(np.float32))
# intermediate activations of step 0 (the engine's MMDIT_DEBUG dumps): the refiner's output, the packed sequence before and
# after block 0, after block 49, and the norm_out output
cap = {}
def save_cap(name):
    def f(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap[name] = t.detach().float().cpu().numpy().reshape(-1, t.shape[-1]).astype(np.float32)
    return f
def save_in(name):
    def f(mod, args):
        t = args[0]
        cap[name] = t.detach().float().cpu().numpy().reshape(-1, t.shape[-1]).astype(np.float32)
    return f
hs_ = [model.token_refiner.final_norm.register_forward_hook(save_cap('tn')),
       model.transformer_blocks[0].register_forward_pre_hook(save_in('b0_in')),
       model.transformer_blocks[0].register_forward_hook(save_cap('b0_out')),
       model.transformer_blocks[49].register_forward_hook(save_cap('b49_out')),
       model.norm_out.register_forward_hook(save_cap('norm_out'))]
for step in (0, 5):
    t = torch.tensor([1.0 - sigmas[step]], dtype=torch.float32, device='cuda')
    with torch.inference_mode():
        out = model(hidden_states=tok.to('cuda', torch.bfloat16)[None], encoder_hidden_states=enc,
                    timestep=t, timestep_indices=torch.zeros(S, dtype=torch.long, device='cuda'),
                    token_tags=tags, position_ids=torch.tensor(pos, dtype=torch.float64, device='cuda'),
                    image_indices=image_idx, text_indices=text_idx, pdd_step_index=step, return_dict=False)[0]
    if step == 0:
        for k, a in cap.items(): a.tofile(f'{OUT}/dbg_{k}.f32')
        print('dumped', {k: a.shape for k, a in cap.items()}, flush=True)
    v = out[0].float().cpu()   # [N, 96]
    vol = v.reshape(hp, wp, 24, 2, 2).permute(2, 0, 3, 1, 4).reshape(24, h, w).numpy().astype(np.float32)
    vol.tofile(f'{OUT}/velocity_step{step}.f32')
    print('step', step, 'velocity', vol.shape, 'norm', float(np.linalg.norm(vol)), flush=True)
# the whole sampler: the six Euler velocity steps from the noise (x += (sigma_i - sigma_{i+1}) * v), the pipeline's rule
xs = x.clone().float()
with torch.inference_mode():
    for step in range(6):
        tok_s = xs.reshape(24, hp, 2, wp, 2).permute(1, 3, 0, 2, 4).reshape(N, 96)
        t = torch.tensor([1.0 - sigmas[step]], dtype=torch.float32, device='cuda')
        out = model(hidden_states=tok_s.to('cuda', torch.bfloat16)[None], encoder_hidden_states=enc, timestep=t,
                    timestep_indices=torch.zeros(S, dtype=torch.long, device='cuda'), token_tags=tags,
                    position_ids=torch.tensor(pos, dtype=torch.float64, device='cuda'), image_indices=image_idx,
                    text_indices=text_idx, pdd_step_index=step, return_dict=False)[0]
        vol = out[0].float().cpu().reshape(hp, wp, 24, 2, 2).permute(2, 0, 3, 1, 4).reshape(24, h, w)
        xs = xs + (sigmas[step] - sigmas[step + 1]) * vol
xs.numpy().astype(np.float32).tofile(f'{OUT}/sample_final.f32')
print('sampler final', tuple(xs.shape), 'norm', float(xs.norm()), flush=True)
np.save(f'{OUT}/meta.npy', np.array([h, w, T], dtype=np.int64))
print('done', flush=True)
