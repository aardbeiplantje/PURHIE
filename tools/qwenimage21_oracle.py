# Qwen-Image 2.1 text to image, end to end in float32, as diffusers 0.43.0's QwenImage21Pipeline runs it, on the ComfyUI files
# (the int8 ConvRot weights dequantised exactly: W = (q s) H per 256-block of the input dim, H the normalised regular Hadamard;
# the BF16 weights as they are). Every stage is written for the engine's checks (tests/qwenimage21_pipe_verify.cpp):
#   ids.i32        the prompt's token ids (the t2i template with the system turn)       drop.txt  the system tokens dropped
#   emb.f32        [L][4096] the text encoder's last layer before its norm, the system tokens dropped
#   noise.f32      [N][64] the packed initial latent: torch.randn(1, 1, 64, gh, gw) after manual_seed(seed), packed
#   sigmas.f32     the scheduler's sigmas (steps + 1)     v0.f32, x1.f32  the first step's velocity and the latent after it
#   latent.f32     [N][64] the final packed latent        dec.f32  [4][H][W] the VAE's output (RGBA in [-1, 1])     image.png
# The scheduler is FlowMatchEulerDiscreteScheduler with the model's scheduler_config.json (Qwen/Qwen-Image-2.1 on Hugging Face):
# the exponential dynamic shift (base 256 -> 0.5, max 8192 -> 0.9), shift_terminal 0.02; the sigmas linspace(1, 1/steps, steps).
# Usage: PYTHONPATH=/data/zimage-lib2 python3 tools/qwenimage21_oracle.py --prompt "..." --width 512 --height 512 --steps 8 \
#        --seed 0 --out /data/zimage-ref/qwenimage2/pipe
import argparse, json, os, re
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument('--prompt', default='a red cube on a wooden table, soft studio light')
ap.add_argument('--width', type=int, default=512)
ap.add_argument('--height', type=int, default=512)
ap.add_argument('--steps', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='/data/zimage-ref/qwenimage2/pipe')
ap.add_argument('--te', default='/comfyui/text_encoders/qwen3vl_8b_int8_convrot.safetensors')
ap.add_argument('--dit', default='/comfyui/diffusion_models/qwen_image_2.1_int8_convrot.safetensors')
ap.add_argument('--vae', default='/comfyui/vae/qwen_image_2.1_vae_bf16.safetensors')
ap.add_argument('--vae-config', default='/data/zimage-ref/qwenimage2/vae/config.json')
ap.add_argument('--tokenizer', default='/data/zimage-ref/qwen3vl')
args = ap.parse_args()
assert args.width % 32 == 0 and args.height % 32 == 0, 'the size: multiples of 32'
os.makedirs(args.out, exist_ok=True)
dev = 'cuda'
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False


def save(name, t):
    np.ascontiguousarray(t.detach().float().cpu().numpy()).astype(np.float32).tofile(os.path.join(args.out, name))


H4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64)
Hm = np.array([[1.0]])
while Hm.shape[0] < 256:
    Hm = np.kron(Hm, H4)
HAD = torch.from_numpy(Hm / 16.0).float().to(dev)


def deq(f, name):
    q = f.get_tensor(name).float().to(dev)
    s = f.get_tensor(name[:-len('.weight')] + '.weight_scale').float().to(dev)
    W = q * s
    out, inn = W.shape
    return (W.reshape(out, inn // 256, 256) @ HAD).reshape(out, inn)


# ---- 1. the prompt and the text encoder (the Qwen3-VL 8B language model)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(args.tokenizer)
sys_text = 'Comprehend and analyze the provided prompt.'
prompt = args.prompt if args.prompt else ' '
text = f'<|im_start|>system\n{sys_text}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids[0]
drop = len(tok('<|im_start|>system\n' + sys_text + '<|im_end|>\n', add_special_tokens=False).input_ids)
np.asarray(ids.numpy(), dtype=np.int32).tofile(os.path.join(args.out, 'ids.i32'))
open(os.path.join(args.out, 'drop.txt'), 'w').write(f'{drop}\n')
print('tokens', len(ids), 'drop', drop, flush=True)

HID, NH, NKV, HD, THETA, EPS = 4096, 32, 8, 128, 5000000.0, 1e-6


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


F = safe_open(args.te, 'pt', device='cpu')
L = len(ids)
inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32, device=dev) / HD))
ang = torch.outer(torch.arange(L, dtype=torch.float32, device=dev), inv)
ang = torch.cat([ang, ang], dim=-1)
cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]


def rot_half(x):
    return torch.cat([-x[..., HD // 2:], x[..., :HD // 2]], dim=-1)


with torch.no_grad():
    x = deq(F, 'model.embed_tokens.weight')[ids.to(dev)]
    mask = torch.triu(torch.full((L, L), float('-inf'), device=dev), diagonal=1)
    for l in range(36):
        p = f'model.layers.{l}.'
        bf = lambda n: F.get_tensor(p + n).float().to(dev)
        h = rms(x, bf('input_layernorm.weight'))
        q = (h @ deq(F, p + 'self_attn.q_proj.weight').T).view(L, NH, HD)
        k = (h @ deq(F, p + 'self_attn.k_proj.weight').T).view(L, NKV, HD)
        v = (h @ deq(F, p + 'self_attn.v_proj.weight').T).view(L, NKV, HD)
        q = rms(q, bf('self_attn.q_norm.weight'))
        k = rms(k, bf('self_attn.k_norm.weight'))
        q = q * cos + rot_half(q) * sin
        k = k * cos + rot_half(k) * sin
        k = k.repeat_interleave(NH // NKV, dim=1)
        v = v.repeat_interleave(NH // NKV, dim=1)
        att = (torch.einsum('qhd,khd->hqk', q, k) / HD ** 0.5 + mask[None]).softmax(-1)
        x = x + torch.einsum('hqk,khd->qhd', att, v).reshape(L, HID) @ deq(F, p + 'self_attn.o_proj.weight').T
        h2 = rms(x, bf('post_attention_layernorm.weight'))
        g = h2 @ deq(F, p + 'mlp.gate_proj.weight').T
        u = h2 @ deq(F, p + 'mlp.up_proj.weight').T
        x = x + (torch.nn.functional.silu(g) * u) @ deq(F, p + 'mlp.down_proj.weight').T
emb = x[drop:].clone()
save('emb.f32', emb)
del x, F
torch.cuda.empty_cache()
print('text', tuple(emb.shape), flush=True)

# ---- 2. the transformer, the noise and the sampler
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel

m = QwenImage21Transformer2DModel(patch_size=1, in_channels=64, out_channels=64, num_layers=32, attention_head_dim=128,
                                  num_attention_heads=32, context_in_dim=4096, mlp_ratio=3, axes_dims_rope=(16, 56, 56), eps=1e-6,
                                  causal_condition=True).to(dev)
sd = m.state_dict()
loaded = set()


def put(name, t):
    assert tuple(sd[name].shape) == tuple(t.shape), (name, sd[name].shape, t.shape)
    sd[name].copy_(t.to(sd[name].device, torch.float32))
    loaded.add(name)


with safe_open(args.dit, 'pt', device='cpu') as f:
    for k in f.keys():
        if k.endswith(('.weight_scale', '.comfy_quant')):
            continue
        t = f.get_tensor(k)
        W = deq(f, k) if t.dtype == torch.int8 else t.float()  # the int8 file dequantised, or the BF16 file as it is
        if k.endswith('img_mlp.gate_up.weight'):
            base = k[:-len('.weight')]
            put(base.replace('gate_up', 'gate_layer') + '.weight', W[:12288])
            put(base.replace('gate_up', 'proj') + '.weight', W[12288:])
        else:
            put(k, W)
print('transformer: loaded', len(loaded), 'of', len(sd), flush=True)
m.eval()

gh, gw = args.height // 16, args.width // 16
N = gh * gw
g = torch.Generator('cpu').manual_seed(args.seed)
noise = torch.randn(1, 1, 64, gh, gw, generator=g)
lat = noise.view(1, 64, N).transpose(1, 2).contiguous().to(dev)  # the pipeline's _pack_latents
save('noise.f32', lat[0])

sched_cfg = {"base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": False, "max_image_seq_len": 8192, "max_shift": 0.9,
             "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": 0.02, "stochastic_sampling": False,
             "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
             "use_exponential_sigmas": False, "use_karras_sigmas": False}
sched = FlowMatchEulerDiscreteScheduler(**sched_cfg)
mm = (0.9 - 0.5) / (8192 - 256)
mu = N * mm + (0.5 - mm * 256)
sched.set_timesteps(sigmas=np.linspace(1.0, 1 / args.steps, args.steps), device=dev, mu=mu)
save('sigmas.f32', sched.sigmas)
print('sigmas', [round(float(s), 5) for s in sched.sigmas], flush=True)

Lt = emb.shape[0]
img_mask = torch.cat([torch.zeros(1, Lt, dtype=torch.bool), torch.ones(1, N // 4, dtype=torch.bool)], dim=1).to(dev)
txt_mask = torch.ones(1, Lt, dtype=torch.long, device=dev)
sched.set_begin_index(0)
with torch.no_grad():
    for i, t in enumerate(sched.timesteps):
        v = m(hidden_states=lat, timestep=(t / 1000).expand(1).float(), encoder_hidden_states=emb[None], encoder_hidden_states_mask=txt_mask,
              img_shapes=[[(1, gh, gw)]], img_mask=img_mask, return_dict=False)[0][:, -N:]
        lat = sched.step(v, t, lat, return_dict=False)[0]
        if i == 0:
            save('v0.f32', v[0])
            save('x1.f32', lat[0])
        print('step', i, float(t), flush=True)
save('latent.f32', lat[0])
del m, sd
torch.cuda.empty_cache()

# ---- 3. the VAE decoder (the ComfyUI BF16 file on diffusers' names)
from diffusers import AutoencoderKLQwenImage21

cfg = json.load(open(args.vae_config))
cfg.pop('_class_name', None)
cfg.pop('_diffusers_version', None)
vae = AutoencoderKLQwenImage21(**cfg).eval()
src = load_file(args.vae)


def rename(k):
    if k.startswith('conv2.'):
        return 'post_quant_conv.' + k[len('conv2.'):]
    if not k.startswith('decoder.'):
        return None
    if k.startswith('decoder.conv1.'):
        return 'decoder.conv_in.' + k[len('decoder.conv1.'):]
    if k.startswith('decoder.head.0.'):
        return 'decoder.norm_out.' + k[len('decoder.head.0.'):]
    if k.startswith('decoder.head.2.'):
        return 'decoder.conv_out.' + k[len('decoder.head.2.'):]
    mt = re.match(r'decoder\.middle\.(\d)\.(.*)', k)
    res = lambda r: r.replace('residual.0.', 'norm1.').replace('residual.2.', 'conv1.').replace('residual.3.', 'norm2.') \
                     .replace('residual.6.', 'conv2.').replace('shortcut.', 'conv_shortcut.')
    if mt:
        i, rest = int(mt.group(1)), mt.group(2)
        if i == 1:
            return 'decoder.mid_block.attentions.0.' + rest
        return f'decoder.mid_block.resnets.{0 if i == 0 else 1}.' + res(rest)
    mt = re.match(r'decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.(.*)', k)
    if mt:
        b, j, rest = int(mt.group(1)), int(mt.group(2)), mt.group(3)
        if j == 3:
            return f'decoder.up_blocks.{b}.upsampler.' + rest
        return f'decoder.up_blocks.{b}.resnets.{j}.' + res(rest)
    return None


vsd = vae.state_dict()
for k, v in src.items():
    nk = rename(k)
    if nk is not None:
        vsd[nk].copy_(v.float().reshape(vsd[nk].shape))
vae = vae.to(dev).float()
mean = torch.tensor(cfg['latents_mean'], device=dev).view(1, 64, 1, 1, 1)
std = torch.tensor(cfg['latents_std'], device=dev).view(1, 64, 1, 1, 1)
with torch.no_grad():
    z = lat.transpose(1, 2).reshape(1, 64, 1, gh, gw) * std + mean  # the pipeline's _unpack_latents, then the denormalisation
    dec = vae.decode(z, return_dict=False)[0][:, :, 0]
save('dec.f32', dec[0])
img = ((dec[0].clamp(-1, 1) / 2 + 0.5) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
from PIL import Image

Image.fromarray(img, 'RGBA' if img.shape[2] == 4 else 'RGB').save(os.path.join(args.out, 'image.png'))
print('done', tuple(dec.shape), flush=True)
