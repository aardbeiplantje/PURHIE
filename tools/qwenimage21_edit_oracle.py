# Qwen-Image 2.1 image edit, end to end in float32, as diffusers 0.43.0's QwenImage21Pipeline runs it with a condition image, on
# the ComfyUI files (the int8 ConvRot weights dequantised exactly, as tools/qwenimage21_oracle.py; the BF16 weights as they are).
# The prompt and the image go through the pipeline's own _get_qwen_prompt_embeds (transformers' Qwen3VLProcessor and
# Qwen3VLForConditionalGeneration: the vision tower, its deepstack features, the M-RoPE positions) and _encode_vae_image; the
# denoising loop is the pipeline's, with its KV cache of the text and condition tokens. Every stage is written for the engine's
# checks (tests/qwenimage21_edit_verify.cpp):
#   cond_rgba.u8     the condition image resized as the pipeline does it (VaeImageProcessor.resize, PIL's LANCZOS), RGBA8
#   cond_size.txt    its width and height; out_size.txt the output's width and height
#   ids.i32          the encoder's token ids (the ti2i template, the image pads)        drop.txt  the system tokens dropped
#   pixels.f32       [Np][1536] the processor's pixel rows      grid.i32  (t, h, w) of the patch grid
#   vis_merged.f32   [Np / 4][4096] the vision tower's merged output    vis_ds{0,1,2}.f32  its deepstack features (layers 8, 16, 24)
#   pos.i32          [3][T] the M-RoPE positions
#   emb.f32          [L][4096] the encoder's last layer before its norm, the system tokens dropped     imgmask.u8  [L] the image pads
#   cond.f32         [Nc][64] the condition image's packed, normalised latent (the VAE encoder's mean)
#   noise.f32        [N][64] the packed initial latent        sigmas.f32  the scheduler's sigmas (steps + 1)
#   v0.f32, x1.f32   the first step's velocity (the target rows) and the latent after it
#   latent.f32       [N][64] the final packed latent          dec.f32  [4][H][W] the VAE's output       image.png
# Usage: PYTHONPATH=/data/zimage-lib2 python3 tools/qwenimage21_edit_oracle.py --image IN.png --prompt "..." --resolution 512 \
#        --steps 8 --seed 0 --out /data/zimage-ref/qwenimage2/edit
import argparse, json, os, re, types
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument('--image', required=True)
ap.add_argument('--prompt', default='make the cube blue')
ap.add_argument('--resolution', type=int, default=1024, help="the pipeline's output_resolution (a pixel budget of its square)")
ap.add_argument('--steps', type=int, default=8)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', default='/data/zimage-ref/qwenimage2/edit')
ap.add_argument('--te', default='/comfyui/text_encoders/qwen3vl_8b_int8_convrot.safetensors')
ap.add_argument('--dit', default='/comfyui/diffusion_models/qwen_image_2.1_int8_convrot.safetensors')
ap.add_argument('--vae', default='/comfyui/vae/qwen_image_2.1_vae_bf16.safetensors')
ap.add_argument('--vae-config', default='/data/zimage-ref/qwenimage2/vae/config.json')
ap.add_argument('--proc', default='/data/zimage-ref/qwen3vl-proc', help='the processor files and text_encoder/config.json')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
dev = 'cuda'
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def save(name, t):
    np.ascontiguousarray(t.detach().float().cpu().numpy()).astype(np.float32).tofile(os.path.join(args.out, name))


def save_i32(name, t):
    np.ascontiguousarray(np.asarray(t.detach().cpu().numpy() if torch.is_tensor(t) else t)).astype(np.int32).tofile(
        os.path.join(args.out, name))


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


from PIL import Image as PILImage
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline, calculate_dimensions

# ---- 1. the condition image, as __call__ prepares it: one resize feeds the text encoder and the VAE
img = PILImage.open(args.image)
img.load()
res = args.resolution
width, height, _ = calculate_dimensions(res * res, img.size[0] / img.size[1])
width, height = width // 32 * 32, height // 32 * 32
image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)
if img.mode != 'RGBA':
    img = img.convert('RGBA')
in_w, in_h, _ = calculate_dimensions(res * res, img.size[0] / img.size[1])
input_image = image_processor.resize(img, width=in_w, height=in_h)
vae_image = image_processor.preprocess(img, width=in_w, height=in_h).unsqueeze(2)
np.asarray(input_image.convert('RGBA'), dtype=np.uint8).tofile(os.path.join(args.out, 'cond_rgba.u8'))
open(os.path.join(args.out, 'cond_size.txt'), 'w').write(f'{in_w} {in_h}\n')
open(os.path.join(args.out, 'out_size.txt'), 'w').write(f'{width} {height}\n')
print('condition', img.size, '->', (in_w, in_h), 'output', (width, height), flush=True)

# ---- 2. the text encoder: Qwen3-VL 8B (the vision tower BF16, the language model int8 ConvRot dequantised), in float32
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

processor = Qwen3VLProcessor.from_pretrained(args.proc)
cfg = Qwen3VLConfig.from_pretrained(args.proc)
cfg._attn_implementation = 'eager'
cfg.text_config._attn_implementation = 'eager'
cfg.vision_config._attn_implementation = 'eager'
with torch.device(dev):
    te = Qwen3VLForConditionalGeneration(cfg).float()
te.eval()
tsd = te.state_dict()
seen = set()
with safe_open(args.te, 'pt', device='cpu') as f:
    for k in f.keys():
        if k.endswith(('.weight_scale', '.comfy_quant')) or k.startswith('lm_head.'):
            continue
        if k.startswith('model.visual.'):
            nk = k
        elif k.startswith('model.'):
            nk = 'model.language_model.' + k[len('model.'):]
        else:
            raise SystemExit('unexpected tensor ' + k)
        t = f.get_tensor(k)
        W = deq(f, k) if t.dtype == torch.int8 else t.float().to(dev)
        assert tuple(tsd[nk].shape) == tuple(W.shape), (nk, tsd[nk].shape, W.shape)
        tsd[nk].copy_(W.reshape(tsd[nk].shape))
        seen.add(nk)
missing = [k for k in tsd if k not in seen and not k.startswith('lm_head.')]
assert not missing, missing[:8]
print('text encoder: loaded', len(seen), 'tensors', flush=True)

cap = {}


class CapProc:
    """the processor, its outputs kept"""

    def __call__(self, **kw):
        out = processor(**kw)
        cap['inputs'] = out
        return out


def vis_hook(mod, inp, out):
    cap['merged'] = out.pooler_output
    cap['ds'] = out.deepstack_features


def rot_hook(mod, inp):
    cap['pos'] = inp[1]


hv = te.model.visual.register_forward_hook(vis_hook)
hr = te.model.language_model.rotary_emb.register_forward_pre_hook(rot_hook)
fake = types.SimpleNamespace()
fake.processor = CapProc()
fake.text_encoder = te
fake.sys_prompt = 'Comprehend and analyze the provided prompt.'
fake.prompt_template_t2i = (f'<|im_start|>system\n{fake.sys_prompt}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n'
                            f'<|im_start|>assistant\n')
fake.prompt_template_ti2i = (f'<|im_start|>system\n{fake.sys_prompt}<|im_end|>\n'
                             f'<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n'
                             f'<|im_start|>assistant\n')
sys_message = [{'role': 'system', 'content': [{'type': 'text', 'text': fake.sys_prompt}]}]
fake._drop_idx = len(processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])
fake._img_token_id = processor.tokenizer.encode('<|image_pad|>')[0]
fake._execution_device = dev
fake._extract_masked_hidden = lambda h, m: QwenImage21Pipeline._extract_masked_hidden(fake, h, m)
with torch.no_grad():
    emb, emb_mask, pad_mask = QwenImage21Pipeline._get_qwen_prompt_embeds(fake, args.prompt, [input_image], dev)
hv.remove()
hr.remove()
inp = cap['inputs']
save_i32('ids.i32', inp.input_ids[0])
open(os.path.join(args.out, 'drop.txt'), 'w').write(f'{fake._drop_idx}\n')
save('pixels.f32', inp.pixel_values)
save_i32('grid.i32', inp.image_grid_thw[0])
save('vis_merged.f32', cap['merged'])
for i, d in enumerate(cap['ds']):
    save(f'vis_ds{i}.f32', d)
save_i32('pos.i32', cap['pos'][:, 0, :])
save('emb.f32', emb[0])
np.asarray(pad_mask[0].cpu().numpy(), dtype=np.uint8).tofile(os.path.join(args.out, 'imgmask.u8'))
print('tokens', inp.input_ids.shape[1], 'drop', fake._drop_idx, 'grid', inp.image_grid_thw[0].tolist(), 'emb', tuple(emb.shape),
      'pads', int(pad_mask.sum()), flush=True)
fake.text_encoder = None
del te, tsd, cap
torch.cuda.empty_cache()

# ---- 3. the VAE (the ComfyUI BF16 file on diffusers' names): the condition image's latent
from diffusers import AutoencoderKLQwenImage21

vcfg = json.load(open(args.vae_config))
vcfg.pop('_class_name', None)
vcfg.pop('_diffusers_version', None)
vae = AutoencoderKLQwenImage21(**vcfg).eval()


def rename(k):
    res = lambda r: r.replace('residual.0.', 'norm1.').replace('residual.2.', 'conv1.').replace('residual.3.', 'norm2.') \
                     .replace('residual.6.', 'conv2.').replace('shortcut.', 'conv_shortcut.')
    if k.startswith('conv1.'):
        return 'quant_conv.' + k[len('conv1.'):]
    if k.startswith('conv2.'):
        return 'post_quant_conv.' + k[len('conv2.'):]
    for part in ('encoder', 'decoder'):
        if not k.startswith(part + '.'):
            continue
        if k.startswith(part + '.conv1.'):
            return part + '.conv_in.' + k[len(part + '.conv1.'):]
        if k.startswith(part + '.head.0.'):
            return part + '.norm_out.' + k[len(part + '.head.0.'):]
        if k.startswith(part + '.head.2.'):
            return part + '.conv_out.' + k[len(part + '.head.2.'):]
        mt = re.match(part + r'\.middle\.(\d)\.(.*)', k)
        if mt:
            i, rest = int(mt.group(1)), mt.group(2)
            if i == 1:
                return part + '.mid_block.attentions.0.' + rest
            return part + f'.mid_block.resnets.{0 if i == 0 else 1}.' + res(rest)
        mt = re.match(r'decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.(.*)', k)
        if mt:
            b, j, rest = int(mt.group(1)), int(mt.group(2)), mt.group(3)
            if j == 3:
                return f'decoder.up_blocks.{b}.upsampler.' + rest
            return f'decoder.up_blocks.{b}.resnets.{j}.' + res(rest)
        mt = re.match(r'encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.(.*)', k)
        if mt:
            b, j, rest = int(mt.group(1)), int(mt.group(2)), mt.group(3)
            if j == 2:
                return f'encoder.down_blocks.{b}.downsampler.' + rest
            return f'encoder.down_blocks.{b}.resnets.{j}.' + res(rest)
    return None


vsd = vae.state_dict()
src = load_file(args.vae)
vseen = set()
for k, v in src.items():
    nk = rename(k)
    assert nk in vsd, (k, nk)
    vsd[nk].copy_(v.float().reshape(vsd[nk].shape))
    vseen.add(nk)
vmissing = [k for k in vsd if k not in vseen]
assert not vmissing, vmissing[:8]
vae = vae.to(dev).float()
fake.vae = vae
fake.latent_channels = 64
gen = torch.Generator('cpu').manual_seed(args.seed)
with torch.no_grad():
    cl = QwenImage21Pipeline._encode_vae_image(fake, vae_image.to(dev, torch.float32), gen)
ch, cw = cl.shape[3], cl.shape[4]
cond = QwenImage21Pipeline._pack_latents(cl, 1, 64, ch, cw)
save('cond.f32', cond[0])
print('condition latent', tuple(cl.shape), flush=True)

# ---- 4. the transformer, the noise and the sampler (the pipeline's loop, with its KV cache)
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

vae.cpu()
torch.cuda.empty_cache()
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
        W = deq(f, k) if t.dtype == torch.int8 else t.float()
        if k.endswith('img_mlp.gate_up.weight'):
            base = k[:-len('.weight')]
            put(base.replace('gate_up', 'gate_layer') + '.weight', W[:12288])
            put(base.replace('gate_up', 'proj') + '.weight', W[12288:])
        else:
            put(k, W)
print('transformer: loaded', len(loaded), 'of', len(sd), flush=True)
m.eval()

gh, gw = height // 16, width // 16
N = gh * gw
noise = torch.randn(1, 1, 64, gh, gw, generator=gen)
lat = noise.view(1, 64, N).transpose(1, 2).contiguous().to(dev)
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

img_shapes = [[(1, ch, cw), (1, gh, gw)]]
img_mask = torch.cat([pad_mask, pad_mask.new_ones(1, N // 4)], dim=1).to(dev)
cache = QwenImage21KVCache(len(m.transformer_blocks))
sched.set_begin_index(0)
with torch.no_grad():
    for i, t in enumerate(sched.timesteps):
        x_in = torch.cat([cond, lat], dim=1)
        v = m(hidden_states=x_in, timestep=(t / 1000).expand(1).float(), encoder_hidden_states=emb, encoder_hidden_states_mask=emb_mask,
              img_shapes=img_shapes, img_mask=img_mask, kv_cache=cache, kv_cache_mode='extract' if i == 0 else 'cached',
              return_dict=False)[0][:, -N:]
        lat = sched.step(v, t, lat, return_dict=False)[0]
        if i == 0:
            save('v0.f32', v[0])
            save('x1.f32', lat[0])
        print('step', i, float(t), flush=True)
save('latent.f32', lat[0])
del m, sd, cache
torch.cuda.empty_cache()

# ---- 5. the VAE decoder
vae = vae.to(dev)
mean = torch.tensor(vcfg['latents_mean'], device=dev).view(1, 64, 1, 1, 1)
std = torch.tensor(vcfg['latents_std'], device=dev).view(1, 64, 1, 1, 1)
with torch.no_grad():
    z = lat.transpose(1, 2).reshape(1, 64, 1, gh, gw) * std + mean
    dec = vae.decode(z, return_dict=False)[0][:, :, 0]
save('dec.f32', dec[0])
img = ((dec[0].clamp(-1, 1) / 2 + 0.5) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
PILImage.fromarray(img, 'RGBA' if img.shape[2] == 4 else 'RGB').save(os.path.join(args.out, 'image.png'))
print('done', tuple(dec.shape), flush=True)
