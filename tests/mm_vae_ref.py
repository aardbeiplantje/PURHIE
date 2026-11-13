# Reference for MiniMax-H3-Image's VAE decoder (M2): diffusers 0.43.0's AutoencoderKLMiniMaxH3Image, float32 on the CPU.
# The pipeline's decoders.py: latents = z * latents_std + latents_mean, then vae.decode (tiled by default).
import numpy as np, torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import AutoencoderKLMiniMaxH3Image
D = '/data/minimax-ref/vae'
OUT = '/data/minimax-ref/vae'
from safetensors.torch import load_file
vae = AutoencoderKLMiniMaxH3Image.from_config(D).float().eval()   # the encoder stays unloaded (decode only)
ck = load_file(D + '/diffusion_pytorch_model.safetensors')
sd = {}
for k, v in ck.items():
    if k.startswith('decoder.x_embedder.'): sd[k.replace('x_embedder', 'proj_in')] = v
    elif k.startswith('post_quant_conv.'): sd[k] = v
    elif '.attn.to_qkv.' in k:
        q, kk, vv = v.chunk(3, dim=0)
        base = k[:k.index('to_qkv')]
        suffix = k.rsplit('.', 1)[1]
        sd[base + 'to_q.' + suffix], sd[base + 'to_k.' + suffix], sd[base + 'to_v.' + suffix] = q, kk, vv
    elif '.attn.to_out.' in k: sd[k.replace('to_out.', 'to_out.0.')] = v
    elif '.ff.w1.' in k: sd[k.replace('ff.w1.', 'ff.net.0.proj.')] = v
    elif '.ff.w2.' in k: sd[k.replace('ff.w2.', 'ff.net.2.')] = v
    elif k.startswith('decoder.') and 'mask_token' not in k: sd[k] = v
dec_sd = {k[len('decoder.'):]: v for k, v in sd.items() if k.startswith('decoder.')}
vae.decoder.load_state_dict(dec_sd, strict=True)
vae.post_quant_conv.load_state_dict({k[len('post_quant_conv.'):]: v for k, v in sd.items() if k.startswith('post_quant_conv.')}, strict=True)
print('loaded decoder strictly:', len(dec_sd), 'tensors', flush=True)
mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
print('use_tiling', vae.use_tiling, 'tile', vae.tile_sample_min_height, vae.tile_sample_min_overlap_height, 'ratio', vae.spatial_compression_ratio, flush=True)
g = torch.Generator().manual_seed(0)
for name, hw in [('small', 16), ('tiled', 40)]:
    z = torch.randn((1, 24, hw, hw), generator=g)           # the normalized latents (the pipeline's input)
    with torch.inference_mode():
        zd = z * std + mean
        out = vae.decode(zd, return_dict=False)[0]           # [1, 3, 16 hw, 16 hw]
        dec = vae.decoder(vae.post_quant_conv(zd))           # the untiled decoder on the whole latent (reference for the tiles)
    z.numpy().astype(np.float32).tofile(f'{OUT}/{name}_z.f32')
    out.numpy().astype(np.float32).tofile(f'{OUT}/{name}_out.f32')
    dec.numpy().astype(np.float32).tofile(f'{OUT}/{name}_untiled.f32')
    print(name, 'latent', tuple(z.shape), 'image', tuple(out.shape), 'untiled', tuple(dec.shape),
          'range', float(out.min()), float(out.max()), 'tiled-vs-untiled max', float((out - dec).abs().max()), flush=True)
print('done', flush=True)
