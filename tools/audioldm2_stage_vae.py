# AudioLDM 2 stage oracle: the audio VAE decoder (vae/, AutoencoderKL) on the UNet's output (unet_out.f32, 8 x 32 x 32) as the pipeline
# does it: latents / scaling_factor, then decode. Writes vae_out.f32 (1 x mel_h x mel_w) to /data/audioldm2-stages. fp32 on the CPU.
import numpy as np, torch
from diffusers import AutoencoderKL
OUT = '/data/audioldm2-stages'
vae = AutoencoderKL.from_pretrained('/data/audioldm2/vae').eval()
z = torch.from_numpy(np.fromfile(f'{OUT}/unet_out.f32', dtype=np.float32)).reshape(1, 8, 32, 32)
with torch.no_grad():
    mel = vae.decode(1.0 / vae.config.scaling_factor * z).sample
mel[0].numpy().astype(np.float32).tofile(f'{OUT}/vae_out.f32')
print('mel', tuple(mel.shape), 'scaling', vae.config.scaling_factor, flush=True)
