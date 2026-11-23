# AudioLDM 2 stage oracle: one call of the UNet (unet/, AudioLDM2UNet2DConditionModel) on fixed inputs, fp32 on the CPU: a latent
# [8, 32, 32] (seed 0), timestep 500, the GPT-2 generated rows (gen_out.f32, 8 x 768: encoder_hidden_states) and the T5 hidden states
# (t5_hidden.f32, L x 1024: encoder_hidden_states_1). Writes /data/audioldm2-stages/: unet_in.f32 (8 x 32 x 32), unet_out.f32 (8 x 32 x 32).
import numpy as np, torch
from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2UNet2DConditionModel
OUT = '/data/audioldm2-stages'
unet = AudioLDM2UNet2DConditionModel.from_pretrained('/data/audioldm2/unet').eval()
g = torch.Generator().manual_seed(0)
x = torch.randn(1, 8, 32, 32, generator=g)
gen = torch.from_numpy(np.fromfile(f'{OUT}/gen_out.f32', dtype=np.float32)).reshape(1, 8, 768)
t5 = torch.from_numpy(np.fromfile(f'{OUT}/t5_hidden.f32', dtype=np.float32)).reshape(1, -1, 1024)
with torch.no_grad():
    out = unet(x, torch.tensor([500]), encoder_hidden_states=gen, encoder_hidden_states_1=t5).sample
x[0].numpy().astype(np.float32).tofile(f'{OUT}/unet_in.f32')
out[0].numpy().astype(np.float32).tofile(f'{OUT}/unet_out.f32')
print('out', tuple(out.shape), 'rms %.4f' % float(out.pow(2).mean().sqrt()), flush=True)
