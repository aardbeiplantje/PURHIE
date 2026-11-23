# AudioLDM 2 reference of the sampler: the pipeline's latent after each denoising step (callback_on_step_end), for the ocean prompt,
# 100 steps, 5 s, guidance 3.5, seed 0 (the same as ref_ocean.wav). Writes /data/audioldm2-stages/steps/diff_step_<k>.f32 (8 x 125 x 16).
import numpy as np, torch
from diffusers import AudioLDM2Pipeline
OUT = '/data/audioldm2-stages/steps'
pipe = AudioLDM2Pipeline.from_pretrained('/data/audioldm2', torch_dtype=torch.float32).to('cuda')
g = torch.Generator(device='cpu').manual_seed(0)
def cb(i, t, lat):
    lat[0].float().cpu().numpy().astype(np.float32).tofile(f'{OUT}/diff_step_{i:03d}.f32')
audio = pipe('ocean waves crashing on a beach', num_inference_steps=100, audio_length_in_s=5.0, generator=g,
             callback=cb, callback_steps=1).audios[0]
print('done', audio.shape, flush=True)
