# AudioLDM 2 reference of the first denoising step (100 steps, 5 s, guidance 3.5, seed 0): the UNet call's inputs and outputs of the
# pipeline's first UNet call (a batch of two: index 0 the unconditional, index 1 the conditional). Writes /data/audioldm2-stages/steps/:
# s0_x.f32 (8 x 125 x 16, the latent), s0_ctx1_u.f32, s0_ctx1_c.f32 (8 x 768), s0_ctx2_u.f32 (1 x 1024), s0_ctx2_c.f32 (9 x 1024),
# s0_eu.f32, s0_ec.f32 (the two predictions, 8 x 125 x 16), and the timestep in s0_t.txt.
import numpy as np, torch
from diffusers import AudioLDM2Pipeline
OUT = '/data/audioldm2-stages/steps'
pipe = AudioLDM2Pipeline.from_pretrained('/data/audioldm2', torch_dtype=torch.float32).to('cuda')
g = torch.Generator(device='cpu').manual_seed(0)
cap = {}
def hook(mod, args, kwargs, out):
    if 'done' in cap:
        return
    cap['done'] = True
    cap['x'] = args[0].detach().float().cpu()
    cap['t'] = float(torch.as_tensor(args[1]).reshape(-1)[0])
    cap['c1'] = kwargs['encoder_hidden_states'].detach().float().cpu()
    cap['c2'] = kwargs['encoder_hidden_states_1'].detach().float().cpu()
    cap['out'] = out[0].detach().float().cpu()
pipe.unet.register_forward_hook(hook, with_kwargs=True)
pipe('ocean waves crashing on a beach', num_inference_steps=100, audio_length_in_s=5.0, generator=g, guidance_scale=3.5)
cap['x'][0].numpy().astype(np.float32).tofile(f'{OUT}/s0_x.f32')
cap['c1'][0].numpy().astype(np.float32).tofile(f'{OUT}/s0_ctx1_u.f32')
cap['c1'][1].numpy().astype(np.float32).tofile(f'{OUT}/s0_ctx1_c.f32')
cap['c2'][0].numpy().astype(np.float32).tofile(f'{OUT}/s0_ctx2_u.f32')
cap['c2'][1].numpy().astype(np.float32).tofile(f'{OUT}/s0_ctx2_c.f32')
cap['out'][0].numpy().astype(np.float32).tofile(f'{OUT}/s0_eu.f32')
cap['out'][1].numpy().astype(np.float32).tofile(f'{OUT}/s0_ec.f32')
open(f'{OUT}/s0_t.txt', 'w').write(str(cap['t']))
print('unet first call: t', cap['t'], 'ctx1', tuple(cap['c1'].shape), 'ctx2', tuple(cap['c2'].shape), 'out', tuple(cap['out'].shape), flush=True)
