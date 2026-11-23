# AudioLDM 2 stage oracle: the projection model (projection_model/) on fixed inputs (fp32, CPU). Writes raw float32 to
# /data/audioldm2-stages/: proj_clap.f32 (Tc x 512), proj_t5.f32 (Tt x 1024), proj_out.f32 ((Tc + 2 + Tt + 2) x 768).
import numpy as np, torch
from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2ProjectionModel
OUT = '/data/audioldm2-stages'
model = AudioLDM2ProjectionModel.from_pretrained('/data/audioldm2/projection_model').eval()
g = torch.Generator().manual_seed(0)
Tc, Tt = 7, 9
clap = torch.randn(1, Tc, 512, generator=g)
t5 = torch.randn(1, Tt, 1024, generator=g)
with torch.no_grad():
    out = model(hidden_states=clap, hidden_states_1=t5).hidden_states
clap.numpy().astype(np.float32).tofile(f'{OUT}/proj_clap.f32')
t5.numpy().astype(np.float32).tofile(f'{OUT}/proj_t5.f32')
out[0].numpy().astype(np.float32).tofile(f'{OUT}/proj_out.f32')
print('out', tuple(out.shape), flush=True)
