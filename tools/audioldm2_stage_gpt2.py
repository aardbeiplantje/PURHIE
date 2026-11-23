# AudioLDM 2 stage oracle: the GPT-2 language model (language_model/) on fixed inputs_embeds (teacher-forced, fp32 on CPU).
# Writes raw float32: input (1 x T x 768), hidden6 (after layer 6), last (after the final layer norm).
import numpy as np, torch
from transformers import GPT2Config, GPT2Model
from safetensors.torch import load_file
D = '/data/audioldm2'
OUT = '/data/audioldm2-stages'
cfg = GPT2Config.from_json_file(D + '/language_model/config.json')
model = GPT2Model(cfg)
sd = load_file(D + '/language_model/model.safetensors')
missing, unexpected = model.load_state_dict(sd, strict=False)
print('missing', missing[:4], 'unexpected', unexpected[:4], flush=True)
model.eval()
T = 12
g = torch.Generator().manual_seed(0)
x = torch.randn(1, T, cfg.n_embd, generator=g)
with torch.no_grad():
    out = model(inputs_embeds=x, output_hidden_states=True)
hs = out.hidden_states  # [embeddings, layer 1, ..., layer 12 (before the final norm)], and last_hidden_state is normed
x.numpy().astype(np.float32).tofile(f'{OUT}/gpt2_input.f32')
hs[6].numpy().astype(np.float32).tofile(f'{OUT}/gpt2_hidden6.f32')
out.last_hidden_state.numpy().astype(np.float32).tofile(f'{OUT}/gpt2_last.f32')
print('T', T, 'dtype', next(model.parameters()).dtype, 'last', tuple(out.last_hidden_state.shape), 'hidden6 mean %.4f std %.4f' % (hs[6].mean(), hs[6].std()), flush=True)
