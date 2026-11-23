# AudioLDM 2 stage oracle: the projection of the prompt's CLAP feature and T5 hidden states (the pipeline's inputs), and the
# GPT-2's generated rows (generate_language_model: 8 steps, each the last hidden state appended). Writes to /data/audioldm2-stages/:
# gen_proj.f32 ((Tp x 768): the projection output), gen_out.f32 (8 x 768). fp32 on the CPU (transformers 4.44).
import numpy as np, torch
from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2ProjectionModel
from transformers import GPT2Config, GPT2Model
from safetensors.torch import load_file
OUT = '/data/audioldm2-stages'
proj = AudioLDM2ProjectionModel.from_pretrained('/data/audioldm2/projection_model').eval()
clap = torch.from_numpy(np.fromfile(f'{OUT}/clap_features.f32', dtype=np.float32)).reshape(1, 1, 512)
t5 = torch.from_numpy(np.fromfile(f'{OUT}/t5_hidden.f32', dtype=np.float32)).reshape(1, -1, 1024)
with torch.no_grad():
    seq = proj(hidden_states=clap, hidden_states_1=t5).hidden_states
lm = GPT2Model(GPT2Config.from_json_file('/data/audioldm2/language_model/config.json'))
lm.load_state_dict(load_file('/data/audioldm2/language_model/model.safetensors'), strict=False)
lm.eval()
gen = []
with torch.no_grad():
    cur = seq
    for _ in range(8):
        h = lm(inputs_embeds=cur, output_hidden_states=True, return_dict=True).hidden_states[-1]
        nxt = h[:, -1:, :]
        gen.append(nxt)
        cur = torch.cat([cur, nxt], dim=1)
seq[0].numpy().astype(np.float32).tofile(f'{OUT}/gen_proj.f32')
torch.cat(gen, dim=1)[0].numpy().astype(np.float32).tofile(f'{OUT}/gen_out.f32')
print('projection rows', seq.shape[1], 'generated', 8, flush=True)
