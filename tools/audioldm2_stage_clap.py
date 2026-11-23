# AudioLDM 2 stage oracle: the CLAP text encoder (text_encoder/) on the pipeline's own tokenization of a prompt (padded to the
# tokenizer's 512). Writes to /data/audioldm2-stages/: clap_ids.i32, clap_mask.i32 (512 each), clap_hidden.f32 (512 x 768: the
# last hidden state of the text model), clap_pooled.f32 (768: the pooler), clap_features.f32 (512: get_text_features, the
# pipeline's prompt embedding). fp32 on the CPU (transformers 4.44 for the pipeline's generate API; ClapModel is unchanged).
import sys
import numpy as np, torch
from transformers import ClapModel, RobertaTokenizer
OUT = '/data/audioldm2-stages'
prompt = sys.argv[1] if len(sys.argv) > 1 else 'ocean waves crashing on a beach'
tok = RobertaTokenizer.from_pretrained('/data/audioldm2/tokenizer')
inputs = tok(prompt, padding='max_length', max_length=tok.model_max_length, truncation=True, return_tensors='pt')
ids, mask = inputs.input_ids, inputs.attention_mask
model = ClapModel.from_pretrained('/data/audioldm2/text_encoder').eval()
with torch.no_grad():
    tm = model.text_model(input_ids=ids, attention_mask=mask)
    feats = model.get_text_features(ids, attention_mask=mask)
ids.numpy().astype(np.int32)[0].tofile(f'{OUT}/clap_ids.i32')
mask.numpy().astype(np.int32)[0].tofile(f'{OUT}/clap_mask.i32')
tm.last_hidden_state[0].numpy().astype(np.float32).tofile(f'{OUT}/clap_hidden.f32')
tm.pooler_output[0].numpy().astype(np.float32).tofile(f'{OUT}/clap_pooled.f32')
feats[0].numpy().astype(np.float32).tofile(f'{OUT}/clap_features.f32')
print('tokens', ids.shape[1], 'valid', int(mask.sum()), 'hidden', tuple(tm.last_hidden_state.shape), 'features', tuple(feats.shape), flush=True)
