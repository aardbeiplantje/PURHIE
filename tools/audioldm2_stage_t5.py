# AudioLDM 2 stage oracle: the T5 encoder (text_encoder_2/) on the pipeline's tokenization of a prompt (padding to the longest,
# i.e. no padding for one prompt). Writes to /data/audioldm2-stages/: t5_ids.i32 (L), t5_hidden.f32 (L x 1024: the encoder's
# last hidden state). fp32 on the CPU (transformers 4.44, as the pipeline ran).
import sys
import numpy as np, torch
from transformers import AutoTokenizer, T5EncoderModel
OUT = '/data/audioldm2-stages'
prompt = sys.argv[1] if len(sys.argv) > 1 else 'ocean waves crashing on a beach'
tok = AutoTokenizer.from_pretrained('/data/audioldm2/tokenizer_2')
inputs = tok(prompt, padding=True, max_length=tok.model_max_length, truncation=True, return_tensors='pt')
ids = inputs.input_ids
model = T5EncoderModel.from_pretrained('/data/audioldm2/text_encoder_2').eval()
with torch.no_grad():
    out = model(input_ids=ids, attention_mask=inputs.attention_mask).last_hidden_state
ids.numpy().astype(np.int32)[0].tofile(f'{OUT}/t5_ids.i32')
out[0].numpy().astype(np.float32).tofile(f'{OUT}/t5_hidden.f32')
print('tokens', ids.shape[1], 'hidden', tuple(out.shape), flush=True)
