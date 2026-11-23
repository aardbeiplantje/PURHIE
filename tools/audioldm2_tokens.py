# AudioLDM 2 tokens (the pipeline's tokenization, done here in Python for now): the prompt and the negative prompt (""), for the CLAP
# text encoder (RoBERTa, padded to 512, with the mask) and the T5 encoder (padding to the longest). Writes to /data/audioldm2-stages/:
# tok_<which>_clap_ids.i32, tok_<which>_clap_mask.i32, tok_<which>_t5_ids.i32, with which = pos (the prompt) or neg (the empty prompt).
import sys
import numpy as np
from transformers import AutoTokenizer, RobertaTokenizer
OUT = '/data/audioldm2-stages'
prompt = sys.argv[1] if len(sys.argv) > 1 else 'ocean waves crashing on a beach'
roberta = RobertaTokenizer.from_pretrained('/data/audioldm2/tokenizer')
t5 = AutoTokenizer.from_pretrained('/data/audioldm2/tokenizer_2')
for which, text in [('pos', prompt), ('neg', '')]:
    c = roberta(text, padding='max_length', max_length=roberta.model_max_length, truncation=True, return_tensors='pt')
    c.input_ids[0].numpy().astype(np.int32).tofile(f'{OUT}/tok_{which}_clap_ids.i32')
    c.attention_mask[0].numpy().astype(np.int32).tofile(f'{OUT}/tok_{which}_clap_mask.i32')
    u = t5(text, padding=True, max_length=t5.model_max_length, truncation=True, return_tensors='pt')
    u.input_ids[0].numpy().astype(np.int32).tofile(f'{OUT}/tok_{which}_t5_ids.i32')
    print(which, 't5 ids', u.input_ids[0].tolist(), flush=True)
