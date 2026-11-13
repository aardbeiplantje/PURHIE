# Reference for the MiniMax-H3 text conditioning: transformers' Qwen3-VL text part, the hidden state after decoder layer 50
# (hidden_states[50]; the card's text_encoder_layer), for prompts with no special tokens and no template.
import sys, numpy as np, torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
D = '/data/minimax-ref/te'
OUT = '/data/minimax-ref/te'
tok = AutoTokenizer.from_pretrained(D)
m = Qwen3VLForConditionalGeneration.from_pretrained(D, dtype=torch.bfloat16)
m.eval()
prompts = ['a photo of a red fox in the snow', 'A  Photo, of CAT\'s!!', 'hello   world 123']
for i, p in enumerate(prompts):
    ids = tok(p, add_special_tokens=False, return_tensors='pt').input_ids.to('cpu')
    with torch.inference_mode():
        out = m.model.language_model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states[50][0].float().cpu().numpy()
    np.save(f'{OUT}/hs50_{i}.npy', hs)
    np.save(f'{OUT}/ids_{i}.npy', ids[0].cpu().numpy().astype(np.int32))
    print(i, repr(p), 'tokens', ids.shape[1], 'hidden', hs.shape, 'norm', float(np.linalg.norm(hs)), flush=True)
print('done', flush=True)
