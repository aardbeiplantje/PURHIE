# Nanonets-OCR2-3B (Qwen2.5-VL, 3B) on a picture as transformers runs it, in float32 on the GPU: the reference for the engine's
# OCR path (the vision tower's merged tokens, and the greedy transcription). Outputs in OUT (every file is float32 or text):
#   pixel_values.f32  [N][1176]   the processor's patch rows (patch 14, temporal 2, merge 2)
#   grid.i32          (t, h, w) of the patch grid
#   vis_merged.f32    [N / 4][2048] the vision tower's merged output (the language model's image tokens)
#   ids.i32           the prompt's token ids (the chat template with the picture)
#   answer.txt        the greedy transcription (max_new_tokens tokens at most)
# Usage: python3 tools/ocr_qwen25vl_oracle.py --model DIR --image PICTURE --out OUT [--max-new 512] [--prompt TEXT]
import argparse, os
import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--image', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--max-new', type=int, default=512)
ap.add_argument('--prompt', default='Read all the text in this picture exactly as written, line by line. Output only the text.')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False


def save(name, t):
    np.ascontiguousarray(t.detach().float().cpu().numpy()).astype(np.float32).tofile(os.path.join(args.out, name))


proc = AutoProcessor.from_pretrained(args.model)
# The checkpoint ties lm_head to embed_tokens, but only text_config says so; transformers reads the top-level flag,
# and without it lm_head is random (the transcription comes out as noise).
cfg = AutoConfig.from_pretrained(args.model)
cfg.tie_word_embeddings = True
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, config=cfg, dtype=torch.float32).cuda().eval()
assert model.lm_head.weight is model.get_input_embeddings().weight, 'lm_head is not tied to the embeddings'
img = Image.open(args.image).convert('RGB')
msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': args.prompt}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], images=[img], return_tensors='pt').to('cuda')
np.asarray(inputs['input_ids'][0].cpu().numpy(), dtype=np.int32).tofile(os.path.join(args.out, 'ids.i32'))
np.asarray(inputs['image_grid_thw'][0].cpu().numpy(), dtype=np.int32).tofile(os.path.join(args.out, 'grid.i32'))
save('pixel_values.f32', inputs['pixel_values'])
with torch.no_grad():
    vis = model.model.visual(inputs['pixel_values'], grid_thw=inputs['image_grid_thw']).pooler_output
    save('vis_merged.f32', vis)
    out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
answer = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
open(os.path.join(args.out, 'answer.txt'), 'w').write(answer)
print('grid', inputs['image_grid_thw'][0].tolist(), 'image tokens', int(vis.shape[0]), 'tokens', inputs['input_ids'].shape[1])
print(answer)
