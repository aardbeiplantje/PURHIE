# AudioLDM 2 stage oracle: the vocoder (vocoder/, the SpeechT5 HiFi-GAN) on a fixed mel [frames x 64] (seed 0, standard normal).
# Writes /data/audioldm2-stages/voc_mel.f32 (frames x 64) and voc_out.f32 (the waveform, samples). fp32 on the CPU (transformers 4.44).
import numpy as np, torch
from transformers import SpeechT5HifiGan
OUT = '/data/audioldm2-stages'
voc = SpeechT5HifiGan.from_pretrained('/data/audioldm2/vocoder').eval()
g = torch.Generator().manual_seed(0)
T = 128
mel = torch.randn(T, 64, generator=g)
with torch.no_grad():
    wav = voc(mel)
mel.numpy().astype(np.float32).tofile(f'{OUT}/voc_mel.f32')
wav.numpy().astype(np.float32).tofile(f'{OUT}/voc_out.f32')
print('waveform', tuple(wav.shape), 'rms %.4f' % float(wav.pow(2).mean().sqrt()), flush=True)
