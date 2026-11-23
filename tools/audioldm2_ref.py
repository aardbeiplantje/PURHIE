# AudioLDM 2 reference: a text-to-audio clip with the diffusers pipeline, from the local files in /data/audioldm2.
import sys, wave, torch, numpy as np
from diffusers import AudioLDM2Pipeline
prompt = sys.argv[1] if len(sys.argv) > 1 else "ocean waves crashing on a beach"
out = sys.argv[2] if len(sys.argv) > 2 else "/data/audioldm2/ref_ocean.wav"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
pipe = AudioLDM2Pipeline.from_pretrained("/data/audioldm2", torch_dtype=torch.float32).to("cuda")
g = torch.Generator(device="cpu").manual_seed(seed)
audio = pipe(prompt, num_inference_steps=100, audio_length_in_s=5.0, generator=g).audios[0]
pcm = (np.clip(audio, -1, 1) * 32767).astype('<i2')
with wave.open(out, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm.tobytes())
print("wrote", out, audio.shape, "peak %.3f" % float(np.abs(audio).max()), flush=True)
