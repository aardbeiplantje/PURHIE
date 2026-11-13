#!/usr/bin/env python3
# A voice for the engine's speech (/v1/audio/speech, src/qwen3tts_speech.cpp): the reference clip's codec frames, its
# transcript and its speaker embedding, computed with the official qwen-tts (0.1.1: the speaker encoder and the speech
# tokenizer's encoder, which the engine does not have yet). Writes a directory:
#   voice.json  {"ref_text", "frames", "codebooks", "n_embd"}
#   codes.i32   frames x codebooks int32, little-endian (frame-major)
#   spk.f32     n_embd float32, little-endian
#
#   tools/qwen3tts_voice.py <Qwen3-TTS-12Hz-0.6B-Base dir> <reference.wav> "<the reference's transcript>" <voice dir>
#
# Run it where qwen-tts runs (transformers 4.57.3; /data/tts-site in this checkout's reference setup). The reference clip
# should be a few seconds of one speaker: the codes are the voice's sound, the transcript must say what is in the clip.
import json
import os
import sys

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel


def main():
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    model_dir, ref_wav, ref_text, out = sys.argv[1:]
    torch.manual_seed(0)
    model = Qwen3TTSModel.from_pretrained(model_dir, device_map="cpu", dtype=torch.float32)
    item = model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text, x_vector_only_mode=False)[0]
    if item.ref_code is None or item.icl_mode is not True:
        sys.exit("qwen-tts gave no in-context codes for this clip")
    codes = item.ref_code.detach().cpu().numpy().astype("<i4")
    spk = item.ref_spk_embedding.detach().cpu().numpy().astype("<f4").reshape(-1)
    os.makedirs(out, exist_ok=True)
    codes.tofile(os.path.join(out, "codes.i32"))
    spk.tofile(os.path.join(out, "spk.f32"))
    with open(os.path.join(out, "voice.json"), "w") as f:
        json.dump({"ref_text": ref_text, "frames": int(codes.shape[0]), "codebooks": int(codes.shape[1]),
                   "n_embd": int(spk.shape[0])}, f, indent=1)
    print(f"{out}: {codes.shape[0]} frames x {codes.shape[1]} codebooks, speaker embedding {spk.shape[0]}")


if __name__ == "__main__":
    main()
