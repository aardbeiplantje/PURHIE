# rocm-inference ComfyUI nodes

Copy this folder to the ComfyUI custom nodes directory (in the container: `/comfyui-user/custom_nodes/rocm_inference`).

The nodes call the rocm-inference HTTP service (`rocm-inference-server`, `/v1/images/generations`). They hold no model code.
The server address: the node's `server` input, or the `ROCM_INFERENCE_URL` environment variable (default `http://rocm-inference:8080`).

Nodes:
- `ROCm Inference: Image`: prompt, model name (as the service lists it), size WIDTHxHEIGHT (z-image-turbo: 1024x1024 by default), steps (z-image-turbo: 9, the ComfyUI workflow's), seed (42 matches the workflow's image: torch's noise for the seed) -> IMAGE.
- `ROCm Inference: Audio`: prompt, negative prompt, seconds (1 to 10, 5 by default), steps (1 to 200, AudioLDM 2's 100 by default), guidance scale (3.5 by default) and seed -> AUDIO (16 kHz, mono). Calls `/v1/audio/generations`; the server needs `--audioldm2` (run.sh: `AUDIOLDM2_DIR`).

Not yet: the stage endpoints (encode, sample, decode) that would let a workflow use the engine's pieces separately.
