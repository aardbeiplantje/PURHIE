# rocm-inference ComfyUI nodes

Copy this folder to the ComfyUI custom nodes directory (in the container: `/comfyui-user/custom_nodes/rocm_inference`).

The nodes call the rocm-inference HTTP service (`rocm-inference-server`, `/v1/images/generations`). They hold no model code.
The server address: the node's `server` input, or the `ROCM_INFERENCE_URL` environment variable (default `http://rocm-inference:8080`).

Nodes:
- `ROCm Inference: Image`: prompt, model name (as the service lists it), size WIDTHxHEIGHT, seed -> IMAGE.

Not yet: the stage endpoints (encode, sample, decode) that would let a workflow use the engine's pieces separately.
