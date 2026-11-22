"""ComfyUI custom nodes that call the rocm-inference HTTP service (perl/ROCm-Inference bin/rocm-inference-server).

The node holds no model code: it sends the request to the service and returns the image. The service URL comes from the
node's `server` input, or from the ROCM_INFERENCE_URL environment variable when that is empty.
"""
import base64
import io
import json
import os
import urllib.request

import numpy as np
import torch
from PIL import Image

DEFAULT_SERVER = os.environ.get("ROCM_INFERENCE_URL", "http://rocm-inference:8080")


class RocmInferenceImage:
    """Text to image through /v1/images/generations (the service's model name, e.g. z-image-turbo or sd-turbo)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a red square on a blue background"}),
                "model": ("STRING", {"default": "z-image-turbo"}),
                "size": ("STRING", {"default": "256x256"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**53 - 1}),
            },
            "optional": {
                "server": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "rocm-inference"

    def generate(self, prompt, model, size, seed, server=""):
        base = (server or DEFAULT_SERVER).rstrip("/")
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "size": size,
            "seed": int(seed),
            "response_format": "b64_json",
        }).encode("utf-8")
        req = urllib.request.Request(base + "/v1/images/generations", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=3600) as resp:
            reply = json.loads(resp.read().decode("utf-8"))
        png = base64.b64decode(reply["data"][0]["b64_json"])
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(rgb)[None, ...],)


NODE_CLASS_MAPPINGS = {
    "RocmInferenceImage": RocmInferenceImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RocmInferenceImage": "ROCm Inference: Image",
}
