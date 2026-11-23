"""ComfyUI custom nodes that call the rocm-inference HTTP service (perl/ROCm-Inference bin/rocm-inference-server).

The node holds no model code: it sends the request to the service and returns the image. The service URL comes from the
node's `server` input, or from the ROCM_INFERENCE_URL environment variable when that is empty.
"""
import base64
import io
import json
import os
import socket
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image

DEFAULT_SERVER = os.environ.get("ROCM_INFERENCE_URL", "http://rocm-inference:8080")

class RocmInferenceError(RuntimeError):
    """A request to the service failed: one line for the ComfyUI log, without the library's chained traceback."""


def _fail(message):
    raise RocmInferenceError("[rocm-inference] " + message) from None


def _service_error(server, err):
    """The message of an HTTP error: the service's JSON {error: {message}} when it sent one, else the status line."""
    try:
        detail = json.loads(err.read().decode("utf-8", "replace"))["error"]["message"]
    except (ValueError, KeyError, TypeError):
        detail = err.reason or "no detail"
    return f"{server}: HTTP {err.code}: {detail}"


class RocmInferenceImage:
    """Text to image through /v1/images/generations (the service's model name, e.g. z-image-turbo or sd-turbo)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a red square on a blue background"}),
                "model": ("STRING", {"default": "z-image-turbo"}),
                "size": ("STRING", {"default": "1024x1024"}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**53 - 1}),
            },
            "optional": {
                "server": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "rocm-inference"

    def generate(self, prompt, model, size, steps, seed, server=""):
        base = (server or DEFAULT_SERVER).rstrip("/")
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "size": size,
            "steps": int(steps),  # z-image-turbo: the sampler's steps (9 is the ComfyUI workflow's)
            "seed": int(seed),
            "response_format": "b64_json",
        }).encode("utf-8")
        req = urllib.request.Request(base + "/v1/images/generations", data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                reply = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            _fail(_service_error(base, err))
        except urllib.error.URLError as err:
            _fail(f"{base}: cannot connect ({err.reason}); check the server address (the node's server input or ROCM_INFERENCE_URL)")
        except (socket.timeout, TimeoutError):
            _fail(f"{base}: no answer within the timeout (the generation is still running on the server or the server hung)")
        except ValueError:
            _fail(f"{base}: the reply is not JSON (is the address the service's /rocm prefix?)")
        try:
            png = base64.b64decode(reply["data"][0]["b64_json"])
        except (KeyError, IndexError, TypeError, ValueError):
            _fail(f"{base}: the reply has no image")
        rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32) / 255.0
        return (torch.from_numpy(rgb)[None, ...],)


NODE_CLASS_MAPPINGS = {
    "RocmInferenceImage": RocmInferenceImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RocmInferenceImage": "ROCm Inference: Image",
}
