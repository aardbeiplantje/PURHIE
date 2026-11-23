#!/usr/bin/env python3
"""generate_image: text to image through the rocm-inference service (Z-Image-Turbo), for an AI harness or the shell.

One call: POST /v1/images/generations, the PNG written to a file. No ComfyUI.

  python3 tools/generate_image.py "a red square on a blue background" -o out.png
  python3 tools/generate_image.py "..." --size 512x512 --steps 9 --seed 7 --server http://host:8000/rocm

The server address: --server, or the ROCM_INFERENCE_URL environment variable (the base URL including the /rocm prefix
where nginx serves it; the path /v1/images/generations is added here). Standard library only.

As a harness tool: TOOL_SCHEMA is the tool definition (name, description, JSON Schema of the input) and generate_image()
is the function to call with the parsed input. It returns the file path, which the harness can show or attach.
"""
import argparse
import base64
import json
import os
import re
import socket
import sys
import urllib.error
import urllib.request

DEFAULT_SERVER = os.environ.get("ROCM_INFERENCE_URL", "http://127.0.0.1:8080")
MODEL = "z-image-turbo"
TIMEOUT_S = 3600  # one image takes about 30 s on the device; the queue can add more

TOOL_SCHEMA = {
    "name": "generate_image",
    "description": ("Generate an image from a text prompt (Z-Image-Turbo, on the local GPU). Returns the path of the "
                    "PNG file. A 1024x1024 image takes about 30 seconds."),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "What the image shows."},
            "size": {"type": "string", "default": "1024x1024",
                     "description": "WIDTHxHEIGHT, multiples of 16 from 64 to 1024."},
            "steps": {"type": "integer", "default": 9, "minimum": 1, "maximum": 50,
                      "description": "Sampler steps (9 is the model's default)."},
            "seed": {"type": "integer", "default": 42, "minimum": 0,
                     "description": "The same prompt, size, steps and seed give the same image."},
            "output": {"type": "string", "description": "Where to write the PNG (default: image_<seed>.png)."},
        },
        "required": ["prompt"],
    },
}


class ImageError(RuntimeError):
    """A generation failed: the message is one line for the caller."""


def _check_size(size):
    m = re.fullmatch(r"(\d+)x(\d+)", size or "")
    if not m:
        raise ImageError(f"size must be WIDTHxHEIGHT, not {size!r}")
    w, h = int(m.group(1)), int(m.group(2))
    if not (64 <= w <= 1024 and 64 <= h <= 1024 and w % 16 == 0 and h % 16 == 0):
        raise ImageError(f"size {size}: multiples of 16 from 64 to 1024")
    return size


def _service_error(server, err):
    try:
        detail = json.loads(err.read().decode("utf-8", "replace"))["error"]["message"]
    except (ValueError, KeyError, TypeError):
        detail = err.reason or "no detail"
    return f"{server}: HTTP {err.code}: {detail}"


def generate_image(prompt, size="1024x1024", steps=9, seed=42, output=None, server=None):
    """Generate the image and write it as a PNG; returns the file path."""
    if not prompt or not prompt.strip():
        raise ImageError("the prompt is empty")
    size = _check_size(size)
    steps = int(steps)
    if not 1 <= steps <= 50:
        raise ImageError(f"steps must be 1 to 50, not {steps}")
    seed = int(seed)
    if seed < 0:
        raise ImageError("seed must be 0 or more")
    base = (server or DEFAULT_SERVER).rstrip("/")
    body = json.dumps({"model": MODEL, "prompt": prompt, "size": size, "steps": steps, "seed": seed,
                       "response_format": "b64_json"}).encode("utf-8")
    req = urllib.request.Request(base + "/v1/images/generations", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            reply = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise ImageError(_service_error(base, err)) from None
    except urllib.error.URLError as err:
        raise ImageError(f"{base}: cannot connect ({err.reason}); check the server address") from None
    except (socket.timeout, TimeoutError):
        raise ImageError(f"{base}: no answer within {TIMEOUT_S} s") from None
    except ValueError:
        raise ImageError(f"{base}: the reply is not JSON (is the address the service's /rocm prefix?)") from None
    try:
        png = base64.b64decode(reply["data"][0]["b64_json"])
    except (KeyError, IndexError, TypeError, ValueError):
        raise ImageError(f"{base}: the reply has no image") from None
    path = output or f"image_{seed}.png"
    with open(path, "wb") as f:
        f.write(png)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate an image with the rocm-inference service (Z-Image-Turbo).")
    ap.add_argument("prompt")
    ap.add_argument("-o", "--output", help="the PNG file (default: image_<seed>.png)")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--server", default=None, help=f"the service's base URL (default {DEFAULT_SERVER})")
    args = ap.parse_args(argv)
    try:
        path = generate_image(args.prompt, size=args.size, steps=args.steps, seed=args.seed, output=args.output,
                              server=args.server)
    except ImageError as err:
        print(f"generate_image: {err}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
