"""The ComfyUI workflow's input preparation for TRELLIS.2, for engine runs on a photo: RemoveBackground (BiRefNet) then ImageCropToMask
(a square crop around the mask with a margin, the object over black, resized). The engine has no background removal; this is the reference
step that gives it the same kind of input as the workflow (the workflow's node 312: 1024 x 1024, margin 1.1, black).

BiRefNet's model code is the Hugging Face repository's (ZhengPeng7/BiRefNet: birefnet.py and BiRefNet_config.py, downloaded to --code-dir,
not part of this repository); the weights are the local ComfyUI copy. Runs on the CPU (no GPU contention with an engine run).

Usage: python3 -I tools/bg_crop_birefnet.py input_image out_dir --code-dir DIR [--weights FILE] [--size 1024] [--margin 1.1]
Writes out_dir/mask.png (the BiRefNet mask at the input's size) and out_dir/crop_SIZE.png (RGB, the object over black).
"""
import argparse
import importlib
import importlib.util
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms


def load_birefnet_module(code_dir):
    # the downloaded files import each other relatively: load the directory as a package
    name = "birefnet_remote"
    spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [code_dir]
    sys.modules[name] = pkg
    return importlib.import_module(name + ".birefnet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--weights", default="/comfyui/background_removal/birefnet.safetensors")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--margin", type=float, default=1.1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mod = load_birefnet_module(os.path.abspath(args.code_dir))
    model = mod.BiRefNet(bb_pretrained=False)
    missing, unexpected = model.load_state_dict(load_file(args.weights), strict=False)
    print("weights: missing", len(missing), "unexpected", len(unexpected), flush=True)
    assert not missing, missing[:5]
    model.eval()

    image = Image.open(args.input).convert("RGB")
    # BiRefNet's preprocessing (TRELLIS.2's rembg wrapper): 1024 x 1024, ImageNet normalisation; the sigmoid of the last prediction
    tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        pred = model(tf(image).unsqueeze(0))[-1].sigmoid()[0, 0]
    mask = transforms.ToPILImage()(pred).resize(image.size)
    mask.save(os.path.join(args.out_dir, "mask.png"))

    # ImageCropToMask: the mask's bounding box, square, times the margin, centred; the object over black; resized
    a = np.asarray(mask).astype(np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0 + 1, y1 - y0 + 1) * args.margin
    rgb = np.asarray(image).astype(np.float32) * a[:, :, None]
    canvas = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
    box = (int(round(cx - side / 2)), int(round(cy - side / 2)), int(round(cx + side / 2)), int(round(cy + side / 2)))
    crop = canvas.crop(box)  # PIL pads outside the image with black
    crop = crop.resize((args.size, args.size), Image.Resampling.LANCZOS)
    out = os.path.join(args.out_dir, "crop_%d.png" % args.size)
    crop.save(out)
    print("mask bbox", (int(x0), int(y0), int(x1), int(y1)), "crop box", box, "->", out, flush=True)


if __name__ == "__main__":
    main()
