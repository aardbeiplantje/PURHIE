"""The ComfyUI workflow's input preparation for TRELLIS.2, for engine runs on a photo: RemoveBackground (BiRefNet) then ImageCropToMask
(a square crop around the mask with a margin, the object over black, resized). The engine has no background removal; this is the reference
step that gives it the same kind of input as the workflow (the workflow's node 312: 1024 x 1024, margin 1.1, black).

BiRefNet's model code is the Hugging Face repository's (ZhengPeng7/BiRefNet: birefnet.py and BiRefNet_config.py, downloaded to --code-dir,
not part of this repository); the weights are the local ComfyUI copy. Runs on the CPU (no GPU contention with an engine run).

Usage: python3 -I tools/bg_crop_birefnet.py input_image out_dir --code-dir DIR [--weights FILE] [--size 1024] [--margin 1.1] [--dump DIR]
Writes out_dir/mask.png (the BiRefNet mask at the input's size) and out_dir/crop_SIZE.png (RGB, the object over black). With --dump,
the oracle of the engine's port (src/birefnet.cpp, tests/birefnet_verify.cpp): the model's input and its stages as float32 files in DIR
(index.json: each name with its shape), from forward hooks on the backbone (both scales), the squeeze block and the decoder's blocks.
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
    ap.add_argument("--dump", default=None)
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
    x = tf(image).unsqueeze(0)
    stages = {}
    if args.dump:
        os.makedirs(args.dump, exist_ok=True)
        calls = {"bb": 0}

        def keep(name):
            def hook(module, inputs, output):
                if name == "bb":  # the backbone runs twice: the image, then the half-size image
                    calls["bb"] += 1
                    for i, t in enumerate(output):
                        stages["bb%d_x%d" % (calls["bb"], i + 1)] = t
                else:
                    stages[name] = output
            return hook

        model.bb.register_forward_hook(keep("bb"))
        model.squeeze_module.register_forward_hook(keep("squeeze"))
        d = model.decoder
        for n in ("ipt_blk5", "ipt_blk4", "ipt_blk3", "ipt_blk2", "ipt_blk1", "decoder_block4", "decoder_block3", "decoder_block2",
                  "decoder_block1", "gdt_convs_attn_4", "gdt_convs_attn_3", "gdt_convs_attn_2", "lateral_block4", "lateral_block3",
                  "lateral_block2", "conv_out1"):
            getattr(d, n).register_forward_hook(keep(n))
        stages["input"] = x
    with torch.no_grad():
        logits = model(x)[-1]
        pred = logits.sigmoid()[0, 0]
    if args.dump:
        index = {}
        for name, t in stages.items():
            a = t.detach().float().contiguous().numpy()[0]
            a.astype(np.float32).tofile(os.path.join(args.dump, name + ".f32"))
            index[name] = list(a.shape)
        import json
        with open(os.path.join(args.dump, "index.json"), "w") as f:
            json.dump(index, f, indent=1)
        print("dumped", len(index), "stages to", args.dump, flush=True)
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
