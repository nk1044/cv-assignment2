"""Turn a training checkpoint into the deployed parameter file weights/model.pt.

    python -m src.export --ckpt checkpoints/base.pt [--tune] [--flip-tta] [--fp16]

1. folds BatchNorm and the 1/255 input scale into the convs;
2. with --tune, picks the logit threshold that maximises validation Q (never DUTS-TE);
3. stores weights as int8 + per-channel scales (default) or float16, with the threshold;
4. checks the deployed CPU pipeline against the original float32 model on validation photos.
"""
import argparse
import os

import numpy as np
import torch

from src.config import WEIGHTS_FILE
from src.data import load_set, load_split
from src.metrics import SegMeter
from src.model import SegNet, fuse_for_inference
from src.pipeline import Segmenter
from src.weights_io import save_params

THRESHOLDS = np.round(np.arange(-1.5, 1.51, 0.1), 2)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # our own training checkpoint
    model = SegNet(**ckpt["config"])
    model.load_state_dict(ckpt["state"])
    return model.eval(), ckpt


def val_logits(seg, limit=0):
    images, masks, _ = load_set("tr")
    _, val = load_split()
    val = val[:limit] if limit else val
    return [seg.logits(np.array(images[i])) for i in val], [np.array(masks[i]) for i in val]


def tune_threshold(logits, masks):
    best = None
    for t in THRESHOLDS:
        meter = SegMeter()
        for lg, gt in zip(logits, masks):
            meter.update(lg > t, gt)
        s = meter.summary()
        print(f"  threshold {t:+.1f}: mIoU {s['mIoU']:.4f} BF1 {s['BF1']:.4f} Q {s['Q']:.4f}")
        if best is None or s["Q"] > best[1]["Q"]:
            best = (float(t), s)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=str(WEIGHTS_FILE))
    p.add_argument("--fp16", action="store_true", help="float16 weights instead of int8")
    p.add_argument("--flip-tta", action="store_true", help="average with the mirrored photo at inference")
    p.add_argument("--tune", action="store_true", help="tune the logit threshold on validation")
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()

    model, ckpt = load_checkpoint(args.ckpt)
    reference = SegNet(**ckpt["config"])
    reference.load_state_dict(ckpt["state"])
    reference.eval()
    fused = fuse_for_inference(model)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def save(threshold):
        save_params(args.out, ckpt["config"], fused.state_dict(), threshold, args.flip_tta, int8=not args.fp16)

    threshold = args.threshold
    if args.tune:
        save(0.0)
        print("tuning the threshold on validation:")
        threshold, s = tune_threshold(*val_logits(Segmenter(args.out)))
        print(f"best threshold {threshold:+.1f} (val Q {s['Q']:.4f})")
    save(threshold)

    # sanity check: deployed pipeline (fused, quantised, cv2 upsampling) vs float32 model
    seg = Segmenter(args.out)
    images, _, _ = load_set("tr")
    _, val = load_split()
    agree = []
    with torch.no_grad():
        for i in val[:50]:
            img = np.array(images[i])
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
            x = torch.nn.functional.interpolate(x, size=(seg.res, seg.res), mode="area")
            ref = (reference(x)[0, 0] > threshold).numpy()
            agree.append((seg(img) == ref).mean())
    print(f"saved {args.out}: {os.path.getsize(args.out):,} bytes, threshold {threshold:+.2f}, "
          f"flip_tta {args.flip_tta} (checkpoint epoch {ckpt['epoch']}, val Q {ckpt['val']['Q']:.4f})")
    print(f"pixel agreement deployed vs float32 on 50 val photos: {np.mean(agree):.5f}"
          + ("" if not args.flip_tta else " (reference has no TTA)"))


if __name__ == "__main__":
    main()
