"""Turn a training checkpoint into a deployed parameter file.

    python -m src.export --ckpt checkpoints/base.pt [--out weights/model.pt] [--tune] [--flip-tta] [--fp16]

1. folds BatchNorm and the 1/255 input scale into the convs;
2. with --tune, picks the logit threshold that maximises validation Q (never DUTS-TE);
3. stores weights as int8 + per-channel scales (default) or float16, plus the threshold;
4. checks the deployed CPU pipeline against the original float32 model on validation photos.
Writes results/export_<out name>.json (threshold sweep, agreement, file size) for the report.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.config import RESULTS, WEIGHTS_FILE
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


def val_logits(seg):
    """Deployed-pipeline logits on every validation photo: (N,256,256) float32 tensor + masks."""
    images, masks, _ = load_set("tr")
    _, val = load_split()
    logits = torch.from_numpy(np.stack([seg.logits(np.array(images[i])) for i in val]))
    return logits, torch.from_numpy(np.array(masks[val])).bool()


def sweep(logits, masks, thresholds=THRESHOLDS, chunk=128):
    rows = []
    for t in thresholds:
        meter = SegMeter()
        for s in range(0, len(logits), chunk):
            meter.update(logits[s:s + chunk] > t, masks[s:s + chunk])
        rows.append({"threshold": float(t), **meter.summary()})
    return rows


def agreement(seg, reference, threshold, n=50):
    images, _, _ = load_set("tr")
    _, val = load_split()
    agree = []
    with torch.no_grad():
        for i in val[:n]:
            img = np.array(images[i])
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
            x = torch.nn.functional.interpolate(x, size=(seg.res, seg.res), mode="area")
            agree.append((seg(img) == (reference(x)[0, 0] > threshold).numpy()).mean())
    return float(np.mean(agree))


def export(ckpt_path, out, fp16=False, flip_tta=False, tune=False, threshold=0.0, verbose=True):
    model, ckpt = load_checkpoint(ckpt_path)
    reference, _ = load_checkpoint(ckpt_path)
    fused = fuse_for_inference(model)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    def save(t):
        save_params(out, ckpt["config"], fused.state_dict(), t, flip_tta, int8=not fp16)

    save(0.0)
    rows = sweep(*val_logits(Segmenter(out)), thresholds=THRESHOLDS if tune else [threshold])
    best = max(rows, key=lambda r: r["Q"])
    if verbose and tune:
        for r in rows:
            print(f"  threshold {r['threshold']:+.1f}: mIoU {r['mIoU']:.4f} BF1 {r['BF1']:.4f} Q {r['Q']:.4f}")
    save(best["threshold"])

    info = {"ckpt": str(ckpt_path), "out": str(out), "config": ckpt["config"], "train_args": ckpt.get("args"),
            "epoch": ckpt["epoch"], "ckpt_val": ckpt["val"], "storage": "float16" if fp16 else "int8",
            "flip_tta": flip_tta, "threshold": best["threshold"], "val": {k: best[k] for k in ("mIoU", "BF1", "Q", "n")},
            "S_disk_bytes": os.path.getsize(out), "agreement_vs_fp32": None if flip_tta else
            agreement(Segmenter(out), reference, best["threshold"]), "sweep": rows}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"export_{Path(out).stem}.json").write_text(json.dumps(info, indent=1))
    if verbose:
        print(f"saved {out}: {info['S_disk_bytes']:,} bytes, {info['storage']}, threshold {best['threshold']:+.1f}, "
              f"flip_tta {flip_tta}, val Q {best['Q']:.4f} (checkpoint epoch {ckpt['epoch']})")
        if info["agreement_vs_fp32"] is not None:
            print(f"pixel agreement deployed vs float32 on 50 val photos: {info['agreement_vs_fp32']:.5f}")
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=str(WEIGHTS_FILE))
    p.add_argument("--fp16", action="store_true", help="float16 weights instead of int8")
    p.add_argument("--flip-tta", action="store_true", help="average with the mirrored photo at inference")
    p.add_argument("--tune", action="store_true", help="tune the logit threshold on validation")
    p.add_argument("--threshold", type=float, default=0.0)
    args = p.parse_args()
    export(args.ckpt, args.out, args.fp16, args.flip_tta, args.tune, args.threshold)


if __name__ == "__main__":
    main()
