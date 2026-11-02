"""Evaluate the deployed CPU pipeline on validation, DUTS-TE and the captured photos, and
combine it with results/timing.json into the final results table.

    python -m src.evaluate [--weights weights/model.pt] [--sets val te captured]

Run DUTS-TE ("te") only once, for the final model. Writes results/metrics.json,
results/per_image_<set>.json and results/captured_pred/<name>.png.
"""
import src.threads  # noqa: F401

import argparse
import json
import os

import cv2
import numpy as np
import torch

from src.captured import load_captured
from src.config import RESULTS, WEIGHTS_FILE
from src.data import load_set, load_split
from src.metrics import SegMeter, boundary_f1, ces
from src.pipeline import Segmenter


def image_scores(pred, gt):
    p, g = torch.from_numpy(pred).bool()[None], torch.from_numpy(gt).bool()[None]
    inter, union = int((p & g).sum()), int((p | g).sum())
    return {"IoU_obj": inter / union if union else 1.0, "BF1": float(boundary_f1(p, g)[0])}


def run_set(seg, images, masks, names):
    meter, rows, preds = SegMeter(), [], []
    for img, gt, name in zip(images, masks, names):
        pred, gt = seg(np.array(img)), np.array(gt)
        meter.update(pred, gt)
        rows.append({"name": name, **image_scores(pred, gt)})
        preds.append(pred)
    return meter.summary(), rows, preds


def get_set(which):
    if which == "captured":
        return load_captured()
    images, masks, names = load_set("te" if which == "te" else "tr")
    if which == "val":
        idx = load_split()[1]
        return images[idx], masks[idx], [names[i] for i in idx]
    return images, masks, names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(WEIGHTS_FILE))
    p.add_argument("--sets", nargs="+", default=["val", "captured"], choices=["val", "te", "captured"])
    args = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    seg = Segmenter(args.weights)

    out_path = RESULTS / "metrics.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for which in args.sets:
        images, masks, names = get_set(which)
        if not len(names):
            print(f"{which}: no photos found, skipped")
            continue
        summary, rows, preds = run_set(seg, images, masks, names)
        results[which] = summary
        (RESULTS / f"per_image_{which}.json").write_text(json.dumps(rows, indent=1))
        if which == "captured":
            (RESULTS / "captured_pred").mkdir(exist_ok=True)
            for name, pred in zip(names, preds):
                cv2.imwrite(str(RESULTS / "captured_pred" / f"{name}.png"), pred * 255)
        print(f"{which:9s} n={summary['n']:5d}  mIoU {summary['mIoU']:.4f}  BF1 {summary['BF1']:.4f}  Q {summary['Q']:.4f}")

    timing_path = RESULTS / "timing.json"
    if timing_path.exists() and "te" in results:
        timing = json.loads(timing_path.read_text())
        s_disk = os.path.getsize(args.weights)
        results["timing"] = timing
        results["S_disk_bytes"] = s_disk
        results["CES"] = ces(results["te"]["Q"], timing["T_CPU_ms"], s_disk)
        print(f"T_CPU {timing['T_CPU_ms']:.2f} ms ({timing['cpu']}), S_disk {s_disk:,} B, CES {results['CES']:.4f}")
    out_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
