"""Validation-only tuning of an already deployed parameter file: logit threshold and flip TTA.

    python -m src.tune_deployed [--weights weights/model.pt] [--apply]

For flip TTA off and on, it sweeps the threshold on the validation photos, times the pipeline
on 500 validation photos (1 CPU thread) and computes validation CES. Writes
results/tuning_<weights name>.json. With --apply, the setting with the best validation CES is
written back into the parameter file (only its "infer" entry changes, not the weights).
"""
import src.threads  # noqa: F401

import argparse
import json
from pathlib import Path

import torch

from src.config import RESULTS, WEIGHTS_FILE
from src.data import load_set, load_split
from src.export import sweep, val_logits
from src.metrics import ces
from src.pipeline import Segmenter
from src.time_cpu import cpu_name, time_photos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(WEIGHTS_FILE))
    p.add_argument("--time-n", type=int, default=500)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    params = torch.load(args.weights, map_location="cpu", weights_only=True)
    original_infer = dict(params["infer"])
    images, _, _ = load_set("tr")
    val = load_split()[1]
    rows = []
    tmp = Path(args.weights).with_suffix(".tune.pt")
    for tta in (False, True):
        params["infer"] = {"threshold": 0.0, "flip_tta": tta}
        torch.save(params, tmp)
        seg = Segmenter(str(tmp))
        t_ms = time_photos(seg, images, val[:args.time_n])
        for r in sweep(*val_logits(seg)):
            rows.append({"flip_tta": tta, **r, "T_ms": t_ms,
                         "val_CES": ces(r["Q"], t_ms, tmp.stat().st_size)})
        best = max((r for r in rows if r["flip_tta"] == tta), key=lambda r: r["val_CES"])
        print(f"flip_tta={tta}: {t_ms:.2f} ms, best threshold {best['threshold']:+.1f}: mIoU {best['mIoU']:.4f} "
              f"BF1 {best['BF1']:.4f} Q {best['Q']:.4f} val CES {best['val_CES']:.4f}", flush=True)
    tmp.unlink()

    base = next(r for r in rows if not r["flip_tta"] and r["threshold"] == 0.0)
    best = max(rows, key=lambda r: r["val_CES"])
    out = {"weights": args.weights, "cpu": cpu_name(), "n_val": len(val), "original_infer": original_infer,
           "default": base, "best": best, "rows": rows}
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"tuning_{Path(args.weights).stem}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"default (threshold 0, no TTA): Q {base['Q']:.4f}, val CES {base['val_CES']:.4f}")
    print(f"best: threshold {best['threshold']:+.1f}, flip_tta {best['flip_tta']}: Q {best['Q']:.4f}, "
          f"val CES {best['val_CES']:.4f} -> {path}")
    if args.apply:
        params["infer"] = {"threshold": best["threshold"], "flip_tta": best["flip_tta"]}
        torch.save(params, args.weights)
        print(f"applied to {args.weights}")
    else:
        params["infer"] = original_infer


if __name__ == "__main__":
    main()
