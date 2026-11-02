"""Model selection on VALIDATION only: export each checkpoint, time it on validation photos
on this CPU (one thread) and rank by validation CES.

    python -m src.compare checkpoints/*.pt [--tta both] [--time-n 500]

For each checkpoint (and each flip-TTA setting with --tta both) it writes
weights/candidates/<name>[_tta].pt and results/export_<...>.json, and keeps a running table
in results/experiments.json and results/experiments.md (the ablation table for the report).
Already-compared entries are kept, so you can run this after every new training run.
"""
import src.threads  # noqa: F401

import argparse
import json
from pathlib import Path

from src.config import RESULTS, WEIGHTS
from src.data import load_set, load_split
from src.export import export
from src.metrics import ces
from src.pipeline import Segmenter
from src.time_cpu import time_photos

TABLE_JSON = RESULTS / "experiments.json"
TABLE_MD = RESULTS / "experiments.md"


def write_table(rows):
    rows = sorted(rows.values(), key=lambda r: -r["val_CES"])
    TABLE_JSON.write_text(json.dumps(rows, indent=1))
    lines = ["| run | widths / depths | in_res | TTA | thr | S_disk bytes | T ms (val) | mIoU | BF1 | Q | val CES | epoch | train args |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c, a = r["config"], r.get("train_args") or {}
        extra = ", ".join(f"{k}={a[k]}" for k in ("epochs", "boundary_weight", "dice_weight", "ema", "lr") if k in a)
        lines.append(f"| {r['name']} | {c['widths']} / {c['depths']} | {c['in_res']} | {'yes' if r['flip_tta'] else 'no'} "
                     f"| {r['threshold']:+.1f} | {r['S_disk_bytes']:,} | {r['T_ms']:.2f} | {r['val']['mIoU']:.4f} "
                     f"| {r['val']['BF1']:.4f} | {r['val']['Q']:.4f} | **{r['val_CES']:.4f}** | {r['epoch']} | {extra} |")
    TABLE_MD.write_text("Validation-only model comparison (T measured on validation photos, 1 CPU thread)\n\n"
                        + "\n".join(lines) + "\n")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--tta", choices=["off", "on", "both"], default="both")
    p.add_argument("--time-n", type=int, default=500, help="validation photos to time")
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    rows = {r["name"]: r for r in json.loads(TABLE_JSON.read_text())} if TABLE_JSON.exists() else {}
    images, _, _ = load_set("tr")
    val = load_split()[1][:args.time_n]
    for ckpt in args.ckpts:
        for tta in {"off": [False], "on": [True], "both": [False, True]}[args.tta]:
            name = Path(ckpt).stem + ("_tta" if tta else "") + ("_fp16" if args.fp16 else "")
            out = WEIGHTS / "candidates" / f"{name}.pt"
            info = export(ckpt, str(out), fp16=args.fp16, flip_tta=tta, tune=True, verbose=False)
            t_ms = time_photos(Segmenter(str(out)), images, val)
            row = {"name": name, **{k: v for k, v in info.items() if k != "sweep"}, "T_ms": t_ms}
            row["val_CES"] = ces(info["val"]["Q"], t_ms, info["S_disk_bytes"])
            rows[name] = row
            print(f"{name}: val Q {info['val']['Q']:.4f}, {t_ms:.2f} ms, {info['S_disk_bytes']:,} B "
                  f"-> val CES {row['val_CES']:.4f}", flush=True)
            write_table(rows)
    print("\n".join(write_table(rows)))
    print(f"table: {TABLE_MD}")


if __name__ == "__main__":
    main()
