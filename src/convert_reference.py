"""Load the REFERENCE model shipped with the sample code into this project's format.

The sample code's TinyUNet has the same layer structure as SegNet (strided stem, three
depthwise-separable encoder stages, global context, three U-Net decoder blocks, 1x1 head),
only with different parameter names. This script renames its (already BN-fused, int8)
parameters so the model runs in this project's own pipeline, predict.py, timing and
evaluation code. The weights were trained by the sample code's author, not by this project;
results obtained with them must be reported as "reference model" results.

    python -m src.convert_reference [--src sample-code/weights/model.pt] [--out weights/model.pt]
    python -m src.convert_reference --reference-split   # also write the sample's train/val split

--reference-split overwrites data/cache/split.npz with the split the reference model was
trained with (seed 2024, 1,053 validation photos), so validation metrics are not inflated by
photos it saw during training. Needs data/cache/ from `python -m src.data`.
"""
import argparse
import json
import os
import re

import numpy as np
import torch

from src.config import CACHE, RESULTS, ROOT, WEIGHTS_FILE
from src.weights_io import SCALE

REFERENCE_SPLIT_SEED, REFERENCE_VAL_COUNT = 2024, 1053
RENAMES = [(r"^enc(\d)\.", lambda m: f"encoder.{int(m.group(1)) - 1}."),
           (r"^dec(\d)\.", lambda m: f"decoder.{2 - int(m.group(1))}.")]  # dec2 runs first


def rename(key):
    for pattern, repl in RENAMES:
        key = re.sub(pattern, repl, key)
    return key


def convert(src, out):
    ref = torch.load(src, map_location="cpu", weights_only=True)
    config = {k: ref["config"][k] for k in ("widths", "depths", "in_res", "out_size")}
    if ref["config"].get("refine"):
        raise SystemExit("reference model uses a refine stage, which SegNet does not have")
    state = {rename(k): v for k, v in ref["state"].items()}
    params = {"config": config, "infer": {"threshold": 0.0, "flip_tta": False}, "state": state}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save(params, out)

    # load check: every SegNet parameter must be present with the right shape
    from src.model import build_fused
    from src.weights_io import unpack_state
    build_fused(config).load_state_dict(unpack_state(state), strict=True)
    n_params = sum(v.numel() for k, v in state.items() if not k.endswith(SCALE))
    info = {"source": str(src), "out": str(out), "config": config, "params": n_params,
            "S_disk_bytes": os.path.getsize(out), "storage": "int8 conv weights + per-channel scales, float16 biases"}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "reference_model.json").write_text(json.dumps(info, indent=1))
    print(f"converted {src} -> {out}: {n_params:,} parameters, {info['S_disk_bytes']:,} bytes, config {config}")


def write_reference_split():
    n = len((CACHE / "tr_names.txt").read_text().split())
    perm = np.random.default_rng(REFERENCE_SPLIT_SEED).permutation(n)
    val, train = np.sort(perm[:REFERENCE_VAL_COUNT]), np.sort(perm[REFERENCE_VAL_COUNT:])
    np.savez(CACHE / "split.npz", train=train, val=val)
    print(f"reference split written: {len(train)} train / {len(val)} val (seed {REFERENCE_SPLIT_SEED})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=str(ROOT / "sample-code" / "weights" / "model.pt"))
    p.add_argument("--out", default=str(WEIGHTS_FILE))
    p.add_argument("--reference-split", action="store_true")
    args = p.parse_args()
    convert(args.src, args.out)
    if args.reference_split:
        write_reference_split()


if __name__ == "__main__":
    main()
