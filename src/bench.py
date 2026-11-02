"""Single-thread CPU speed and file size of candidate architectures, using random weights,
to decide which ones are worth training. Not the official T_CPU (see src.time_cpu).

    python -m src.bench                     # every candidate below
    python -m src.bench r128_w32 r160_w32   # selected ones
    python -m src.bench --flip-tta          # include mirrored-photo averaging
The table is also written to results/bench[_tta].md.
"""
import src.threads  # noqa: F401

import argparse
import os
import tempfile
import time

import numpy as np

from src.config import RESULTS
from src.model import SegNet, count_params, fuse_for_inference
from src.pipeline import Segmenter
from src.weights_io import save_params

CANDIDATES = {
    "r128_w16": dict(widths=(16, 24, 40, 64), depths=(2, 2, 3), in_res=128),
    "r128_w32": dict(widths=(32, 48, 96, 128), depths=(2, 3, 4), in_res=128),
    "r128_w48": dict(widths=(48, 64, 128, 192), depths=(2, 3, 4), in_res=128),
    "r128_w48_5s": dict(widths=(48, 64, 128, 192, 256), depths=(2, 3, 4, 2), in_res=128),
    "r160_w32": dict(widths=(32, 48, 96, 128), depths=(2, 3, 4), in_res=160),
    "r192_w32": dict(widths=(32, 48, 96, 128), depths=(2, 3, 4), in_res=192),
    "r256_w24": dict(widths=(24, 32, 64, 128), depths=(2, 3, 4), in_res=256),
}


def bench(config, flip_tta, runs=60):
    model = fuse_for_inference(SegNet(**config))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        save_params(path, model.config, model.state_dict(), flip_tta=flip_tta)
        size = os.path.getsize(path)
        seg = Segmenter(path)
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    for _ in range(5):
        seg(img)
    t0 = time.process_time()
    for _ in range(runs):
        seg(img)
    return 1000 * (time.process_time() - t0) / runs, size, count_params(SegNet(**config))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--flip-tta", action="store_true")
    args = p.parse_args()
    lines = ["| candidate | widths | depths | in_res | params | bytes (int8) | ms |", "|---|---|---|---|---|---|---|"]
    print(lines[0])
    for name in args.names or CANDIDATES:
        c = CANDIDATES[name]
        ms, size, params = bench(c, args.flip_tta)
        lines.append(f"| {name} | {list(c['widths'])} | {list(c['depths'])} | {c['in_res']} | {params:,} | {size:,} | {ms:.2f} |")
        print(lines[-1], flush=True)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"bench{'_tta' if args.flip_tta else ''}.md"
    out.write_text(f"Single-thread CPU time per photo, random weights{' with flip TTA' if args.flip_tta else ''}\n\n"
                   + "\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
