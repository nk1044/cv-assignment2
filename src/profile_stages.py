"""Where the CPU time of the timed pipeline goes, and the effect of the memory layout.

    python -m src.profile_stages [--weights weights/model.pt] [--n 300]

Times each stage of Segmenter.__call__ (downscale, tensor conversion, network, upsample and
threshold) separately on validation photos, and compares contiguous vs channels_last memory
layout for the whole pipeline. One CPU thread. Writes results/profile_stages.json.
"""
import src.threads  # noqa: F401

import argparse
import json
import platform
import time

import cv2
import numpy as np
import torch

from src.config import RESULTS, WEIGHTS_FILE
from src.data import load_set, load_split
from src.pipeline import Segmenter
from src.time_cpu import cpu_name, time_photos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(WEIGHTS_FILE))
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()
    images, _, _ = load_set("tr")
    val = load_split()[1][:args.n]

    layouts = {name: time_photos(Segmenter(args.weights, channels_last=cl), images, val)
               for name, cl in (("contiguous", False), ("channels_last", True))}

    seg = Segmenter(args.weights)
    seg(np.array(images[val[0]]))
    stages = dict.fromkeys(("downscale", "to_tensor", "network", "upsample_threshold"), 0.0)
    with torch.inference_mode():
        for i in val:
            img = np.array(images[i])
            t0 = time.process_time()
            small = cv2.resize(img, (seg.res, seg.res), interpolation=cv2.INTER_AREA)
            t1 = time.process_time()
            x = torch.from_numpy(small).permute(2, 0, 1)[None].float().contiguous()
            t2 = time.process_time()
            logit = seg.model(x)[0, 0].numpy()
            t3 = time.process_time()
            (cv2.resize(logit, (seg.out_size, seg.out_size), interpolation=cv2.INTER_LINEAR) > seg.threshold).view(np.uint8)
            t4 = time.process_time()
            for key, dt in zip(stages, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
                stages[key] += dt
    stages = {k: 1000 * v / len(val) for k, v in stages.items()}

    out = {"weights": args.weights, "n": len(val), "cpu": cpu_name(), "os": platform.platform(),
           "torch": torch.__version__, "threads": torch.get_num_threads(), "layout_ms": layouts, "stage_ms": stages}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "profile_stages.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
