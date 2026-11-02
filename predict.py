"""Segment one photo into main object (255) and background (0).

    python predict.py --image photo.jpg --out mask.png [--weights weights/model.pt]

Loads the parameters, resizes the photo to 256x256 with cv2.INTER_AREA, segments it on one
CPU thread and saves the mask. Then times 20 more runs on the same photo and prints exactly
two lines: the average CPU time per run and the parameter file size.
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import time  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(1)

from src.pipeline import Segmenter  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "model.pt")
RUNS = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    photo = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if photo is None:
        raise SystemExit(f"cannot read {args.image}")
    img = cv2.resize(photo, (256, 256), interpolation=cv2.INTER_AREA)
    segment = Segmenter(args.weights)

    mask = segment(img)  # first run, not timed: libraries warm up
    cv2.imwrite(args.out, mask * 255)

    t0 = time.process_time()
    for _ in range(RUNS):
        segment(img)
    t_ms = 1000 * (time.process_time() - t0) / RUNS
    print(f"T CPU ms: {t_ms:.2f}")
    print(f"S disk bytes: {os.path.getsize(args.weights)}")


if __name__ == "__main__":
    main()
