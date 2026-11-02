"""Report figures. Run src.evaluate first (it writes per-image scores and captured predictions).

    python -m src.figures captured            # true vs predicted overlays for every captured photo
    python -m src.figures worst --set val     # the 8 lowest-IoU photos of a set
    python -m src.figures worst --set te --names a b c d    # chosen photos (failure analysis)
    python -m src.figures curve --run base    # training curve from results/train_<run>.json

Figures go to report/figures/.
"""
import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.captured import load_captured  # noqa: E402
from src.config import FIGURES, RESULTS, WEIGHTS_FILE  # noqa: E402
from src.metrics import boundary  # noqa: E402

TRUE_RGB = np.array([0, 200, 0])     # green
PRED_RGB = np.array([230, 40, 200])  # magenta
ALPHA = 0.45


def overlay(bgr, mask, rgb):
    """Semi-transparent colour over the object pixels, plus a solid outline."""
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - ALPHA) * out[m] + ALPHA * rgb
    out[boundary(torch.from_numpy(m)[None])[0].numpy()] = rgb
    return out.astype(np.uint8)


def side_by_side(rows, path, title, per_row=1):
    """rows: (bgr photo, true mask, predicted mask, label). per_row photo pairs side by side."""
    n_rows = -(-len(rows) // per_row)
    fig, axes = plt.subplots(n_rows, 2 * per_row, figsize=(3.2 * per_row, 2.05 * n_rows + 0.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (img, gt, pred, label) in enumerate(rows):
        a, b = axes[k // per_row, 2 * (k % per_row)], axes[k // per_row, 2 * (k % per_row) + 1]
        a.imshow(overlay(img, gt, TRUE_RGB))
        a.set_title(f"{label[0]}\ntrue", fontsize=6.5)
        b.imshow(overlay(img, pred, PRED_RGB))
        b.set_title(f"{label[1]}\npredicted", fontsize=6.5)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def scores_by_name(which):
    return {r["name"]: r for r in json.loads((RESULTS / f"per_image_{which}.json").read_text())}


def fig_captured():
    images, masks, names = load_captured()
    scores = scores_by_name("captured")
    rows = []
    for img, gt, name in zip(images, masks, names):
        pred = cv2.imread(str(RESULTS / "captured_pred" / f"{name}.png"), cv2.IMREAD_GRAYSCALE) > 0
        s = scores[name]
        rows.append((img, gt, pred, (name, f"IoU {s['IoU_obj']:.2f}, BF1 {s['BF1']:.2f}")))
    side_by_side(rows, FIGURES / "captured_overlays.png", "Own photos: true (green) vs predicted (magenta)", per_row=2)


def fig_worst(which, k=8, picked=None):
    """The k lowest-IoU photos of a set, or the photos named in `picked`."""
    from src.evaluate import get_set
    from src.pipeline import Segmenter
    images, masks, names = get_set(which)
    index = {n: i for i, n in enumerate(names)}
    scores = scores_by_name(which)
    worst = [scores[n] for n in picked] if picked else sorted(scores.values(), key=lambda r: r["IoU_obj"])[:k]
    seg = Segmenter(str(WEIGHTS_FILE))
    rows = []
    for r in worst:
        i = index[r["name"]]
        img = np.array(images[i])
        rows.append((img, np.asarray(masks[i]), seg(img), (r["name"], f"IoU {r['IoU_obj']:.2f}, BF1 {r['BF1']:.2f}")))
    name = f"picked_{which}.png" if picked else f"worst_{which}.png"
    title = f"Selected {which} failure cases" if picked else f"Lowest-IoU {which} photos"
    side_by_side(rows, FIGURES / name, title + ": true (green) vs predicted (magenta)", per_row=2)


def fig_curve(run):
    hist = json.loads((RESULTS / f"train_{run}.json").read_text())["history"]
    ep = [h["epoch"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    a.plot(ep, [h["loss"] for h in hist])
    a.set(xlabel="epoch", ylabel="training loss")
    for key in ("mIoU", "BF1", "Q"):
        b.plot(ep, [h[key] for h in hist], label=key)
    b.set(xlabel="epoch", ylabel="validation")
    b.legend()
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"curve_{run}.png", dpi=130)
    print(f"wrote {FIGURES / f'curve_{run}.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["captured", "worst", "curve"])
    p.add_argument("--set", default="val", choices=["val", "te", "captured"])
    p.add_argument("--run", default="base")
    p.add_argument("--k", type=int, default=8, help="worst: number of photos")
    p.add_argument("--names", nargs="+", help="worst: show these photos instead of the k lowest-IoU ones")
    args = p.parse_args()
    if args.what == "captured":
        fig_captured()
    elif args.what == "worst":
        fig_worst(args.set, args.k, args.names)
    else:
        fig_curve(args.run)


if __name__ == "__main__":
    main()
