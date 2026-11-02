"""Report figures. Run src.evaluate first (it writes per-image scores and captured predictions).

    python -m src.figures captured            # true vs predicted overlays for every captured photo
    python -m src.figures worst --set val     # the 8 lowest-IoU photos of a set
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

from src.captured import load_captured  # noqa: E402
from src.config import FIGURES, RESULTS, WEIGHTS_FILE  # noqa: E402

TRUE_RGB = np.array([0, 200, 0])     # green
PRED_RGB = np.array([230, 40, 200])  # magenta
ALPHA = 0.45


def overlay(bgr, mask, rgb):
    """Semi-transparent colour over the object pixels, plus a solid outline."""
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    m = mask.astype(bool)
    out[m] = (1 - ALPHA) * out[m] + ALPHA * rgb
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, tuple(int(c) for c in rgb), 1)  # drawing only, not used for scores
    return out


def side_by_side(rows, path, title):
    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 3 * len(rows)), squeeze=False)
    for (img, gt, pred, label), (a, b) in zip(rows, axes):
        a.imshow(overlay(img, gt, TRUE_RGB))
        a.set_title(f"{label}: true mask", fontsize=9)
        b.imshow(overlay(img, pred, PRED_RGB))
        b.set_title("predicted mask", fontsize=9)
        a.axis("off")
        b.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
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
        rows.append((img, gt, pred, f"{name} (IoU {s['IoU_obj']:.2f}, BF1 {s['BF1']:.2f})"))
    side_by_side(rows, FIGURES / "captured_overlays.png", "Captured photos: true (green) vs predicted (magenta)")


def fig_worst(which, k=8):
    from src.evaluate import get_set
    from src.pipeline import Segmenter
    images, masks, names = get_set(which)
    index = {n: i for i, n in enumerate(names)}
    worst = sorted(scores_by_name(which).values(), key=lambda r: r["IoU_obj"])[:k]
    seg = Segmenter(str(WEIGHTS_FILE))
    rows = []
    for r in worst:
        i = index[r["name"]]
        img = np.array(images[i])
        rows.append((img, np.asarray(masks[i]), seg(img), f"{r['name']} (IoU {r['IoU_obj']:.2f}, BF1 {r['BF1']:.2f})"))
    side_by_side(rows, FIGURES / f"worst_{which}.png", f"Lowest-IoU {which} photos")


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
    args = p.parse_args()
    if args.what == "captured":
        fig_captured()
    elif args.what == "worst":
        fig_worst(args.set)
    else:
        fig_curve(args.run)


if __name__ == "__main__":
    main()
