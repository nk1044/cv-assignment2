"""Your own phone photos (Section 1c).

    python -m src.captured prepare   # captured/original/* -> captured/images_256/<name>.png
    python -m src.captured check     # verify names, sizes and mask values

Photos are resized with the same rule as DUTS (cv2.INTER_AREA to 256x256). Draw each mask
on the resized photo (python -m src.annotate, or GIMP/LabelMe) and save it as
captured/masks_256/<name>.png, object 255 and background 0. These photos are test-only.
"""
import sys

import cv2
import numpy as np

from src.config import CAPTURED, MASK_THRESHOLD, SIZE
from src.data import resize_photo

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ORIGINAL, IMAGES, MASKS = CAPTURED / "original", CAPTURED / "images_256", CAPTURED / "masks_256"


def originals():
    return sorted(p for p in ORIGINAL.iterdir() if p.suffix.lower() in PHOTO_EXT)


def prepare():
    IMAGES.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    heic = [p.name for p in ORIGINAL.iterdir() if p.suffix.lower() == ".heic"]
    if heic:
        print(f"OpenCV cannot read HEIC, convert first (sips -s format jpeg x.heic --out x.jpg): {heic}")
    for path in originals():
        photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if photo is None:
            print(f"cannot read {path.name}, skipped")
            continue
        cv2.imwrite(str(IMAGES / f"{path.stem}.png"), resize_photo(photo))
        print(f"{path.name}: {photo.shape[1]}x{photo.shape[0]} -> images_256/{path.stem}.png")


def check():
    ok = True
    names = [p.stem for p in originals()]
    if not 5 <= len(names) <= 10:
        print(f"! {len(names)} photos, the assignment asks for 5 to 10")
        ok = False
    for name in names:
        img = cv2.imread(str(IMAGES / f"{name}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(MASKS / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[:2] != (SIZE, SIZE):
            print(f"! {name}: images_256/{name}.png missing or not {SIZE}x{SIZE}")
            ok = False
        if mask is None:
            print(f"! {name}: masks_256/{name}.png missing")
            ok = False
            continue
        if mask.ndim != 2 or mask.shape != (SIZE, SIZE):
            print(f"! {name}: mask must be single-channel {SIZE}x{SIZE}, got {mask.shape}")
            ok = False
        grey = int(((mask > 0) & (mask < 255)).sum())
        share = float((mask >= MASK_THRESHOLD).mean())
        print(f"  {name}: object {100 * share:.1f}% of pixels" + (f", {grey} grey pixels (thresholded at 128)" if grey else ""))
    print("all good" if ok else "fix the issues above")


def load_captured():
    """(images, masks in {0,1}, names) of every captured photo that has a mask."""
    names = sorted(p.stem for p in MASKS.glob("*.png") if (IMAGES / p.name).exists())
    images = [cv2.imread(str(IMAGES / f"{n}.png"), cv2.IMREAD_COLOR) for n in names]
    masks = [(cv2.imread(str(MASKS / f"{n}.png"), cv2.IMREAD_GRAYSCALE) >= MASK_THRESHOLD).astype(np.uint8)
             for n in names]
    return images, masks, names


if __name__ == "__main__":
    {"prepare": prepare, "check": check}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
