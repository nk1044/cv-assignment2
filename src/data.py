"""Data preparation: resize DUTS to 256x256, cache as .npy, split DUTS-TR into train/val.

    python -m src.data            # build data/cache/ from data/DUTS-TR and data/DUTS-TE

Resizing rule (Section 1b): photos with cv2.INTER_AREA, masks with cv2.INTER_NEAREST,
mask pixel is object if value >= 128. Photos are kept in OpenCV's BGR order.

Cache layout (data/cache/):
    {tr,te}_images.npy  (N,256,256,3) uint8
    {tr,te}_masks.npy   (N,256,256)   uint8 in {0,1}
    {tr,te}_names.txt   one base name per line
    split.npz           index arrays `train` and `val` into the tr arrays
"""
import argparse

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CACHE, DUTS, MASK_THRESHOLD, SIZE, SPLIT_SEED, VAL_FRACTION


def resize_photo(img):
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def resize_mask(mask):
    mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask >= MASK_THRESHOLD).astype(np.uint8)


def read_photo(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask


def build_cache(prefix, img_dir, mask_dir):
    names = sorted(p.stem for p in img_dir.glob("*.jpg"))
    if not names:
        raise SystemExit(f"no .jpg photos in {img_dir}")
    n = len(names)
    images = np.lib.format.open_memmap(CACHE / f"{prefix}_images.npy", "w+", np.uint8, (n, SIZE, SIZE, 3))
    masks = np.lib.format.open_memmap(CACHE / f"{prefix}_masks.npy", "w+", np.uint8, (n, SIZE, SIZE))
    for i, name in enumerate(names):
        images[i] = resize_photo(read_photo(img_dir / f"{name}.jpg"))
        masks[i] = resize_mask(read_mask(mask_dir / f"{name}.png"))
        if (i + 1) % 1000 == 0:
            print(f"{prefix}: {i + 1}/{n}", flush=True)
    images.flush()
    masks.flush()
    (CACHE / f"{prefix}_names.txt").write_text("\n".join(names))
    print(f"{prefix}: {n} photos cached")
    return n


def make_split(n):
    perm = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_val = round(n * VAL_FRACTION)
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def load_set(prefix):
    """Memory-mapped (images, masks, names) for 'tr' or 'te'."""
    images = np.load(CACHE / f"{prefix}_images.npy", mmap_mode="r")
    masks = np.load(CACHE / f"{prefix}_masks.npy", mmap_mode="r")
    names = (CACHE / f"{prefix}_names.txt").read_text().split()
    return images, masks, names


def load_split():
    split = np.load(CACHE / "split.npz")
    return split["train"], split["val"]


class SegDataset(Dataset):
    """uint8 tensors (3,256,256) BGR and (256,256) {0,1}; augmentation happens on the device."""

    def __init__(self, images, masks, indices):
        self.images, self.masks, self.indices = images, masks, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        img = torch.from_numpy(np.array(self.images[j])).permute(2, 0, 1)
        return img, torch.from_numpy(np.array(self.masks[j]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true", help="reuse cached arrays, only redo the split")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    sizes = {}
    for prefix, (img_dir, mask_dir) in DUTS.items():
        names_file = CACHE / f"{prefix}_names.txt"
        if args.skip_existing and names_file.exists():
            sizes[prefix] = len(names_file.read_text().split())
        else:
            sizes[prefix] = build_cache(prefix, img_dir, mask_dir)

    train, val = make_split(sizes["tr"])
    np.savez(CACHE / "split.npz", train=train, val=val)
    print(f"split: {len(train)} train / {len(val)} val (seed {SPLIT_SEED}); test {sizes['te']}")


if __name__ == "__main__":
    main()
