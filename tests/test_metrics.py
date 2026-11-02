"""Check the vectorised metrics against a direct, pixel-by-pixel reading of the definitions.

    python -m tests.test_metrics
"""
import numpy as np
import torch

from src.metrics import SegMeter, boundary_f1, ces


def slow_boundary(m):
    h, w = m.shape
    b = np.zeros_like(m, dtype=bool)
    for y in range(h):
        for x in range(w):
            if m[y, x]:
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < h and 0 <= xx < w) or not m[yy, xx]:
                        b[y, x] = True
    return b


def slow_hit(src, ref):
    """For each True pixel of src: is a True pixel of ref inside its 5x5 window?"""
    h, w = src.shape
    ys, xs = np.nonzero(src)
    return sum(ref[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)].any() for y, x in zip(ys, xs))


def slow_bf1(pred, gt):
    pb, gb = slow_boundary(pred), slow_boundary(gt)
    if not pb.any() and not gb.any():
        return 1.0
    if not pb.any() or not gb.any():
        return 0.0
    p, r = slow_hit(pb, gb) / pb.sum(), slow_hit(gb, pb) / gb.sum()
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def random_blob(rng, size=40):
    yy, xx = np.mgrid[:size, :size]
    cy, cx, r = rng.uniform(0, size, 3) * [1, 1, 0.5]
    return (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2 + rng.normal(0, 20, (size, size))


def test_bf1_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(40):
        pred, gt = random_blob(rng), random_blob(rng)
        fast = boundary_f1(torch.from_numpy(pred)[None], torch.from_numpy(gt)[None]).item()
        assert abs(fast - slow_bf1(pred, gt)) < 1e-9


def test_bf1_edge_cases():
    empty, full = np.zeros((16, 16), bool), np.ones((16, 16), bool)
    square = empty.copy()
    square[4:10, 4:10] = True
    t = lambda a: torch.from_numpy(a)[None]  # noqa: E731
    assert boundary_f1(t(empty), t(empty)).item() == 1.0       # neither has a boundary
    assert boundary_f1(t(empty), t(square)).item() == 0.0      # only one has a boundary
    assert boundary_f1(t(square), t(square)).item() == 1.0
    assert slow_boundary(full).sum() == 4 * 16 - 4              # outside the image is background
    assert boundary_f1(t(full), t(full)).item() == 1.0


def test_miou_uses_summed_confusion():
    a = np.zeros((4, 4), np.uint8)
    b = a.copy()
    b[0, 0] = 1
    meter = SegMeter()
    meter.update(a, a)
    meter.update(b, b)
    meter.update(b, a)   # one false positive
    tp, fp, fn, tn = 1, 1, 0, 48 - 2
    expected = 0.5 * (tp / (tp + fp + fn) + tn / (tn + fn + fp))
    assert abs(meter.miou() - expected) < 1e-12


def test_ces_example_from_handout():
    # Q = 0.70 at 20 ms with a 0.5 MiB file gives 0.635 (Section 2d)
    assert abs(ces(0.70, 20, 0.5 * 4194304 / 4) - 0.70 * (1 - 0.08 - 0.0125)) < 1e-12
    assert ces(0.9, 50.01, 0) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
