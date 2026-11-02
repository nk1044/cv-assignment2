"""Evaluation metrics, written from scratch (Section 3, "Evaluation Metrics").

mIoU: one 2x2 confusion matrix summed over all images of a set; IoU of background and of
      object, averaged.
BF1:  per image boundary F1 with a 5x5 tolerance window, averaged over the set. The window
      test is vectorised: a 5x5 max pool over a boundary map is 1 exactly where some boundary
      pixel lies inside that pixel's 5x5 window.
"""
import numpy as np
import torch
import torch.nn.functional as F

FOUR_MIB = 4 * 1024 * 1024


def boundary(masks):
    """Object pixels with a background pixel directly above, below, left or right.

    masks: (N,H,W) bool tensor. Pixels outside the image count as background.
    """
    m = F.pad(masks.float()[:, None], (1, 1, 1, 1), value=0.0)
    centre = m[:, :, 1:-1, 1:-1]
    neighbours_all_object = m[:, :, :-2, 1:-1] * m[:, :, 2:, 1:-1] * m[:, :, 1:-1, :-2] * m[:, :, 1:-1, 2:]
    return ((centre > 0) & (neighbours_all_object == 0))[:, 0]


def within_window(b, size=5):
    """True where any True pixel of b lies in the size x size window around it."""
    return F.max_pool2d(b.float()[:, None], size, stride=1, padding=size // 2)[:, 0] > 0


def boundary_f1(pred, gt):
    """Per image BF1. pred, gt: (N,H,W) bool tensors. Returns (N,) float64."""
    pb, gb = boundary(pred), boundary(gt)
    n_pred = pb.flatten(1).sum(1).double()
    n_gt = gb.flatten(1).sum(1).double()
    correct = (pb & within_window(gb)).flatten(1).sum(1).double()
    found = (gb & within_window(pb)).flatten(1).sum(1).double()

    precision = correct / n_pred.clamp(min=1)
    recall = found / n_gt.clamp(min=1)
    pr = precision + recall
    f1 = torch.where(pr > 0, 2 * precision * recall / pr.clamp(min=1e-12), torch.zeros_like(pr))
    f1 = torch.where((n_pred == 0) & (n_gt == 0), torch.ones_like(f1), f1)   # neither has a boundary
    f1 = torch.where((n_pred == 0) ^ (n_gt == 0), torch.zeros_like(f1), f1)  # only one has a boundary
    return f1


class SegMeter:
    """Accumulates the confusion matrix and the per image BF1 values of a set."""

    def __init__(self):
        self.conf = np.zeros((2, 2), dtype=np.int64)  # conf[true, pred]
        self.bf1 = []

    def update(self, pred, gt):
        """pred, gt: (H,W) or (N,H,W) arrays/tensors with values {0,1} or bool."""
        pred, gt = torch.as_tensor(pred).bool(), torch.as_tensor(gt).bool()
        if pred.dim() == 2:
            pred, gt = pred[None], gt[None]
        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        tn = gt.numel() - tp - fp - fn
        self.conf += np.array([[tn, fp], [fn, tp]], dtype=np.int64)
        self.bf1.extend(boundary_f1(pred, gt).tolist())

    def miou(self):
        (tn, fp), (fn, tp) = self.conf
        iou_obj = tp / max(tp + fp + fn, 1)
        iou_bg = tn / max(tn + fn + fp, 1)
        return 0.5 * (iou_obj + iou_bg)

    def summary(self):
        miou = float(self.miou())
        bf1 = float(np.mean(self.bf1)) if self.bf1 else 0.0
        return {"mIoU": miou, "BF1": bf1, "Q": quality(miou, bf1), "n": len(self.bf1)}


def quality(miou, bf1):
    return 0.70 * miou + 0.30 * bf1


def ces(q, t_cpu_ms, s_disk_bytes):
    if t_cpu_ms > 50:
        return 0.0
    return q * (1 - 0.20 * t_cpu_ms / 50 - 0.10 * min(1.0, s_disk_bytes / FOUR_MIB))
