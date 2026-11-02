"""Train SegNet from scratch on the DUTS-TR training split; keep the epoch with the best
validation Q. DUTS-TE is never read here.

    python -m src.train --name base --in-res 128 --widths 32 48 96 128 --depths 2 3 4 --epochs 80

Runs on CUDA (Colab/Kaggle), Apple MPS or CPU (--device, default: best available).
Augmentation (flip, affine zoom/rotate/shift, colour and lighting jitter, blur, noise) runs
on the device. Loss: pixel-wise BCE, optionally up-weighted near the true outline, + soft Dice.
Writes checkpoints/<name>.pt and results/train_<name>.json.
"""
import argparse
import copy
import json
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import CHECKPOINTS, RESULTS
from src.data import SegDataset, load_set, load_split
from src.metrics import SegMeter, boundary, within_window
from src.model import SegNet, count_params


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_float(img_u8, mask_u8, device):
    x = img_u8.to(device, non_blocking=True).float() / 255.0
    m = mask_u8.to(device, non_blocking=True)[:, None].float()
    return x, m


def augment(x, m):
    """x: (B,3,H,W) in [0,1]; m: (B,1,H,W) in {0,1}. Same geometry for photo and mask."""
    b, dev = x.shape[0], x.device

    def u(lo, hi, *shape):
        return torch.empty(b, *shape, device=dev).uniform_(lo, hi)

    zoom = torch.exp(u(math.log(0.75), math.log(1.4)))
    angle = u(-10, 10) * math.pi / 180
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.stack([torch.stack([flip * cos, -sin, u(-0.1, 0.1)], 1),
                         torch.stack([flip * sin, cos, u(-0.1, 0.1)], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    m = (F.grid_sample(m, grid, mode="bilinear", padding_mode="reflection", align_corners=False) > 0.5).float()

    x = x * u(0.7, 1.3, 1, 1, 1)                                   # brightness
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * u(0.7, 1.3, 1, 1, 1) + mean                   # contrast
    grey = x.mean(1, keepdim=True)
    x = (x - grey) * u(0.6, 1.4, 1, 1, 1) + grey                   # saturation
    x = x * u(0.88, 1.12, 3, 1, 1)                                 # colour cast (daylight / tube light)
    x = x.clamp(0, 1) ** u(0.75, 1.35, 1, 1, 1)                    # gamma
    blur = (torch.rand(b, 1, 1, 1, device=dev) < 0.2).float()
    x = blur * F.avg_pool2d(x, 3, 1, 1, count_include_pad=False) + (1 - blur) * x
    x = x + (torch.rand(b, 1, 1, 1, device=dev) < 0.3).float() * torch.randn_like(x) * u(0, 0.04, 1, 1, 1)
    return x.clamp(0, 1), m


def downscale(x, res):
    return x if x.shape[-1] == res else F.interpolate(x, size=(res, res), mode="area")


def loss_fn(logit, m, boundary_weight=0.0, dice_weight=1.0):
    weight = None
    if boundary_weight > 0:
        weight = 1 + boundary_weight * within_window(boundary(m[:, 0] > 0.5)).float()[:, None]
    bce = F.binary_cross_entropy_with_logits(logit, m, weight=weight)
    p = torch.sigmoid(logit)
    dice = 1 - (2 * (p * m).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1)
    return bce + dice_weight * dice.mean()


@torch.no_grad()
def evaluate(model, loader, res, device):
    model.eval()
    meter = SegMeter()
    for img_u8, mask_u8 in loader:
        x, m = to_float(img_u8, mask_u8, device)
        logit = model(downscale(x, res))
        meter.update((logit[:, 0] > 0).cpu(), (m[:, 0] > 0.5).cpu())
    return meter.summary()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--in-res", type=int, default=128)
    p.add_argument("--widths", type=int, nargs="+", default=[16, 24, 40, 64])
    p.add_argument("--depths", type=int, nargs="+", default=[2, 2, 3])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--boundary-weight", type=float, default=2.0)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights, e.g. 0.999 (0 = off)")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=0, help="debug: use only this many training photos")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True

    images, masks, _ = load_set("tr")
    train_idx, val_idx = load_split()
    if args.max_train:
        train_idx = train_idx[:args.max_train]
    train_loader = DataLoader(SegDataset(images, masks, train_idx), args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.workers, pin_memory=cuda,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(SegDataset(images, masks, val_idx), 64, num_workers=args.workers, pin_memory=cuda)

    model = SegNet(args.widths, args.depths, in_res=args.in_res).to(device)
    ema = copy.deepcopy(model).eval() if args.ema else None
    print(f"{args.name}: {count_params(model):,} params, train {len(train_idx)}, val {len(val_idx)}, {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps, warmup = args.epochs * len(train_loader), len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup else
                                              0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    CHECKPOINTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    history, best_q = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for img_u8, mask_u8 in train_loader:
            x, m = augment(*to_float(img_u8, mask_u8, device))
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                logit = model(downscale(x, args.in_res))
            loss = loss_fn(logit.float(), m, args.boundary_weight, args.dice_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                        if pe.dtype.is_floating_point:
                            pe.lerp_(pm, 1 - args.ema)
                        else:
                            pe.copy_(pm)

        scored = ema if ema is not None else model
        val = evaluate(scored, val_loader, args.in_res, device)
        row = {"epoch": epoch, "loss": running / len(train_loader), **val, "sec": time.time() - t0}
        history.append(row)
        mark = ""
        if val["Q"] > best_q:
            best_q, mark = val["Q"], " *"
            torch.save({"config": scored.config, "state": scored.state_dict(), "epoch": epoch, "val": val,
                        "args": vars(args)}, CHECKPOINTS / f"{args.name}.pt")
        print(f"ep {epoch:3d} loss {row['loss']:.4f} val mIoU {val['mIoU']:.4f} BF1 {val['BF1']:.4f} "
              f"Q {val['Q']:.4f} ({row['sec']:.0f}s){mark}", flush=True)
        (RESULTS / f"train_{args.name}.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=1))
    print(f"best val Q {best_q:.4f} -> {CHECKPOINTS / (args.name + '.pt')}")


if __name__ == "__main__":
    main()
