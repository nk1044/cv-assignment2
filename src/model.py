"""Small fully convolutional U-Net for main-object segmentation (Track B), trained from scratch.

    photo (in_res) -> stem 3x3/2 -> [stage: DS block /2 + (depth-1) DS blocks] x len(depths)
                   -> + global context (mean pool -> 1x1 conv, added to the deepest map)
                   -> decoder: bilinear x2, concat encoder skip, DS block   (U-Net style)
                   -> 1x1 head: one object logit at in_res/2 -> bilinear up to 256x256

DS block = depthwise 3x3 + BN + ReLU, pointwise 1x1 + BN, residual add when shapes match.
One logit with binary cross entropy is the two-class pixel-wise cross entropy.

For deployment `fuse_for_inference` folds each BatchNorm into its conv and the 1/255 input
scale into the stem, so the CPU pipeline feeds raw uint8 values to plain convolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, groups=1, relu=True):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )


class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = ConvBNReLU(cin, cin, 3, stride, groups=cin)
        self.pw = ConvBNReLU(cin, cout, 1, relu=False)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.pw(self.dw(x))
        return F.relu(y + x if self.residual else y)


def up_cat(x, skip):
    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([x, skip], 1)


class SegNet(nn.Module):
    """widths: channels of the stem and of each encoder stage (len(depths) + 1 values)."""

    def __init__(self, widths=(16, 24, 40, 64), depths=(2, 2, 3), in_res=128, out_size=256):
        super().__init__()
        assert len(widths) == len(depths) + 1
        self.config = {"widths": list(widths), "depths": list(depths), "in_res": in_res, "out_size": out_size}
        self.out_size = out_size
        self.stem = ConvBNReLU(3, widths[0], 3, stride=2)
        self.encoder = nn.ModuleList(
            nn.Sequential(DSBlock(cin, cout, 2), *[DSBlock(cout, cout) for _ in range(d - 1)])
            for cin, cout, d in zip(widths[:-1], widths[1:], depths))
        self.context = nn.Conv2d(widths[-1], widths[-1], 1)
        self.decoder = nn.ModuleList(
            DSBlock(cdeep + cskip, cskip) for cdeep, cskip in zip(widths[:0:-1], widths[-2::-1]))
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.upsample_output = True  # the CPU pipeline upsamples with cv2.resize instead

    def forward(self, x):
        """x: (B,3,in_res,in_res) BGR in [0,1] for training, raw [0,255] after fusing."""
        skips = [self.stem(x)]
        for stage in self.encoder:
            skips.append(stage(skips[-1]))
        d = skips.pop()
        d = d + self.context(d.mean((2, 3), keepdim=True))
        for block in self.decoder:
            d = block(up_cat(d, skips.pop()))
        logit = self.head(d)
        if not self.upsample_output:
            return logit
        return F.interpolate(logit, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def _fold_bn(conv, bn):
    scale = bn.weight / (bn.running_var + bn.eps).sqrt()
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, groups=conv.groups, bias=True)
    fused.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)
    fused.bias.data = bn.bias.data - bn.running_mean * scale
    return fused


@torch.no_grad()
def fuse_for_inference(model):
    """In place: fold every BN into its conv and the 1/255 input scale into the stem conv."""
    model.eval()
    for m in model.modules():
        if isinstance(m, ConvBNReLU) and isinstance(m[1], nn.BatchNorm2d):
            m[0] = _fold_bn(m[0], m[1])
            m[1] = nn.Identity()
    model.stem[0].weight.data /= 255.0
    model.upsample_output = False
    return model


def build_fused(config):
    """Architecture with BN already replaced by biased convs, ready for fused weights."""
    model = SegNet(**config)
    for m in model.modules():
        if isinstance(m, ConvBNReLU):
            c = m[0]
            m[0] = nn.Conv2d(c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding,
                             groups=c.groups, bias=True)
            m[1] = nn.Identity()
    model.upsample_output = False
    return model.eval()
