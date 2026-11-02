"""The timed CPU pipeline: 256x256 BGR uint8 photo in memory -> 256x256 {0,1} mask in memory.

Everything inside Segmenter.__call__ is timed (downscale, tensor conversion, network,
upsampling, threshold). Loading parameters and tracing in __init__ is not.

Speed choices:
  * BN and 1/255 folded into the convs; the network takes raw uint8 BGR values, so there is
    no colour conversion or normalisation step.
  * TorchScript trace + freeze removes Python overhead between layers.
  * the single-channel logit is upsampled with cv2.resize (bilinear, half-pixel centres like
    F.interpolate(align_corners=False)), which is cheaper than a torch op on one channel.
"""
import warnings

import cv2
import numpy as np
import torch

from src.model import build_fused
from src.weights_io import load_params

warnings.filterwarnings("ignore", module="torch.jit")


class Segmenter:
    def __init__(self, weights_path, jit=True, channels_last=False):
        config, infer, state = load_params(weights_path)
        self.res, self.out_size = config["in_res"], config["out_size"]
        self.threshold = infer["threshold"]
        self.flip_tta = infer["flip_tta"]
        self.memory_format = torch.channels_last if channels_last else torch.contiguous_format

        model = build_fused(config)
        model.load_state_dict(state)
        model = model.to(memory_format=self.memory_format)
        if jit:
            batch = 2 if self.flip_tta else 1
            example = torch.zeros(batch, 3, self.res, self.res).contiguous(memory_format=self.memory_format)
            with torch.inference_mode():
                model = torch.jit.freeze(torch.jit.trace(model, example))
                for _ in range(3):  # let the profiling executor settle now, not on the first photos
                    model(example)
        self.model = model

    @torch.inference_mode()
    def logits(self, img):
        """img: (256,256,3) uint8 BGR -> (256,256) float32 object logits."""
        if img.shape[0] != self.res:
            img = cv2.resize(img, (self.res, self.res), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float()
        if self.flip_tta:
            x = torch.cat([x, x.flip(3)])
        x = x.contiguous(memory_format=self.memory_format)
        out = self.model(x)
        logit = out[0, 0]
        if self.flip_tta:
            logit = 0.5 * (logit + out[1, 0].flip(1))
        return cv2.resize(logit.numpy(), (self.out_size, self.out_size), interpolation=cv2.INTER_LINEAR)

    def __call__(self, img):
        """img: (256,256,3) uint8 BGR -> (256,256) uint8 mask of 0s and 1s."""
        return (self.logits(img) > self.threshold).view(np.uint8)
