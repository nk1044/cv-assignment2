"""The deployed parameter file (weights/model.pt): plain torch.save, no extra compression.

    {"config":  SegNet kwargs,
     "infer":   {"threshold": logit threshold, "flip_tta": bool}   (learned on validation),
     "state":   BN-fused state dict, float16, or int8 conv weights + one float32 scale per
                output channel}

Every learned value lives in this file (Section 2c). Values are cast back to float32 once at
load time, which is not timed.
"""
import torch

SCALE = ":scale"


def pack_state(state, int8=False):
    packed = {}
    for key, value in state.items():
        value = value.detach().float().cpu()
        if int8 and value.dim() == 4:
            scale = (value.abs().amax(dim=(1, 2, 3)) / 127).clamp(min=1e-12)
            packed[key] = torch.round(value / scale.view(-1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
            packed[key + SCALE] = scale
        else:
            packed[key] = value.half()
    return packed


def unpack_state(packed):
    state = {}
    for key, value in packed.items():
        if key.endswith(SCALE):
            continue
        if value.dtype == torch.int8:
            state[key] = value.float() * packed[key + SCALE].float().view(-1, 1, 1, 1)
        else:
            state[key] = value.float()
    return state


def save_params(path, config, fused_state, threshold=0.0, flip_tta=False, int8=True):
    torch.save({"config": config,
                "infer": {"threshold": float(threshold), "flip_tta": bool(flip_tta)},
                "state": pack_state(fused_state, int8=int8)}, path)


def load_params(path):
    params = torch.load(path, map_location="cpu", weights_only=True)
    return params["config"], params["infer"], unpack_state(params["state"])
