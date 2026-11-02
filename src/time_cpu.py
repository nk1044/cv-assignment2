"""Official T_CPU (Section 2b): segment all 5,019 DUTS-TE photos one at a time on one thread.

Each photo is a 256x256 array in memory before its timer starts; the timer stops when the
256x256 {0,1} mask is in memory. T_CPU = total process time / number of photos.

    python -m src.time_cpu [--weights weights/model.pt] [--limit N] [--set val]

--set val times validation photos instead, for comparing candidate models without
touching DUTS-TE. Only the full DUTS-TE run writes results/timing.json.
"""
import src.threads  # noqa: F401  (must come before NumPy/OpenCV/PyTorch)

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch

from src.config import RESULTS, WEIGHTS_FILE
from src.data import load_set, load_split
from src.pipeline import Segmenter


def cpu_name():
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if system == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if system == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def time_photos(seg, images, indices):
    seg(np.array(images[indices[0]]))  # warm-up run, not timed
    total = 0.0
    for i in indices:
        img = np.array(images[i])      # reading from disk is not timed
        t0 = time.process_time()
        mask = seg(img)
        total += time.process_time() - t0
        assert mask.shape == (256, 256)
    return 1000 * total / len(indices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(WEIGHTS_FILE))
    p.add_argument("--set", choices=["te", "val"], default="te")
    p.add_argument("--limit", type=int, default=0, help="time only the first N photos")
    args = p.parse_args()

    if args.set == "te":
        images, _, _ = load_set("te")
        indices = np.arange(len(images))
    else:
        images, _, _ = load_set("tr")
        indices = load_split()[1]
    if args.limit:
        indices = indices[:args.limit]

    t_ms = time_photos(Segmenter(args.weights), images, indices)
    info = {"T_CPU_ms": t_ms, "S_disk_bytes": os.path.getsize(args.weights), "set": args.set, "n": len(indices),
            "cpu": cpu_name(), "os": platform.platform(), "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__}
    print(json.dumps(info, indent=1))
    if args.set == "te" and not args.limit:
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / "timing.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
