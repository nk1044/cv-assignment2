"""Single-thread setup required by Section 2b.

Import this module FIRST, before NumPy, OpenCV or PyTorch are imported anywhere:

    import src.threads  # noqa: F401
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import cv2  # noqa: E402
import torch  # noqa: E402

cv2.setNumThreads(1)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:  # already set (can only be called once per process)
    pass
