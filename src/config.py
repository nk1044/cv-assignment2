"""Project paths and constants shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
CAPTURED = ROOT / "captured"
WEIGHTS = ROOT / "weights"
WEIGHTS_FILE = WEIGHTS / "model.pt"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"

SIZE = 256              # every photo and mask is resized to SIZE x SIZE (Section 1b)
MASK_THRESHOLD = 128    # a mask pixel is object if its value is >= 128
VAL_FRACTION = 0.10     # share of DUTS-TR held out for validation
SPLIT_SEED = 0

# Raw DUTS layout expected under data/ (as unpacked from the official zips)
DUTS = {
    "tr": (DATA / "DUTS-TR" / "DUTS-TR-Image", DATA / "DUTS-TR" / "DUTS-TR-Mask"),
    "te": (DATA / "DUTS-TE" / "DUTS-TE-Image", DATA / "DUTS-TE" / "DUTS-TE-Mask"),
}
