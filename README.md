# CSL7360 Assignment 2: Cost-Efficient Segmentation (Track B)

A small fully convolutional U-Net, trained from scratch on DUTS, that labels every pixel of a 256×256 photo as **main object** or **background**. It runs on **one CPU thread**, well under the 50 ms budget.

```
Q   = 0.70·mIoU + 0.30·BF1
CES = Q · [1 − 0.20·T_CPU/50 ms − 0.10·min(1, S_disk/4 MiB)]      (0 if T_CPU > 50 ms)
```

The full task breakdown and experiment plan are in [plan.md](plan.md), and the method, results and discussion are in `report/`.

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # torch, numpy, opencv-python, matplotlib only
```
Unpack DUTS (http://saliencydetection.net/duts/) into:
```
data/DUTS-TR/DUTS-TR-Image/*.jpg   data/DUTS-TR/DUTS-TR-Mask/*.png
data/DUTS-TE/DUTS-TE-Image/*.jpg   data/DUTS-TE/DUTS-TE-Mask/*.png
```

## Predict one photo
```bash
python predict.py --image photo.jpg --out mask.png
```
This saves a 0/255 PNG mask, then times 20 more runs and prints exactly:
```
T CPU ms: 12.34
S disk bytes: 425767
```

## Workflow
Run every command from the project root.

| Step | Command | Notes |
|---|---|---|
| 1. Cache data and split | `python -m src.data` | Resizes to 256×256 (photos INTER_AREA, masks INTER_NEAREST, ≥128 is object) and splits DUTS-TR 90/10 into train/val with seed 0. |
| 2. Check metrics | `python -m tests.test_metrics` | Compares against a brute-force reading of the definitions. |
| 3. Compare architecture speed | `python -m src.bench [names] [--flip-tta]` | Uses random weights and one thread. Edit `CANDIDATES` in [src/bench.py](src/bench.py) to add more. |
| 4. Train (GPU/MPS/CPU) | `python -m src.train --name base --in-res 128 --widths 32 48 96 128 --depths 2 3 4 --epochs 80` | Keeps the best epoch by **val** Q in `checkpoints/base.pt`. Add `--ema 0.999` and tune `--boundary-weight`. |
| 5. Export parameters | `python -m src.export --ckpt checkpoints/base.pt --tune [--flip-tta] [--fp16]` | Folds BN, tunes the threshold on val, and writes int8 weights to `weights/model.pt`. |
| 6. Time on val (model selection) | `python -m src.time_cpu --set val` | Compares candidates without touching DUTS-TE. |
| 7. Prepare captured photos | `python -m src.captured prepare`, then `python -m src.annotate`, then `python -m src.captured check` | See below. |
| 8. Evaluate val and captured | `python -m src.evaluate` | |
| 9. **Final run only:** official T_CPU | `python -m src.time_cpu` | All 5,019 DUTS-TE photos, written to `results/timing.json`. |
| 10. **Final run only:** DUTS-TE metrics and CES | `python -m src.evaluate --sets val te captured` | Written to `results/metrics.json`. |
| 11. Report figures | `python -m src.figures captured`, `python -m src.figures worst --set val`, `python -m src.figures curve --run base` | Written to `report/figures/`. |

DUTS-TE is read only by steps 9–10 and by `src.figures worst --set te`. Never use it to pick a model, threshold or epoch.

### Training on Colab or Kaggle
Upload the repo and `data/cache/` (built by step 1, about 3 GB). Run step 4 there with `--device cuda --workers 4`, then download `checkpoints/<name>.pt` and run steps 5 onward **on the laptop**, since timing must come from your own CPU. Use the same torch version in both places.

## Captured photos
1. Put 5 to 10 of your own phone photos in `captured/original/` as JPEG, using base names such as `bottle.jpg`. Convert iPhone HEIC files with `sips -s format jpeg x.heic --out x.jpg`.
2. Run `python -m src.captured prepare` to write `captured/images_256/<name>.png`.
3. Run `python -m src.annotate` and draw polygons on the resized photo. Left-click adds a point, right-click closes the polygon, `x` toggles cut mode for holes, and `s` saves to `captured/masks_256/<name>.png` with object 255 and background 0.
4. Run `python -m src.captured check` to check names, sizes, mask values and the photo count.

## Layout
```
predict.py               required CLI (single thread, prints T CPU ms / S disk bytes)
weights/model.pt         deployed parameters: config, threshold, BN-fused int8 weights
src/config.py            paths and constants          src/threads.py    single-thread setup
src/data.py              resize rule, cache, split, Dataset
src/model.py             SegNet (tiny U-Net), BN folding
src/weights_io.py        parameter file format (int8 / float16, plain torch.save)
src/pipeline.py          the timed CPU pipeline (Segmenter)
src/train.py             training with augmentation, BCE + soft Dice
src/export.py            checkpoint -> weights/model.pt, threshold tuning on val
src/metrics.py           mIoU, Boundary F1, Q, CES (from scratch)
src/time_cpu.py          official T_CPU          src/evaluate.py   results table
src/bench.py             architecture speed      src/figures.py    report figures
src/captured.py          captured photo prep and checks
src/annotate.py          polygon mask annotator
tests/test_metrics.py    metric correctness against brute force
captured/{original,images_256,masks_256}/
results/                 metrics.json, timing.json, per-image scores, training logs
report/                  report and figures
```

## How the rules are followed
* **Libraries:** only PyTorch, NumPy, OpenCV (I/O, resize, colour conversion, drawing in the annotator) and Matplotlib. There are no pretrained weights, no torchvision models, and no library segmentation, clustering, morphology, connected-component or metric code. The boundary map uses shifted-tensor comparisons, and the 5×5 window test uses `F.max_pool2d`.
* **Single thread:** `OMP/MKL/OPENBLAS/VECLIB_*_NUM_THREADS=1` are set before NumPy, OpenCV and PyTorch are imported (in `predict.py` and `src/threads.py`), followed by `cv2.setNumThreads(1)` and `torch.set_num_threads(1)`. There is no multiprocessing in the timed path.
* **Timing:** the timer starts with a 256×256 photo in memory and stops at the 256×256 {0,1} mask. Downscaling, tensor conversion, the network, upsampling and thresholding are all timed. A warm-up run comes first.
* **Parameters:** every learned value, including the decision threshold, is in `weights/model.pt`, saved with plain `torch.save` and no compression.
* **Data:** design choices use the validation split of DUTS-TR only. The captured photos are for testing only.
