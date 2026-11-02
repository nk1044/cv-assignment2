# CSL7360 Assignment 2: Cost-Efficient Segmentation, Plan

> **Update (2026-11-02):** the sample code has been removed and replaced by a fresh codebase (see [README.md](README.md)). Sections 0–3 below describe the sample as it was and are kept for reference. File names have changed: `prepare_data` is now `src.data`, `TinyUNet` is now `SegNet`, `bench_arch` is now `src.bench`, `make_figures` is now `src.figures`, and `prepare_captured` is now `src.captured`. The Mac fixes from section 2 are already done: device-agnostic training, macOS CPU name, a HEIC warning, and the threshold and flip-TTA settings stored in the parameter file.

Goal: a pipeline that labels every pixel of a 256×256 photo as **object** or **background**, running on **one CPU thread** in **at most 50 ms on average** per photo, with the **CES score on DUTS-TE as high as possible**.

```
Q   = 0.70·mIoU + 0.30·BF1
CES = Q · [1 − 0.20·(T_CPU / 50 ms) − 0.10·min(1, S_disk / 4 MiB)]   (0 if T_CPU > 50 ms)
```

---

## 0. TL;DR

* **Track:** stay on **Track B (tiny FCN / U-Net)**. The sample code already uses it, and Track A (clustering plus a hand-written graph cut) usually lands around Q ≈ 0.5 on DUTS.
* **Baseline (sample code, measured on someone else's AMD Ryzen 5 5600H):** DUTS-TE mIoU 0.725, BF1 0.507, Q 0.660, T_CPU 4.79 ms, S_disk 337 KB, **CES 0.642**.
* **Where the points are:** at this operating point the time and size penalties together cost only about 2.7% of Q. **Q matters most, and BF1 matters most within Q** (0.51 on TE). Speed matters much less.
* **What has to be done whatever the model:** (1) make the code run on the Apple M3 Mac and re-time everything there; (2) take and annotate **your own 5 to 10 photos** (the sample has only 4, which breaks the rule, and they aren't yours); (3) write the report, since the README's `report/` folder doesn't exist in the sample.
* **What could improve the score:** threshold tuning, flip TTA, longer training with EMA, a wider receptive field and context module, a full-resolution refinement head for BF1, and a loss tuned for boundaries. All of these get chosen on **validation CES**, and DUTS-TE gets evaluated only once at the end.

---

## 1. Requirements checklist (from the PDF)

### Data
- [ ] Download DUTS-TR (10,553) and DUTS-TE (5,019) from http://saliencydetection.net/duts/.
- [ ] Split DUTS-TR into train and val. **All design choices use val only. Never use DUTS-TE for training, tuning or early stopping.**
- [ ] Resize photos to 256×256 with `cv2.INTER_AREA` and masks with `cv2.INTER_NEAREST`. A mask pixel is object if its value is ≥ 128. Non-square photos get squashed, which the assignment intends.
- [ ] Take 5 to 10 phone photos, each with one clear everyday object, covering **at least 3 of these 5** conditions: indoor + outdoor, daylight + tube light, near + far, cluttered background, partial occlusion. Use them **only for testing**.
- [ ] Draw each mask on the **resized** 256×256 photo and save it as a PNG with object 255 and background 0.

### Rules
- [ ] Allowed: stdlib, PyTorch (tensors, autograd, optimizers, `torch.nn` and `F.*` including `F.max_pool2d`), NumPy, OpenCV for I/O, resize, colour conversion and `cv2.filter2D`, Matplotlib for plots. TorchScript and `torch.compile` are allowed.
- [ ] Not allowed: pretrained weights, torchvision.models, sklearn, skimage, scipy.ndimage, networkx, PyMaxflow, `cv2.kmeans`, `cv2.grabCut`, `cv2.dilate` or `cv2.erode`, library morphology, metric or connected-component code, ONNX Runtime or any other inference engine.
- [ ] The timed pipeline goes from a 256×256 photo in memory to a 256×256 {0,1} mask in memory. Every step in between counts.
- [ ] Single thread: set `OMP/MKL/OPENBLAS/VECLIB_*_NUM_THREADS=1` **before** importing numpy, cv2 or torch, then call `cv2.setNumThreads(1)` and `torch.set_num_threads(1)`. No multiprocessing.
- [ ] Do one warm-up run, then time all 5,019 TE photos one at a time with `time.process_time()`. T_CPU is the total divided by 5,019 and must be ≤ 50 ms.
- [ ] S_disk is `os.path.getsize(param_file)`. **All learned values must live in that file.** float16 or int8 storage is fine, extra compression (gzip, `savez_compressed`) is not.

### Deliverables
- [ ] `captured/original/`, `captured/images_256/`, `captured/masks_256/`, using the same base names in all three.
- [ ] Clean, documented code for training, evaluation and timing, plus the parameter file.
- [ ] `predict.py --image photo.jpg --out mask.png` loads the parameters, resizes, segments and saves a 0/255 PNG, then times 20 more runs and prints **exactly two lines**:
  ```
  T CPU ms: 12.34
  S disk bytes: 425767
  ```
- [ ] Report covering: the method, how it stays fast, train and val split sizes, **CPU model**, and one table with mIoU, BF1 and Q for **val, DUTS-TE and captured**, plus T_CPU, S_disk and CES. It also needs side-by-side semi-transparent overlays of the true and predicted masks for **every** captured photo, and **at least two failure cases** with likely causes.

---

## 2. Audit of the sample (baseline) code

### What it does
| Part | File | Notes |
|---|---|---|
| Data cache + split | `src/prepare_data.py` | Writes uint8 `.npy` memmaps in BGR order. Split is 9,500 train / 1,053 val with seed 2024. Follows the resize rule correctly. |
| Model | `src/model.py` | TinyUNet: strided stem, then 3 depthwise-separable encoder stages down to 1/16, a global-average context vector, 3 decoder stages with concat skips, and a head at 1/2 of the input resolution. There is an optional `refine` full-res stage that was never used in the final model. |
| Training | `src/train.py` | AdamW with cosine LR and warmup, BCE (up-weighted near boundaries) + soft Dice, GPU augmentation, best epoch chosen by val Q. **CUDA only**, because autocast and GradScaler are hard-coded to `"cuda"`. |
| Deployment | `src/model.py: fuse_for_inference`, `src/export.py`, `src/weights_io.py` | Folds BN and the 1/255 scale into the convs, stores int8 per-channel weights, and dequantizes at load time so inference runs in fp32. |
| Timed pipeline | `src/pipeline.py` | Takes uint8 BGR, runs INTER_AREA down to 128, applies a frozen TorchScript model in channels_last, upsamples the logit with `cv2.resize`, and thresholds at 0. |
| Metrics | `src/metrics.py`, `tests/test_metrics.py` | Implements a global confusion matrix for mIoU and BF1 with a 5×5 window through `F.max_pool2d`. The test checks them against a brute-force reference. |
| Timing, eval, figures | `src/time_cpu.py`, `src/evaluate.py`, `src/make_figures.py` | Official T_CPU, the results table, overlays. |
| Captured tools | `src/prepare_captured.py`, `src/annotate.py` | Resize script and an OpenCV polygon annotator. |

Final config: `in_res=128, widths=(48,64,128,192), depths=(2,3,4), boundary_weight=2`, 60 epochs.

### Results
| Set | mIoU | BF1 | Q |
|---|---|---|---|
| Val (1,053) | 0.877 | 0.687 | 0.820 |
| DUTS-TE (5,019) | 0.725 | 0.507 | 0.660 |
| Captured (4) | 0.816 | 0.416 | 0.696 |

T_CPU is 4.79 ms on a Ryzen 5 5600H, S_disk is 336,642 B, and CES is 0.642.

Sweep (`results/compare_sweep.txt`): widening from 16 to 48 channels helped val Q by about 0.055. Raising the resolution from 128 to 192 or 256 did **not** help at the same width, and boundary-weight 2 helped BF1 by about 0.015.

### Problems and gaps to fix
1. **The captured photos aren't yours, and there are only 4.** The assignment needs 5 to 10 photos covering at least 3 conditions. Take new ones (section 7).
2. **No report.** The README links `report/report.md`, `report/make_report_assets.py` and `report/build_report.py`, but none of them are in the folder.
3. **Different machine.** Your laptop is an **Apple M3 (arm64, macOS)**. All timing has to be redone on it. PyTorch on arm64 doesn't use the x86 oneDNN paths, so choices like `channels_last` and the JIT freeze mode need re-benchmarking.
4. **`cpu_name()` in `time_cpu.py`** returns just `"arm"` on macOS. Use `sysctl -n machdep.cpu.brand_string`, which gives "Apple M3".
5. **Training is CUDA only.** Either train on Colab or Kaggle (T4/P100, simplest) or make `train.py` device-agnostic for MPS: no `GradScaler`, autocast only on CUDA, `pin_memory` only on CUDA.
6. **Environment:** torch is not installed on this Mac yet, and the README's setup is for Windows.
7. **`.heic` in `prepare_captured.py`:** `cv2.imread` can't read HEIC. Export iPhone photos as JPEG, or convert them with `sips -s format jpeg x.heic --out x.jpg`.
8. **Large val→TE gap** (Q 0.82 → 0.66). DUTS-TE has more cluttered scenes, more multi-object scenes and larger objects, so val is optimistic. Gains on val may shrink on TE, so the plan should favour changes that improve *generalization* (context, augmentation, regularization) over ones that only overfit val.
9. **Some TE images score IoU = 0** (complete misses). Many low-IoU cases are probably empty predictions or the wrong object being picked. Look at the per-image files to confirm.

---

## 3. Score analysis: where to spend effort

At Q ≈ 0.66 on TE:

| Change | CES effect |
|---|---|
| +1 ms of T_CPU | −0.2/50 · Q ≈ **−0.0026** |
| +1 MiB of S_disk (below 4 MiB) | −0.025 · Q ≈ **−0.0165** |
| +0.01 TE mIoU | +0.007 · 0.97 ≈ **+0.0068** |
| +0.01 TE BF1 | +0.003 · 0.97 ≈ **+0.0029** |

What follows from these numbers:
* **Spending time is cheap.** Doubling the network from 5 ms to 10 ms costs about 0.013 CES, so it only needs about +0.02 TE Q to pay off. Flip TTA (×2 time) pays off if it adds ≥ 0.013 Q.
* **Spending size is fairly expensive.** Each MiB is worth about 6 ms. Keep int8 storage. Parameters above about 1 to 1.5 MB need a clear Q gain to justify them.
* **Break-even rule** for comparing two candidates a and b: pick b if `CES_val(b) > CES_val(a)`, where `CES_val` uses **val Q and T measured on the M3**. All of this is automated in `src/compare_runs.py`, which needs re-timing on the M3.
* **The ceiling is Q.** At about 0.70 Q, even a perfect 0 ms, 0 byte pipeline only reaches 0.70. Realistic target: **TE Q 0.69 to 0.72 → CES ≈ 0.66 to 0.70**.

---

## 4. Phase-by-phase plan

### Phase 0: Environment (½ day)
1. Set up a Python env on the Mac:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install torch numpy opencv-python matplotlib
   ```
   Note the torch version in the report. Use the same version for training (Colab) and timing (Mac), because TorchScript and weights must load across both.
2. Pick where to train. **Recommended: Kaggle or Colab GPU** (copy the repo plus the `data/cache/*.npy` files, about 2.9 GB for TR and TE images). The backup is to make `train.py` device-agnostic and use MPS on the M3, which is slower but works for small models.
3. Patch `train.py`: `--device {cuda,mps,cpu}`, turn off autocast and GradScaler when not on CUDA, and set `pin_memory` only for CUDA.
4. Patch `time_cpu.py: cpu_name()` for macOS.
5. Initialise git and add a `.gitignore` covering `data/`, `checkpoints/` and `.venv/`.

### Phase 1: Data (½ day)
1. Download DUTS-TR and DUTS-TE and place them at `data/DUTS-TR/{DUTS-TR-Image,DUTS-TR-Mask}` and `data/DUTS-TE/{DUTS-TE-Image,DUTS-TE-Mask}`.
2. Run `python -m src.prepare_data`. Confirm counts of 10,553 and 5,019, that every image has a mask, and that the masks are binary after `>=128`.
3. Keep the **9,500 / 1,053** split with seed 2024 and record it in the report. Optionally look at whether val covers DUTS-TR's image sources, for example ImageNet `n*` and `ILSVRC*` names.
4. Spot-check 20 cached pairs visually with a quick matplotlib grid.

### Phase 2: Metric correctness (1 hour)
1. Run `python -m tests.test_metrics`, which compares against the brute-force reference.
2. Add edge-case tests: an all-zero prediction against an all-zero GT should give BF1 = 1, a prediction with no boundary against a GT with one should give 0, a mask touching the image border should count border pixels as boundary, and mIoU should use the **summed** confusion matrix rather than a per-image average.
3. These are the metrics reported, so they must match the spec exactly.

### Phase 3: Reproduce the baseline on the M3 (½ day)
1. Run `python predict.py --image image.png --out mask.png` with the shipped `weights/model.pt` and check the two-line output.
2. Run `python -m src.time_cpu` to get the official T_CPU on all 5,019 photos on the **M3**, and record it.
3. Run `python -m src.evaluate` and check it reproduces val and TE Q (it should match the numbers above to about 1e-4).
4. Run `python -m src.bench_arch` on the M3 to re-rank architectures by **M3** time, and also check:
   * `channels_last` on versus off (arm64 can behave differently from x86)
   * `jit="freeze"` versus `"optimize"` versus eager versus `torch.compile`
   * whether the INTER_AREA downscale and `cv2.resize` upsample are a meaningful fraction of the total, using `time.process_time` around each step
5. **Baseline row for the log:** the M3 T_CPU, val Q and val CES. Every later experiment gets compared against it.

### Phase 4: Experiments, all judged on validation CES (3 to 5 days, mostly GPU time)
Keep an experiment log in `results/experiments.md` with one row per run: config, params, bytes, M3 ms, val mIoU, BF1, Q, val CES. Change **one thing at a time** where possible. Run these roughly in order of expected gain per hour.

| # | Experiment | Why | Cost |
|---|---|---|---|
| E1 | **Threshold tuning** on val logits (−1.0 … +1.0 in steps of 0.1) | mIoU and BF1 often peak away from 0. Free at inference. Store the threshold **in the param file** because it's a learned value. | Minutes |
| E2 | **Horizontal-flip TTA** (average the two logits) | Usually +0.5–1.5 Q points. Costs about 2× the network time (about +5 ms ≈ −0.013 CES). Keep it only if val CES rises. Run both flips as **one batch of 2**, which is cheaper than two calls. | Minutes |
| E3 | **Longer training + EMA weights** (100–150 epochs, EMA decay 0.999) | The curve in `log_final.txt` was still rising at epoch 59. EMA reliably helps small nets. | 1–2 GPU-h |
| E4 | **Context / receptive field**: add a 4th downsampling stage (1/32) or a light PPM/ASPP-lite (dilated depthwise convs at rates 2, 4, 8) at the deepest level | The TE gap and IoU-0 misses point to errors about which object is salient, so global context should help. | ~1 GPU-h per variant |
| E5 | **Boundary quality for BF1**: (a) `refine=16` full-input-res stage (already implemented), (b) head at full `in_res` instead of 1/2, (c) `in_res` 160 with refine | BF1 is the weakest term (0.51 TE). Upsampling 64→256 blurs outlines. | ~1 GPU-h each |
| E6 | **Loss**: boundary weight in {2, 4}, add a Lovász-hinge or soft-IoU term (written by hand, which is allowed), and try Dice weight in {0.5, 1} | Lovász directly optimizes IoU. Boundary weighting directly targets BF1. | ~1 GPU-h each |
| E7 | **Augmentation for TE-like scenes**: wider zoom-out range (0.6–1.45), random crops that keep objects large, and *optionally* a 2-image mosaic or copy-paste for multi-object scenes | Narrows the val→TE gap. Check that val doesn't regress. | ~1 GPU-h |
| E8 | **Width/depth retune on the M3**: e.g. widths (48, 96, 160, 256), depths (2, 3, 6). Keep int8 size ≤ about 1 MB | Use the M3 timings from Phase 3. Time is cheap, but size isn't. | 2–3 GPU-h |
| E9 | **Distillation (optional)**: train a big teacher **from scratch on the train split only**, then train the deployed student on teacher soft labels plus GT | Often +1–2 Q for tiny students. This doesn't use pretrained weights, but **confirm with the TA** before using it. | 4–6 GPU-h |
| E10 | **Quantization check**: compare int8 against float16 export on val Q. If int8 loses more than 0.002 Q, use float16 for some layers (e.g. the head and first stem) | Size versus accuracy. | Minutes |

Rejected or low-priority ideas:
* *Post-processing (largest connected component, hole filling):* library CC and morphology are banned, and a pure-Python CC on 256² is slow. A vectorised max-pool "dilation" is possible but probably not worth it. Skip unless the failure analysis shows many small false-positive blobs.
* *Higher in_res at the same width:* this already didn't help in the sweep. Revisit only alongside E5.
* *Track A:* much lower Q ceiling, so it isn't worth doing for score. Optional appendix only.

**Selection rule:** choose the final config by **val CES with M3 timing**. Break near-ties (< 0.002) in favour of the smaller or faster model, because it should generalise better.

### Phase 5: Final model (1 day)
1. Train the chosen config with 2–3 seeds. Pick the best **by val Q** (still no TE).
2. Run `python -m src.export --ckpt checkpoints/final.pt --out weights/model.pt --int8`. Check pixel agreement is ≥ 0.999 against the fp32 model.
3. Make sure **every learned value** (weights, threshold, any normalisation constants) is in `weights/model.pt`, with nothing hard-coded in `pipeline.py`.
4. **Freeze the model.** Only now run the following, once:
   * `python -m src.time_cpu` gives the official M3 T_CPU on all 5,019 TE photos, with laptop plugged in, other apps closed, and the warm-up run done.
   * `python -m src.evaluate` gives val, TE and captured metrics and CES in `results/metrics.json`.
5. Don't change the model after seeing TE numbers. If you do, the report has to say so.

### Phase 6: Captured photos (1 day, can run in parallel with Phase 4)
Take **8 to 10** photos so that a bad one can be dropped and 5 to 10 still remain. Cover **at least 3 conditions**, ideally all 5:

| # | Object | Scene | Light | Distance | Clutter | Occlusion |
|---|---|---|---|---|---|---|
| 1 | Water bottle | Indoor (desk) | Tube light | Near | — | — |
| 2 | Backpack | Indoor (hostel room) | Tube light | Near | ✔ cluttered bed/desk | — |
| 3 | Cycle | Outdoor (parking) | Daylight | Far | ✔ other cycles | partial |
| 4 | Chair | Indoor (classroom) | Tube light | Far | ✔ | partial (desk) |
| 5 | Shoes | Outdoor (corridor) | Daylight | Near | — | — |
| 6 | Plant / pot | Outdoor | Daylight | Far | ✔ | — |
| 7 | Campus cat or dog | Outdoor | Daylight | Near/far | — | — |
| 8 | Mug behind a laptop | Indoor | Tube light | Near | — | ✔ partial |

Steps:
1. Save the originals to `captured/original/<name>.jpg` (JPEG, not HEIC).
2. Run `python -m src.prepare_captured` to get `captured/images_256/<name>.png` (INTER_AREA).
3. Annotate on the **resized** image with `python -m src.annotate` (the polygon tool, which supports cut mode for holes) or with LabelMe or GIMP. Save to `captured/masks_256/<name>.png` with values 0 or 255.
4. Check that the base names match across all three folders and that each mask is single-channel 256×256 {0,255}.
5. Delete the sample's bottle, cat, chair and shoes files first.

### Phase 7: `predict.py` and compliance pass (½ day)
- [ ] `python predict.py --image captured/original/item1.jpg --out /tmp/m.png` prints **exactly** two lines with nothing else, so check for warnings on stdout. The current code already suppresses the TorchScript FutureWarning, but the check is still needed on the M3.
- [ ] The saved mask has values {0, 255} only, at 256×256.
- [ ] Env vars are set before **any** numpy, cv2 or torch import, in `predict.py`, `time_cpu.py` and `evaluate.py`.
- [ ] Grep the codebase for banned APIs: `torchvision`, `sklearn`, `skimage`, `scipy`, `cv2.kmeans`, `grabCut`, `dilate`, `erode`, `morphologyEx`, `connectedComponents`, `onnx`.
- [ ] Parameter file is plain `torch.save` with no compression.
- [ ] No DUTS-TE access in `train.py`, `compare_runs.py` or threshold tuning.
- [ ] Run predict.py from a fresh clone and fresh venv to catch missing files.

### Phase 8: Report (1 to 1.5 days)
Structure (Markdown, exported to PDF):
1. **Introduction**: the task, Track B, and why.
2. **Data**: the DUTS split (9,500 / 1,053, seed), the resizing rule, augmentation, and the captured-photo table with conditions.
3. **Method**: an architecture diagram with a table of layers, channels, resolutions and params. Also cover the loss, optimizer and schedule, and the training hardware (GPU).
4. **Keeping it fast**: 128 input, depthwise-separable blocks, BN and 1/255 folding, no colour conversion (BGR kept), channels_last, TorchScript freeze, one-channel logit upsampled with cv2, int8 storage. Include a **timing breakdown per stage** on the M3.
5. **Experiments**: an ablation table (Phase 4 log) on **val**, showing what helped and what didn't, plus the training curve.
6. **Results table** (required format):

   | Set | mIoU | BF1 | Q |
   |---|---|---|---|
   | Validation (1,053) | | | |
   | DUTS-TE (5,019) | | | |
   | Captured (N) | | | |
   | **T_CPU** (ms, Apple M3, 1 thread) | | **S_disk** (bytes) | **CES** |

   Also state the CPU model (Apple M3), the OS, and the torch, OpenCV and NumPy versions.
7. **Captured photos**: for **every** photo, show the true mask and the predicted mask **side by side as semi-transparent overlays** (`src/make_figures.py` already does green for GT and magenta for prediction), with per-image IoU and BF1 under each.
8. **Failure analysis (≥ 2 cases)**: take the worst captured photos and a few of the worst TE and val images from `results/per_image_*.json`. Name the likely cause for each, such as low light or tube-light colour cast, clutter or similar colours, small or far object, occlusion, or multiple salient objects. Expect the cat photo to fail, as it did in the sample (IoU 0.12).
9. **Generalisation discussion**: the val, TE and captured gap and the reasons for it.
10. **Rules compliance** statement.

### Phase 9: Packaging
```
predict.py  README.md  weights/model.pt
src/  tests/  results/  captured/{original,images_256,masks_256}/
report/report.pdf (+ figures)
```
Leave out `data/`, `checkpoints/` and `.venv/`. Update the README with Mac and Linux setup commands and the exact command order.

---

## 5. Timeline (about 9 working days)

| Day | Work |
|---|---|
| 1 | Phase 0, 1, 2. Start the DUTS download. |
| 2 | Phase 3 (M3 baseline, bench). Take captured photos (Phase 6, steps 1–2). |
| 3–6 | Phase 4 experiments (GPU runs in the background). Annotate captured masks. Draft report sections 1–4. |
| 7 | Phase 5 final training and export. One TE evaluation. |
| 8 | Phase 7 compliance, figures, failure analysis. |
| 9 | Phase 8 report finished, Phase 9 packaging, fresh-clone test. |

---

## 6. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Training versus timing torch version mismatch breaks loading | Pin the same torch version on Colab and the Mac. The param file is a plain state dict, so the model is rebuilt from `config`. |
| M3 timing noise (thermal, background apps, efficiency cores) | Plug in, close apps, run twice, report the official 5,019-photo run. 50 ms leaves about 10× headroom anyway. |
| TA re-runs on a slower x86 laptop | Keep T_CPU ≤ about 15 ms on the M3 so there's a safety margin. |
| Tuning on val overfits it, so TE gains are smaller | Prefer generalisation changes (E3, E4, E7). Use a fixed selection rule. Look at TE only once. |
| Ambiguity about distillation or `F.grid_sample` augmentation | Ask the TA early. Both are only `torch.nn.functional` and use no pretrained weights. |
| Mask annotation quality hurts captured BF1 | Annotate at ×3 zoom with the polygon tool. Expect BF1's 5×5 tolerance to absorb about 2 px of error. |
| Academic integrity | The sample code is a reference. Make sure the final code, experiments, photos and report are your own work, and cite or acknowledge the sample if your course allows building on it. |

---

## 7. Immediate next steps
1. Create the venv and install torch, opencv-python, numpy and matplotlib.
2. Start the DUTS download.
3. Patch `train.py` (device-agnostic) and `time_cpu.py` (macOS CPU name).
4. Reproduce the baseline on the M3 by running `predict.py`, `time_cpu`, `evaluate` and `bench_arch`.
5. Take the 8 to 10 photos this week while daylight and campus conditions are easy to get.
