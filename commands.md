# Commands: full run on the Mac (Apple M3)

Run everything from the project root. Training uses the Mac GPU (MPS), and timing uses one CPU thread. Every step writes its output under `results/` or `report/figures/`, so nothing needed for the report is lost. The last section lists which file feeds which part of the report.

**Measured on this M3:** one training epoch (about 9,500 photos plus validation) takes about 40 s for the `w32` and `w48` models at 128 px. An 80-epoch run takes about 1 hour. `--amp` (float16) gave no speedup on MPS, so it isn't used below.

---

## 0. One-time setup
```bash
cd ~/Downloads/cv-assignment2
python3.12 -m venv .venv            # already created; skip if .venv/ exists
source .venv/bin/activate           # do this in every new terminal
pip install -r requirements.txt
python -m tests.test_metrics        # all 4 tests must print "ok"
```

For long runs, keep the laptop **plugged in**, turn off **Low Power Mode**, and prefix the command with `caffeinate -i` so the Mac doesn't sleep. For example:
```bash
caffeinate -i python -m src.train ...
```

## 1. Download and cache DUTS
```bash
mkdir -p data && cd data
curl -L -O http://saliencydetection.net/duts/download/DUTS-TR.zip
curl -L -O http://saliencydetection.net/duts/download/DUTS-TE.zip
unzip -q DUTS-TR.zip && unzip -q DUTS-TE.zip
cd ..
ls data/DUTS-TR/DUTS-TR-Image | wc -l      # expect 10553
ls data/DUTS-TE/DUTS-TE-Image | wc -l      # expect 5019

python -m src.data 2>&1 | tee results/data_prep.txt
```
The cache step takes a few minutes. It prints `split: 9498 train / 1055 val (seed 0); test 5019`, which is the split size the report needs.

## 2. CPU speed of candidate architectures (no training needed)
```bash
python -m src.bench                 # -> results/bench.md
python -m src.bench --flip-tta      # -> results/bench_tta.md
```
This shows what each size costs in ms and bytes before you spend an hour training it.

## 3. Training runs (validation only, never DUTS-TE)
Run these one after another. Each run writes:
* `checkpoints/<name>.pt`, the best epoch by validation Q
* `results/train_<name>.json`, the per-epoch history
* `results/logs/train_<name>.log`, the console output

After each run, add it to the comparison table, which exports the model, tunes the threshold on val, times it on 500 val photos with one CPU thread, and computes val CES both with and without flip TTA:
```bash
python -m src.compare checkpoints/<name>.pt      # updates results/experiments.md
```

| # | Purpose | Command | Time |
|---|---|---|---|
| R1 | small baseline | `caffeinate -i python -m src.train --name r1_w16 --widths 16 24 40 64 --depths 2 2 3 --epochs 60` | ~35 min |
| R2 | wider (main candidate) | `caffeinate -i python -m src.train --name r2_w48 --widths 48 64 128 192 --depths 2 3 4 --epochs 80` | ~1 h |
| R3 | R2 + weight EMA | `caffeinate -i python -m src.train --name r3_w48_ema --widths 48 64 128 192 --depths 2 3 4 --epochs 80 --ema 0.999` | ~1 h |
| R4 | stronger boundary loss (for BF1) | `caffeinate -i python -m src.train --name r4_w48_bw4 --widths 48 64 128 192 --depths 2 3 4 --epochs 80 --ema 0.999 --boundary-weight 4` | ~1 h |
| R5 | 5th stage, more context (for TE scenes) | `caffeinate -i python -m src.train --name r5_w48_5s --widths 48 64 128 192 256 --depths 2 3 4 2 --epochs 80 --ema 0.999` | ~1 h |
| R6 | higher input resolution | `caffeinate -i python -m src.train --name r6_w32_r160 --in-res 160 --widths 32 48 96 128 --depths 2 3 4 --epochs 80 --ema 0.999` | ~1.3 h |

Options:
* If R3 to R6 don't beat R2, drop `--ema 0.999` or `--boundary-weight 4` from the later runs. Change one thing at a time.
* To queue several runs overnight, chain them with `;`, then compare all of them in the morning:
  ```bash
  caffeinate -i sh -c 'python -m src.train --name r3_w48_ema --widths 48 64 128 192 --depths 2 3 4 --epochs 80 --ema 0.999; \
                        python -m src.train --name r4_w48_bw4 --widths 48 64 128 192 --depths 2 3 4 --epochs 80 --ema 0.999 --boundary-weight 4'
  python -m src.compare checkpoints/*.pt
  ```
* `python -m src.train ... --max-train 500 --epochs 2` is a quick check that a new configuration runs at all.

## 4. Final model
Take the setting with the best **val CES** in `results/experiments.md`. Retrain it longer, and with a second seed if there's time:
```bash
caffeinate -i python -m src.train --name final --widths <best> --depths <best> [--in-res <best>] \
    --epochs 150 --ema 0.999 [--boundary-weight <best>]
caffeinate -i python -m src.train --name final_s1 --seed 1 <same options>        # optional
python -m src.compare checkpoints/final.pt checkpoints/final_s1.pt
```
Export the winner (best val CES, using the TTA setting that won in the table) to the deployed parameter file:
```bash
python -m src.export --ckpt checkpoints/final.pt --tune [--flip-tta]
#   -> weights/model.pt, results/export_model.json (threshold sweep, size, agreement)
python -m src.time_cpu --set val    # -> results/timing_val_model.json
python -m src.figures curve --run final             # -> report/figures/curve_final.png
```
From here on, **don't change the model.**

## 5. Captured photos (any time before step 6)
```bash
# copy 5-10 of your own JPEGs into captured/original/  (HEIC: sips -s format jpeg x.heic --out x.jpg)
python -m src.captured prepare      # -> captured/images_256/<name>.png
python -m src.annotate              # draw masks -> captured/masks_256/<name>.png
python -m src.captured check        # must end with "all good"
```
Annotator keys: left-click adds a point, right-click closes the polygon, `x` toggles cut mode for holes, `u` undoes a point, `c` clears the mask, `s` saves, `n` skips the photo, and `q` quits.

## 6. Official numbers (run once, with the final model)
Close other apps, keep the laptop plugged in, and run:
```bash
python -m src.time_cpu 2>&1 | tee results/time_cpu_te.txt       # all 5,019 DUTS-TE photos -> results/timing.json
python -m src.evaluate --sets val te captured 2>&1 | tee results/evaluate.txt
#   -> results/metrics.json, results/final_table.md, results/per_image_{val,te,captured}.json,
#      results/captured_pred/*.png
```

## 7. Report figures
```bash
python -m src.figures captured            # -> report/figures/captured_overlays.png (true vs predicted, every photo)
python -m src.figures worst --set val     # -> report/figures/worst_val.png
python -m src.figures worst --set te      # -> report/figures/worst_te.png (for failure analysis only)
python -m src.figures curve --run r2_w48  # training curves of any other run you want to show
```

## 8. Check `predict.py`
```bash
python predict.py --image captured/original/<name>.jpg --out results/predict_check.png 2>&1 | tee results/predict_check.txt
```
It must print exactly two lines: `T CPU ms: ...` and `S disk bytes: ...`.

---

## What goes where in the report
| Report item | Source |
|---|---|
| Train / val / test split sizes | `results/data_prep.txt` |
| Architecture speed and size trade-off | `results/bench.md`, `results/bench_tta.md` |
| Ablation / experiment table (validation only) | `results/experiments.md` (and `.json`) |
| Training curves | `report/figures/curve_<run>.png`, `results/train_<run>.json`, `results/logs/` |
| Threshold tuning and int8 accuracy check | `results/export_model.json` (sweep, agreement) |
| **Main results table** (val, DUTS-TE, own photos, T_CPU, S_disk, CES) | `results/final_table.md`, `results/metrics.json` |
| CPU model, OS, torch version | `results/timing.json` |
| Captured-photo overlays | `report/figures/captured_overlays.png` |
| Failure cases (≥ 2) | `results/per_image_captured.json`, `results/per_image_te.json` (sort by `IoU_obj`), `report/figures/worst_*.png` |
| `predict.py` output | `results/predict_check.txt` |
