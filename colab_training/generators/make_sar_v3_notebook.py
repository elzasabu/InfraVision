import json, sys
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# InfraVision: person detector v3 (POP + AIResQ + WiSARD + RGBTDronePerson + HIT-UAV), Kaggle

Fine-tunes detector **v2** (the result of `train_sar_yolo_kaggle.ipynb`) on a larger thermal search-and-rescue set:

* **POP** drone thermal (every 2nd frame) + **flood v2** copies, which now include *empty* water blobs so a blob alone no longer means "person"
* **AIResQ** thermal drone flights (tiny people, 7x9 px): full frames + 2x2 tiles
* **WiSARD** wilderness SAR thermal: 31 labelled sequences, FLIR + DJI cameras, top-down and oblique
* **RGBTDronePerson** (tiny people, median 13 px) and **HIT-UAV** (60–130 m altitude), thermal only
* **LLVIP** persons: large, close, oblique people (the v2 model missed them)

Test sets are separate flights / splits, never seen in training: `test_pop`, `test_flood`, `test_airesq`, `test_wisard`, `test_rgbt`, `test_hituav`, `test_llvip`.

**Run**
1. **Datasets → New Dataset** → upload `sar_v3_yolo.zip` (private). Kaggle unzips it.
   **Second dataset**: upload `start.pt` (detector v2, made on the laptop by `scripts/install_detector.py`).
2. **Code → New Notebook → File → Import Notebook** → this file.
3. Right panel: **Add Input** → both datasets; **Accelerator → GPU T4 x2**; **Internet → On**.
4. **Save Version → Save & Run All (Commit)**. You can close the browser. The log prints an ETA after epoch 1.
5. Output tab → download `sar_v3.zip`. Put it in `datasets/outputs/`.

If the session stops early: download the output's `runs/sar_v3/weights/last.pt`, **rename it `resume.pt`**, upload it as a dataset, add it as an input and run again; it resumes. (Only a file named `resume.pt` triggers resuming.)

A run that fails in the first minutes with `HEALTH CHECK FAILED` is broken: send Claude the log.
""")
md("## 1. GPU")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch
assert torch.cuda.is_available(), "No GPU. Set Accelerator -> GPU T4 x2 in the right panel."
print("GPU ready:", torch.cuda.get_device_name(0))
""")
md("## 2. Same Ultralytics version as the laptop")
code("""
import subprocess, sys
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.191"], capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-2000:])
assert r.returncode == 0, "pip install failed. Right panel -> Internet -> On (phone-verified account), then run again."
import ultralytics
print("ultralytics", ultralytics.__version__)
""")
md("## 3. Dataset and settings")
code("""
import json
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
sar = [p.parent for p in INPUT.rglob("train.txt") if (p.parent / "images" / "val").is_dir()]
assert sar, "sar dataset not found: Add Input -> the uploaded sar_yolo.zip dataset."
SAR = sar[0]
# Prefer a separately uploaded start.pt (detector v2) over one packed inside the data zip.
starts = sorted(INPUT.rglob("start.pt"), key=lambda p: SAR.parent in p.parents)
assert starts, "start.pt (detector v2) not found: upload it as a dataset and Add Input."
START_WEIGHTS = starts[0]
previous = [p for p in INPUT.rglob("resume.pt")]
PREVIOUS_LAST = previous[0] if previous else None

def data_yaml(test_split, train="train.txt"):
    y = WORK / f"sar_{test_split}.yaml"
    y.write_text(f"path: {SAR}\\ntrain: {SAR / train}\\nval: images/val\\ntest: images/{test_split}\\n\\nnames:\\n  0: person\\n")
    return y

TESTS = [s for s in ("test_pop", "test_flood", "test_airesq", "test_wisard", "test_rgbt", "test_hituav", "test_llvip")
         if (SAR / "images" / s).is_dir()]
DATA_YAML = data_yaml("test_pop")
RUNS_DIR, RUN_NAME = WORK / "runs", "sar_v3"
RUN_DIR = RUNS_DIR / RUN_NAME

IMGSZ = 1024     # small people stay visible
EPOCHS = 20      # fine-tune from detector v2
PATIENCE = 8
WORKERS = 4      # Kaggle has 4 CPU cores; v2 used 2 and was data-loading bound (1.5 it/s)
BATCH = 16       # v2 used 4.9 GB of 15 GB at batch 8

n_train = sum(1 for l in (SAR / "train.txt").read_text().split("\\n") if l.strip())
print("Dataset      :", SAR)
print("train list   :", n_train, "images;", json.loads((SAR / "splits.json").read_text()).get("train_mix"))
for s in ["val"] + TESTS:
    print(f"{s:13s}: {len(list((SAR / 'images' / s).iterdir()))} images")
print("Start weights:", START_WEIGHTS)
print("Previous run :", PREVIOUS_LAST or "none (fresh start)")
""")
md("""
## 4. Evaluation helpers

mAP on every test set, plus the **flood shortcut check**: how often an *empty* water blob in `test_flood`
is detected as a person (conf ≥ 0.25). Gate: < 2%. The old model detected 24% of them.
""")
code("""
from ultralytics import YOLO

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0

def decoy_rate(weights, conf=0.25):
    dec = {k.split("/", 1)[1]: v for k, v in json.loads((SAR / "flood_decoys.json").read_text()).items()
           if k.startswith("test_flood/")}
    files = sorted(f for f in (SAR / "images" / "test_flood").iterdir() if f.name in dec)
    model, n, hit = YOLO(str(weights)), 0, 0
    for i in range(0, len(files), 16):
        for f, r in zip(files[i:i + 16], model.predict([str(f) for f in files[i:i + 16]], imgsz=IMGSZ, conf=conf,
                                                         device=0, verbose=False)):
            preds = r.boxes.xyxy.cpu().numpy().tolist()
            for d in dec[f.name]:
                n += 1
                hit += any(iou(p, d) >= 0.1 or (d[0] <= (p[0] + p[2]) / 2 <= d[2] and d[1] <= (p[1] + p[3]) / 2 <= d[3])
                           for p in preds)
    return {"decoys": n, "detected": hit, "rate": hit / max(1, n)}

def evaluate_all(weights, tag):
    out = {}
    for s in TESTS:
        for sz in ([640, IMGSZ] if s in ("test_airesq", "test_wisard", "test_rgbt", "test_hituav", "test_llvip") else [IMGSZ]):
            m = YOLO(str(weights)).val(data=str(data_yaml(s)), split="test", imgsz=sz, batch=BATCH, device=0,
                                       plots=False, project=str(WORK / "val"), name=f"{tag}_{s}_{sz}", exist_ok=True)
            out[f"{s}@{sz}"] = {"P": round(m.box.mp, 4), "R": round(m.box.mr, 4),
                                "mAP50": round(m.box.map50, 4), "mAP50_95": round(m.box.map, 4)}
    out["flood_empty_blobs"] = decoy_rate(weights)
    return out

def show(res):
    for k, v in res.items():
        if "mAP50" in v:
            print(f"{k:18s} P {v['P']:.3f}  R {v['R']:.3f}  mAP50 {v['mAP50']:.3f}  mAP50-95 {v['mAP50_95']:.3f}")
    d = res["flood_empty_blobs"]
    print(f"empty water blobs detected: {d['detected']}/{d['decoys']} = {100 * d['rate']:.1f}%  (gate < 2%)")
""")
md("## 5. Before: the starting model on every test set")
code("""
before = evaluate_all(START_WEIGHTS, "before")
show(before)
""")
md("""
## 6. Train

12.2k training images. The health check prints the measured time per epoch and an ETA after every epoch.
Stops early when val mAP stops improving for 8 epochs (val is balanced across sources).
Health check after every epoch: stops if the loss turns NaN, or val mAP50 is under 0.3 at epoch 3
(the starting model is already well above that).
""")
code("""
import math, time
T0 = time.time()

def health_check(trainer, min_map50=0.3, by_epoch=3):
    epoch = trainer.epoch + 1
    loss = trainer.tloss
    values = loss.values() if isinstance(loss, dict) else [loss] if loss is not None else []
    if not all(math.isfinite(float(v.sum())) if hasattr(v, "sum") else math.isfinite(float(v)) for v in values):
        raise RuntimeError(f"HEALTH CHECK FAILED: epoch {epoch}: training loss is NaN/inf ({loss})")
    map50 = trainer.metrics.get("metrics/mAP50(B)", 0.0)
    if epoch >= by_epoch and not map50 >= min_map50:
        raise RuntimeError(f"HEALTH CHECK FAILED: epoch {epoch}: val mAP50 {map50:.3f} < {min_map50}")
    per = (time.time() - T0) / epoch
    print(f"health check ok: epoch {epoch}, mAP50 {map50:.3f} | {per / 60:.1f} min/epoch, "
          f"at most {max(0, EPOCHS - epoch) * per / 3600:.1f} h of training left (+~20 min final evaluation)", flush=True)

if PREVIOUS_LAST is not None:
    print("Resuming from:", PREVIOUS_LAST)
    try:
        model = YOLO(str(PREVIOUS_LAST))
        model.add_callback("on_fit_epoch_end", health_check)
        model.train(resume=True, data=str(DATA_YAML), save_dir=str(RUN_DIR), device=0, workers=WORKERS)
    except AssertionError as e:
        print("Nothing to resume (training already finished):", e)
else:
    print("Starting from:", START_WEIGHTS)
    model = YOLO(str(START_WEIGHTS))
    model.add_callback("on_fit_epoch_end", health_check)
    model.train(data=str(DATA_YAML), imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, batch=BATCH, device=0,
                workers=WORKERS, project=str(RUNS_DIR), name=RUN_NAME, exist_ok=True, seed=0,
                flipud=0.5,      # top-down drone view: vertical flips are realistic
                mosaic=1.0, scale=0.5,
                # Fine-tune gently: v2 used the automatic SGD lr 0.01 and peaked at epoch 2.
                optimizer="SGD", lr0=0.002, lrf=0.1, cos_lr=True, warmup_epochs=1)
""")
md("## 7. After: every test set, compared with before")
code("""
best = RUN_DIR / "weights" / "best.pt"
after = evaluate_all(best, "after")
print(f"{'':18s} {'before':>8s} {'after':>8s}   (mAP50)")
for k in after:
    if "mAP50" in after[k]:
        print(f"{k:18s} {before[k]['mAP50']:8.3f} {after[k]['mAP50']:8.3f}")
print(f"{'empty blobs hit':18s} {100 * before['flood_empty_blobs']['rate']:7.1f}% {100 * after['flood_empty_blobs']['rate']:7.1f}%")
(RUN_DIR / "sar_eval.json").write_text(json.dumps({"before": before, "after": after, "start": str(START_WEIGHTS),
                                                   "imgsz": IMGSZ, "epochs": EPOCHS}, indent=2))
""")
md("## 8. Training curve and example detections")
code("""
import pandas as pd
from IPython.display import Image as Show, display
df = pd.read_csv(RUN_DIR / "results.csv"); df.columns = df.columns.str.strip()
display(df[["epoch", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]].tail(10))
display(Show(str(RUN_DIR / "results.png"), width=900))
for s in TESTS:
    files = sorted((SAR / "images" / s).iterdir())
    YOLO(str(best)).predict(files[:: max(1, len(files) // 3)][:3], imgsz=IMGSZ, conf=0.25, save=True,
                            project=str(WORK / "pred"), name=s, exist_ok=True, verbose=False)
    for p in sorted((WORK / "pred" / s).iterdir()):
        display(Show(str(p), width=640))
""")
md("## 9. Package: download `sar_v3.zip` from the Output tab")
code("""
import shutil
archive = shutil.make_archive(str(WORK / "sar_v3"), "zip", RUN_DIR)
print("Saved:", archive)
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
