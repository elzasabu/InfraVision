"""Build colab_training/train_colorizer_v2_kaggle.ipynb.

    python3 colab_training/generators/make_colorizer_v2_notebook.py colab_training/train_colorizer_v2_kaggle.ipynb
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# InfraVision: colorizer v2 (daytime aerial colour), Kaggle

The v1 colorizer learned from LLVIP (night street cameras), so it paints a dark night look. v2 learns
daytime aerial colour from **CART** (Caltech Aerial RGB-Thermal, ECCV 2024: co-registered drone colour +
thermal over rivers, lakes, coastlines, deserts, forests; CC BY-NC 4.0), plus optional extra paired sets
(KAIST / M3FD street scenes with people, FLAME 2 forest + wildfire, LLVIP night).

* CART is **downloaded inside this notebook** from data.caltech.edu (4.3 GB, MD5-checked), so you upload nothing
  for it. If you uploaded a packed CART set (`cart/train/ir ...`), it is used instead.
* Loss: L1 + edge + VGG + chroma, plus a conditional **PatchGAN** from epoch 3 (realistic, less grey colour).
* Best checkpoint = best **CART** test PSNR (held-out flights). Every set is scored each epoch.

**Run**
1. Optional: **Add Input** → the `colorizer_extra` dataset (KAIST / M3FD / FLAME packs from the laptop).
2. Session options: **GPU T4** (one is enough), **Internet On** (needed for the CART download and the git clone).
3. **Save Version → Save & Run All (Commit)**. About 3–5 h. The log prints the time per epoch and what is left.
4. Output tab → `colorizer_v2.zip` → put it in `datasets/outputs/`.

If the session stops early: download `colorizer_v2/last.pth` from the output, rename it **`resume.pth`**,
upload it as a dataset, add it as an input and run again (only a file named `resume.pth` is resumed).
A run that stops in the first minutes with `HEALTH CHECK FAILED` or an assertion: send Claude the log.
""")
md("## 1. Code, attached sets, resume")
code("""
import hashlib, json, shutil, subprocess, sys, time
from pathlib import Path
import torch

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
REPO, OUT = WORK / "InfraVision", WORK / "colorizer_v2"
DATA, TMP = Path("/tmp/colorizer_data"), Path("/tmp/cart_download")
assert torch.cuda.is_available(), "No GPU: Session options -> Accelerator -> GPU T4"
if not REPO.exists():
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", "feature/rev1",
                        "https://github.com/elzasabu/InfraVision.git", str(REPO)], capture_output=True, text=True)
    assert r.returncode == 0, "git clone failed (Internet On?): " + r.stderr[-500:]
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# Packed paired sets attached as datasets: <name>/{train,test}/{ir,rgb}
sets = {}
for p in sorted(INPUT.rglob("ir")):
    if p.parent.name == "train" and (p.parent / "rgb").is_dir():
        sets.setdefault(p.parent.parent.name, p.parent.parent)
print("attached paired sets:", {k: str(v) for k, v in sets.items()} or "none")

resume = list(INPUT.rglob("resume.pth"))
if resume and not (OUT / "last.pth").exists():
    shutil.copy(resume[0], OUT / "last.pth")
    print("resuming from", resume[0])
print("GPU:", torch.cuda.get_device_name(0))
""")
md("## 2. CART: download from Caltech (skipped if a packed `cart` set is attached)")
code("""
URL = "https://data.caltech.edu/api/records/cks6g-ps927/files/labeled_rgbt_pairs.zip/content"
MD5 = "22f42923694e724d4b7e354bace12389"
if "cart" not in sets:
    TMP.mkdir(parents=True, exist_ok=True)
    z = TMP / "labeled_rgbt_pairs.zip"
    t0 = time.time()
    p = subprocess.Popen(["bash", str(REPO / "scripts" / "pdl.sh"), URL, str(z), "16", "32"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    assert p.wait() == 0 and z.is_file(), "CART download failed (see above); rerun to resume it"
    h = hashlib.md5()
    with open(z, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    assert h.hexdigest() == MD5, f"CART zip MD5 mismatch: {h.hexdigest()}"
    print(f"CART downloaded and verified in {(time.time() - t0) / 60:.0f} min")
    r = subprocess.run(["unzip", "-q", "-o", str(z), "labeled_rgbt_pairs/color/*", "labeled_rgbt_pairs/thermal8/*",
                        "-d", str(TMP)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    z.unlink()
    r = subprocess.run([sys.executable, "scripts/pack_cart.py", "--src", str(TMP / "labeled_rgbt_pairs"),
                        "--out_dir", str(DATA)], cwd=REPO, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])
    assert r.returncode == 0, "pack_cart.py failed"
    sets["cart"] = DATA / "cart"
for k, v in sets.items():
    print(f"{k:8s} train {len(list((v / 'train' / 'ir').iterdir())):6d}  test "
          f"{len(list((v / 'test' / 'ir').iterdir())) if (v / 'test' / 'ir').is_dir() else 0:5d}  {v}")
""")
md("""
## 3. Train

CART first = primary set (best checkpoint by its test PSNR). Sampling weights: CART 4, FLAME 2, M3FD 1, KAIST 0.5,
LLVIP 0.5 (each set gets that share of every epoch, whatever its size).
""")
code("""
WEIGHTS = {"cart": 4, "flame": 2, "m3fd": 1, "kaist": 0.5, "llvip": 0.5}
order = ["cart"] + sorted(k for k in sets if k != "cart")
cmd = [sys.executable, "-m", "infravision.train.thermovision", "--data",
       *[f"{k}={sets[k]}:{WEIGHTS.get(k, 1)}" for k in order],
       "--out", str(OUT), "--amp", "--resume", "--workers", "4", "--epochs", "40", "--batch", "16",
       "--size", "320", "--steps_per_epoch", "400", "--max_val", "300",
       "--gan_weight", "0.01", "--gan_start", "3"]
print(" ".join(cmd))
p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:   # live log here and in the Logs tab
    print(line, end="", flush=True)
assert p.wait() == 0, "training failed: see the error above"
""")
md("## 4. Scores per set and sample images (rows: thermal | prediction | real colour)")
code("""
from IPython.display import Image as Show, display
m = json.loads((OUT / "metrics.json").read_text())
print("best epoch:", m["best"]["epoch"], "| CART test PSNR", round(m["best"]["val_psnr"], 2))
for h in m["history"]:
    per = h.get("per_set", {})
    print(f"epoch {h['epoch']:3d}  " + " | ".join(f"{k} {v['psnr']:.2f} dB sat {v['saturation']:.3f}/{v['target_saturation']:.3f}"
                                               for k, v in per.items()))
last = m["history"][-1]["epoch"]
for f in sorted(OUT.glob(f"val_samples_epoch{last:03d}_*.png")):
    print(f.name)
    display(Show(str(f)))
""")
md("## 5. Package: download `colorizer_v2.zip` from the Output tab (`last.pth` stays there for resuming)")
code("""
pack = WORK / "colorizer_v2_pack"
pack.mkdir(exist_ok=True)
for f in OUT.iterdir():
    if f.name != "last.pth":
        shutil.copy(f, pack / f.name)
print("Saved:", shutil.make_archive(str(WORK / "colorizer_v2"), "zip", pack))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                              "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
