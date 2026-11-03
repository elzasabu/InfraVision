"""Build colab_training/train_colorizer_v3_kaggle.ipynb.

    python3 colab_training/generators/make_colorizer_v3_notebook.py colab_training/train_colorizer_v3_kaggle.ipynb
"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# InfraVision: colorizer v3 (aerial + flood / disaster colours), Kaggle

v2 learned daytime colour from real aligned colour/thermal pairs, but only a few landscapes, mostly tilted
views ("blue = top" bias), no floods. v3 adds **pseudo pairs** from colour-only disaster drone photos:

1. Real aligned pairs: **CART** (downloaded here), **FLAME 2** + **M3FD** (`colorizer_extra` input).
2. A **reverse model** (colour -> thermal) is trained on those real pairs (~1 h).
3. It makes a thermal frame for every view of **FloodNet** (Hurricane Harvey floods, `flood_rgb` input) and
   **RescueNet** (Hurricane Michael damage, CC0, downloaded here): flooded roads, houses, debris, rooftops.
4. **Colorizer v3** trains on real + pseudo pairs with random rotation / flip (no "sky on top" shortcut),
   thermal contrast / gamma / noise (other cameras) and PatchGAN (~3 h).

Best checkpoint = best **CART** test PSNR (real pairs, held-out flights). Every set is scored each epoch.

**Run**
1. **Add Input** → `colorizer-extra` (FLAME 2 + M3FD, uploaded for v2) and the new `flood-rgb` dataset (`datasets/flood_rgb.zip`).
2. Session options: **GPU T4**, **Internet On** (CART + RescueNet downloads, git clone).
3. **Save Version → Save & Run All (Commit)**. About 4–5 h.
4. Output tab → `colorizer_v3.zip` → put it in `datasets/outputs/`.

If the session stops early: download `colorizer_v3/last.pth` (rename to **`resume.pth`**) and
`reverse/rgb2thermal_best.pth` from the output, upload both as one dataset, add it as an input and run
again: the reverse model is reused and the colorizer resumes.
""")
md("## 1. Code, attached sets, resume")
code("""
import hashlib, json, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import torch

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
REPO, OUT, REV = WORK / "InfraVision", WORK / "colorizer_v3", WORK / "reverse"
DATA, RGBDIR, TMP = Path("/tmp/colorizer_data"), Path("/tmp/flood_rgb"), Path("/tmp/dl")
assert torch.cuda.is_available(), "No GPU: Session options -> Accelerator -> GPU T4"
if not REPO.exists():
    r = subprocess.run(["git", "clone", "--depth", "1", "-b", "feature/rev1",
                        "https://github.com/elzasabu/InfraVision.git", str(REPO)], capture_output=True, text=True)
    assert r.returncode == 0, "git clone failed (Internet On?): " + r.stderr[-500:]
for d in (OUT, REV, DATA, RGBDIR, TMP):
    d.mkdir(parents=True, exist_ok=True)

def run(cmd, **kw):
    print(" ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(list(map(str, cmd)), cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kw)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()

sets, rgb_sets = {}, {}          # paired <name>/train/{ir,rgb}; colour-only <name>/train/rgb
for p in sorted(INPUT.rglob("rgb")):
    if p.parent.name == "train":
        (sets if (p.parent / "ir").is_dir() else rgb_sets).setdefault(p.parent.parent.name, p.parent.parent)
print("paired sets:", {k: str(v) for k, v in sets.items()} or "none")
print("colour-only sets:", {k: str(v) for k, v in rgb_sets.items()} or "none")
resume = list(INPUT.rglob("resume.pth"))
if resume and not (OUT / "last.pth").exists():
    shutil.copy(resume[0], OUT / "last.pth")
    print("colorizer resumes from", resume[0])
rev_given = list(INPUT.rglob("rgb2thermal_best.pth"))
print("GPU:", torch.cuda.get_device_name(0))

def fetch(url, out, size, md5):
    \"\"\"Parallel ranged download (scripts/pdl.sh), judged by size + MD5, not by exit code.\"\"\"
    rc = run(["bash", REPO / "scripts" / "pdl.sh", url, out, "16", "32"])
    ok = out.is_file() and not Path(str(out) + ".chunks").exists() and out.stat().st_size == size
    assert ok, f"{out.name} incomplete (pdl.sh exit {rc}); run the notebook again"
    h = hashlib.md5()
    with open(out, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    assert h.hexdigest() == md5, f"{out.name} MD5 mismatch: {h.hexdigest()}"
    print(f"{out.name}: downloaded and verified", flush=True)
""")
md("## 2. CART (real aligned pairs; skipped if a packed `cart` set is attached)")
code("""
if "cart" not in sets:
    z = TMP / "labeled_rgbt_pairs.zip"
    fetch("https://data.caltech.edu/api/records/cks6g-ps927/files/labeled_rgbt_pairs.zip/content", z,
          4287170398, "22f42923694e724d4b7e354bace12389")
    with zipfile.ZipFile(z) as zz:
        for n in zz.namelist():
            if n.startswith(("labeled_rgbt_pairs/color/", "labeled_rgbt_pairs/thermal8/")) and not n.endswith("/"):
                zz.extract(n, TMP)
    z.unlink()
    assert run([sys.executable, "scripts/pack_cart.py", "--src", TMP / "labeled_rgbt_pairs", "--out_dir", DATA]) == 0
    sets["cart"] = DATA / "cart"
""")
md("## 3. RescueNet (Hurricane Michael, CC0; skipped if a `rescuenet` colour set is attached)")
code("""
RESCUENET = [("https://ndownloader.figshare.com/files/40582916", "segmentation-validationset.zip", 2373908074,
              "51360be8ea3f73233605a767801c9d1c"),
             ("https://ndownloader.figshare.com/files/40583084", "segmentation-testset.zip", 2387196841,
              "6dbd095cf14d9ddfdf1312d5f1949aeb")]
if "rescuenet" not in rgb_sets:
    photos = TMP / "rescuenet_photos"
    for url, name, size, md5 in RESCUENET:
        z = TMP / name
        fetch(url, z, size, md5)
        with zipfile.ZipFile(z) as zz:   # photos only (the *-label-img folders hold masks)
            for n in zz.namelist():
                if "-org-img/" in n and n.lower().endswith(".jpg"):
                    zz.extract(n, photos)
        z.unlink()
    assert run([sys.executable, "scripts/pack_rgb_views.py", "rescuenet", photos, "--out_dir", RGBDIR]) == 0
    rgb_sets["rescuenet"] = RGBDIR / "rescuenet"
for k, v in rgb_sets.items():
    print(f"{k:10s} colour views: train {len(list((v / 'train' / 'rgb').iterdir()))}, "
          f"test {len(list((v / 'test' / 'rgb').iterdir()))}")
""")
md("## 4. Reverse model: colour -> thermal, on the real pairs (~1 h; reused if attached)")
code("""
REAL_W = {"cart": 4, "flame": 2, "m3fd": 1}
real = ["cart"] + sorted(k for k in sets if k in REAL_W and k != "cart")
if rev_given:
    REV_CKPT = rev_given[0]
    print("reusing reverse model", REV_CKPT)
else:
    rc = run([sys.executable, "-m", "infravision.train.thermovision", "--reverse", "--data",
              *[f"{k}={sets[k]}:{REAL_W[k]}" for k in real], "--out", REV, "--amp", "--workers", "4",
              "--epochs", "15", "--batch", "16", "--size", "320", "--steps_per_epoch", "400", "--max_val", "200",
              "--gan_weight", "0.01", "--gan_start", "3", "--geo_aug", "0.5"])
    assert rc == 0, "reverse model training failed: see above"
    REV_CKPT = REV / "rgb2thermal_best.pth"
""")
md("## 5. Pseudo pairs: a thermal frame for every flood / disaster colour view")
code("""
src = Path("/tmp/flood_rgb_all")
src.mkdir(exist_ok=True)
for k, v in rgb_sets.items():          # one folder holding every colour-only set
    if not (src / k).exists():
        (src / k).symlink_to(v)
assert run([sys.executable, "scripts/make_pseudo_pairs.py", "--model", REV_CKPT, "--src", src, "--out_dir", DATA]) == 0
for k in rgb_sets:
    sets[k] = DATA / k
from IPython.display import Image as Show, display
import cv2, numpy as np
rows = []
for k in rgb_sets:
    fs = sorted((DATA / k / "test" / "rgb").glob("*.jpg"))[:: 7][:3]
    for f in fs:
        c, t = cv2.imread(str(f)), cv2.imread(str(DATA / k / "test" / "ir" / f.name))
        rows.append(np.hstack([c, t]))
if rows:
    cv2.imwrite(str(WORK / "pseudo_pairs.jpg"), np.vstack(rows))
    print("colour photo | generated thermal")
    display(Show(str(WORK / "pseudo_pairs.jpg"), width=700))
""")
md("""
## 6. Colorizer v3

Sampling weights: CART 4, FLAME 2, M3FD 1 (real) and FloodNet 2, RescueNet 2 (pseudo). CART first =
primary set. Random rotation / flip of half the pairs, thermal contrast / gamma / noise, PatchGAN 0.02.
""")
code("""
WEIGHTS = {"cart": 4, "flame": 2, "m3fd": 1, "floodnet": 2, "rescuenet": 2}
order = ["cart"] + sorted(k for k in sets if k != "cart" and k in WEIGHTS)
rc = run([sys.executable, "-m", "infravision.train.thermovision", "--data",
          *[f"{k}={sets[k]}:{WEIGHTS[k]}" for k in order],
          "--out", OUT, "--amp", "--resume", "--workers", "4", "--epochs", "40", "--batch", "16", "--size", "320",
          "--steps_per_epoch", "400", "--max_val", "300", "--gan_weight", "0.02", "--gan_start", "3",
          "--geo_aug", "0.5", "--thermal_aug"])
assert rc == 0, "colorizer training failed: see above"
""")
md("## 7. Scores per set and sample images (rows: thermal | prediction | real colour)")
code("""
m = json.loads((OUT / "metrics.json").read_text())
print("best epoch:", m["best"]["epoch"], "| CART test PSNR", round(m["best"]["val_psnr"], 2))
for h in m["history"][::4] + [m["history"][-1]]:
    print(f"epoch {h['epoch']:3d}  " + " | ".join(f"{k} {v['psnr']:.2f} dB sat {v['saturation']:.3f}/{v['target_saturation']:.3f}"
                                               for k, v in h.get("per_set", {}).items()))
last = m["history"][-1]["epoch"]
for f in sorted(OUT.glob(f"val_samples_epoch{last:03d}_*.png")):
    print(f.name)
    display(Show(str(f)))
""")
md("## 8. Package: download `colorizer_v3.zip` (`last.pth` stays in the output for resuming)")
code("""
pack = WORK / "colorizer_v3_pack"
pack.mkdir(exist_ok=True)
for f in OUT.iterdir():
    if f.name != "last.pth":
        shutil.copy(f, pack / f.name)
for f in (Path(REV_CKPT), WORK / "pseudo_pairs.jpg"):
    if f.is_file():
        shutil.copy(f, pack / f.name)
print("Saved:", shutil.make_archive(str(WORK / "colorizer_v3"), "zip", pack))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"},
                                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                              "isGpuEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
json.dump(nb, open(sys.argv[1], "w"), indent=1)
