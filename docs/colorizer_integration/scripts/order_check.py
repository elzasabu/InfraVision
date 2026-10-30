"""Quick check (step 2.6): what should the ThermoVision colorizer see inside branch A?

    raw       the original thermal frame (what the U-Net was trained on); colour resized to the SR size
    enhanced  the CLAHE + denoise + SR frame (the old branch A order)
each with and without the SR detail add-back (colorizer.detail 0.5 / 0). Metrics against the real RGB, both
resized to 640 x 512 (INTER_AREA), on 5 paired frames (3 LLVIP test, 2 FLIR val).

usage: python -B order_check.py --tv <ThermoVision dir> --out docs/colorizer_integration [--extra-site <dir>]
(--extra-site appends a site-packages folder that has torchmetrics, for LPIPS)
"""
import argparse
import csv
import json
import os
import site
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--tv", required=True)
p.add_argument("--out", required=True)
p.add_argument("--extra-site", default=None)
a = p.parse_args()
if a.extra_site:
    site.addsitedir(a.extra_site)   # appended: this env's own torch etc. still win

import cv2  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO)
from infravision.config import load_config, merge  # noqa: E402
from infravision.enhance import Enhancer  # noqa: E402
from infravision.io import read_rgb, read_thermal  # noqa: E402
from infravision.metrics import psnr, ssim  # noqa: E402

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).eval()
except Exception as e:  # noqa: BLE001
    print("LPIPS unavailable:", e)
    lpips = None

data = os.path.join(REPO, "data", "paired_disaster")
rows = {}
for r in csv.DictReader(open(os.path.join(data, "manifest.csv"), newline="")):
    if r["split"] == "test":
        rows.setdefault(r["id"], r)
ids = sorted(rows)
pairs = [(os.path.join(data, rows[i]["thermal"]), os.path.join(data, rows[i]["rgb"]), i)
         for i in (ids[len(ids) // 6], ids[len(ids) // 2], ids[5 * len(ids) // 6])]
flir = [ln.strip().split("_PreviewData")[0] for ln in open(os.path.join(a.tv, "dataset/align/align_validation.txt")) if ln.strip()]
pairs += [(os.path.join(a.tv, "dataset/align/JPEGImages", i + "_PreviewData.jpeg"),
           os.path.join(a.tv, "dataset/align/JPEGImages", i + "_RGB.jpg"), i) for i in (flir[100], flir[700])]

base = load_config(overrides={"device": "cpu"})
variants = {f"{inp}, detail {d}": merge(base, {"colorizer": {"backend": "thermovision", "input": inp, "detail": d}})
            for inp in ("raw", "enhanced") for d in (0.5, 0.0)}
SIZE = (640, 512)


def t(img):
    return torch.from_numpy(cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA).astype(np.float32) / 255).permute(2, 0, 1)[None]


res = {}
for name, cfg in variants.items():
    enh = Enhancer(cfg)
    acc = {"psnr": [], "ssim": [], "lpips": [], "sat": []}
    for tp, vp, _ in pairs:
        g, _ = read_thermal(tp)
        out = enh(g)["enhanced_rgb"]
        x, y = t(out), t(read_rgb(vp))
        acc["psnr"].append(psnr(x, y).item()); acc["ssim"].append(ssim(x, y).item())
        if lpips is not None:
            with torch.no_grad():
                acc["lpips"].append(lpips(x, y).item())
        mx, mn = x[0].max(0).values, x[0].min(0).values
        acc["sat"].append(((mx - mn) / mx.clamp_min(1e-6)).mean().item())
    res[name] = {k: float(np.mean(v)) if v else None for k, v in acc.items()}
    print(name, res[name], flush=True)
res["images"] = [i for _, _, i in pairs]
res["sr"] = Enhancer(base).sr.name
json.dump(res, open(os.path.join(a.out, "order_check.json"), "w"), indent=1)
