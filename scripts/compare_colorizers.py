"""Compare colorizer checkpoints: real ground truth, orientation, and real thermal from other cameras.

    .venv/bin/python scripts/compare_colorizers.py        # v2 vs v3 -> results/colorizer_v3_compare.{json,jpg}

1. Held-out CART flights (real colour photo = ground truth): PSNR / SSIM upright, upside down and
   rotated 90 degrees. A big drop when turned means the model uses position, not content.
2. Real thermal from cameras it never paired with colour (POP, WiSARD, HIT-UAV, AIResQ), no ground truth:
   - rotation disagreement: colour of f(x) vs rot180(f(rot180(x))) in LAB a/b (0 = content only)
   - vertical band: mean over frames of the signed LAB a/b difference, top third - bottom third.
     Straight-down footage has no "up", so a consistent difference is a painted-on band.
   - saturation.
3. Grid through the dashboard pipeline: thermal | model A | model B, for the photo and the luma views.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pack_cart import SRC, pairs, split  # noqa: E402
from infravision.colorizer import ColorizerAdapter  # noqa: E402
from infravision.metrics import psnr, ssim  # noqa: E402

DATA = ROOT.parent / "datasets"
DET = ROOT / "human_detection" / "datasets"


def real_sets(n):
    pick = lambda fs: fs[:: max(1, len(fs) // n)][:n]
    return {"POP test": pick(sorted(glob.glob(str(DET / "pop/images/test/*")))),
            "WiSARD": pick(sorted(glob.glob(str(DATA / "WiSARD/WiSARDv1/210924_FHL_Enterprise_IR_0567/*.jp*g")))
                           + sorted(glob.glob(str(DATA / "WiSARD/WiSARDv1/200929_Karen_FLIR_IR_5/*.jp*g")))),
            "HIT-UAV": pick(sorted(glob.glob(str(DET / "hituav/images/test/*.jpg")))),
            "AIResQ": pick(sorted(glob.glob(str(DATA / "AIResQ/Benchmark/images/*"))))}


def lab_ab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1:].astype(np.float32) - 128


def sat(rgb):
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].mean() / 255)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", default=str(ROOT / "checkpoints/colorizer/thermovision_v2.pth"))
    p.add_argument("--b", default=str(ROOT / "checkpoints/colorizer/thermovision_v3.pth"))
    p.add_argument("--names", nargs=2, default=["v2", "v3"])
    p.add_argument("--per_flight", type=int, default=8)
    p.add_argument("--n_real", type=int, default=30)
    p.add_argument("--out", default=str(ROOT / "results/colorizer_v3_compare.json"))
    a = p.parse_args()
    models = {a.names[0]: ColorizerAdapter(a.a, "cpu", 256, verbose=False),
              a.names[1]: ColorizerAdapter(a.b, "cpu", 256, verbose=False)}
    t = lambda x: torch.from_numpy(x.astype(np.float32) / 255).permute(2, 0, 1)[None]
    bf = pairs(SRC)
    frames = [fr for f in sorted(split(bf)) for fr in bf[f][:: max(1, len(bf[f]) // a.per_flight)][: a.per_flight]]
    turns = {"upright": lambda x: x, "upside down": lambda x: np.ascontiguousarray(x[::-1]),
             "rotated 90": lambda x: np.ascontiguousarray(np.rot90(x))}
    res = {"cart_ground_truth": {k: {o: {"psnr": [], "ssim": []} for o in turns} for k in models}}
    for _, tp, cp in frames:
        th = cv2.imread(str(tp), 0)
        gt = cv2.cvtColor(cv2.imread(str(cp)), cv2.COLOR_BGR2RGB)
        th = cv2.resize(th, (gt.shape[1], gt.shape[0]))
        for k, m in models.items():
            for o, f in turns.items():
                x, g = f(th), f(gt)
                y = m.colorize(x, (x.shape[1], x.shape[0]))
                res["cart_ground_truth"][k][o]["psnr"].append(psnr(t(y), t(g)).item())
                res["cart_ground_truth"][k][o]["ssim"].append(ssim(t(y), t(g)).item())
    for k in models:
        for o in turns:
            d = res["cart_ground_truth"][k][o]
            d["psnr"], d["ssim"] = round(float(np.mean(d["psnr"])), 2), round(float(np.mean(d["ssim"])), 3)
    res["cart_frames"] = len(frames)
    res["real_thermal"] = {}
    for name, files in real_sets(a.n_real).items():
        res["real_thermal"][name] = {"frames": len(files)}
        for k, m in models.items():
            dis, band, sa = [], [], []
            for f in files:
                g = cv2.imread(f, 0)
                if g is None:
                    continue
                y = m.colorize(g)
                r = m.colorize(np.ascontiguousarray(g[::-1, ::-1]))[::-1, ::-1]
                ab, abr = lab_ab(y), lab_ab(np.ascontiguousarray(r))
                dis.append(float(np.abs(ab - abr).mean()))
                h = ab.shape[0] // 3
                band.append(ab[:h].mean((0, 1)) - ab[-h:].mean((0, 1)))
                sa.append(sat(y))
            res["real_thermal"][name][k] = {"rotation_disagreement": round(float(np.mean(dis)), 2),
                                            "vertical_band": round(float(np.linalg.norm(np.mean(band, 0))), 2),
                                            "saturation": round(float(np.mean(sa)), 3)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
