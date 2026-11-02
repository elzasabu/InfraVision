"""Compare colorizer checkpoints on held-out CART flights (real daytime colour as ground truth).

    .venv/bin/python scripts/eval_colorizer.py                 # v1 (LLVIP) vs v2 -> results/colorizer_v2_eval.json + grid
    .venv/bin/python scripts/eval_colorizer.py --extra a.png b.png   # also colourise these frames (no ground truth)

CART test flights = the split scripts/pack_cart.py makes (seed 0), so these frames were never
trained on. Each model sees the 8-bit thermal frame and is scored against the co-registered colour
photo at the photo's size: PSNR / SSIM (higher = closer), mean saturation (the real photos' value
is printed too), mean brightness. Grid rows: thermal | v1 | v2 | real colour.
"""
import argparse
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

MODELS = {"v1 (LLVIP night)": ROOT / "checkpoints/colorizer/thermovision_llvip_v1.pth",
          "v2 (CART+FLAME+M3FD)": ROOT / "checkpoints/colorizer/thermovision_v2.pth"}


def sat(rgb):
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].mean() / 255)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--per_flight", type=int, default=12, help="frames per held-out flight")
    p.add_argument("--grid_rows", type=int, default=6)
    p.add_argument("--extra", nargs="*", default=[], help="thermal frames to colourise (no ground truth)")
    p.add_argument("--out", default=str(ROOT / "results" / "colorizer_v2_eval.json"))
    a = p.parse_args()
    by_flight = pairs(Path(a.src))
    test = sorted(split(by_flight))
    frames = [fr for f in test for fr in by_flight[f][:: max(1, len(by_flight[f]) // a.per_flight)][: a.per_flight]]
    print(f"{len(frames)} frames from {len(test)} held-out CART flights: {test}")
    models = {k: ColorizerAdapter(v, "cpu", 256, verbose=False) for k, v in MODELS.items() if v.is_file()}
    t = lambda x: torch.from_numpy(x.astype(np.float32) / 255).permute(2, 0, 1)[None]
    res = {k: {"psnr": [], "ssim": [], "sat": [], "bright": []} for k in models}
    real = {"sat": [], "bright": []}
    rows = []
    for i, (_, tp, cp) in enumerate(frames):
        th = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
        gt = cv2.cvtColor(cv2.imread(str(cp)), cv2.COLOR_BGR2RGB)
        h, w = gt.shape[:2]
        th = cv2.resize(th, (w, h), interpolation=cv2.INTER_AREA)
        real["sat"].append(sat(gt))
        real["bright"].append(float(gt.mean()))
        outs = {}
        for k, m in models.items():
            o = m.colorize(th, (w, h))
            outs[k] = o
            res[k]["psnr"].append(psnr(t(o), t(gt)).item())
            res[k]["ssim"].append(ssim(t(o), t(gt)).item())
            res[k]["sat"].append(sat(o))
            res[k]["bright"].append(float(o.mean()))
        if i % max(1, len(frames) // a.grid_rows) == 0 and len(rows) < a.grid_rows:
            cells = [label(cv2.cvtColor(th, cv2.COLOR_GRAY2RGB), "thermal")]
            cells += [label(o, k) for k, o in outs.items()] + [label(gt, "real colour")]
            rows.append(np.hstack([cv2.resize(c, (400, round(400 * h / w))) for c in cells]))
    summary = {k: {m: round(float(np.mean(v)), 4) for m, v in d.items()} for k, d in res.items()}
    summary["real colour photos"] = {m: round(float(np.mean(v)), 4) for m, v in real.items()}
    summary["frames"], summary["test_flights"] = len(frames), test
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    grid = out.with_suffix(".jpg")
    cv2.imwrite(str(grid), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k:24s} " + "  ".join(f"{m} {x}" for m, x in v.items()))
    print("->", out, grid)
    if a.extra:
        ex = []
        for f in a.extra:
            th = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            cells = [label(cv2.cvtColor(th, cv2.COLOR_GRAY2RGB), Path(f).stem)]
            cells += [label(m.colorize(th), k) for k, m in models.items()]
            ex.append(np.hstack([cv2.resize(c, (420, round(420 * th.shape[0] / th.shape[1]))) for c in cells]))
        width = max(e.shape[1] for e in ex)
        ex = [np.pad(e, ((0, 0), (0, width - e.shape[1]), (0, 0))) for e in ex]
        ef = out.with_name(out.stem + "_extra.jpg")
        cv2.imwrite(str(ef), cv2.cvtColor(np.vstack(ex), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("->", ef)


if __name__ == "__main__":
    main()
