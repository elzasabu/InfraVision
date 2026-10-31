"""SR gate on full POP test frames (RUN_ORDER Phase 1): EDSR x2 vs bicubic x2.

    .venv/bin/python scripts/eval_sr_test.py                 # 30 frames -> results/sr_test_eval.json
    .venv/bin/python scripts/eval_sr_test.py --n 5 --out /tmp/x.json

The full 1280x1024 test frame is the ground truth. Two low-res inputs per frame:
  clean    : INTER_AREA x2 downscale (classic SR protocol)
  degraded : blur, x2 downscale, noise and JPEG at low res (seed 0), like compressed drone video
Both are bicubic-upscaled back to full size; bicubic is scored as is, EDSR restores it.
  train_deg: the exact training degradation (infravision.data.pop.degrade_for_sr) on the full
             frame, i.e. the val-metric protocol on whole test frames.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infravision.config import load_config  # noqa: E402
from infravision.data.pop import degrade_for_sr, pop_splits  # noqa: E402
from infravision.enhance import SuperResolver  # noqa: E402
from infravision.io import read_thermal  # noqa: E402
from infravision.metrics import psnr, ssim  # noqa: E402


def degrade_lr(clean, rng):
    """Training degradation (infravision.data.pop.degrade_for_sr) but kept at low res."""
    h, w = clean.shape
    img = cv2.GaussianBlur(clean, (0, 0), sigmaX=float(rng.uniform(0.6, 1.4)))
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = np.clip(small + rng.normal(0, rng.uniform(0.0, 4.0), small.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 95))])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--out", default="results/sr_test_eval.json")
    args = p.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = load_config()
    edsr = SuperResolver(**{**cfg["sr"], "backend": "edsr", "scale": 2, "device": "cpu"})
    assert edsr.model is not None, "no EDSR weights found"
    restore = SuperResolver(**{**cfg["sr"], "backend": "edsr", "scale": 1, "device": "cpu"})
    recs = pop_splits(existing_only=True)["test"]
    recs = recs[:: max(1, len(recs) // args.n)][: args.n]
    t = lambda a: torch.from_numpy(a.astype(np.float32) / 255)[None, None]
    rows = {m: {k: {"psnr": [], "ssim": []} for k in ("bicubic", "edsr")} for m in ("clean", "degraded", "train_deg")}
    ms = []
    for i, r in enumerate(recs):
        gt, _ = read_thermal(r["path"])
        h, w = gt.shape
        gt = gt[: h // 2 * 2, : w // 2 * 2]
        h, w = gt.shape
        lrs = {"clean": cv2.resize(gt, (w // 2, h // 2), interpolation=cv2.INTER_AREA), "degraded": degrade_lr(gt, rng)}
        for mode, lr in lrs.items():
            bic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
            t0 = time.perf_counter()
            out = edsr(lr)
            ms.append((time.perf_counter() - t0) * 1000)
            for k, img in (("bicubic", bic), ("edsr", out)):
                rows[mode][k]["psnr"].append(psnr(t(img), t(gt)).item())
                rows[mode][k]["ssim"].append(ssim(t(img), t(gt)).item())
        bic = degrade_for_sr(gt, 2, rng)
        for k, img in (("bicubic", bic), ("edsr", restore(bic))):
            rows["train_deg"][k]["psnr"].append(psnr(t(img), t(gt)).item())
            rows["train_deg"][k]["ssim"].append(ssim(t(img), t(gt)).item())
        print(f"{i + 1}/{len(recs)} {r['name']}: clean bic {rows['clean']['bicubic']['psnr'][-1]:.2f} "
              f"edsr {rows['clean']['edsr']['psnr'][-1]:.2f} | degraded bic {rows['degraded']['bicubic']['psnr'][-1]:.2f} "
              f"edsr {rows['degraded']['edsr']['psnr'][-1]:.2f} | train_deg bic {rows['train_deg']['bicubic']['psnr'][-1]:.2f} "
              f"edsr {rows['train_deg']['edsr']['psnr'][-1]:.2f}", flush=True)
    res = {m: {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in md.items()} for m, md in rows.items()}
    for m, md in rows.items():
        res[m]["edsr_wins_psnr"] = int(sum(e > b for e, b in zip(md["edsr"]["psnr"], md["bicubic"]["psnr"])))
    res.update(frames=len(recs), frame_size=f"{w}x{h}", weights=str(edsr.path), seed=0,
               edsr_ms_per_frame_cpu=float(np.mean(ms)), flights=sorted({r["flight"] for r in recs}))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
