"""Evaluation report: SR and colorizer metrics, a qualitative POP grid, CPU timings.

    python -m infravision.evaluate --out docs/results            # writes metrics.json + images
    python -m infravision.evaluate --out docs/results --grid 10 --timing 3

Detection mAP comes from `python -m infravision.train.yolo eval/small`
(see docs/RESULTS.md); this script covers the rest:

- SR: PSNR/SSIM on fixed POP val crops (val flights only), model vs bicubic.
- Colorizer: PSNR/SSIM on held-out pairs of data/paired_disaster (synthetic
  set, real LLVIP RGB), the trained model vs two baselines (grey thermal
  copied to 3 channels, and the false-colour map the demo falls back to).
- A grid of N POP test frames: thermal | fused output.
- CPU timings of the full pipeline per stage.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, short
from .data.paired import PairedColor, disaster_pairs
from .data.pop import POPSRCrops, pop_splits
from .enhance import COLORMAPS, SuperResolver, colorize_tensor, load_colorizer, find_weights
from .metrics import psnr, ssim


def eval_sr(cfg, n_val, crop, seed=0):
    recs = pop_splits(existing_only=True)["val"]
    recs = recs[:: max(1, len(recs) // n_val)][:n_val]
    scale = cfg["sr"]["scale"]
    ds = POPSRCrops(recs, crop, scale, 1, train=False, seed=seed)
    sr = SuperResolver(device="cpu", **{**cfg["sr"], "scale": 1})   # input is already bicubic-upscaled
    rows = dict(bicubic=[[], []], model=[[], []])
    for x, y in DataLoader(ds, batch_size=1):
        inp = (x[0, 0].numpy() * 255).round().astype(np.uint8)
        out = sr(inp) if sr.model is not None else inp
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255)[None, None]
        for k, img in (("bicubic", inp), ("model", out)):
            rows[k][0].append(psnr(to_t(img), y).item())
            rows[k][1].append(ssim(to_t(img), y).item())
    res = {k: dict(psnr=float(np.mean(v[0])), ssim=float(np.mean(v[1]))) for k, v in rows.items()}
    res.update(weights=short(sr.path) if sr.path else None, crops=len(ds), crop=crop, scale=scale,
               val_flights=sorted({r["flight"] for r in recs}))
    if sr.path is None:
        res.pop("model")
    return res


def eval_thermovision(cfg, pairs, split):
    """ThermoVision U-Net on whole frames (as trained): raw thermal and RGB at input_size x input_size."""
    from .colorizer import ColorizerAdapter

    c = cfg["colorizer"]
    path = find_weights(c["checkpoint"])
    if path is None:
        return None
    ad = ColorizerAdapter(path, "cpu", c["input_size"])
    s = ad.input_size
    cmap = COLORMAPS.get(c["fallback_colormap"], cv2.COLORMAP_INFERNO)
    acc = {}
    to_t = lambda a: torch.from_numpy(a / 255.0).float().permute(2, 0, 1)[None]
    for tp, vp, _ in pairs:
        g = cv2.imdecode(np.fromfile(str(tp), np.uint8), cv2.IMREAD_GRAYSCALE)
        v = cv2.cvtColor(cv2.imdecode(np.fromfile(str(vp), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        tgt = to_t(cv2.resize(v, (s, s), interpolation=cv2.INTER_AREA))
        g = cv2.resize(g, (s, s), interpolation=cv2.INTER_AREA)
        preds = {"grey thermal (baseline)": np.repeat(g[..., None], 3, 2),
                 "false-colour map (baseline)": cv2.cvtColor(cv2.applyColorMap(g, cmap), cv2.COLOR_BGR2RGB),
                 "colorizer": ad.colorize(g)}
        for k, rgb in preds.items():
            a = acc.setdefault(k, [[], []])
            a[0].append(psnr(to_t(rgb), tgt).item())
            a[1].append(ssim(to_t(rgb), tgt).item())
    res = {k: dict(psnr=float(np.mean(v[0])), ssim=float(np.mean(v[1]))) for k, v in acc.items()}
    return dict(methods=res, pairs=len(pairs), split=split, synthetic_set=True, backend="thermovision",
                protocol=f"whole frame, raw thermal, {s}x{s}", weights=short(path))


def eval_colorizer(cfg, n_val, crop=256):
    pairs = disaster_pairs("test") or []
    split = "test"
    if not pairs:
        pairs, split = disaster_pairs("val"), "val"
    pairs = pairs[:: max(1, len(pairs) // n_val)][:n_val]
    if cfg["colorizer"].get("backend") == "thermovision":
        res = eval_thermovision(cfg, pairs, split)
        if res is not None:
            return res
    work_side = cfg["colorizer"]["work_side"]
    ds = PairedColor(pairs, crop, work_side, train=False, pre=cfg["pre_enhance"])
    path = find_weights(cfg["colorizer"]["weights"])
    model = load_colorizer(path, "cpu")[0] if path else None
    cmap = COLORMAPS.get(cfg["colorizer"]["fallback_colormap"], cv2.COLORMAP_INFERNO)
    acc = {}

    def add(name, pred, tgt):
        a = acc.setdefault(name, [[], []])
        a[0].append(psnr(pred, tgt).item())
        a[1].append(ssim(pred, tgt).item())

    for t, v in DataLoader(ds, batch_size=1):
        tgt = (v + 1) / 2
        g = ((t[0, 0].numpy() + 1) * 127.5).round().astype(np.uint8)
        add("grey thermal (baseline)", torch.from_numpy(g / 255.0).float()[None, None].repeat(1, 3, 1, 1), tgt)
        fc = cv2.cvtColor(cv2.applyColorMap(g, cmap), cv2.COLOR_BGR2RGB)
        add("false-colour map (baseline)", torch.from_numpy(fc / 255.0).float().permute(2, 0, 1)[None], tgt)
        if model is not None:
            rgb = colorize_tensor(model, g, "cpu")
            add("colorizer", torch.from_numpy(rgb / 255.0).float().permute(2, 0, 1)[None], tgt)
    res = {k: dict(psnr=float(np.mean(v[0])), ssim=float(np.mean(v[1]))) for k, v in acc.items()}
    return dict(methods=res, pairs=len(ds), split=split, synthetic_set=True,
                weights=short(path) if path else None)


def make_grid(pipe, recs, out, width=1600):
    """Rows of thermal (ground-truth people in cyan) | fused output (model boxes)."""
    rows = []
    for rec in recs:
        f = rec["path"]
        r = pipe.run(f)
        t = cv2.cvtColor(r["thermal"], cv2.COLOR_GRAY2RGB)
        for x1, y1, x2, y2 in rec["boxes"].astype(int):
            cv2.rectangle(t, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 255, 255), 3)
        fused = r["fused"]
        h = width // 2 * t.shape[0] // t.shape[1]
        row = np.hstack([cv2.resize(t, (width // 2, h), interpolation=cv2.INTER_AREA),
                         cv2.resize(fused, (width // 2, h), interpolation=cv2.INTER_AREA)])
        cv2.putText(row, f"{Path(f).name}  ground truth={len(rec['boxes'])}  detected={r['detections']['count']}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        rows.append(row)
    grid = np.vstack(rows)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return [Path(rec["path"]).name for rec in recs]


def timing(cfg_over, files, repeat):
    from .pipeline import Pipeline

    out = {}
    for label, over in cfg_over.items():
        pipe = Pipeline(overrides={"device": "cpu", **over})
        pipe.run(files[0])   # warm-up
        runs = [pipe.run(f)["timings"] for f in files[:repeat]]
        out[label] = {k: round(float(np.mean([r[k] for r in runs])), 2) for k in runs[0]}
        out[label]["images"] = len(runs)
        pipe.close()
    out["cpu_threads"] = torch.get_num_threads()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/results")
    p.add_argument("--sr_val", type=int, default=100)
    p.add_argument("--color_val", type=int, default=80)
    p.add_argument("--grid", type=int, default=10)
    p.add_argument("--timing", type=int, default=3)
    p.add_argument("--skip", nargs="*", default=[], choices=["sr", "colorizer", "grid", "timing"])
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(overrides={"device": "cpu"})
    report_path = out / "metrics.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    if "sr" not in args.skip:
        print("SR ...", flush=True)
        report["sr"] = eval_sr(cfg, args.sr_val, 192)
        print(json.dumps(report["sr"], indent=2))
        report_path.write_text(json.dumps(report, indent=2))
    if "colorizer" not in args.skip:
        print("colorizer ...", flush=True)
        report["colorizer"] = eval_colorizer(cfg, args.color_val)
        print(json.dumps(report["colorizer"], indent=2))
        report_path.write_text(json.dumps(report, indent=2))
    test = pop_splits(existing_only=True)["test"]
    picks = [r for r in test[:: max(1, len(test) // max(args.grid, 1))][: args.grid]]
    if "grid" not in args.skip and picks:
        from .pipeline import Pipeline

        print("grid ...", flush=True)
        pipe = Pipeline(cfg)
        report["grid"] = dict(file=str(out / "pop_test_grid.jpg"),
                              images=make_grid(pipe, picks, out / "pop_test_grid.jpg"),
                              models=pipe.describe())
        pipe.close()
        report_path.write_text(json.dumps(report, indent=2))
    if "timing" not in args.skip and picks:
        print("timing ...", flush=True)
        report["timing_cpu"] = timing({"full (2x SR)": {}, "fast (1x)": {"sr": {"scale": 1}}},
                                      [r["path"] for r in picks], args.timing)
        print(json.dumps(report["timing_cpu"], indent=2))
    report_path.write_text(json.dumps(report, indent=2))
    print("->", report_path)


if __name__ == "__main__":
    main()
