"""Phase 0 latency baseline: ms per pipeline stage, CPU vs CUDA, 256x256 crops and full frames.

    python scripts/benchmark_baseline.py                  # 10 POP test frames, writes results/baseline_timing.json
    python scripts/benchmark_baseline.py --images 3 --devices cpu

Uses POP test frames when POP/ exists, else seeded random frames (latency only).
Quality metrics come from the existing tools:
    python -m infravision.evaluate --skip grid timing     # SR PSNR/SSIM vs bicubic, colorizer
    python -m infravision.train.yolo eval                 # detection mAP@0.5 on POP test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infravision.config import ROOT  # noqa: E402
from infravision.data.pop import pop_splits  # noqa: E402
from infravision.io import read_thermal  # noqa: E402
from infravision.pipeline import Pipeline  # noqa: E402


def frames(n, seed=0):
    """n full-size uint8 thermal frames and their centre 256x256 crops."""
    try:
        test = pop_splits(existing_only=True)["test"]
    except FileNotFoundError:
        test = []
    if test:
        full = [read_thermal(r["path"])[0] for r in test[:: max(1, len(test) // n)][:n]]
        source = "POP test"
    else:
        rng = np.random.default_rng(seed)
        full = [rng.integers(0, 256, (1024, 1280), dtype=np.uint8) for _ in range(n)]
        source = "synthetic (POP not found)"
    crops = [f[(f.shape[0] - 256) // 2:][:256, (f.shape[1] - 256) // 2:][:, :256] for f in full]
    return {"256x256": crops, "full": full}, source


def bench(device, imgs):
    pipe = Pipeline(overrides={"device": device})
    pipe.run(imgs[0])   # warm-up (cuDNN autotune, lazy init)
    runs = [pipe.run(im)["timings"] for im in imgs]
    pipe.close()
    return {k: round(1000 * float(np.mean([r[k] for r in runs])), 1) for k in runs[0]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=int, default=10)
    p.add_argument("--devices", nargs="*", default=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
    p.add_argument("--out", default="results/baseline_timing.json")
    args = p.parse_args(argv)
    torch.manual_seed(0)

    sets, source = frames(args.images)
    report = dict(source=source, images=args.images, unit="ms (mean per image)",
                  cpu_threads=torch.get_num_threads(),
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  models=None, timings={})
    for dev in args.devices:
        for size, imgs in sets.items():
            print(f"{dev} {size} ...", flush=True)
            report["timings"][f"{dev}/{size}"] = bench(dev, imgs)
    report["models"] = Pipeline(overrides={"device": "cpu"}).describe()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["timings"], indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
