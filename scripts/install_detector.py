"""Install a Kaggle detector run next to the old weights, compare, and prepare the next run.

    .venv/bin/python scripts/install_detector.py ../datasets/outputs/sar_1024.zip --name sar_v2
    .venv/bin/python scripts/install_detector.py ../datasets/outputs/sar_v3.zip --name sar_v3 --sweep

1. Unpacks weights/best.pt -> checkpoints/yolo/<name>.pt (never overwrites an existing file),
   plus results.csv / args.yaml / sar_eval.json next to it.
2. Prints the notebook's before/after table (sar_eval.json) and copies it to
   results/detection_<name>_eval.json.
3. Writes human_detection/<name>_start/start.pt: the file to upload to Kaggle as the
   starting model of the next run (train_sar_v3_yolo_kaggle.ipynb).
4. --sweep (task C6, CPU, slow): the pipeline's own Detector (so tiling counts) at
   imgsz x tiles on a spread-out subset of each local test set, old vs new weights.
   -> results/detection_<name>_sweep.json. Pick the best setting for the dashboard from it.

The dashboard / config are not switched automatically: put checkpoints/yolo/<name>.pt first in
detection.weights once the numbers say it is better.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "human_detection" / "datasets"
# Local copies of the test sets (same splits as make_sar_zip.py).
TEST_SETS = {
    "pop_test": (DATA / "pop", "test"),
    "pop_flood": (DATA / "pop", "test_flood"),
    "hituav_test": (DATA / "hituav", "test"),
    "rgbt_val": (DATA / "rgbt", "val"),
}


def install(zip_path, name):
    dst = ROOT / "checkpoints" / "yolo"
    dst.mkdir(parents=True, exist_ok=True)
    pt = dst / f"{name}.pt"
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        best = next((n for n in names if n.endswith("weights/best.pt")), None)
        if best is None:
            sys.exit(f"{zip_path}: no weights/best.pt inside ({len(names)} entries)")
        if pt.exists():
            print(f"{pt} exists: not overwritten")
        else:
            pt.write_bytes(z.read(best))
            print("installed", pt)
        for extra in ("results.csv", "args.yaml", "sar_eval.json"):
            n = next((m for m in names if m.endswith(extra) and "/val/" not in m), None)
            if n:
                out = dst / f"{name}_{extra}"
                if not out.exists():
                    out.write_bytes(z.read(n))
    ev = dst / f"{name}_sar_eval.json"
    if ev.is_file():
        e = json.loads(ev.read_text())
        print(f"\n{'test set':20s} {'before':>8s} {'after':>8s}   mAP50 (notebook)")
        for k, v in e["after"].items():
            if "mAP50" in v:
                b = e["before"].get(k, {}).get("mAP50", float("nan"))
                print(f"{k:20s} {b:8.3f} {v['mAP50']:8.3f}")
        fb, fa = e["before"]["flood_empty_blobs"], e["after"]["flood_empty_blobs"]
        print(f"{'empty flood blobs':20s} {100 * fb['rate']:7.1f}% {100 * fa['rate']:7.1f}%   (gate < 2%)")
        (ROOT / "results").mkdir(exist_ok=True)
        shutil.copy2(ev, ROOT / "results" / f"detection_{name}_eval.json")
    start = ROOT / "human_detection" / f"{name}_start" / "start.pt"
    start.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pt, start)
    print(f"\nNext Kaggle run: upload {start} as its own dataset (it must be called start.pt).")
    return pt


def yolo_records(root, split, max_images):
    import cv2
    files = sorted(f for f in (root / "images" / split).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    files = files[:: max(1, len(files) // max_images)][:max_images] if max_images else files
    recs = []
    for f in files:
        h, w = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).shape
        rows = []
        lbl = root / "labels" / split / (f.stem + ".txt")
        if lbl.is_file():
            for line in lbl.read_text().split("\n"):
                p = line.split()
                if len(p) == 5:
                    xc, yc, bw, bh = (float(v) for v in p[1:])
                    rows.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
        recs.append(dict(path=str(f), name=f.name, boxes=np.array(rows, np.float32).reshape(-1, 4)))
    return recs


def sweep(weights_list, sizes, tiles, max_images, out):
    from infravision.config import load_config
    from infravision.detect import Detector
    from infravision.train.yolo import evaluate
    cfg = load_config()["detection"]
    rows = []
    for set_name, (root, split) in TEST_SETS.items():
        if not (root / "images" / split).is_dir():
            print("skip (not on this machine):", set_name)
            continue
        recs = yolo_records(root, split, max_images)
        for w in weights_list:
            for size in sizes:
                for tile in tiles:
                    det = Detector(**{**cfg, "weights": [str(w)], "imgsz": size, "tile": tile, "conf": 0.001,
                                      "device": "cpu"})
                    m, _ = evaluate(recs, det, [cfg.get("conf", 0.25)])
                    row = dict(set=set_name, weights=Path(w).name, imgsz=size, tile=tile, images=len(recs),
                               ap50=round(m["ap50"], 4), **{k: round(v, 4) if isinstance(v, float) else v
                                                           for k, v in m["at_conf"][0].items()},
                               sec_per_image=round(m["sec_per_image"], 2))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    Path(out).write_text(json.dumps(rows, indent=1))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zip")
    p.add_argument("--name", required=True, help="e.g. sar_v2 -> checkpoints/yolo/sar_v2.pt")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--compare", default=str(ROOT / "checkpoints" / "yolo" / "best.pt"), help="old weights")
    p.add_argument("--sizes", type=int, nargs="+", default=[1024])
    p.add_argument("--tiles", type=int, nargs="+", default=[0, 2])
    p.add_argument("--max_images", type=int, default=100, help="per test set (CPU time: ~1 h for the defaults)")
    a = p.parse_args()
    pt = install(Path(a.zip), a.name)
    if a.sweep:
        sweep([a.compare, pt], a.sizes, a.tiles, a.max_images, ROOT / "results" / f"detection_{a.name}_sweep.json")


if __name__ == "__main__":
    main()
