"""Flood shortcut check (C1 v2): how often is an EMPTY water blob detected as a person?

    .venv/bin/python scripts/eval_flood_decoys.py checkpoints/yolo/best.pt
    .venv/bin/python scripts/eval_flood_decoys.py best.pt --pop /kaggle/input/.../pop --device 0

Uses images/test_flood and flood_decoys.json written by scripts/prepare_flood_aug.py.
A decoy counts as detected when a prediction at conf >= 0.25 has IoU >= 0.1 with it or its
centre lies inside it. Gate: < 2% of decoys detected. Also reports the person recall at the
same threshold, so a model cannot pass by simply not firing near water.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def hit(pred, box, min_iou):
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    return iou(pred, box) >= min_iou or (box[0] <= cx <= box[2] and box[1] <= cy <= box[3])


def read_people(path, w, h):
    out = []
    if path.is_file():
        for line in path.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                xc, yc, bw, bh = (float(v) for v in p[1:])
                out.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def evaluate(weights, pop, imgsz=1024, conf=0.25, device="cpu", batch=8):
    from ultralytics import YOLO
    pop = Path(pop)
    decoys = {k.split("/", 1)[1]: v for k, v in json.loads((pop / "flood_decoys.json").read_text()).items()
              if k.startswith("test_flood/")}
    files = sorted(f for f in (pop / "images" / "test_flood").iterdir() if f.name in decoys)
    model = YOLO(str(weights))
    n_dec = n_dec_hit = n_ppl = n_ppl_hit = 0
    for i in range(0, len(files), batch):
        for f, r in zip(files[i:i + batch], model.predict([str(f) for f in files[i:i + batch]], imgsz=imgsz,
                                                           conf=conf, device=device, verbose=False)):
            preds = r.boxes.xyxy.cpu().numpy().tolist()
            h, w = r.orig_shape
            for d in decoys[f.name]:
                n_dec += 1
                n_dec_hit += any(hit(p, d, 0.1) for p in preds)
            for b in read_people(pop / "labels" / "test_flood" / (f.stem + ".txt"), w, h):
                n_ppl += 1
                n_ppl_hit += any(iou(p, b) >= 0.5 for p in preds)
    return dict(decoys=n_dec, decoys_detected=n_dec_hit, decoy_rate=n_dec_hit / max(1, n_dec),
                people=n_ppl, people_found=n_ppl_hit, person_recall=n_ppl_hit / max(1, n_ppl),
                images=len(files), conf=conf, imgsz=imgsz, gate_pass=n_dec_hit / max(1, n_dec) < 0.02)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights")
    p.add_argument("--pop", default=str(ROOT / "human_detection" / "datasets" / "pop"))
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="")
    a = p.parse_args()
    res = evaluate(a.weights, a.pop, a.imgsz, a.conf, a.device)
    res["weights"] = str(a.weights)
    print(json.dumps(res, indent=2))
    print(f"empty blobs detected: {100 * res['decoy_rate']:.1f}% (gate < 2%: {'PASS' if res['gate_pass'] else 'FAIL'}), "
          f"flooded people found: {100 * res['person_recall']:.1f}%")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    np.random.seed(0)
    main()
