"""Person detector on POP: train, evaluate, tune thresholds, check small people.

    python -m infravision.train.yolo train --epochs 60 --imgsz 1024        # GPU
    python -m infravision.train.yolo train --smoke --device cpu             # 2 tiny epochs
    python -m infravision.train.yolo eval  --split val                      # mAP@0.5, P, R
    python -m infravision.train.yolo tune  --split val                      # conf + NMS IoU
    python -m infravision.train.yolo small --split val --sizes 1024 1280 --tiles 0 2

`train` converts POP (COCO json) to YOLO labels under human_detection/datasets/pop
(the same folder convert_pop.py writes) if needed, trains a single-class
"person" YOLO11s and copies best.pt to checkpoints/yolo/best.pt.
`eval`/`tune`/`small` run the pipeline's own Detector (so tiling is covered)
and match boxes at IoU 0.5. `tune` writes checkpoints/yolo/thresholds.json,
which the pipeline config picks up automatically.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from ..config import ROOT, load_config, pick_device
from ..data.pop import POP_ROOT, pop_splits
from ..io import read_thermal
from .common import CHECKPOINTS

YOLO_DS = ROOT / "human_detection" / "datasets" / "pop"
SIZE_BINS = [(0, 16), (16, 32), (32, 64), (64, 1e9)]


# ---------------------------------------------------------------- data

def write_yolo_dataset(pop_root, out, max_per_split=0, copy=False):
    """POP COCO -> YOLO layout (images linked or copied, labels as txt)."""
    out = Path(out)
    splits = pop_splits(pop_root, existing_only=True)
    for split, recs in splits.items():
        if max_per_split:
            recs = recs[:: max(1, len(recs) // max_per_split)][:max_per_split]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in recs:
            dst = out / "images" / split / r["name"]
            if not dst.exists():
                try:
                    if copy:
                        raise OSError
                    dst.hardlink_to(r["path"])
                except OSError:
                    shutil.copy2(r["path"], dst)
            W, H = r["width"], r["height"]
            lines = [f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                     for x1, y1, x2, y2 in r["boxes"]]
            (out / "labels" / split / (Path(r["name"]).stem + ".txt")).write_text("\n".join(lines))
        print(f"{split}: {len(recs)} images")
    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
                         "names:\n  0: person\n")
    return yaml_path


# ---------------------------------------------------------------- matching

def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred, scores, gt, thr=0.5):
    """Greedy by score. -> (tp flags per prediction, matched flags per gt)."""
    order = np.argsort(-scores)
    pred, scores = pred[order], scores[order]
    ious = iou_matrix(pred, gt)
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(pred), bool)
    for i in range(len(pred)):
        if not len(gt):
            break
        cand = np.where(~used & (ious[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(ious[i, cand])]
            used[j] = tp[i] = True
    return scores, tp, used


def average_precision(scores, tps, n_gt):
    """All-point interpolated AP (VOC 2010+/COCO style at one IoU)."""
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-scores)
    tp = tps[order].astype(np.float64)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(records, detect, conf_grid=None, iou_thr=0.5):
    """Run detect(gray) on every record. Returns metrics and per-image raw results."""
    all_s, all_tp, raw = [], [], []
    n_gt = 0
    t = time.perf_counter()
    for k, r in enumerate(records, 1):
        gray, _ = read_thermal(r["path"])
        d = detect(gray)
        s, tp, used = match(d["boxes"], d["scores"], r["boxes"], iou_thr)
        all_s.append(s)
        all_tp.append(tp)
        n_gt += len(r["boxes"])
        raw.append(dict(scores=s, tp=tp, gt=r["boxes"], gt_matched_at=_gt_best_score(d, r["boxes"], iou_thr)))
        if k % 50 == 0:
            print(f"  {k}/{len(records)} images", flush=True)
    secs = (time.perf_counter() - t) / max(len(records), 1)
    s, tp = np.concatenate(all_s), np.concatenate(all_tp)
    out = dict(images=len(records), people=n_gt, ap50=average_precision(s, tp, n_gt), sec_per_image=secs)
    out["at_conf"] = pr_at(s, tp, n_gt, conf_grid or [0.25])
    return out, raw


def _gt_best_score(d, gt, thr):
    """For each gt box, the highest score of a prediction overlapping it at >= thr (0 if none)."""
    ious = iou_matrix(gt, d["boxes"])
    best = np.zeros(len(gt), np.float32)
    for j in range(len(gt)):
        hit = ious[j] >= thr
        if hit.any():
            best[j] = d["scores"][hit].max()
    return best


def pr_at(scores, tps, n_gt, confs):
    rows = []
    for c in confs:
        keep = scores >= c
        tp = int(tps[keep].sum())
        fp = int(keep.sum() - tp)
        p = tp / max(tp + fp, 1)
        r = tp / max(n_gt, 1)
        rows.append(dict(conf=round(float(c), 3), precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9),
                         tp=tp, fp=fp, fn=n_gt - tp))
    return rows


def recall_by_size(raw, conf):
    out = {}
    for lo, hi in SIZE_BINS:
        hit = tot = 0
        for r in raw:
            g = r["gt"]
            if not len(g):
                continue
            side = np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))
            m = (side >= lo) & (side < hi)
            tot += int(m.sum())
            hit += int((r["gt_matched_at"][m] >= conf).sum())
        out[f"{int(lo)}-{'inf' if hi > 1e8 else int(hi)}px"] = dict(people=tot, recall=hit / tot if tot else None)
    return out


def make_detector(args, **over):
    from ..detect import Detector

    det = load_config(args.config)["detection"]
    det.update({k: v for k, v in over.items() if v is not None})
    if args.weights:
        det["weights"] = [args.weights]
    return Detector(device=args.device, **det)


def records_for(args):
    recs = pop_splits(args.pop, existing_only=True)[args.split]
    if args.max_images:
        recs = recs[:: max(1, len(recs) // args.max_images)][: args.max_images]
    if not recs:
        raise SystemExit(f"No POP {args.split} images found under {args.pop}")
    return recs


# ---------------------------------------------------------------- commands

def _keep_good_copy(trainer):
    """After Ultralytics saves last.pt, keep a complete copy (a disconnect can cut last.pt mid-write)."""
    last = Path(trainer.last)
    if last.exists():
        tmp = last.with_name("last_good.pt.tmp")
        shutil.copy2(last, tmp)
        os.replace(tmp, last.with_name("last_good.pt"))


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None


def cmd_train(args):
    from ultralytics import YOLO

    if args.smoke:
        args.epochs, args.imgsz, args.batch, args.max_per_split = 2, 320, 2, 8
        data_root = CHECKPOINTS / "smoke" / "yolo_data"
    else:
        data_root = Path(args.data_root)
    yaml_path = data_root / "data.yaml"
    if args.smoke or not yaml_path.exists() or args.rebuild:
        yaml_path = write_yolo_dataset(args.pop, data_root, args.max_per_split)
    device = pick_device(args.device)
    project = Path(args.project or CHECKPOINTS / ("smoke" if args.smoke else "yolo_runs"))
    name = "yolo_run" if args.smoke else args.name
    last = project / name / "weights" / "last.pt"
    good = last.with_name("last_good.pt")
    ck = _load(last) if args.resume and last.exists() else None
    if args.resume and last.exists() and ck is None and good.exists():
        print(f"{last.name} is damaged (cut off mid-write); using {good.name}")
        shutil.copy2(good, last)
        ck = _load(last)
    # Ultralytics strips the optimizer from last.pt when a run finishes.
    finished = ck is not None and (ck.get("optimizer") is None or ck.get("epoch", -1) == -1)
    if finished:
        print(f"{last.parent.parent.name} already finished; keeping its best.pt")
        best = last.with_name("best.pt")
    elif ck is not None:
        # Ultralytics saves last.pt after every epoch; resume=True continues with the same settings.
        print("resuming YOLO from", last)
        model = YOLO(str(last))
        model.add_callback("on_model_save", _keep_good_copy)
        model.train(resume=True)
        best = Path(model.trainer.best)
    else:
        if args.resume:
            print(f"--resume: no {last} yet, starting a new run")
        model = YOLO(args.start)
        model.add_callback("on_model_save", _keep_good_copy)
        model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    patience=args.patience, workers=args.workers, single_cls=True, seed=args.seed,
                    device=0 if device.type == "cuda" else "cpu", amp=device.type == "cuda",
                    project=str(project), name=name, exist_ok=True, plots=not args.smoke)
        best = Path(model.trainer.best)
    out = Path(args.out or CHECKPOINTS / ("smoke/yolo" if args.smoke else "yolo"))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out / "best.pt")
    (out / "DONE").write_text(str(best))
    print("Best weights ->", out / "best.pt")


def cmd_eval(args):
    # Low conf so AP covers the whole precision/recall curve; P/R are then read
    # off at the configured threshold (and a few others) from the same run.
    det = make_detector(args, imgsz=args.imgsz, tile=args.tile, conf=0.001)
    print("Detector:", det.name)
    conf = load_config(args.config)["detection"]["conf"]
    m, raw = evaluate(records_for(args), det, [conf, 0.1, 0.25, 0.4, 0.5])
    m["recall_by_size"] = recall_by_size(raw, conf)
    m.update(detector=det.name, split=args.split)
    print(json.dumps(m, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(m, indent=2))


def cmd_tune(args):
    recs = records_for(args)
    best, table = None, []
    for iou in args.ious:
        det = make_detector(args, conf=0.01, iou=iou, imgsz=args.imgsz, tile=args.tile)
        print(f"NMS IoU {iou}: running {len(recs)} images ...")
        m, _ = evaluate(recs, det, list(np.round(np.arange(0.05, 0.80, 0.025), 3)))
        for row in m["at_conf"]:
            row = dict(iou=iou, ap50=m["ap50"], **row)
            table.append(row)
            if best is None or row["f1"] > best["f1"]:
                best = row
    res = dict(conf=best["conf"], iou=best["iou"], selected_by="max F1 at IoU 0.5 match", split=args.split,
               images=len(recs), precision=best["precision"], recall=best["recall"], f1=best["f1"],
               ap50=best["ap50"], weights=str(det.path), imgsz=args.imgsz, tile=args.tile)
    out = Path(args.out) / "thresholds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    (Path(args.out) / "threshold_sweep.json").write_text(json.dumps(table, indent=2))
    print(json.dumps(res, indent=2), "\n->", out)
    if Path(det.path).parent != out.parent:
        print(f"NOTE: the pipeline reads thresholds.json next to the weights ({det.path.parent}); "
              f"copy it there or pass --out {det.path.parent}")


def cmd_small(args):
    recs = records_for(args)
    conf = load_config(args.config)["detection"]["conf"]
    rows = []
    for size in args.sizes:
        for tile in args.tiles:
            det = make_detector(args, imgsz=size, tile=tile)
            det.conf = 0.001
            m, raw = evaluate(recs, det, [conf])
            rows.append(dict(imgsz=size, tile=tile, ap50=m["ap50"], **m["at_conf"][0],
                             sec_per_image=m["sec_per_image"], recall_by_size=recall_by_size(raw, conf)))
            print(json.dumps(rows[-1], indent=2), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--pop", default=str(POP_ROOT))
        sp.add_argument("--device", default="auto")
        sp.add_argument("--config", default=None)
        sp.add_argument("--weights", default="", help="YOLO .pt (default: first in the config list)")
        sp.add_argument("--split", default="val", choices=["train", "val", "test"])
        sp.add_argument("--max_images", type=int, default=0, help="evaluate a spread-out subset")
        sp.add_argument("--imgsz", type=int, default=None)
        sp.add_argument("--tile", type=int, default=None)
        sp.add_argument("--json", default="")
        return sp

    t = sub.add_parser("train")
    t.add_argument("--pop", default=str(POP_ROOT))
    t.add_argument("--data_root", default=str(YOLO_DS))
    t.add_argument("--rebuild", action="store_true")
    t.add_argument("--max_per_split", type=int, default=0)
    t.add_argument("--start", default="yolo11s.pt", help="starting weights (FLIR model or yolo11s.pt)")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--imgsz", type=int, default=1024)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    t.add_argument("--name", default="pop_1024")
    t.add_argument("--project", default=None,
                   help="folder for the Ultralytics run (default checkpoints/yolo_runs); put it on Drive to survive a disconnect")
    t.add_argument("--resume", action="store_true", help="continue <project>/<name>/weights/last.pt if it exists")
    t.add_argument("--out", default=None, help="default checkpoints/yolo (checkpoints/smoke/yolo with --smoke)")
    t.add_argument("--smoke", action="store_true")
    common(sub.add_parser("eval"))
    tu = common(sub.add_parser("tune"))
    tu.add_argument("--ious", type=float, nargs="+", default=[0.45, 0.5, 0.6, 0.7])
    tu.add_argument("--out", default=str(CHECKPOINTS / "yolo"))
    sm = common(sub.add_parser("small"))
    sm.add_argument("--sizes", type=int, nargs="+", default=[1024, 1280])
    sm.add_argument("--tiles", type=int, nargs="+", default=[0, 2])
    args = p.parse_args(argv)
    dict(train=cmd_train, eval=cmd_eval, tune=cmd_tune, small=cmd_small)[args.cmd](args)


if __name__ == "__main__":
    main()
