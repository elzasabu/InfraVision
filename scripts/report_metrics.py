"""Report numbers for InfraVision, computed by running the code on small fixed subsets.

    python scripts/report_metrics.py --pop_images <dir of POP test JPGs> --pop_ann <test.json>

Writes docs/report_metrics/{metrics.json, results_tables.tex, NOTES.md,
figures/infravision_output.png}. Every number comes from this run or from a
file it reads (annotation JSONs, manifest.csv, saved training args). Values
that cannot be computed are written as null / "n/a" with the reason.
"""
import argparse
import csv
import json
import platform
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infravision.config import load_config  # noqa: E402
from infravision.enhance import Colorizer, SuperResolver, pre_enhance  # noqa: E402
from infravision.fusion import rescale_boxes  # noqa: E402
from infravision.metrics import psnr, ssim  # noqa: E402
from infravision.pipeline import Pipeline  # noqa: E402

OUT = ROOT / "docs" / "report_metrics"
GROUPS = {"Clear": None, "Smoke/Fog": ("smoke", "fog"), "Dust": ("dust",), "Darkness": ("dark",)}


# ---------------------------------------------------------------- helpers

def t01(img):
    """uint8 HxW or HxWx3 -> 1xCxHxW float in [0, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
    return x[None, None] if x.ndim == 2 else x.permute(2, 0, 1)[None]


class Lpips:
    """LPIPS (AlexNet) when the lpips package and its weights are available, else None + reason."""

    def __init__(self):
        self.fn, self.reason = None, None
        try:
            import lpips
            self.fn = lpips.LPIPS(net="alex", verbose=False).eval()
        except Exception as e:  # noqa: BLE001 - any failure means "not computed"
            self.reason = f"LPIPS not computed: {type(e).__name__}: {str(e).splitlines()[0][:160]}"

    @torch.inference_mode()
    def __call__(self, a, b):
        if self.fn is None:
            return None
        x, y = t01(a), t01(b)
        if x.shape[1] == 1:
            x, y = x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1)
        return float(self.fn(x * 2 - 1, y * 2 - 1).mean())


def quality(pred, ref, lp):
    p, r = t01(pred), t01(ref)
    return dict(psnr=float(psnr(p, r)), ssim=float(ssim(p, r)), lpips=lp(pred, ref))


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def read_gray(p):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(p)
    return img


def read_rgb(p):
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- detection scoring

def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(boxes, scores, gt, thr=0.5):
    """Greedy by score: -> is_tp per detection (in score order), sorted scores."""
    order = np.argsort(-scores)
    boxes, scores = boxes[order], scores[order]
    ious = iou_matrix(boxes, gt)
    used, tp = set(), np.zeros(len(boxes), bool)
    for i in range(len(boxes)):
        if ious.shape[1] == 0:
            break
        cand = [(ious[i, j], j) for j in range(len(gt)) if j not in used and ious[i, j] >= thr]
        if cand:
            used.add(max(cand)[1]); tp[i] = True
    return tp, scores


def det_scores(per_image, n_gt_total, conf):
    """per_image: list of (is_tp, scores). -> precision/recall at `conf` and all-point AP@0.5."""
    tp = np.concatenate([t for t, _ in per_image]) if per_image else np.zeros(0, bool)
    sc = np.concatenate([s for _, s in per_image]) if per_image else np.zeros(0)
    keep = sc >= conf
    ntp, nfp = int(tp[keep].sum()), int((~tp[keep]).sum())
    order = np.argsort(-sc)
    ctp, cfp = np.cumsum(tp[order]), np.cumsum(~tp[order])
    rec = ctp / max(1, n_gt_total)
    prec = ctp / np.maximum(1, ctp + cfp)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return dict(precision=ntp / max(1, ntp + nfp), recall=ntp / max(1, n_gt_total), map50=ap,
                tp=ntp, fp=nfp, fn=n_gt_total - ntp, gt=n_gt_total)


# ---------------------------------------------------------------- sections

def section_datasets(a):
    out = {}
    pop = {}
    for split in ("train", "val", "test"):
        p = Path(a.pop_ann_dir) / split / "Annotations" / f"{split}.json"
        if p.exists():
            d = json.load(open(p))
            pop[split] = dict(images=len(d["images"]), persons=len(d["annotations"]))
    pop["total"] = dict(images=sum(v["images"] for v in pop.values()),
                        persons=sum(v["persons"] for v in pop.values()))
    out["POP"] = dict(splits=pop, role="Target domain: drone thermal SAR; SR and detection evaluation",
                      source="POP/{split}/Annotations/{split}.json")

    flir = {}
    for split in ("train", "test"):
        p = Path(a.flir_ann) / f"{split}.json"
        if p.exists():
            d = json.load(open(p))
            pid = [c["id"] for c in d["categories"] if c["name"] == "person"][0]
            flir[split] = dict(images=len(d["images"]),
                               persons=sum(1 for x in d["annotations"] if x["category_id"] == pid))
    flir["total"] = dict(images=sum(v["images"] for v in flir.values()),
                         persons=sum(v["persons"] for v in flir.values()))
    out["FLIR"] = dict(splits=flir, role="Detector pre-training (thermal street scenes)",
                       source="flir_align/coco_annotations/{train,test}.json (no val split in this copy)")

    rows = list(csv.DictReader(open(a.manifest, newline="")))
    pairs = defaultdict(set)
    for r in rows:
        pairs[r["split"]].add(r["id"])
    llvip = {s: dict(pairs=len(v)) for s, v in pairs.items()}
    llvip["total"] = dict(pairs=sum(len(v) for v in pairs.values()))
    out["LLVIP"] = dict(splits=llvip, role="Aligned thermal/RGB pairs: colorizer training and test",
                        source="data/paired_disaster/manifest.csv (pairs used; raw LLVIP folder not on this machine)")
    cond = Counter(r["condition"] for r in rows)
    sev = Counter(r["severity"] for r in rows)
    cond_split = defaultdict(Counter)
    for r in rows:
        cond_split[r["condition"]][r["split"]] += 1
    out["synthetic"] = dict(degraded_images=len(rows), per_condition=dict(cond),
                            per_condition_split={k: dict(v) for k, v in cond_split.items()},
                            per_severity=dict(sev), source=str(a.manifest))
    return out


def section_training(a, sr, col, det):
    out = {}
    # EDSR: constants of the script that wrote the checkpoint (SR/train_ssim.py).
    src = (ROOT / "SR" / "train_ssim.py").read_text()
    const = {k: re.search(rf"^{k}\s*=\s*(.+)$", src, re.M).group(1).strip()
             for k in ("EPOCHS", "BATCH_SIZE", "LEARNING_RATE")}
    out["EDSR"] = dict(
        checkpoint=str(sr.path.relative_to(ROOT)) if sr.path else None,
        input_size="not recorded", epochs=int(const["EPOCHS"]), batch=int(const["BATCH_SIZE"]),
        optimizer="Adam", lr=float(const["LEARNING_RATE"]), loss="0.9 L1 + 0.1 (1 - SSIM)",
        device="not recorded (script uses CUDA if available, else CPU)",
        data="not POP (older SR/checkpoints run; training data not recorded)",
        status="preliminary", source="SR/train_ssim.py constants")

    m = ROOT / "checkpoints" / "colorizer" / "metrics.json"
    if col.path and m.exists():
        d = json.load(open(m))
        c = d.get("config", {})
        loss = "L1"
        if c.get("perceptual_weight"):
            loss += f" + {c['perceptual_weight']} VGG perceptual"
        if c.get("ssim_weight"):
            loss += f" + {c['ssim_weight']} (1 - SSIM)"
        if c.get("adv_weight"):
            loss += f" + {c['adv_weight']} adversarial"
        out["Colorizer"] = dict(
            checkpoint=str(col.path.relative_to(ROOT)),
            input_size=f"{c.get('crop')} px crops (frames resized to {c.get('work_side')} px)",
            epochs=c.get("epochs"), batch=c.get("batch"), optimizer="Adam (betas 0.5, 0.999), cosine LR",
            lr=c.get("lr"), loss=loss, device=c.get("device"),
            data=f"{d.get('train_pairs', a.col_train_pairs)} train / {d.get('val_pairs', a.col_val_pairs)} val LLVIP pairs",
            minutes=d.get("minutes"), best=d.get("best"),
            status=f"preliminary ({c.get('epochs')} epochs, {a.col_train_pairs} images)",
            source=str(m.relative_to(ROOT)))
    else:
        out["Colorizer"] = dict(status="missing", note="no colorizer weights; pipeline uses a false-colour map")

    args_yaml = Path(a.yolo_args)
    y = {}
    for line in args_yaml.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            y[k.strip()] = v.strip()
    res = list(csv.DictReader(open(args_yaml.parent / "results.csv")))
    last = {k.strip(): v for k, v in res[-1].items()}
    out["YOLO"] = dict(
        checkpoint=str(det.path.relative_to(ROOT)), model=y.get("model"),
        input_size=int(y["imgsz"]), epochs=int(y["epochs"]), epochs_run=len(res), batch=int(y["batch"]),
        optimizer=y.get("optimizer"), lr=float(y["lr0"]), loss="Ultralytics default (box CIoU + cls BCE + DFL)",
        device=f"GPU (device={y.get('device')}, Colab)", data="FLIR ADAS thermal (person), not POP",
        flir_val_last=dict(precision=float(last["metrics/precision(B)"]), recall=float(last["metrics/recall(B)"]),
                           map50=float(last["metrics/mAP50(B)"])),
        status="fully trained on FLIR; not fine-tuned on POP", source=str(args_yaml))
    return out


def section_sr(a, files, sr, lp):
    rows = []
    for p in files:
        hr = read_gray(p)
        h, w = hr.shape
        h, w = h - h % sr.scale, w - w % sr.scale
        hr = hr[:h, :w]
        lr = cv2.resize(hr, (w // sr.scale, h // sr.scale), interpolation=cv2.INTER_AREA)
        bic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        out = sr(lr)
        rows.append(dict(file=p.name, bicubic=quality(bic, hr, lp), edsr=quality(out, hr, lp)))
        print(f"  SR {len(rows)}/{len(files)} {p.name}", flush=True)
    res = {}
    for k in ("bicubic", "edsr"):
        res[k] = {m: mean_of([r[k] for r in rows], m) for m in ("psnr", "ssim", "lpips")}
    return dict(n=len(rows), scale=sr.scale, downscale="cv2.INTER_AREA", reference="original POP thermal",
                metric_space="8-bit grey", means=res, per_image=rows)


def section_color(a, col, lp):
    rows = list(csv.DictReader(open(Path(a.paired) / "manifest.csv", newline="")))
    test = [r for r in rows if r["split"] == "test"]
    ids = sorted({r["id"] for r in test})
    random.Random(a.seed).shuffle(ids)
    ids = sorted(ids[: a.n_color])
    by_id = defaultdict(list)
    for r in test:
        if r["id"] in ids:
            by_id[r["id"]].append(r)
    cfg = load_config()["pre_enhance"]
    root = Path(a.paired)
    pair_rows, deg_rows = [], []
    side = col.work_side
    for i, pid in enumerate(ids):
        r0 = by_id[pid][0]
        th = read_gray(root / r0["thermal"].replace("\\", "/"))
        rgb = read_rgb(root / r0["rgb"].replace("\\", "/"))
        h, w = th.shape
        s = side / max(h, w)
        size = (round(w * s), round(h * s))
        th_s = cv2.resize(pre_enhance(th, **cfg), size, interpolation=cv2.INTER_AREA)
        ref = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        pred = col(th_s)
        fc = cv2.cvtColor(cv2.applyColorMap(th_s, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        pair_rows.append(dict(id=pid, colorizer=quality(pred, ref, lp), false_colour=quality(fc, ref, lp)))
        for r in by_id[pid]:
            deg = cv2.resize(read_rgb(root / r["rgb_degraded"].replace("\\", "/")), size,
                             interpolation=cv2.INTER_AREA)
            deg_rows.append(dict(id=pid, condition=r["condition"], severity=r["severity"],
                                 colorizer=pair_rows[-1]["colorizer"], degraded_visible=quality(deg, ref, lp)))
        print(f"  colour {i + 1}/{len(ids)} {pid}", flush=True)

    def agg(rs, key):
        return {m: mean_of([r[key] for r in rs], m) for m in ("psnr", "ssim", "lpips")}

    groups = {}
    for g, conds in GROUPS.items():
        if conds is None:
            groups[g] = dict(n_pairs=len(pair_rows), n_images=len(pair_rows), colorizer=agg(pair_rows, "colorizer"),
                             false_colour=agg(pair_rows, "false_colour"), degraded_visible=None)
        else:
            rs = [r for r in deg_rows if r["condition"] in conds]
            groups[g] = dict(n_pairs=len({r["id"] for r in rs}), n_images=len(rs), colorizer=agg(rs, "colorizer"),
                             degraded_visible=agg(rs, "degraded_visible"))
    combos = {}
    for c in sorted({r["condition"] for r in deg_rows}):
        rs = [r for r in deg_rows if r["condition"] == c]
        combos[c] = dict(n_images=len(rs), colorizer=agg(rs, "colorizer"), degraded_visible=agg(rs, "degraded_visible"))
    return dict(n_pairs=len(pair_rows), n_degraded=len(deg_rows), resolution=f"{side} px longer side",
                reference="clean RGB of the pair", groups=groups, per_condition=combos,
                note="The colorizer input is thermal, which the synthetic conditions do not change; "
                     "per-condition colorizer rows differ only by which pairs carry that condition.",
                ids=ids)


def load_gt(ann_path):
    d = json.load(open(ann_path))
    by_img = defaultdict(list)
    for x in d["annotations"]:
        bx, by, bw, bh = x["bbox"]
        by_img[x["image_id"]].append([bx, by, bx + bw, by + bh])
    return {im["file_name"]: np.array(by_img[im["id"]], np.float32).reshape(-1, 4) for im in d["images"]}


def run_detector(det, img_bgr, conf):
    keep = det.conf
    det.conf = conf
    try:
        return det._predict(np.ascontiguousarray(img_bgr), det.imgsz)
    finally:
        det.conf = keep


# ---------------------------------------------------------------- figure

def image_row(panels, labels, panel_w=3.2, gap=0.06, label_size=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h0, w0 = panels[0].shape[:2]
    ph, label_h = panel_w * h0 / w0, 0.28
    fw = len(panels) * panel_w + (len(panels) - 1) * gap
    fig = plt.figure(figsize=(fw, ph + label_h))
    for i, (img, lab) in enumerate(zip(panels, labels)):
        ax = fig.add_axes([i * (panel_w + gap) / fw, label_h / (ph + label_h), panel_w / fw, ph / (ph + label_h)])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255, interpolation="lanczos")
        ax.set_axis_off()
        ax.text(0.5, -0.03, lab, transform=ax.transAxes, ha="center", va="top", fontsize=label_size)
    return fig


# ---------------------------------------------------------------- tables

def f(v, d):
    return "n/a" if v is None else f"{v:.{d}f}"


def write_tex(M, path):
    D, T, S, C, E, R = (M[k] for k in ("datasets", "training", "sr", "colorization", "detection", "runtime"))
    L = ["% Generated by scripts/report_metrics.py. Requires \\usepackage{booktabs}.", ""]

    def sp(d, k, key):
        v = d["splits"].get(k)
        return "--" if v is None else f"{v[key]:,}".replace(",", "{,}")

    L += [r"\begin{table}[t]", r"\centering", r"\caption{Datasets and their roles}", r"\label{tab:datasets}",
          r"\small", r"\begin{tabular}{llrrrr}", r"\toprule",
          r"Dataset & Role & Total & Train & Val & Test \\", r"\midrule"]
    L.append(f"POP (images) & {D['POP']['role']} & {sp(D['POP'], 'total', 'images')} & {sp(D['POP'], 'train', 'images')} & "
             f"{sp(D['POP'], 'val', 'images')} & {sp(D['POP'], 'test', 'images')} \\\\")
    L.append(f"FLIR (images) & {D['FLIR']['role']} & {sp(D['FLIR'], 'total', 'images')} & {sp(D['FLIR'], 'train', 'images')} & "
             f"-- & {sp(D['FLIR'], 'test', 'images')} \\\\")
    L.append(f"LLVIP (pairs) & {D['LLVIP']['role']} & {sp(D['LLVIP'], 'total', 'pairs')} & {sp(D['LLVIP'], 'train', 'pairs')} & "
             f"{sp(D['LLVIP'], 'val', 'pairs')} & {sp(D['LLVIP'], 'test', 'pairs')} \\\\")
    syn = D["synthetic"]
    cs = syn["per_condition_split"]
    L.append(r"\midrule")
    for c in ("smoke", "fog", "dust", "dark", "smoke+dark", "dust+dark"):
        v = cs.get(c, {})
        n = lambda k: f"{v.get(k, 0):,}".replace(",", "{,}")  # noqa: E731
        L.append(f"Synthetic: {c} & Degraded RGB (LLVIP) & {n_tot(syn['per_condition'].get(c, 0))} & {n('train')} & {n('val')} & {n('test')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    L += [r"\begin{table}[t]", r"\centering", r"\caption{Training configuration of each model}", r"\label{tab:train}",
          r"\small", r"\begin{tabular}{lllllll}", r"\toprule",
          r"Model & Input & Epochs & Batch & Optimizer & LR & Loss / device \\", r"\midrule"]
    e = T["EDSR"]
    L.append(f"EDSR ($\\times$2) & {e['input_size']} & {e['epochs']} & {e['batch']} & {e['optimizer']} & {e['lr']:g} & "
             f"{e['loss']}; {e['device']} \\\\")
    c = T["Colorizer"]
    if c.get("status") != "missing":
        L.append(f"Colorizer & {c['input_size']} & {c['epochs']} & {c['batch']} & Adam & {c['lr']:g} & "
                 f"{c['loss']}; {c['device'].upper()} \\\\")
    else:
        L.append(r"Colorizer & \multicolumn{6}{l}{not trained (no weights)} \\")
    y = T["YOLO"]
    L.append(f"YOLO11s & {y['input_size']} & {y['epochs']} & {y['batch']} & {y['optimizer']} & {y['lr']:g} & "
             f"{y['loss']}; {y['device']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    L += [r"\begin{table}[t]", r"\centering", r"\caption{Super-resolution quality on the POP test split}",
          r"\label{tab:sr}", r"\begin{tabular}{lccc}", r"\toprule",
          r"Method & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ \\", r"\midrule"]
    for k, name in (("bicubic", "Bicubic"), ("edsr", "EDSR")):
        m = S["means"][k]
        L.append(f"{name} & {f(m['psnr'], 2)} & {f(m['ssim'], 3)} & {f(m['lpips'], 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    L += [r"\begin{table}[t]", r"\centering", r"\caption{Colorization quality on the synthetic paired test set}",
          r"\label{tab:color}", r"\small", r"\begin{tabular}{lrcccccc}", r"\toprule",
          r" & & \multicolumn{3}{c}{Colorized thermal} & \multicolumn{3}{c}{Degraded visible} \\",
          r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
          r"Condition & $n$ & PSNR & SSIM & LPIPS & PSNR & SSIM & LPIPS \\", r"\midrule"]
    for g, v in C["groups"].items():
        a_, b_ = v["colorizer"], v["degraded_visible"]
        right = "-- & -- & --" if b_ is None else f"{f(b_['psnr'], 2)} & {f(b_['ssim'], 3)} & {f(b_['lpips'], 2)}"
        L.append(f"{g} & {v['n_images']} & {f(a_['psnr'], 2)} & {f(a_['ssim'], 3)} & {f(a_['lpips'], 2)} & {right} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    L += [r"\begin{table}[t]", r"\centering", r"\caption{Human detection on the POP test split}", r"\label{tab:det}",
          r"\begin{tabular}{lrccc}", r"\toprule",
          r"Detector input & Images & Precision & Recall & mAP@0.5 \\", r"\midrule"]
    for k, name in (("thermal_subset", "Original thermal"), ("thermal_runtime", "Original thermal"),
                    ("colorized", "Colorized + enhanced")):
        v = E.get(k)
        if v:
            L.append(f"{name} & {v['images']} & {f(v['precision'], 3)} & {f(v['recall'], 3)} & {f(v['map50'], 3)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    L += [r"\begin{table}[t]", r"\centering", r"\caption{Mean per-image runtime}", r"\label{tab:runtime}",
          r"\begin{tabular}{lc}", r"\toprule", f"Stage ({R['device_label']}) & Seconds / image \\\\", r"\midrule"]
    for k, name in (("branch_a", "Branch A (enhance, SR, colorize)"), ("branch_b", "Branch B (YOLO)"),
                    ("sequential", "Full pipeline, sequential"), ("parallel", "Full pipeline, parallel")):
        L.append(f"{name} & {f(R['mean_s'][k], 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(L))


def n_tot(v):
    return f"{v:,}".replace(",", "{,}")


def write_notes(M, path):
    T, S, C, E, R = (M[k] for k in ("training", "sr", "colorization", "detection", "runtime"))
    col = T["Colorizer"]
    sb, se = S["means"]["bicubic"], S["means"]["edsr"]
    cl = C["groups"]["Clear"]
    ta, tb = E["thermal_runtime"], E["colorized"]
    txt = (
        "# Report metrics: notes\n\n"
        f"Models: the YOLO11s detector is fully trained ({T['YOLO']['epochs_run']} epochs) but on FLIR street scenes, "
        "not on POP; the EDSR checkpoint is preliminary (an older 10-epoch run on non-POP data); the colorizer is "
        f"preliminary ({col.get('epochs')} epochs on {col.get('data', 'n/a')}, CPU, {col.get('minutes')} min). "
        f"Subsets (seed {M['seed']}): SR on {S['n']} POP test images, colorization on {C['n_pairs']} synthetic test pairs "
        f"({C['n_degraded']} degraded images), detection on {E['thermal_subset']['images']} POP test images for the thermal "
        f"input and on {tb['images']} images for the colorized input (the same images as the runtime test), runtime on "
        f"{R['images_timed']} images after {R['warmup_skipped']} warm-up runs, on {R['device']}. "
        f"Surprising results: EDSR is worse than bicubic ({se['psnr']:.2f} vs {sb['psnr']:.2f} dB PSNR), and the detector "
        f"finds few POP people (recall {E['thermal_subset']['recall']:.3f}, mAP@0.5 {E['thermal_subset']['map50']:.3f} on "
        f"thermal; {tb['recall']:.3f} / {tb['map50']:.3f} on colorized vs {ta['recall']:.3f} / {ta['map50']:.3f} on thermal "
        "for the same images). "
        "Interpretation: these numbers measure untrained-on-target models, so they show the pipeline runs end to end but "
        "not what it can reach once EDSR, the colorizer and YOLO are trained on POP and the paired set on a GPU.\n"
    )
    path.write_text(txt)


# ---------------------------------------------------------------- main

def main():
    U = "/mnt/user-data/uploads/Infravision"
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop_images", default=f"{U}/JPEGImages")
    p.add_argument("--pop_ann_dir", default=f"{U}/POP", help="POP/{split}/Annotations/{split}.json")
    p.add_argument("--flir_ann", default=f"{U}/human_detection/datasets/flir_align/coco_annotations")
    p.add_argument("--manifest", default=f"{U}/data/paired_disaster/manifest.csv", help="full synthetic manifest")
    p.add_argument("--paired", default=str(ROOT / "data" / "paired_disaster"), help="local pairs used for D")
    p.add_argument("--yolo_args", default=f"{U}/human_detection/runs/thermal/colab_resumed/args.yaml")
    p.add_argument("--n_pop", type=int, default=100)
    p.add_argument("--n_sr", type=int, default=30, help="EDSR is ~13 s per 1280x1024 frame on CPU")
    p.add_argument("--n_color", type=int, default=100)
    p.add_argument("--n_runtime", type=int, default=13)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--col_train_pairs", type=int, default=75)
    p.add_argument("--col_val_pairs", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--from_json", action="store_true", help="only rewrite the .tex and NOTES.md from metrics.json")
    p.add_argument("--skip", default="", help="comma list of sections to skip (sr,color,det,runtime)")
    a = p.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    if a.from_json:
        M = json.load(open(OUT / "metrics.json"))
        write_tex(M, OUT / "results_tables.tex")
        write_notes(M, OUT / "NOTES.md")
        return

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    lp = Lpips()
    pipe = Pipeline()
    sr, col, det = pipe.enhancer.sr, pipe.enhancer.colorizer, pipe.detector
    gt = load_gt(Path(a.pop_ann_dir) / "test" / "Annotations" / "test.json")
    avail = sorted(x for x in Path(a.pop_images).iterdir() if x.name in gt)
    rng = random.Random(a.seed)
    pop = sorted(rng.sample(avail, min(a.n_pop, len(avail))), key=lambda x: x.name)
    runtime_files = pop[: a.n_runtime]
    device = "GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else \
        f"CPU ({platform.processor() or platform.machine()}, {torch.get_num_threads()} threads)"

    M = dict(seed=a.seed, device=device, torch=torch.__version__, lpips_note=lp.reason, pipeline=pipe.describe(),
             subsets=dict(pop_test_available=len(avail), pop_test_used=len(pop), runtime_images=len(runtime_files),
                          runtime_warmup=a.warmup, pop_files=[x.name for x in pop]))
    print("A datasets", flush=True)
    M["datasets"] = section_datasets(a)
    M["training"] = section_training(a, sr, col, det)

    if "sr" not in skip:
        print("C super-resolution", flush=True)
        M["sr"] = section_sr(a, pop[: a.n_sr], sr, lp)
        M["subsets"]["sr_images"] = M["sr"]["n"]
    if "color" not in skip:
        print("D colorization", flush=True)
        M["colorization"] = section_color(a, col, lp)
        M["subsets"]["color_pairs_used"] = M["colorization"]["n_pairs"]

    print("E detection (original thermal)", flush=True)
    conf_pr, conf_ap = det.conf, 0.001
    det_a = {}
    for x in pop:
        g = read_gray(x)
        b, s = run_detector(det, np.repeat(g[..., None], 3, 2), conf_ap)
        det_a[x.name] = match(b, s, gt[x.name])
    E = dict(iou=0.5, conf_for_pr=conf_pr, conf_for_map=conf_ap, nms_iou=det.iou, imgsz=det.imgsz,
             detector=det.name)
    E["thermal_subset"] = dict(images=len(pop), **det_scores(list(det_a.values()),
                                                             sum(len(gt[x.name]) for x in pop), conf_pr))
    rt = [x.name for x in runtime_files]
    E["thermal_runtime"] = dict(images=len(rt), **det_scores([det_a[n] for n in rt], sum(len(gt[n]) for n in rt), conf_pr))

    print("F runtime + E colorized", flush=True)
    times = dict(parallel=[], sequential=[], branch_a=[], branch_b=[])
    det_b = {}
    for i, x in enumerate(runtime_files):
        r_seq = pipe.run(x, parallel=False)
        r_par = pipe.run(x, parallel=True)
        if i >= a.warmup:
            times["sequential"].append(r_seq["timings"]["total"])
            times["parallel"].append(r_par["timings"]["total"])
            times["branch_a"].append(r_seq["timings"]["branch_a"])
            times["branch_b"].append(r_seq["timings"]["branch_b"])
        rgb = r_seq["enhanced_rgb"]
        b, s = run_detector(det, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), conf_ap)
        b = rescale_boxes(b, (rgb.shape[1], rgb.shape[0]), r_seq["detections"]["input_size"])
        det_b[x.name] = match(np.asarray(b, np.float32).reshape(-1, 4), s, gt[x.name])
        print(f"  run {i + 1}/{len(runtime_files)} {x.name} seq {r_seq['timings']['total']:.2f}s "
              f"par {r_par['timings']['total']:.2f}s", flush=True)
    E["colorized"] = dict(images=len(rt), input="enhanced_rgb of branch A (pre-enhance, EDSR x2, colorizer), "
                          "boxes mapped back to the thermal frame",
                          **det_scores([det_b[n] for n in rt], sum(len(gt[n]) for n in rt), conf_pr))
    M["detection"] = E
    M["runtime"] = dict(device_label="GPU" if torch.cuda.is_available() else "CPU", device=device,
                        images_timed=len(times["parallel"]), warmup_skipped=a.warmup,
                        mean_s={k: float(np.mean(v)) if v else None for k, v in times.items()},
                        std_s={k: float(np.std(v)) if v else None for k, v in times.items()},
                        note="branch_a/branch_b are from the sequential runs (each branch alone)")

    # Sample figure: the subset image with the most true positives and no false positives at conf_pr.
    def tp_fp(n):
        t, s = det_a[n]
        k = s >= conf_pr
        return int(t[k].sum()), int((~t[k]).sum())
    best = max((n for n in det_a if tp_fp(n)[1] == 0), key=lambda n: (tp_fp(n)[0], -len(gt[n]), n), default=None)
    if best:
        r = pipe.run(Path(a.pop_images) / best)
        fig = image_row([r["thermal"], r["fused"]], ["(a) Thermal input", "(b) InfraVision output"])
        fig.savefig(OUT / "figures" / "infravision_output.png", dpi=300, bbox_inches="tight", pad_inches=0.01,
                    facecolor="white")
        tp, fp = tp_fp(best)
        M["figure"] = dict(file="docs/report_metrics/figures/infravision_output.png", source_image=best,
                           gt_people=int(len(gt[best])), tp=tp, fp=fp, detections=r["detections"]["count"],
                           rule="most TP with zero FP at conf_for_pr among the POP subset", notes=r["notes"])

    json.dump(M, open(OUT / "metrics.json", "w"), indent=2, default=float)
    if all(k in M for k in ("sr", "colorization")):
        write_tex(M, OUT / "results_tables.tex")
        write_notes(M, OUT / "NOTES.md")
    print("done", flush=True)


if __name__ == "__main__":
    main()
