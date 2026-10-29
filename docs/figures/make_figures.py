"""Report figures, made by running the real InfraVision code.

    python docs/figures/make_figures.py select    --pop POP/test      # rank POP test frames
    python docs/figures/make_figures.py outputs   --pop POP/test      # output_1..3.png, stages.png
    python docs/figures/make_figures.py synthetic --data data/paired_disaster
    python docs/figures/make_figures.py architecture

`select` runs the pipeline's detection branch (the same Detector and config the
full pipeline uses, so the boxes are identical) on every POP test frame,
matches the detections to the ground truth at IoU 0.5 and writes
docs/figures/selection.json with the chosen close-up, group and small-person
frames. `outputs` runs the full parallel pipeline on those three frames.
Re-run all four after training to regenerate the figures with the new weights.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "figures"
SELECTION = OUT / "selection.json"
DPI = 300

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.facecolor": "white", "figure.facecolor": "white"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, bbox_inches="tight", pad_inches=0.01, facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def image_row(panels, labels, panel_w=3.2, gap=0.06, label_size=9):
    """Panels side by side at the same height, '(a) ...' under each, nothing else."""
    h0, w0 = panels[0].shape[:2]
    ph = panel_w * h0 / w0
    label_h = 0.28
    fw = len(panels) * panel_w + (len(panels) - 1) * gap
    fig = plt.figure(figsize=(fw, ph + label_h))
    for i, (img, lab) in enumerate(zip(panels, labels)):
        ax = fig.add_axes([i * (panel_w + gap) / fw, label_h / (ph + label_h), panel_w / fw, ph / (ph + label_h)])
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255, interpolation="lanczos")
        ax.set_axis_off()
        ax.text(0.5, -0.03, lab, transform=ax.transAxes, ha="center", va="top", fontsize=label_size)
    return fig


# ---------------------------------------------------------------- POP selection

def iou_matrix(a, b):
    a, b = np.asarray(a, np.float32).reshape(-1, 4), np.asarray(b, np.float32).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(det, scores, gt, thr=0.5):
    """Greedy match by score -> (tp scores, n fp, matched gt idx)."""
    if not len(det) or not len(gt):
        return [], len(det), []
    m = iou_matrix(det, gt)
    used, tps, fp = set(), [], 0
    for i in np.argsort(-scores):
        j = int(np.argmax(np.where([k in used for k in range(len(gt))], -1, m[i])))
        if m[i, j] >= thr and j not in used:
            used.add(j)
            tps.append(float(scores[i]))
        else:
            fp += 1
    return tps, fp, sorted(used)


def cmd_select(args):
    from infravision.config import load_config
    from infravision.detect import Detector
    from infravision.io import read_thermal

    pop = Path(args.pop)
    coco = json.loads((pop / "Annotations" / "test.json").read_text())
    gt = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    cfg = load_config()
    det = Detector(device=cfg.get("device", "auto"), **cfg["detection"])
    imgs = args.images and Path(args.images) or pop / "JPEGImages"
    rows = []
    for k, im in enumerate(coco["images"]):
        path = imgs / im["file_name"]
        if not path.exists():
            continue
        g = np.asarray(gt.get(im["id"], []), np.float32).reshape(-1, 4)
        d = det(read_thermal(path)[0])
        tps, fp, used = match(d["boxes"], d["scores"], g)
        heights = (g[:, 3] - g[:, 1]).tolist()
        rows.append(dict(file=im["file_name"], n_gt=len(g), tp=len(tps), fp=fp, min_tp_score=min(tps, default=0),
                         mean_tp_score=float(np.mean(tps)) if tps else 0.0,
                         gt_median_h=float(np.median(heights)) if heights else 0.0,
                         matched_min_h=float(min(heights[j] for j in used)) if used else 0.0))
        if k % 100 == 0:
            print(k, im["file_name"], rows[-1])
    if not rows:
        sys.exit(f"no POP test images found in {imgs}")

    clean = [r for r in rows if r["tp"] == r["n_gt"] and r["fp"] == 0 and r["tp"] > 0]
    pool = clean or [r for r in rows if r["tp"] > 0]
    # Close-up: biggest people, every one found, no false alarms.
    close = max(pool, key=lambda r: (r["gt_median_h"], r["mean_tp_score"]))
    # Group: most people found in one frame (ties: fewest false alarms, then score).
    group = max((r for r in rows if r["file"] != close["file"]),
                key=lambda r: (r["tp"], -r["fp"], r["tp"] == r["n_gt"], r["mean_tp_score"]))
    # Hard: the smallest person that was still found correctly.
    hard = min((r for r in rows if r["tp"] > 0 and r["file"] not in (close["file"], group["file"])),
               key=lambda r: (r["matched_min_h"], r["fp"]))
    tot = lambda k: int(sum(r[k] for r in rows))  # noqa: E731
    summary = dict(images=len(rows), people=tot("n_gt"), found=tot("tp"), false_alarms=tot("fp"),
                   frames_all_found_no_fp=len(clean), detector=det.name)
    sel = dict(close_up=close, group=group, hard_small=hard, summary=summary)
    OUT.mkdir(parents=True, exist_ok=True)
    SELECTION.write_text(json.dumps(sel, indent=2))
    (OUT / "selection_all.json").write_text(json.dumps(rows))
    print(json.dumps(sel, indent=2))


# ---------------------------------------------------------------- POP outputs + stages

def cmd_outputs(args):
    from infravision.pipeline import Pipeline

    sel = json.loads(SELECTION.read_text())
    imgs = Path(args.images) if args.images else Path(args.pop) / "JPEGImages"
    pipe = Pipeline()
    info = dict(pipeline=pipe.describe(), frames={})
    for n, key in enumerate(("close_up", "group", "hard_small"), 1):
        f = sel[key]["file"]
        out = pipe.run(imgs / f)
        panels, labels = [out["thermal"], out["fused"]], ["(a) Thermal input", "(b) InfraVision output"]
        # POP is thermal only; a reference visible panel is added only when one exists.
        ref = next((p for p in [imgs.parent / "visible" / f] if p.exists()), None)
        if ref is not None:
            from infravision.io import read_rgb
            panels.append(read_rgb(ref))
            labels.append("(c) Reference visible")
        save(image_row(panels, labels), f"output_{n}.png")
        d = out["detections"]
        info["frames"][f"output_{n}.png"] = dict(file=f, kind=key, people=d["count"],
                                                 scores=[round(float(s), 3) for s in d["scores"]],
                                                 timings={k: round(v, 2) for k, v in out["timings"].items()},
                                                 notes=out["notes"])
        if key == args.stages_from:
            stages = [out["thermal"], out["pre_enhanced"], out["sr"], out["enhanced_rgb"], out["fused"]]
            names = ["(a) Thermal", "(b) Pre-enhanced", "(c) Super-resolved", "(d) Colorized",
                     "(e) Final with detections"]
            save(image_row(stages, names, panel_w=2.2, label_size=8), "stages.png")
            info["stages_from"] = f
    pipe.close()
    (OUT / "figures_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


# ---------------------------------------------------------------- synthetic dataset

def cmd_synthetic(args):
    import csv

    import cv2

    sys.path.insert(0, str(ROOT / "paired_data"))
    from degrade import degrade

    root = Path(args.data)
    by_id = {}
    with open(root / "manifest.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            by_id.setdefault(r["id"], r)
    ids = args.ids.split(",")
    conds = [("smoke", "(c) Smoke"), ("fog", "(d) Fog"), ("dust", "(e) Dust"), ("dark", "(f) Darkness")]
    heads = ["(a) Thermal", "(b) Clear RGB"] + [c[1] for c in conds]
    pw, gap = 1.6, 0.04
    first = cv2.imread(str(root / by_id[ids[0]]["thermal"].replace("\\", "/")), cv2.IMREAD_GRAYSCALE)
    ph = pw * first.shape[0] / first.shape[1]
    head_h = 0.22
    fw = len(heads) * pw + (len(heads) - 1) * gap
    fh_ = len(ids) * ph + (len(ids) - 1) * gap + head_h
    fig = plt.figure(figsize=(fw, fh_))
    used = []
    for r, pid in enumerate(ids):
        row = by_id[pid]
        p = lambda k: root / row[k].replace("\\", "/")  # noqa: E731
        th = cv2.imread(str(p("thermal")), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(cv2.imread(str(p("rgb"))), cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng(int(pid.split("_")[-1]))
        # The dataset's own generator (paired_data/degrade.py) at medium severity, so
        # every row shows all four conditions; the stored set keeps 3 random ones per pair.
        panels = [th, rgb] + [degrade(rgb, c, "medium", rng) for c, _ in conds]
        h, w = th.shape
        boxes = []
        if row["labels"] and p("labels").exists():
            for line in p("labels").read_text().splitlines():
                c, cx, cy, bw, bh = map(float, line.split())
                boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
        used.append(dict(id=pid, people=len(boxes)))
        y0 = 1 - (head_h + r * (ph + gap) + ph) / fh_
        for c, img in enumerate(panels):
            ax = fig.add_axes([c * (pw + gap) / fw, y0, pw / fw, ph / fh_])
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
            for x, y, bw, bh in boxes:
                ax.add_patch(Rectangle((x, y), bw, bh, fill=False, ec="#00ff00", lw=0.9))
                ax.add_patch(Rectangle((x, y), bw, bh, fill=False, ec="black", lw=0.3, ls="-"))
            ax.set_axis_off()
            if r == 0:
                ax.text(0.5, 1.03, heads[c], transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
    save(fig, "synthetic_samples.png")
    print(used)


# ---------------------------------------------------------------- architecture

def cmd_architecture(_args):
    from infravision.config import load_config

    cfg = load_config()
    sr, det = cfg["sr"], cfg["detection"]
    fig = plt.figure(figsize=(3.5, 4.75))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 136)
    ax.set_axis_off()
    fs, small = 7.2, 5.9

    def box(x, y, w, h, title, sub=None, fc="#ffffff", ec="#333333", bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=2",
                                    fc=fc, ec=ec, lw=0.8))
        if sub:
            ax.text(x, y + 1.9, title, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")
            ax.text(x, y - 2.6, sub, ha="center", va="center", fontsize=small, color="#444444")
        else:
            ax.text(x, y, title, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")
        return (x, y, w, h)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#333333",
                                     shrinkA=0, shrinkB=0))

    blue, orange, grey = "#e8f0fb", "#fdf0e2", "#f2f2f2"
    xa, xb = 29, 76
    box(50, 128, 62, 10, "Thermal IR frame", "8/16-bit, grey (POP 1280×1024)", fc=grey, bold=True)

    # Branch frames
    ax.add_patch(FancyBboxPatch((3, 42), 52, 72, boxstyle="round,pad=0,rounding_size=2.5", fc="none",
                                ec="#5b8bd0", lw=0.8, ls=(0, (3, 2))))
    ax.add_patch(FancyBboxPatch((59, 42), 38, 72, boxstyle="round,pad=0,rounding_size=2.5", fc="none",
                                ec="#d08a3c", lw=0.8, ls=(0, (3, 2))))
    # Titles sit beside the incoming arrows, not across them.
    ax.text(5, 111.5, "Branch A\nenhancement", ha="left", va="top", fontsize=fs, weight="bold", color="#2f5f9f",
            linespacing=1.1)
    ax.text(95, 111.5, "Branch B\ndetection", ha="right", va="top", fontsize=fs, weight="bold", color="#9f5f1f",
            linespacing=1.1)

    a1 = box(xa, 97, 46, 12, "Pre-enhance", "NL-means denoise + CLAHE", fc=blue)
    a2 = box(xa, 77, 46, 12, "Super-resolution", f"bicubic ×{sr['scale']} + EDSR", fc=blue)
    a3 = box(xa, 57, 46, 12, "Colorization", "thermal → RGB generator", fc=blue)
    b1 = box(xb, 77, 33, 22, "YOLO11s", f"person, imgsz {det['imgsz']}\non the input frame", fc=orange)

    # Split into the two branches
    arrow((50, 123), (50, 120))
    ax.plot([xa, xb], [120, 120], color="#333333", lw=0.8)
    arrow((xa, 120), (xa, a1[1] + 6))
    arrow((xb, 120), (xb, b1[1] + 11))
    ax.text(50, 118.8, "run in parallel (2 threads)", ha="center", va="top", fontsize=small, style="italic",
            color="#444444", bbox=dict(fc="white", ec="none", pad=0.6))
    arrow((xa, a1[1] - 6), (xa, a2[1] + 6))
    arrow((xa, a2[1] - 6), (xa, a3[1] + 6))

    r = box(xb, 31, 38, 12, "Box rescaling", "x·W_out/W_in, y·H_out/H_in", fc=grey)
    arrow((xb, b1[1] - 11), (xb, r[1] + 6))
    ax.text(xb + 1.5, 46.5, "boxes + scores\n(input pixels)", ha="left", va="center", fontsize=small,
            color="#444444")
    o = box(50, 14, 54, 11, "Overlay", "boxes, confidences, people count", fc=grey)
    arrow((xa, a3[1] - 6), (xa, o[1] + 5.5))
    ax.text(xa - 1.5, 37, "colorized\nSR frame", ha="right", va="center", fontsize=small, color="#444444")
    arrow((xb, r[1] - 6), (xb - 8, o[1] + 5.5))
    arrow((50, o[1] - 5.5), (50, 2.6))
    ax.text(50, 1.2, "Output: fused rescue view", ha="center", va="center", fontsize=fs, weight="bold")
    save(fig, "architecture.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("select", "outputs"):
        s = sub.add_parser(name)
        s.add_argument("--pop", default=str(ROOT / "POP" / "test"), help="POP split folder with Annotations/")
        s.add_argument("--images", default=None, help="folder with the frames (default <pop>/JPEGImages)")
        if name == "outputs":
            s.add_argument("--stages-from", default="close_up", choices=("close_up", "group", "hard_small"))
    s = sub.add_parser("synthetic")
    s.add_argument("--data", default=str(ROOT / "data" / "paired_disaster"))
    s.add_argument("--ids", required=True, help="comma-separated pair ids from manifest.csv")
    sub.add_parser("architecture")
    args = p.parse_args()
    dict(select=cmd_select, outputs=cmd_outputs, synthetic=cmd_synthetic, architecture=cmd_architecture)[args.cmd](args)


if __name__ == "__main__":
    main()
