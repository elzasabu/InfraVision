"""Step 3.2/3.4: check the CLI outputs of both colorizers and draw the before/after figure.

usage: python -B verify_outputs.py --inputs <folder of thermal frames> --new <infer --all out (thermovision)>
           --old <infer --all out (infravision_tiny)> --rgb <json {stem: rgb path}> --out docs/colorizer_integration

Per output image: share of flat black / white pixels, a checkerboard score (share of luma power at the
period-2 frequencies, ppm, next to the same score of the SR frame it came from), a seam score at the SR tile borders (gradient there / mean gradient, ~1 = none), PSNR
against the real RGB and the PSNR change if R and B were swapped (negative = channel order is right), and
whether the boxes equal the input boxes scaled by the output/input size ratio.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--inputs", required=True)
p.add_argument("--new", required=True)
p.add_argument("--old", required=True)
p.add_argument("--rgb", required=True)
p.add_argument("--out", required=True)
p.add_argument("--tile", type=int, default=1280, help="sr.max_input_side (tile size on the upscaled frame)")
a = p.parse_args()
rgb_map = json.loads(Path(a.rgb).read_text(encoding="utf-8-sig"))


def imread(path, flag=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if flag == cv2.IMREAD_COLOR else img


def luma(img):
    return img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def checker(img):
    """Share of the luma power (parts per million) at the period-2 frequencies (3x3 bins around each Nyquist
    peak), at full resolution. A transposed-conv checkerboard shows up as a large share there."""
    L = luma(img) if img.ndim == 3 else img.astype(np.float32)
    P = np.abs(np.fft.fft2(L - L.mean())) ** 2
    H, W = L.shape
    pk = sum(P[(cy + dy) % H, (cx + dx) % W] for cy, cx in ((H // 2, W // 2), (H // 2, 0), (0, W // 2))
             for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return float(1e6 * pk / P.sum())


def seam(img, tile):
    L = luma(img)
    H, W = L.shape
    gx, gy = np.abs(np.diff(L, axis=1)), np.abs(np.diff(L, axis=0))
    xs = np.linspace(0, W, -(-W // tile) + 1).round().astype(int)[1:-1]
    ys = np.linspace(0, H, -(-H // tile) + 1).round().astype(int)[1:-1]
    vals = [gx[:, x - 1].mean() / gx.mean() for x in xs] + [gy[y - 1, :].mean() / gy.mean() for y in ys]
    return float(max(vals)) if vals else None


def psnr(x, y):
    return float(10 * np.log10(255 ** 2 / max(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2), 1e-10)))


report = {}
for tag, folder in (("thermovision", a.new), ("infravision_tiny", a.old)):
    det = json.loads((Path(folder) / "detections.json").read_text())
    rows = {}
    for r in det["results"]:
        stem = Path(r["file"]).stem
        enh = imread(Path(folder) / f"{stem}_enhanced.png")
        fused = imread(Path(folder) / f"{stem}_fused.png")
        (w_in, h_in), (w_out, h_out) = r["input_size"], r["output_size"]
        row = dict(size=[w_out, h_out], people=r["people"], total_s=r["timings"]["total"],
                   colorize_s=r["timings"]["colorize"],
                   black=float((enh.max(2) == 0).mean()), white=float((enh.min(2) == 255).mean()),
                   checker_ppm=checker(enh), checker_ppm_sr_input=checker(imread(Path(folder) / f"{stem}_sr.png")),
                   seam=seam(enh, a.tile), mean_rgb=enh.reshape(-1, 3).mean(0).round(1).tolist(),
                   fused_matches_size=list(fused.shape[:2]) == [h_out, w_out])
        if stem in rgb_map:
            tgt = cv2.resize(imread(rgb_map[stem]), (640, 512), interpolation=cv2.INTER_AREA)
            e = cv2.resize(enh, (640, 512), interpolation=cv2.INTER_AREA)
            row["psnr"] = psnr(e, tgt)
            row["swap_rb_psnr_change"] = psnr(e[..., ::-1], tgt) - row["psnr"]
        rows[stem] = row
    report[tag] = dict(models=det["models"], images=rows,
                       mean_total_s=float(np.mean([v["total_s"] for v in rows.values()])),
                       mean_colorize_s=float(np.mean([v["colorize_s"] for v in rows.values()])))

# boxes: fused-image boxes must be the input-frame boxes x (output / input size)
errs, n_boxes = [], 0
for folder in (a.new, a.old):
    for r in json.loads((Path(folder) / "detections.json").read_text())["results"]:
        b, bi = np.array(r["boxes"], np.float32).reshape(-1, 4), np.array(r.get("boxes_input", []), np.float32).reshape(-1, 4)
        if len(bi) != len(b):
            continue
        (w_in, h_in), (w_out, h_out) = r["input_size"], r["output_size"]
        errs.append(np.abs(b - bi * [w_out / w_in, h_out / h_in, w_out / w_in, h_out / h_in]).max() if len(b) else 0.0)
        n_boxes += len(b)
report["box_check"] = dict(boxes=n_boxes, max_abs_error_px=float(max(errs)) if errs else None)
Path(a.out, "verify_outputs.json").write_text(json.dumps(report, indent=1))

# ---------------------------------------------------------------- before/after figure
stems = [s for s in rgb_map][:3] + [Path(f).stem for f in sorted(Path(a.inputs).glob("*.JPG"))][:2]
T = 400
cols = ["thermal input", "before: InfraVision tiny", "after: ThermoVision exp7 last", "real RGB"]
canvas = []
for stem in stems:
    src = next(Path(a.inputs).glob(stem + ".*"))
    th = imread(src, cv2.IMREAD_GRAYSCALE)
    h = round(T * th.shape[0] / th.shape[1])
    tiles = [cv2.cvtColor(cv2.resize(th, (T, h), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2RGB)]
    for folder in (a.old, a.new):
        tiles.append(cv2.resize(imread(Path(folder) / f"{stem}_fused.png"), (T, h), interpolation=cv2.INTER_AREA))
    tiles.append(cv2.resize(imread(rgb_map[stem]), (T, h), interpolation=cv2.INTER_AREA) if stem in rgb_map
                 else np.full((h, T, 3), 60, np.uint8))
    row = np.hstack(tiles)
    cv2.putText(row, stem + ("" if stem in rgb_map else "  (POP: no RGB)"), (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1, cv2.LINE_AA)
    canvas.append(row)
head = np.full((30, T * 4, 3), 255, np.uint8)
for k, c in enumerate(cols):
    cv2.putText(head, c, (T * k + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
fig = np.vstack([head] + canvas)
cv2.imencode(".png", cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))[1].tofile(str(Path(a.out, "before_after.png")))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "models"} if isinstance(v, dict) else v
                  for k, v in report.items()}, indent=1))
