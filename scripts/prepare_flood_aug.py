"""Synthetic flood occlusion for POP person detection (RUN_ORDER 2.2).

    python scripts/prepare_flood_aug.py                # 30% of train + all of test
    python scripts/prepare_flood_aug.py --frac 0.5 --grid results/flood_grid.jpg

The lower 30-60% of every person box is covered with "water": a cool intensity taken
from the ground around each person, plus ripple stripes and noise. The water patch is wider
than the box, blob-shaped, with a wavy, soft top edge, so the model cannot learn "a rectangle = a person".
Labels are copied unchanged (the box still covers the whole person).

Writes, inside human_detection/datasets/pop (from convert_pop.py):
  images/train/flood_<name>, labels/train/flood_<name>.txt   extra training frames
  images/test_flood, labels/test_flood, data_flood.yaml      flooded copy of the test split
Re-running replaces the earlier flood_ files. Seeded (seed 0).
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POP = ROOT / "human_detection" / "datasets" / "pop"
PREFIX = "flood_"


def read_boxes(label_path, w, h):
    """YOLO txt -> list of (x0, y0, x1, y1) pixel boxes."""
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            xc, yc, bw, bh = (float(v) for v in parts[1:])
            boxes.append((round((xc - bw / 2) * w), round((yc - bh / 2) * h),
                          round((xc + bw / 2) * w), round((yc + bh / 2) * h)))
    return boxes


def flood(gray, boxes, rng):
    """Cover the lower 30-60% of each box with cool rippled water. gray: uint8 HxW."""
    h, w = gray.shape
    out = gray.astype(np.float32)
    for x0, y0, x1, y1 in boxes:
        bw, bh = x1 - x0, y1 - y0
        if bw < 2 or bh < 2:
            continue
        top = y1 - bh * rng.uniform(0.3, 0.6)                # water line
        pad = max(4, round(bw * rng.uniform(0.4, 0.9)))     # patch wider than the box
        px0, px1 = max(0, x0 - 2 * pad), min(w, x1 + 2 * pad)
        reach = round((y1 - top) + pad * 0.6 + min(bw, bh) * 0.4)   # ellipse + blur below the box
        py0, py1 = max(0, int(top - bh * 0.2) - 4), min(h, y1 + reach)
        if px1 <= px0 or py1 <= py0:
            continue
        # Water is a little cooler than the ground around this person, not a fixed dark value.
        ring = gray[py0:py1, px0:px1].copy()
        ring[max(0, y0 - py0):max(0, y1 - py0), max(0, x0 - px0):max(0, x1 - px0)] = 0
        around = ring[ring > 0]
        level = (np.percentile(around, 40) if around.size else np.median(gray)) - rng.uniform(4, 15)
        yy, xx = np.mgrid[py0:py1, px0:px1].astype(np.float32)

        # Hard mask: below a wavy water line, inside an ellipse around the box; then blur it soft.
        wave = top + bh * 0.06 * np.sin(xx[0] / max(2.0, bw * rng.uniform(0.2, 0.5)) + rng.uniform(0, 6.3))
        cx, rx = (x0 + x1) / 2, bw / 2 + pad * 0.8
        cy, ry = y1, (y1 - top) + pad * 0.6
        inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
        mask = ((yy >= wave[None]) & inside).astype(np.float32)
        alpha = cv2.GaussianBlur(mask, (0, 0), max(1.5, min(bw, bh) * 0.12))

        ripple = 3 * np.sin(yy * rng.uniform(0.6, 1.2) + 0.3 * np.sin(xx * rng.uniform(0.1, 0.3)))
        noise = cv2.GaussianBlur(rng.normal(0, 3, yy.shape).astype(np.float32), (0, 0), 1.0)
        water = level + ripple + noise
        patch = out[py0:py1, px0:px1]
        out[py0:py1, px0:px1] = patch * (1 - alpha) + water * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def clear(paths):
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()


def write_split(names, src_split, dst_split, prefix, rng):
    img_dst, lbl_dst = POP / "images" / dst_split, POP / "labels" / dst_split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    done = []
    for name in names:
        img_path = POP / "images" / src_split / name
        lbl_path = POP / "labels" / src_split / (Path(name).stem + ".txt")
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise OSError(f"unreadable image: {img_path}")
        boxes = read_boxes(lbl_path, gray.shape[1], gray.shape[0])
        if not boxes:
            continue
        cv2.imwrite(str(img_dst / (prefix + name)), flood(gray, boxes, rng), [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copyfile(lbl_path, lbl_dst / (prefix + Path(name).stem + ".txt"))
        done.append((img_path, img_dst / (prefix + name), boxes))
    return done


def save_grid(samples, out, n=6, crop=256):
    """Before / after crops around a person, boxes drawn in green."""
    tiles = []
    for before, after, boxes in samples[:n]:
        pair = []
        x0, y0, x1, y1 = boxes[0]
        for path in (before, after):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            for b in boxes:
                cv2.rectangle(img, b[:2], b[2:], (0, 255, 0), 1)
            h, w = img.shape[:2]
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            cx0, cy0 = min(max(0, cx - crop // 2), w - crop), min(max(0, cy - crop // 2), h - crop)
            pair.append(img[cy0:cy0 + crop, cx0:cx0 + crop])
        tiles.append(np.hstack(pair))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(tiles))
    print("Grid (left = original, right = flooded):", out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frac", type=float, default=0.3, help="share of train frames that get a flooded copy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", default=str(ROOT / "results" / "flood_aug_grid.jpg"))
    args = p.parse_args(argv)
    if not (POP / "images" / "train").is_dir():
        sys.exit(f"{POP} not found. Run human_detection/datasets/convert_pop.py first.")
    rng = np.random.default_rng(args.seed)

    # Remove earlier output so re-runs do not pile up copies.
    clear([*(POP / "images" / "train").glob(PREFIX + "*"), *(POP / "labels" / "train").glob(PREFIX + "*"),
           POP / "images" / "test_flood", POP / "labels" / "test_flood"])

    train = sorted(f.name for f in (POP / "images" / "train").iterdir())
    pick = sorted(rng.choice(train, size=round(len(train) * args.frac), replace=False))
    tr = write_split(pick, "train", "train", PREFIX, rng)
    print(f"train: {len(tr)} flooded copies added ({len(train)} originals)")

    te = write_split(sorted(f.name for f in (POP / "images" / "test").iterdir()), "test", "test_flood", "", rng)
    (POP / "data_flood.yaml").write_text(
        f"path: {POP}\ntrain: images/train\nval: images/val\ntest: images/test_flood\n\nnames:\n  0: person\n")
    print(f"test_flood: {len(te)} flooded test frames; evaluate with data_flood.yaml, split=test")

    save_grid(tr[::max(1, len(tr) // 6)], Path(args.grid))


if __name__ == "__main__":
    main()
