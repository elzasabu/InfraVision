"""Pack POP + AIResQ (+ WiSARD) into sar_yolo.zip for colab_training/train_sar_yolo_kaggle.ipynb.

    .venv/bin/python human_detection/datasets/make_sar_zip.py
    .venv/bin/python human_detection/datasets/make_sar_zip.py --wisard ../datasets/WiSARD/WiSARDv1

Splits are by flight / sequence, never by frame, so no near-duplicate frame is in train and test.
  POP     : official train (+ flood v2 copies) / val / test / test_flood (+ flood_decoys.json)
  AIResQ  : 17 flights (grouped by capture time); 4 test flights, 2 val flights, the rest train.
            People are tiny (median 7x9 px at 640x512), so train also gets 2x2 overlapping tiles
            (each 1.6x larger in the frame after resize), matching tiled inference (detection.tile).
  WiSARD  : without --wisard: the Multi Modal Sample flight (MtErie IR_0004, 264 frames) = test_wisard.
            --wisard DIR (thermal folders of WiSARDv1, `unzip WiSARDv1.zip '*_IR_*'`): labelled IR
            sequences split by sequence: WISARD_TEST (4 sites/cameras, incl. the sample flight, all
            frames), WISARD_VAL (2), rest train. Video frames are near-duplicates, so train/val keep
            every --wisard_step-th frame with people and every --wisard_empty_step-th empty frame.
            Sequences without labels or with zero boxes (unannotated) are skipped.

Zip layout (Kaggle unzips it):
  sar/train.txt                 balanced training list (only these train images are packed)
  sar/images/{train,val,test_pop,test_flood,test_airesq,test_wisard}/  sar/labels/...
  sar/flood_decoys.json         empty-blob boxes for scripts/eval_flood_decoys.py
  sar/splits.json               which flights / sequences went where
  weights/start.pt              starting model (--weights; `none` = leave it out and upload start.pt separately)
  RGBTDronePerson / HIT-UAV     --rgbt / --hituav (thermal tiny-person sets): test_rgbt, test_hituav
"""
import argparse
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POP = HERE / "pop"
AIRESQ = REPO.parent / "datasets" / "AIResQ" / "Benchmark"
WISARD_SAMPLE = REPO.parent / "datasets" / "WiSARD" / "WiSARD_Multi_Modal_Sample" / "210417_MtErie_Enterprise_IR_0004"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# AIResQ flights by index in time order (see airesq_flights); chosen to cover 3 capture days.
AIRESQ_TEST = {4, 7, 14, 16}
AIRESQ_VAL = {10, 15}


def airesq_flights(names, gap_s=120, gap_n=20):
    """Group AIResQ files into flights: a new flight starts after a time (or index) gap."""
    groups, prev = [], None
    for n in sorted(names):
        m = re.match(r"DJI_(\d{14})_(\d+)", n)
        m2 = re.match(r"DJI_(\d+)_T", n)
        key = ("ts", datetime.strptime(m.group(1), "%Y%m%d%H%M%S")) if m else ("n", int(m2.group(1))) if m2 else ("o", n)
        new = (prev is None or key[0] != prev[0] or key[0] == "o"
               or (key[0] == "ts" and (key[1] - prev[1]).total_seconds() > gap_s)
               or (key[0] == "n" and key[1] - prev[1] > gap_n))
        if new:
            groups.append([])
        groups[-1].append(n)
        prev = key
    return groups


def read_yolo(path):
    rows = []
    if path.is_file():
        for line in path.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                rows.append([float(v) for v in p[1:]])
    return rows


def fmt(rows):
    return "".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in rows)


def tiles(img, rows, n=2, overlap=0.2, keep=0.5):
    """n x n overlapping crops with their boxes (clipped; kept if >= `keep` of the box is inside)."""
    H, W = img.shape[:2]
    th, tw = int(H / (n - (n - 1) * overlap)), int(W / (n - (n - 1) * overlap))
    out = []
    for iy in range(n):
        for ix in range(n):
            y0 = round(iy * (H - th) / max(1, n - 1))
            x0 = round(ix * (W - tw) / max(1, n - 1))
            new = []
            for xc, yc, bw, bh in rows:
                bx0, by0, bx1, by1 = (xc - bw / 2) * W, (yc - bh / 2) * H, (xc + bw / 2) * W, (yc + bh / 2) * H
                cx0, cy0, cx1, cy1 = max(bx0, x0), max(by0, y0), min(bx1, x0 + tw), min(by1, y0 + th)
                if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < keep * (bx1 - bx0) * (by1 - by0):
                    continue
                new.append([((cx0 + cx1) / 2 - x0) / tw, ((cy0 + cy1) / 2 - y0) / th, (cx1 - cx0) / tw, (cy1 - cy0) / th])
            out.append((f"t{iy}{ix}", img[y0:y0 + th, x0:x0 + tw], new))
    return out


class Packer:
    """Writes images + labels; train frames are thinned per source (keep every steps[kind]-th)."""

    def __init__(self, z, steps=None):
        self.z, self.count, self.boxes, self.train = z, {}, {}, []
        self.steps, self.seen = steps or {}, {}

    def add(self, split, name, rows, src=None, data=None, kind="other"):
        if split == "train":
            self.seen[kind] = self.seen.get(kind, 0) + 1
            if self.seen[kind] % self.steps.get(kind, 1):
                return
        stem = Path(name).stem
        if src is not None:
            self.z.write(src, f"sar/images/{split}/{name}")
        else:
            self.z.writestr(f"sar/images/{split}/{name}", data)
        self.z.writestr(f"sar/labels/{split}/{stem}.txt", fmt(rows))
        self.count[split] = self.count.get(split, 0) + 1
        self.boxes[split] = self.boxes.get(split, 0) + len(rows)
        if split == "train":
            self.train.append((f"./images/train/{name}", kind, len(rows) > 0))


def jpg(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # thermal: grey, and drops DJI radiometric blocks
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert ok
    return buf.tobytes()


def images_in(d):
    return sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)


WISARD_TEST = {"210417_MtErie_Enterprise_IR_0004", "200929_Karen_FLIR_IR_5",
               "210924_FHL_Enterprise_IR_0567", "210327_Airfield_FLIR_IR_2"}
WISARD_VAL = {"200910_Carnation_FLIR_IR_2", "210924_FHL_Enterprise_IR_0406"}


def wisard_ir_sequences(root):
    """Labelled IR sequence folders -> [(folder, [(image, rows), ...])]; skips unlabelled ones."""
    seqs = []
    for d in sorted({p.parent for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT}):
        if not re.search(r"(^|_)IR(_|$)", d.name, re.I):
            continue
        frames = [(f, read_yolo(f.with_suffix(".txt"))) for f in images_in(d) if f.with_suffix(".txt").is_file()]
        if sum(len(r) for _, r in frames) == 0:
            continue
        seqs.append((d, frames))
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "human_detection" / "sar_yolo.zip"))
    ap.add_argument("--weights", default=str(REPO / "checkpoints" / "yolo" / "best.pt"))
    ap.add_argument("--wisard", default="", help="full WiSARD folder (thermal sequences are found by name)")
    ap.add_argument("--wisard_step", type=int, default=4, help="train/val: keep every n-th WiSARD frame with people")
    ap.add_argument("--wisard_empty_step", type=int, default=12, help="train/val: keep every n-th empty WiSARD frame")
    ap.add_argument("--pop_step", type=int, default=2, help="keep every n-th POP train frame in train.txt")
    ap.add_argument("--rgbt", default="", help="RGBTDronePerson in YOLO form (convert_rgbtdroneperson.py): "
                                               "train -> train, val -> test_rgbt")
    ap.add_argument("--hituav", default="", help="HIT-UAV in YOLO form (convert_hituav.py): train / val / test_hituav")
    ap.add_argument("--llvip", default="", help="LLVIP persons in YOLO form (convert_llvip_persons.py): "
                                                "train -> train, test -> test_llvip (large, near, oblique people)")
    ap.add_argument("--steps", default="", help="train.txt: keep every n-th frame per source, e.g. "
                                                "pop=3,pop_flood=3,airesq_tile=2,wisard=2,rgbt=2 (default pop/pop_flood = --pop_step)")
    args = ap.parse_args()
    steps = {"pop": args.pop_step, "pop_flood": args.pop_step}
    steps.update({k: int(v) for k, v in (kv.split("=") for kv in args.steps.split(",") if kv)})
    for d in (POP / "images" / "train", AIRESQ / "images", WISARD_SAMPLE):
        if not d.is_dir():
            sys.exit(f"missing {d}")
    if not (POP / "flood_decoys.json").is_file():
        sys.exit("run scripts/prepare_flood_aug.py first (flood v2 with decoys)")
    splits = {"airesq": {}, "wisard": {}}
    out = Path(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        pk = Packer(z, steps)
        # POP
        for src, dst in (("train", "train"), ("val", "val"), ("test", "test_pop"), ("test_flood", "test_flood")):
            for f in images_in(POP / "images" / src):
                kind = "pop_flood" if f.name.startswith("flood_") else "pop"
                pk.add(dst, f.name, read_yolo(POP / "labels" / src / (f.stem + ".txt")), src=f, kind=kind)
        dec = json.loads((POP / "flood_decoys.json").read_text())
        z.writestr("sar/flood_decoys.json", json.dumps(dec))

        # AIResQ
        groups = airesq_flights([f.name for f in images_in(AIRESQ / "images")])
        n_empty = 0
        for gi, g in enumerate(groups):
            split = "test_airesq" if gi in AIRESQ_TEST else "val" if gi in AIRESQ_VAL else "train"
            splits["airesq"][f"flight{gi:02d}_{g[0]}..{g[-1]}"] = f"{split} ({len(g)})"
            for n in g:
                f = AIRESQ / "images" / n
                rows = read_yolo(AIRESQ / "labels" / (f.stem + ".txt"))
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                pk.add(split, f"airesq_{f.stem}.jpg", rows, data=jpg(img), kind="airesq")
                if split == "train":
                    for tag, crop, trows in tiles(img, rows):
                        if not trows:                      # keep every 4th empty tile only
                            n_empty += 1
                            if n_empty % 4:
                                continue
                        pk.add("train", f"airesq_{f.stem}_{tag}.jpg", trows, data=jpg(crop), kind="airesq_tile")

        # WiSARD
        if not args.wisard:
            for f in images_in(WISARD_SAMPLE):
                pk.add("test_wisard", "wisard_" + f.name, read_yolo(f.with_suffix(".txt")), src=f)
            splits["wisard"][WISARD_SAMPLE.name] = f"test_wisard ({len(images_in(WISARD_SAMPLE))})"
        else:
            for d, frames in wisard_ir_sequences(args.wisard):
                split = "test_wisard" if d.name in WISARD_TEST else "val" if d.name in WISARD_VAL else "train"
                n_p = n_e = kept = 0
                for f, rows in frames:
                    if split != "test_wisard":
                        if rows:
                            n_p += 1
                            if n_p % args.wisard_step:
                                continue
                        else:
                            n_e += 1
                            if n_e % args.wisard_empty_step:
                                continue
                    name = f"wisard_{d.name}_{f.stem}.jpg".replace(" ", "_")
                    pk.add(split, name, rows, data=jpg(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)), kind="wisard")
                    kept += 1
                splits["wisard"][d.name] = f"{split} ({kept} of {len(frames)})"
            missing = (WISARD_TEST | WISARD_VAL) - set(splits["wisard"])
            if missing:
                sys.exit(f"WiSARD test/val sequences not found: {missing}")

        # Extra thermal person sets (already YOLO, person only; see their converters).
        for root, kind, plan in ((args.rgbt, "rgbt", (("train", "train"), ("val", "test_rgbt"))),
                                 (args.hituav, "hituav", (("train", "train"), ("val", "val"), ("test", "test_hituav"))),
                                 (args.llvip, "llvip", (("train", "train"), ("test", "test_llvip")))):
            if not root:
                continue
            for src, dst in plan:
                files = images_in(Path(root) / "images" / src)
                if not files:
                    sys.exit(f"no images in {Path(root) / 'images' / src}")
                for f in files:
                    pk.add(dst, f.name, read_yolo(Path(root) / "labels" / src / (f.stem + ".txt")), src=f, kind=kind)
                splits.setdefault(kind, {})[src] = f"{dst} ({len(files)})"

        # Balanced list (consecutive video frames are near-duplicates): Packer kept every
        # steps[kind]-th train frame of each source (default: every 2nd POP frame and flood copy).
        train, mix = [], {}
        for path, kind, people in pk.train:
            train.append(path)
            mix[kind] = mix.get(kind, 0) + 1
        z.writestr("sar/train.txt", "\n".join(train) + "\n")
        z.writestr("sar/splits.json", json.dumps({**splits, "counts": pk.count, "boxes": pk.boxes,
                                                   "train_list": len(train), "train_mix": mix}, indent=1))
        if args.weights != "none" and Path(args.weights).is_file():
            z.write(args.weights, "weights/start.pt")
    for s in sorted(pk.count):
        print(f"{s:12s} {pk.count[s]:6d} images {pk.boxes[s]:7d} boxes")
    print(f"train.txt {len(train)} entries {mix}; source frames seen {pk.seen}; start weights: {args.weights}")
    print(f"Wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
