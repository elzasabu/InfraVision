"""RGBTDronePerson (thermal half) -> YOLO person dataset.

    .venv/bin/python human_detection/datasets/convert_rgbtdroneperson.py
    .venv/bin/python human_detection/datasets/convert_rgbtdroneperson.py --src ../datasets/RGBTDronePerson --out human_detection/datasets/rgbt

Source: Zhang et al., "Drone-based RGBT tiny person detection", ISPRS J. 2023, CC BY 4.0,
https://nnnnerd.github.io/RGBTDronePerson/ (Google Drive: RGBTDronePerson.zip + *_thermal.json).
Unzip only the thermal part: unzip RGBTDronePerson.zip 'train/thermal/*' 'val/thermal/*'.

Classes: person and rider -> 0 (person). `crowd` (one box over a group) and `uncertain` boxes
are not labelled as people; their pixels are inpainted with the surrounding background so the
model is not punished for finding the people inside them. 640x512 grey, median person 13 px.
Writes <out>/images/{train,val}, <out>/labels/{train,val}.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
KEEP = {"person", "rider"}
MASK = {"crowd", "uncertain"}


def convert(src, out, split, pad=2):
    d = json.loads((src / f"{split}_thermal.json").read_text())
    names = {c["id"]: c["name"] for c in d["categories"]}
    by_img = {}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n_img = n_box = n_mask = 0
    for im in d["images"]:
        f = src / split / "thermal" / im["file_name"]
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        rows, mask = [], np.zeros((h, w), np.uint8)
        for a in by_img.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            name = names[a["category_id"]]
            if name in KEEP and bw > 1 and bh > 1:
                x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                rows.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            elif name in MASK:
                mask[max(0, int(y) - pad):int(y + bh) + pad, max(0, int(x) - pad):int(x + bw) + pad] = 255
                n_mask += 1
        if mask.any():
            img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        stem = f"rgbt_{split}_{Path(im['file_name']).stem}"
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n_img += 1
        n_box += len(rows)
    print(f"{split}: {n_img} images, {n_box} person boxes, {n_mask} crowd/uncertain regions inpainted")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(REPO.parent / "datasets" / "RGBTDronePerson"))
    p.add_argument("--out", default=str(REPO / "human_detection" / "datasets" / "rgbt"))
    a = p.parse_args()
    for split in ("train", "val"):
        convert(Path(a.src), Path(a.out), split)


if __name__ == "__main__":
    main()
