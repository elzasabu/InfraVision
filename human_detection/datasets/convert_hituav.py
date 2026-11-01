"""HIT-UAV -> YOLO person dataset.

    .venv/bin/python human_detection/datasets/convert_hituav.py

Source: Suo et al., "HIT-UAV: A high-altitude infrared thermal dataset for UAV-based object
detection", Sci. Data 2023. Zenodo 7633120 (v1.2, md5 3cf99b2787dcdf6228bd513a36bd70db).
Unzip the json part: unzip HIT-UAV-v1.2.zip '*/normal_json/*' -d x
2898 thermal frames 640x512, 60-130 m altitude, schools / roads / parking lots (no water).

Person -> 0. `Bicycle` is inpainted with the background: in thermal a visible bicycle is almost
always ridden and the rider has no Person box, so it would teach the model to ignore people.
`DontCare` is inpainted too; cars / other vehicles are left as background.
Writes <out>/images/{train,val,test}, <out>/labels/{train,val,test} (file names hituav_*).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO.parent / "datasets" / "HIT-UAV" / "x" / "suojiashun-HIT-UAV-Infrared-Thermal-Dataset-f6acd28" / "normal_json"


def convert(src, out, split, pad=2):
    d = json.loads((src / "annotations" / f"{split}.json").read_text())
    names = {c["id"]: c["name"] for c in d["categories"]}
    by_img = {}
    for a in d["annotation"]:
        by_img.setdefault(a["image_id"], []).append(a)
    for kind in ("images", "labels"):
        (out / kind / split).mkdir(parents=True, exist_ok=True)
    n_img = n_box = n_mask = 0
    for im in d["images"]:
        img = cv2.imread(str(src / split / im["filename"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        rows, mask = [], np.zeros((h, w), np.uint8)
        for a in by_img.get(im["id"], []):
            x, y, bw, bh = a["bbox"]
            name = names[a["category_id"]]
            if name == "Person" and bw > 1 and bh > 1:
                x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)
                rows.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            elif name in ("Bicycle", "DontCare"):
                mask[max(0, int(y) - pad):int(y + bh) + pad, max(0, int(x) - pad):int(x + bw) + pad] = 255
                n_mask += 1
        if mask.any():
            img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        stem = f"hituav_{Path(im['filename']).stem}"
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n_img += 1
        n_box += len(rows)
    print(f"{split}: {n_img} images, {n_box} person boxes, {n_mask} bicycle/dontcare regions inpainted")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--out", default=str(REPO / "human_detection" / "datasets" / "hituav"))
    a = p.parse_args()
    for split in ("train", "val", "test"):
        convert(Path(a.src), Path(a.out), split)


if __name__ == "__main__":
    main()
