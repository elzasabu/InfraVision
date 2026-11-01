"""LLVIP infrared + person boxes -> YOLO person dataset (large, near, oblique people).

    .venv/bin/python human_detection/datasets/convert_llvip_persons.py --step 8 --test_step 4

LLVIP (Jia et al. 2021): 12025 train / 3463 test thermal frames, 1280x1024, from ~18 fixed
street cameras a few metres up; 42k person boxes (VOC xml in Annotations/, shared by the
visible and infrared images). The SAR detector only saw small, far, top-down people and missed
large close ones; LLVIP adds them. Consecutive frames are near-duplicates, so only every
--step-th train frame (--test_step-th test frame) is kept. Frames are resized to 640x512 grey.
Writes <out>/images/{train,test}, <out>/labels/{train,test} (file names llvip_*).
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]


def convert(src, out, split, step, side=640):
    files = sorted((src / "infrared" / split).glob("*.jpg"))[::step]
    for kind in ("images", "labels"):
        (out / kind / split).mkdir(parents=True, exist_ok=True)
    n_box = 0
    for f in files:
        xml = src / "Annotations" / (f.stem + ".xml")
        rows = []
        if xml.is_file():
            root = ET.parse(xml).getroot()
            W, H = int(root.find("size/width").text), int(root.find("size/height").text)
            for o in root.iter("object"):
                if o.findtext("name") != "person":
                    continue
                b = o.find("bndbox")
                x0, y0, x1, y1 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                if x1 - x0 > 1 and y1 - y0 > 1:
                    rows.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (side, round(side * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        stem = f"llvip_{split}_{f.stem}"
        cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n_box += len(rows)
    print(f"{split}: {len(files)} images, {n_box} person boxes")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(REPO.parent / "datasets" / "LLVIP"))
    p.add_argument("--out", default=str(REPO / "human_detection" / "datasets" / "llvip"))
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--test_step", type=int, default=4)
    a = p.parse_args()
    convert(Path(a.src), Path(a.out), "train", a.step)
    convert(Path(a.src), Path(a.out), "test", a.test_step)


if __name__ == "__main__":
    main()
