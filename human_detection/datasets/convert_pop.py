"""Convert POP (COCO json, JPEGImages/) into an Ultralytics YOLO dataset at datasets/pop."""
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POP = ROOT / "POP"
OUT = Path(__file__).resolve().parent / "pop"

for split in ("train", "val", "test"):
    data = json.load(open(POP / split / "Annotations" / f"{split}.json", encoding="utf-8"))
    info = {im["id"]: im for im in data["images"]}
    boxes = defaultdict(list)
    for ann in data["annotations"]:
        boxes[ann["image_id"]].append(ann["bbox"])

    img_dir = OUT / "images" / split
    lbl_dir = OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    missing = empty = dropped = 0
    for iid, im in info.items():
        name = Path(im["file_name"]).name
        src = POP / split / "JPEGImages" / name
        if not src.exists():
            missing += 1
            continue
        dst = img_dir / name
        if not dst.exists():
            try:
                os.link(src, dst)          # same volume: no extra disk space
            except OSError:
                shutil.copy2(src, dst)

        W, H = im["width"], im["height"]
        lines = []
        for x, y, w, h in boxes.get(iid, []):
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 - x1 < 1 or y2 - y1 < 1:
                dropped += 1
                continue
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        if not lines:
            empty += 1
        (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(lines))

    print(f"{split}: images={len(info)} missing={missing} empty(background)={empty} dropped_boxes={dropped}")

(OUT / "data.yaml").write_text(
    f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: person\n")
print("Wrote", OUT / "data.yaml")
