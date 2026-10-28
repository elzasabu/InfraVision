
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

BASE = Path(__file__).resolve().parent / "flir_align"

ANNOTATION_FILE = BASE / "coco_annotations" / "test.json"
IMAGE_FOLDER = BASE / "thermal" / "test"
LABEL_FOLDER = BASE / "labels" / "test"

LABEL_FOLDER.mkdir(parents=True, exist_ok=True)

with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

person_ids = {
    c["id"] for c in data["categories"]
    if c["name"].strip().lower()
    in {"person", "pedestrian", "people"}
}

if not person_ids:
    raise ValueError("No person category found in test.json")

image_info = {im["id"]: im for im in data["images"]}
boxes_by_image = defaultdict(list)

for ann in data["annotations"]:
    if ann["category_id"] in person_ids:
        boxes_by_image[ann["image_id"]].append(ann["bbox"])

processed = 0
missing = 0

for image_id, info in image_info.items():
    filename = Path(info["file_name"]).name
    image_path = IMAGE_FOLDER / filename

    if not image_path.exists():
        matches = list(IMAGE_FOLDER.rglob(filename))
        if not matches:
            print("Missing:", filename)
            missing += 1
            continue
        image_path = matches[0]

    with Image.open(image_path) as img:
        width, height = img.size

    lines = []

    for x, y, w, h in boxes_by_image.get(image_id, []):
        if w <= 0 or h <= 0:
            continue

        x1 = max(0, min(width, x))
        y1 = max(0, min(height, y))
        x2 = max(0, min(width, x + w))
        y2 = max(0, min(height, y + h))

        if x2 <= x1 or y2 <= y1:
            continue

        xc = ((x1 + x2) / 2) / width
        yc = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        lines.append(
            f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    (LABEL_FOLDER / f"{image_path.stem}.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    processed += 1

print("Test conversion completed!")
print("Images processed:", processed)
print("Images missing:", missing)
print("Labels saved to:", LABEL_FOLDER)
