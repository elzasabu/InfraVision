
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

# Dataset paths
BASE = Path(__file__).resolve().parent / "flir_align"

ANNOTATION_FILE = BASE / "coco_annotations" / "train.json"
IMAGE_FOLDER = BASE / "thermal" / "train"
LABEL_FOLDER = BASE / "labels" / "train"

LABEL_FOLDER.mkdir(parents=True, exist_ok=True)

# Load COCO JSON
with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Print all categories so we can verify the mapping
print("Dataset categories:")
for category in data["categories"]:
    print(category["id"], ":", category["name"])

# Identify person-related categories by name
person_names = {"person", "pedestrian", "people"}
person_ids = {
    category["id"]
    for category in data["categories"]
    if category["name"].strip().lower() in person_names
}

if not person_ids:
    raise ValueError(
        "No person category found. Check the printed categories."
    )

print("\nPerson category IDs:", person_ids)

# Map COCO image IDs to filenames
image_info = {
    image["id"]: image
    for image in data["images"]
}

# Group person boxes by image ID
boxes_by_image = defaultdict(list)

for ann in data["annotations"]:
    if ann["category_id"] in person_ids:
        boxes_by_image[ann["image_id"]].append(ann["bbox"])

processed = 0
missing = 0
empty = 0

# Convert annotations for each image in the JSON
for image_id, info in image_info.items():

    filename = Path(info["file_name"]).name
    image_path = IMAGE_FOLDER / filename

    # Search subfolders if the image is not directly in the folder
    if not image_path.exists():
        matches = list(IMAGE_FOLDER.rglob(filename))

        if not matches:
            print("Image missing:", filename)
            missing += 1
            continue

        image_path = matches[0]

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    lines = []

    for bbox in boxes_by_image.get(image_id, []):
        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            continue

        # Clip box coordinates to image boundaries
        x1 = max(0, min(img_width, x))
        y1 = max(0, min(img_height, y))
        x2 = max(0, min(img_width, x + w))
        y2 = max(0, min(img_height, y + h))

        if x2 <= x1 or y2 <= y1:
            continue

        # YOLO normalized center coordinates and dimensions
        xc = ((x1 + x2) / 2) / img_width
        yc = ((y1 + y2) / 2) / img_height
        bw = (x2 - x1) / img_width
        bh = (y2 - y1) / img_height

        lines.append(
            f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )

    label_path = LABEL_FOLDER / (image_path.stem + ".txt")
    label_path.write_text("\n".join(lines), encoding="utf-8")

    processed += 1

    if not lines:
        empty += 1

print("\nConversion finished!")
print("Images processed:", processed)
print("Images missing:", missing)
print("Images with no person boxes:", empty)
print("Labels saved to:", LABEL_FOLDER)
