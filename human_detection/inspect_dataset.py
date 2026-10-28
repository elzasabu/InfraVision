import json
import os

base = "datasets/flir_align"

json_path = os.path.join(
    base, "coco_annotations", "train.json"
)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Number of images:", len(data.get("images", [])))
print("Number of annotations:", len(data.get("annotations", [])))
print("Categories:", data.get("categories", []))

# Look up image details using image ID
image_map = {
    image["id"]: image
    for image in data.get("images", [])
}

print("\nFirst 5 image records:")
for image in data.get("images", [])[:5]:
    print(image)

print("\nFirst 10 annotations:")
for ann in data.get("annotations", [])[:10]:
    image = image_map.get(ann["image_id"], {})
    print({
        "image": image.get("file_name"),
        "bbox": ann.get("bbox"),
        "category_id": ann.get("category_id")
    })