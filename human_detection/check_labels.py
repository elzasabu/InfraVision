
from pathlib import Path

BASE = Path("datasets/flir_align")

for split in ["train", "test"]:
    image_dir = BASE / "thermal" / split
    label_dir = BASE / "labels" / split

    images = [
        p for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    missing_labels = []
    empty_labels = 0
    person_boxes = 0

    for image in images:
        label = label_dir / f"{image.stem}.txt"

        if not label.exists():
            missing_labels.append(image.name)
            continue

        content = label.read_text(encoding="utf-8").strip()

        if not content:
            empty_labels += 1
        else:
            person_boxes += len(content.splitlines())

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Missing label files:", len(missing_labels))
    print("Empty label files:", empty_labels)
    print("Person boxes:", person_boxes)

    if missing_labels:
        print("Examples:", missing_labels[:5])
