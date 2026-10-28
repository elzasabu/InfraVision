
from pathlib import Path
import random
import shutil

# Project directory
ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "datasets" / "flir_align"
DEST = ROOT / "datasets" / "thermal_person"

# Reproducible split
random.seed(42)
VAL_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def find_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

def copy_pair(image, label_folder, image_dest, label_dest):
    label = label_folder / f"{image.stem}.txt"

    if not label.exists():
        raise FileNotFoundError(
            f"Missing label for {image.name}: {label}"
        )

    shutil.copy2(image, image_dest / image.name)
    shutil.copy2(label, label_dest / label.name)

def main():
    # Create destination folders
    for split in ("train", "val", "test"):
        (DEST / "images" / split).mkdir(
            parents=True, exist_ok=True
        )
        (DEST / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )

    train_images = find_images(SOURCE / "thermal" / "train")
    test_images = find_images(SOURCE / "thermal" / "test")

    train_labels = SOURCE / "labels" / "train"
    test_labels = SOURCE / "labels" / "test"

    if not train_images or not test_images:
        raise RuntimeError(
            "No images found. Check the thermal folder paths."
        )

    # Check labels before copying
    for image in train_images:
        if not (train_labels / f"{image.stem}.txt").exists():
            raise FileNotFoundError(
                f"Missing training label: {image.name}"
            )

    for image in test_images:
        if not (test_labels / f"{image.stem}.txt").exists():
            raise FileNotFoundError(
                f"Missing test label: {image.name}"
            )

    # Split training images into train and validation
    random.shuffle(train_images)

    val_count = round(len(train_images) * VAL_RATIO)
    val_images = train_images[:val_count]
    actual_train_images = train_images[val_count:]

    # Copy training images and labels
    for image in actual_train_images:
        copy_pair(
            image, train_labels,
            DEST / "images" / "train",
            DEST / "labels" / "train"
        )

    # Copy validation images and labels
    for image in val_images:
        copy_pair(
            image, train_labels,
            DEST / "images" / "val",
            DEST / "labels" / "val"
        )

    # Keep original test set separate
    for image in test_images:
        copy_pair(
            image, test_labels,
            DEST / "images" / "test",
            DEST / "labels" / "test"
        )

    print("\nDataset organized successfully!")
    for split in ("train", "val", "test"):
        image_count = len(list(
            (DEST / "images" / split).iterdir()
        ))
        label_count = len(list(
            (DEST / "labels" / split).iterdir()
        ))
        print(
            f"{split}: {image_count} images, "
            f"{label_count} labels"
        )

    print(f"\nDataset location: {DEST}")

if __name__ == "__main__":
    main()
