
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "datasets" / "pop" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_YAML}")

IMAGE_SIZES = [256, 320, 640]

for size in IMAGE_SIZES:
    print(f"\n{'=' * 50}")
    print(f"Training YOLO11s at image size {size}")
    print(f"{'=' * 50}")

    # Start each experiment from the same pretrained checkpoint
    model = YOLO("yolo11s.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=size,
        batch=8,
        patience=10,
        workers=2,
        seed=42,
        project=str(ROOT / "runs" / "pop_sizes"),
        name=f"imgsz_{size}",
        exist_ok=True
    )

    print(f"Finished training at image size {size}")

print("\nAll three experiments completed.")
