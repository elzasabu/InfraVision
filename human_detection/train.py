
from pathlib import Path
from ultralytics import YOLO

# Project directory
ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "datasets" / "pop" / "data.yaml"

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

# Load a pretrained YOLO11 small model
model = YOLO("yolo11s.pt")

# Train on thermal images
results = model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    workers=2,
    project=str(ROOT / "runs" / "pop"),
    name="baseline",
    exist_ok=True
)

print("Training completed!")
print("Best model:", ROOT / "runs" / "pop" / "baseline" / "weights" / "best.pt")
