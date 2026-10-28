
from pathlib import Path
from ultralytics import YOLO
import time

ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT / "runs" / "pop" / "baseline"
    / "weights" / "best.pt"
)

DATA_YAML = ROOT / "datasets" / "pop" / "data.yaml"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

# Load the trained model
model = YOLO(str(MODEL_PATH))

# Evaluate on the held-out test set
metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=8,
    plots=True
)

print("\n========== TEST RESULTS ==========")
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP@50:", metrics.box.map50)
print("mAP@50-95:", metrics.box.map)

print("\nDetailed metrics:")
print(metrics.results_dict)

# Estimate inference latency and FPS on a test image
test_dir = ROOT / "datasets" / "pop" / "images" / "test"
extensions = {".jpg", ".jpeg", ".png", ".bmp"}

test_images = [
    p for p in test_dir.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
]

if test_images:
    # Warm-up
    model.predict(
        source=str(test_images[0]),
        imgsz=640,
        verbose=False
    )

    # Measure repeated single-image inference
    repetitions = min(50, len(test_images))
    start = time.perf_counter()

    for i in range(repetitions):
        image_path = test_images[i % len(test_images)]
        model.predict(
            source=str(image_path),
            imgsz=640,
            verbose=False
        )

    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / repetitions) * 1000
    fps = repetitions / elapsed

    print("\n========== INFERENCE SPEED ==========")
    print(f"Average latency: {latency_ms:.2f} ms/image")
    print(f"Throughput: {fps:.2f} images/second")
else:
    print("No test images found for speed measurement.")
