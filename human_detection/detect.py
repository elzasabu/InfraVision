""" from ultralytics import YOLO

# Load the trained model
model = YOLO(
    "runs/pop/baseline/weights/best.pt"
)

# Run detection
results = model.predict(
    source="test_thermal.png",
    imgsz=640,
    conf=0.25,
    save=True
)

# Print the detections
for result in results:
    count = len(result.boxes)

    print("Humans detected:", count)

    for box in result.boxes:
        confidence = float(box.conf[0])

        print(
            "Confidence:",
            round(confidence * 100, 2),
            "%"
        )
        """

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "pop" / "baseline" / "weights" / "best.pt"
TEST_FOLDER = ROOT / "datasets" / "pop" / "images" / "test"

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(TEST_FOLDER),
    imgsz=640,
    conf=0.25,
    save=True
)

print("Prediction completed!")
