from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

import sys


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# YOLO project
YOLO_MODEL_PATH = (
    ROOT
    / "human_detection"
    / "runs"
    / "pop"
    / "pop_1024"   # POP run from colab_training/train_pop_yolo_*.ipynb
    / "weights"
    / "best.pt"
)

TEST_FOLDER = (
    ROOT
    / "human_detection"
    / "datasets"
    / "pop"   # built by human_detection/datasets/convert_pop.py
    / "images"
    / "test"
)

# SR project (SwinIR is used only for the displayed image)
SR_FOLDER = ROOT / "SR"

SWINIR_CHECKPOINT = (
    SR_FOLDER
    / "checkpoints_pop"   # where SR/train_swinir.py saves
    / "swinir_direct"
    / "swinir_direct_epoch_20.pth"
)

# Output folder
OUTPUT_DIR = ROOT / "integrated_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("INFRAVISION: THERMAL ENHANCEMENT + HUMAN DETECTION")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# 3. VERIFY FILES
# ============================================================

for path in [
    YOLO_MODEL_PATH,
    SWINIR_CHECKPOINT,
    TEST_FOLDER,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# 4. LOAD SWINIR MODEL
# ============================================================

# Add SR folder to Python's import path
sys.path.insert(0, str(SR_FOLDER))

from models.swinir_refine import SwinIRRefinement

swinir = SwinIRRefinement(
    in_channels=3,
    embed_dim=48,
    num_groups=4,
    depth=4,
    num_heads=4,
    window_size=8
).to(DEVICE)

checkpoint = torch.load(
    SWINIR_CHECKPOINT,
    map_location=DEVICE
)

# Support either a raw state_dict or a checkpoint dictionary
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

# Remove DataParallel prefix if present
checkpoint = {
    (k[7:] if k.startswith("module.") else k): v
    for k, v in checkpoint.items()
}

swinir.load_state_dict(checkpoint)

swinir.eval()

print("SwinIR model loaded successfully.")


# ============================================================
# 5. LOAD YOLO MODEL
# ============================================================

yolo = YOLO(str(YOLO_MODEL_PATH))

print("YOLO model loaded successfully.")


# ============================================================
# 6. FIND A TEST IMAGE
# ============================================================

image_extensions = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

test_images = sorted(
    p for p in TEST_FOLDER.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
)

if not test_images:
    raise RuntimeError(
        f"No test images ({', '.join(sorted(image_extensions))}) "
        f"found in:\n{TEST_FOLDER}"
    )

# Use the first image by default. Pass an index into the sorted test
# folder or an image path to pick another one, e.g.:
#   python integrate_infravision.py 200
#   python integrate_infravision.py path/to/image.png
image_choice = sys.argv[1] if len(sys.argv) > 1 else "0"

if image_choice.lstrip("-").isdigit():
    image_index = int(image_choice)

    if not -len(test_images) <= image_index < len(test_images):
        raise IndexError(
            f"Image index {image_index} is out of range: "
            f"{TEST_FOLDER} has {len(test_images)} images "
            f"(valid: 0 to {len(test_images) - 1})."
        )

    input_path = test_images[image_index]
else:
    input_path = Path(image_choice)

    # Allow a bare file name from the test folder.
    if not input_path.exists() and (TEST_FOLDER / input_path).exists():
        input_path = TEST_FOLDER / input_path

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Image not found:\n{image_choice}"
        )

print("\nInput image:", input_path.name)


# ============================================================
# 7. LOAD THERMAL IMAGE
# ============================================================

# Read as RGB because the SwinIR model expects 3 channels.
original_image = Image.open(input_path).convert("RGB")

original_array = np.array(original_image)

transform = transforms.ToTensor()

input_tensor = transform(
    original_image
).unsqueeze(0).to(DEVICE)

print("Input dimensions:", original_array.shape)


# ============================================================
# 8. ENHANCE IMAGE USING SWINIR (for display only)
# ============================================================

# SwinIR needs height and width divisible by its window size,
# so pad with reflection and crop back afterwards.
height, width = input_tensor.shape[-2:]
pad_h = (-height) % swinir.window_size
pad_w = (-width) % swinir.window_size

padded_tensor = torch.nn.functional.pad(
    input_tensor,
    (0, pad_w, 0, pad_h),
    mode="reflect"
)

with torch.no_grad():
    enhanced_tensor = swinir(padded_tensor)[..., :height, :width]

enhanced_array = (
    enhanced_tensor
    .squeeze(0)
    .detach()
    .cpu()
    .permute(1, 2, 0)
    .numpy()
)

# Clip predictions to the valid display range.
enhanced_array = np.clip(
    enhanced_array, 0, 1
)

enhanced_array = (
    enhanced_array * 255
).round().astype(np.uint8)

print("Enhanced dimensions:", enhanced_array.shape)


# ============================================================
# 9. RUN YOLO ON THE ORIGINAL IMAGE
# ============================================================

# YOLO was trained on raw thermal images; giving it the
# SwinIR output roughly halved recall in testing.

results = yolo.predict(
    source=str(input_path),
    # POP images are 1280x1024 with small people; the POP
    # model was trained at 1024.
    imgsz=1024,
    conf=0.25,
    # Stricter overlap filter (default 0.7) removes duplicate
    # boxes on the same person.
    iou=0.5,
    verbose=False
)

result = results[0]

# Detection coordinates are relative to the original image.
boxes = result.boxes

human_count = len(boxes)

print("\nHumans detected:", human_count)


# ============================================================
# 10. PREPARE ENHANCED IMAGE FOR OVERLAY
# ============================================================

# Convert RGB to BGR for OpenCV drawing.
output_bgr = cv2.cvtColor(
    enhanced_array,
    cv2.COLOR_RGB2BGR
)

enhanced_height, enhanced_width = output_bgr.shape[:2]

original_height, original_width = original_array.shape[:2]

# Scale box coordinates if the enhancer changes the image dimensions.
scale_x = enhanced_width / original_width
scale_y = enhanced_height / original_height


# ============================================================
# 11. DRAW HUMAN DETECTIONS
# ============================================================

for box in boxes:

    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    confidence = float(box.conf[0])

    # Transform coordinates to enhanced-image dimensions.
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    # Keep coordinates inside the output image.
    x1 = max(0, min(x1, enhanced_width - 1))
    x2 = max(0, min(x2, enhanced_width - 1))
    y1 = max(0, min(y1, enhanced_height - 1))
    y2 = max(0, min(y2, enhanced_height - 1))

    # Draw bounding box.
    cv2.rectangle(
        output_bgr,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Add confidence score.
    label = f"Person: {confidence * 100:.1f}%"

    cv2.putText(
        output_bgr,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# Add total count.
cv2.putText(
    output_bgr,
    f"Humans detected: {human_count}",
    (15, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 255),
    2
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

enhanced_path = OUTPUT_DIR / "enhanced_thermal.png"

detection_path = OUTPUT_DIR / "infravision_final.png"

comparison_path = OUTPUT_DIR / "infravision_comparison.png"

# Save enhanced image.
Image.fromarray(enhanced_array).save(
    enhanced_path
)

# Save final detection image.
cv2.imwrite(
    str(detection_path),
    output_bgr
)

# Save a comparison figure.
final_rgb = cv2.cvtColor(
    output_bgr,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_array)
plt.title("Original Thermal Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(enhanced_array)
plt.title("SwinIR Enhanced Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(final_rgb)
plt.title(f"InfraVision: {human_count} Humans Detected")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    comparison_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. FINAL STATUS
# ============================================================

print("\n" + "=" * 60)
print("INFRAVISION PROCESSING COMPLETED")
print("=" * 60)

print("Humans detected:", human_count)

print("\nSaved files:")
print("Enhanced image :", enhanced_path)
print("Final output   :", detection_path)
print("Comparison     :", comparison_path)