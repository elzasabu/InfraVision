import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.edsr import EDSR


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_DIR = "dataset_pop/test/input"
TARGET_DIR = "dataset_pop/test/target"

CHECKPOINT = "checkpoints_pop/edsr_model.pth"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load model
# ============================================================

print("=" * 60)
print("EDSR SUPER-RESOLUTION TEST")
print("=" * 60)

print("Device:", DEVICE)

model = EDSR(
    in_channels=3,
    out_channels=3,
    channels=64,
    num_blocks=8
).to(DEVICE)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# Find first test image
# ============================================================

input_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
])

target_files = sorted([
    f for f in os.listdir(TARGET_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
])

if len(input_files) == 0:
    raise RuntimeError("No test input images found.")

print("Test images:", len(input_files))

input_name = input_files[0]

# Find corresponding target
if input_name in target_files:
    target_name = input_name
else:
    target_name = target_files[0]


input_path = os.path.join(
    INPUT_DIR,
    input_name
)

target_path = os.path.join(
    TARGET_DIR,
    target_name
)


# ============================================================
# Load images
# ============================================================

transform = transforms.ToTensor()

input_image = Image.open(
    input_path
).convert("RGB")

target_image = Image.open(
    target_path
).convert("RGB")

input_tensor = transform(
    input_image
).unsqueeze(0).to(DEVICE)

target_tensor = transform(
    target_image
).unsqueeze(0).to(DEVICE)


print()
print("Input image :", input_name)
print("Target image:", target_name)

print(
    "Input tensor :",
    input_tensor.shape
)

print(
    "Target tensor:",
    target_tensor.shape
)


# ============================================================
# Generate SR output
# ============================================================

with torch.no_grad():

    prediction = model(
        input_tensor
    )


print(
    "Output tensor:",
    prediction.shape
)


# ============================================================
# Convert tensors to images
# ============================================================

input_img = (
    input_tensor
    .squeeze(0)
    .cpu()
    .permute(1, 2, 0)
    .numpy()
)

prediction_img = (
    prediction
    .squeeze(0)
    .cpu()
    .permute(1, 2, 0)
    .numpy()
)

target_img = (
    target_tensor
    .squeeze(0)
    .cpu()
    .permute(1, 2, 0)
    .numpy()
)


# Clamp output
prediction_img = prediction_img.clip(
    0, 1
)


# ============================================================
# Display comparison
# ============================================================

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)

plt.imshow(input_img)

plt.title(
    "Degraded Input"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(prediction_img)

plt.title(
    "EDSR Output"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(target_img)

plt.title(
    "HR Target"
)

plt.axis("off")


plt.tight_layout()


# ============================================================
# Save comparison
# ============================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "sr_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=150
)

plt.show()


# ============================================================
# Save SR image separately
# ============================================================

sr_path = os.path.join(
    OUTPUT_DIR,
    "sr_output.png"
)

plt.imsave(
    sr_path,
    prediction_img
)


print()
print("=" * 60)
print("SR TEST COMPLETE")
print("=" * 60)

print("Comparison saved to:")
print(comparison_path)

print()
print("SR output saved to:")
print(sr_path)