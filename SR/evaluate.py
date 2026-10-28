import os
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from models.edsr import EDSR


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEST_INPUT = "dataset_pop/test/input"
TEST_TARGET = "dataset_pop/test/target"

CHECKPOINT = "checkpoints_pop/ssim_residual/edsr_ssim_residual_epoch_10.pth"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Dataset
# ============================================================

class SRDataset(Dataset):

    def __init__(self, input_dir, target_dir):

        self.input_dir = input_dir
        self.target_dir = target_dir

        self.input_images = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

        self.target_images = sorted([
            f for f in os.listdir(target_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

        if len(self.input_images) != len(self.target_images):
            raise RuntimeError(
                f"Input images: {len(self.input_images)}\n"
                f"Target images: {len(self.target_images)}"
            )

        self.transform = transforms.ToTensor()

        print("Test dataset loaded.")
        print("Input images :", len(self.input_images))
        print("Target images:", len(self.target_images))

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, index):

        input_path = os.path.join(
            self.input_dir,
            self.input_images[index]
        )

        target_path = os.path.join(
            self.target_dir,
            self.target_images[index]
        )

        input_image = Image.open(
            input_path
        ).convert("RGB")

        target_image = Image.open(
            target_path
        ).convert("RGB")

        input_image = self.transform(
            input_image
        )

        target_image = self.transform(
            target_image
        )

        return input_image, target_image


# ============================================================
# Start
# ============================================================

print("=" * 60)
print("EDSR MODEL EVALUATION")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# Dataset
# ============================================================

test_dataset = SRDataset(
    TEST_INPUT,
    TEST_TARGET
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = EDSR(
    in_channels=3,
    out_channels=3,
    channels=64,
    num_blocks=8
).to(DEVICE)


# ============================================================
# Load checkpoint
# ============================================================

print()
print("Loading checkpoint:")

print(CHECKPOINT)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )
)

model.eval()

print("Checkpoint loaded successfully.")


# ============================================================
# Metrics
# ============================================================

psnr_metric = PeakSignalNoiseRatio(
    data_range=1.0
)

ssim_metric = StructuralSimilarityIndexMeasure(
    data_range=1.0
)


# ============================================================
# Evaluation
# ============================================================

total_psnr = 0.0
total_ssim = 0.0

num_images = 0


with torch.no_grad():

    for input_image, target_image in test_loader:

        input_image = input_image.to(DEVICE)
        target_image = target_image.to(DEVICE)

        # ----------------------------------------------------
        # EDSR prediction
        # ----------------------------------------------------

        prediction = model(input_image)

        # ----------------------------------------------------
        # Clamp output to valid RGB range
        # ----------------------------------------------------

        prediction = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        psnr = psnr_metric(
            prediction.cpu(),
            target_image.cpu()
        )

        ssim = ssim_metric(
            prediction.cpu(),
            target_image.cpu()
        )

        total_psnr += psnr.item()
        total_ssim += ssim.item()

        num_images += 1


# ============================================================
# Average metrics
# ============================================================

average_psnr = total_psnr / num_images
average_ssim = total_ssim / num_images


print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(
    f"Number of test images: {num_images}"
)

print(
    f"Average PSNR: {average_psnr:.4f} dB"
)

print(
    f"Average SSIM: {average_ssim:.4f}"
)


# ============================================================
# Visual comparison
# ============================================================

print()
print("Generating visual comparison...")


# Use first test image
input_image, target_image = test_dataset[0]

input_tensor = input_image.unsqueeze(0).to(DEVICE)

with torch.no_grad():

    prediction = model(
        input_tensor
    )

prediction = torch.clamp(
    prediction,
    0.0,
    1.0
)


# Convert tensors to images

input_img = input_image.permute(
    1, 2, 0
).numpy()

prediction_img = prediction.squeeze(
    0
).cpu().permute(
    1, 2, 0
).numpy()

target_img = target_image.permute(
    1, 2, 0
).numpy()


# ============================================================
# Plot
# ============================================================

plt.figure(
    figsize=(15, 5)
)

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


comparison_path = os.path.join(
    OUTPUT_DIR,
    "edsr_test_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=150
)

plt.show()


# ============================================================
# Save EDSR output separately
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "edsr_output.png"
)

plt.imsave(
    prediction_path,
    prediction_img
)


print()
print("Comparison saved to:")
print(comparison_path)

print()
print("EDSR output saved to:")
print(prediction_path)

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)