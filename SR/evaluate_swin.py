import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure
)

from models.edsr import EDSR

from models.swinir_refine import SwinIRRefinement


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_DIR = "dataset_pop/test/input"
TARGET_DIR = "dataset_pop/test/target"

EDSR_CHECKPOINT = (
    "checkpoints_pop/ssim_residual/"
    "edsr_ssim_residual_epoch_10.pth"
)

SWIN_CHECKPOINT = (
    "checkpoints_pop/edsr_swin/"
    "edsr_swin_epoch_10.pth"
)


# ============================================================
# DATASET
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

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        # ----------------------------------------------------
        # REQUIRE SAME SIZE
        # ----------------------------------------------------

        if input_image.shape != target_image.shape:

            raise RuntimeError(
                f"Size mismatch!\n"
                f"Input : {input_image.shape}\n"
                f"Target: {target_image.shape}"
            )

        return input_image, target_image


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, path):

    print()
    print("Loading checkpoint:")
    print(path)

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# START
# ============================================================

print("=" * 70)
print("EDSR + SWINIR REFINEMENT EVALUATION")
print("=" * 70)

print()
print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

dataset = SRDataset(
    INPUT_DIR,
    TARGET_DIR
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ============================================================
# EDSR
# ============================================================

print()
print("Creating EDSR...")

edsr = EDSR(
    in_channels=3,
    out_channels=3,
    channels=64,
    num_blocks=8
).to(DEVICE)


edsr = load_checkpoint(
    edsr,
    EDSR_CHECKPOINT
)

edsr.eval()

# Freeze EDSR
for parameter in edsr.parameters():
    parameter.requires_grad = False


# ============================================================
# SWINIR REFINEMENT
# ============================================================

print()
print("Creating SwinIR Refinement...")

swinir = SwinIRRefinement(
    in_channels=3,
    embed_dim=48,
    num_groups=4,
    depth=4,
    num_heads=4,
    window_size=8
).to(DEVICE)


swinir = load_checkpoint(
    swinir,
    SWIN_CHECKPOINT
)

swinir.eval()


# ============================================================
# EVALUATION
# ============================================================

total_psnr_input = 0.0
total_ssim_input = 0.0

total_psnr_edsr = 0.0
total_ssim_edsr = 0.0

total_psnr_swin = 0.0
total_ssim_swin = 0.0

num_images = 0


# First image for visualization
visual_input = None
visual_edsr = None
visual_swin = None
visual_target = None


print()
print("=" * 70)
print("RUNNING TEST")
print("=" * 70)


with torch.no_grad():

    for batch_idx, (input_image, target_image) in enumerate(loader):

        input_image = input_image.to(DEVICE)
        target_image = target_image.to(DEVICE)

        # ----------------------------------------------------
        # Check original size
        # ----------------------------------------------------

        if input_image.shape[-2:] != (256, 256):

            raise RuntimeError(
                f"Expected input 256x256.\n"
                f"Received: {input_image.shape[-2:]}"
            )

        if target_image.shape[-2:] != (256, 256):

            raise RuntimeError(
                f"Expected target 256x256.\n"
                f"Received: {target_image.shape[-2:]}"
            )

        # ====================================================
        # BASELINE: INPUT
        # ====================================================

        input_psnr = peak_signal_noise_ratio(
            input_image,
            target_image,
            data_range=1.0
        )

        input_ssim = structural_similarity_index_measure(
            input_image,
            target_image,
            data_range=1.0
        )


        # ====================================================
        # STAGE 1: EDSR
        # ====================================================

        edsr_output = edsr(
            input_image
        )

        # ----------------------------------------------------
        # EDSR must preserve 256x256
        # ----------------------------------------------------

        if edsr_output.shape[-2:] != (256, 256):

            raise RuntimeError(
                "EDSR changed image size!\n"
                f"Input : {input_image.shape}\n"
                f"EDSR  : {edsr_output.shape}"
            )

        edsr_output = torch.clamp(
            edsr_output,
            0.0,
            1.0
        )


        edsr_psnr = peak_signal_noise_ratio(
            edsr_output,
            target_image,
            data_range=1.0
        )

        edsr_ssim = structural_similarity_index_measure(
            edsr_output,
            target_image,
            data_range=1.0
        )


        # ====================================================
        # STAGE 2: SWINIR REFINEMENT
        # ====================================================

        swin_output = swinir(
            edsr_output
        )

        # ----------------------------------------------------
        # SwinIR must preserve 256x256
        # ----------------------------------------------------

        if swin_output.shape[-2:] != (256, 256):

            raise RuntimeError(
                "SwinIR changed image size!\n"
                f"EDSR  : {edsr_output.shape}\n"
                f"Swin  : {swin_output.shape}"
            )

        swin_output = torch.clamp(
            swin_output,
            0.0,
            1.0
        )


        # ====================================================
        # FINAL METRICS
        # ====================================================

        swin_psnr = peak_signal_noise_ratio(
            swin_output,
            target_image,
            data_range=1.0
        )

        swin_ssim = structural_similarity_index_measure(
            swin_output,
            target_image,
            data_range=1.0
        )


        total_psnr_input += input_psnr.item()
        total_ssim_input += input_ssim.item()

        total_psnr_edsr += edsr_psnr.item()
        total_ssim_edsr += edsr_ssim.item()

        total_psnr_swin += swin_psnr.item()
        total_ssim_swin += swin_ssim.item()

        num_images += 1


        # ====================================================
        # SAVE FIRST IMAGE
        # ====================================================

        if visual_input is None:

            visual_input = input_image[0].cpu()
            visual_edsr = edsr_output[0].cpu()
            visual_swin = swin_output[0].cpu()
            visual_target = target_image[0].cpu()


        # ====================================================
        # PROGRESS
        # ====================================================

        if (batch_idx + 1) % 25 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1}/{len(loader)}"
            )


# ============================================================
# AVERAGES
# ============================================================

avg_input_psnr = (
    total_psnr_input / num_images
)

avg_input_ssim = (
    total_ssim_input / num_images
)

avg_edsr_psnr = (
    total_psnr_edsr / num_images
)

avg_edsr_ssim = (
    total_ssim_edsr / num_images
)

avg_swin_psnr = (
    total_psnr_swin / num_images
)

avg_swin_ssim = (
    total_ssim_swin / num_images
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print()
print(
    f"Number of test images: {num_images}"
)

print()
print("INPUT")
print(
    f"PSNR : {avg_input_psnr:.4f} dB"
)
print(
    f"SSIM : {avg_input_ssim:.4f}"
)

print()
print("EDSR")
print(
    f"PSNR : {avg_edsr_psnr:.4f} dB"
)
print(
    f"SSIM : {avg_edsr_ssim:.4f}"
)

print()
print("EDSR + SWINIR REFINEMENT")
print(
    f"PSNR : {avg_swin_psnr:.4f} dB"
)
print(
    f"SSIM : {avg_swin_ssim:.4f}"
)


# ============================================================
# VISUALIZATION
# ============================================================

def tensor_to_image(tensor):

    image = tensor.permute(
        1, 2, 0
    ).numpy()

    return np.clip(
        image,
        0.0,
        1.0
    )


input_np = tensor_to_image(
    visual_input
)

edsr_np = tensor_to_image(
    visual_edsr
)

swin_np = tensor_to_image(
    visual_swin
)

target_np = tensor_to_image(
    visual_target
)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(
    figsize=(20, 5)
)


plt.subplot(1, 4, 1)

plt.imshow(input_np)

plt.title(
    "Degraded Input"
)

plt.axis("off")


plt.subplot(1, 4, 2)

plt.imshow(edsr_np)

plt.title(
    "EDSR Output"
)

plt.axis("off")


plt.subplot(1, 4, 3)

plt.imshow(swin_np)

plt.title(
    "EDSR + SwinIR Refinement"
)

plt.axis("off")


plt.subplot(1, 4, 4)

plt.imshow(target_np)

plt.title(
    "HR Target"
)

plt.axis("off")


plt.tight_layout()

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("EDSR + SWINIR REFINEMENT EVALUATION COMPLETE")
print("=" * 70)