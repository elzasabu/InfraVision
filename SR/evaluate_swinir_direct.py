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

# ------------------------------------------------------------
# Test dataset
# ------------------------------------------------------------

INPUT_DIR = "dataset_pop/test/input"
TARGET_DIR = "dataset_pop/test/target"

# ------------------------------------------------------------
# EDSR checkpoint
# ------------------------------------------------------------

EDSR_CHECKPOINT = (
    "checkpoints_pop/ssim_residual/"
    "edsr_ssim_residual_epoch_10.pth"
)

# ------------------------------------------------------------
# DIRECT SWINIR checkpoint
# ------------------------------------------------------------

SWINIR_CHECKPOINT = (
    "checkpoints_pop/swinir_direct/"
    "swinir_direct_epoch_20.pth"
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
        print(
            "Input images :",
            len(self.input_images)
        )

        print(
            "Target images:",
            len(self.target_images)
        )

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

        # ----------------------------------------------------
        # Make sure images are 256 x 256
        # ----------------------------------------------------

        if input_image.shape[-2:] != (256, 256):

            raise RuntimeError(
                f"Input image is not 256x256:\n"
                f"{self.input_images[index]}\n"
                f"Size: {input_image.shape[-2:]}"
            )

        if target_image.shape[-2:] != (256, 256):

            raise RuntimeError(
                f"Target image is not 256x256:\n"
                f"{self.target_images[index]}\n"
                f"Size: {target_image.shape[-2:]}"
            )

        return input_image, target_image


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(model, checkpoint_path):

    print()
    print("Loading checkpoint:")
    print(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print("DIRECT SWINIR EVALUATION")
print("=" * 70)

print()
print("Pipeline:")
print("Input -> Direct SwinIR -> Target")

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

print()
print("Test images:", len(dataset))


# ============================================================
# EDSR MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING EDSR")
print("=" * 70)

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

for parameter in edsr.parameters():

    parameter.requires_grad = False


# ============================================================
# DIRECT SWINIR MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING DIRECT SWINIR")
print("=" * 70)

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
    SWINIR_CHECKPOINT
)

swinir.eval()


# ============================================================
# METRIC VARIABLES
# ============================================================

total_input_psnr = 0.0
total_input_ssim = 0.0

total_edsr_psnr = 0.0
total_edsr_ssim = 0.0

total_swinir_psnr = 0.0
total_swinir_ssim = 0.0

num_images = 0


# ============================================================
# VISUALIZATION VARIABLES
# ============================================================

visual_input = None
visual_edsr = None
visual_swinir = None
visual_target = None


# ============================================================
# TESTING
# ============================================================

print()
print("=" * 70)
print("RUNNING TEST")
print("=" * 70)

with torch.no_grad():

    for batch_idx, (
        input_image,
        target_image
    ) in enumerate(loader):

        input_image = input_image.to(
            DEVICE
        )

        target_image = target_image.to(
            DEVICE
        )

        # ====================================================
        # INPUT BASELINE
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
        # EDSR
        # ====================================================

        edsr_output = edsr(
            input_image
        )

        # ----------------------------------------------------
        # Check size
        # ----------------------------------------------------

        if edsr_output.shape[-2:] != (256, 256):

            raise RuntimeError(
                "EDSR output size is incorrect.\n"
                f"Input : {input_image.shape}\n"
                f"Output: {edsr_output.shape}"
            )

        edsr_output = torch.clamp(
            edsr_output,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # EDSR metrics
        # ----------------------------------------------------

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
        # DIRECT SWINIR
        # ====================================================

        # IMPORTANT:
        #
        # Direct SwinIR receives the ORIGINAL input.
        #
        # NOT EDSR output.
        #
        # Input -> SwinIR -> Target
        # ====================================================

        swinir_output = swinir(
            input_image
        )

        # ----------------------------------------------------
        # Check size
        # ----------------------------------------------------

        if swinir_output.shape[-2:] != (256, 256):

            raise RuntimeError(
                "SwinIR output size is incorrect.\n"
                f"Input : {input_image.shape}\n"
                f"Output: {swinir_output.shape}"
            )

        swinir_output = torch.clamp(
            swinir_output,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # SwinIR metrics
        # ----------------------------------------------------

        swinir_psnr = peak_signal_noise_ratio(
            swinir_output,
            target_image,
            data_range=1.0
        )

        swinir_ssim = structural_similarity_index_measure(
            swinir_output,
            target_image,
            data_range=1.0
        )

        # ====================================================
        # ACCUMULATE METRICS
        # ====================================================

        total_input_psnr += (
            input_psnr.item()
        )

        total_input_ssim += (
            input_ssim.item()
        )

        total_edsr_psnr += (
            edsr_psnr.item()
        )

        total_edsr_ssim += (
            edsr_ssim.item()
        )

        total_swinir_psnr += (
            swinir_psnr.item()
        )

        total_swinir_ssim += (
            swinir_ssim.item()
        )

        num_images += 1

        # ====================================================
        # SAVE FIRST IMAGE
        # ====================================================

        if visual_input is None:

            visual_input = (
                input_image[0]
                .cpu()
            )

            visual_edsr = (
                edsr_output[0]
                .cpu()
            )

            visual_swinir = (
                swinir_output[0]
                .cpu()
            )

            visual_target = (
                target_image[0]
                .cpu()
            )

        # ====================================================
        # PROGRESS
        # ====================================================

        if (batch_idx + 1) % 25 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1}/"
                f"{len(loader)}"
            )


# ============================================================
# CALCULATE AVERAGES
# ============================================================

avg_input_psnr = (
    total_input_psnr /
    num_images
)

avg_input_ssim = (
    total_input_ssim /
    num_images
)

avg_edsr_psnr = (
    total_edsr_psnr /
    num_images
)

avg_edsr_ssim = (
    total_edsr_ssim /
    num_images
)

avg_swinir_psnr = (
    total_swinir_psnr /
    num_images
)

avg_swinir_ssim = (
    total_swinir_ssim /
    num_images
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print()
print(
    "Number of test images:",
    num_images
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
print("DIRECT SWINIR")

print(
    f"PSNR : {avg_swinir_psnr:.4f} dB"
)

print(
    f"SSIM : {avg_swinir_ssim:.4f}"
)


# ============================================================
# IMPROVEMENT OVER EDSR
# ============================================================

psnr_difference = (
    avg_swinir_psnr -
    avg_edsr_psnr
)

ssim_difference = (
    avg_swinir_ssim -
    avg_edsr_ssim
)

print()
print("=" * 70)
print("SWINIR vs EDSR")
print("=" * 70)

print(
    f"PSNR difference : "
    f"{psnr_difference:+.4f} dB"
)

print(
    f"SSIM difference : "
    f"{ssim_difference:+.4f}"
)


# ============================================================
# CONVERT TENSORS TO IMAGES
# ============================================================

def tensor_to_image(tensor):

    image = tensor.permute(
        1, 2, 0
    ).numpy()

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


input_np = tensor_to_image(
    visual_input
)

edsr_np = tensor_to_image(
    visual_edsr
)

swinir_np = tensor_to_image(
    visual_swinir
)

target_np = tensor_to_image(
    visual_target
)


# ============================================================
# VISUAL COMPARISON
# ============================================================

plt.figure(
    figsize=(20, 5)
)


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    1
)

plt.imshow(
    input_np
)

plt.title(
    "Degraded Input"
)

plt.axis("off")


# ------------------------------------------------------------
# EDSR
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    2
)

plt.imshow(
    edsr_np
)

plt.title(
    "EDSR Output"
)

plt.axis("off")


# ------------------------------------------------------------
# Direct SwinIR
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    3
)

plt.imshow(
    swinir_np
)

plt.title(
    "Direct SwinIR"
)

plt.axis("off")


# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

plt.subplot(
    1,
    4,
    4
)

plt.imshow(
    target_np
)

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
print("DIRECT SWINIR EVALUATION COMPLETE")
print("=" * 70)