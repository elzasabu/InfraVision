import os
import random

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image

from torchmetrics.functional.image import (
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

INPUT_DIR = "dataset_pop/train/input"
TARGET_DIR = "dataset_pop/train/target"

# Your already-trained EDSR
EDSR_CHECKPOINT = (
    "checkpoints_pop/ssim_residual/"
    "edsr_ssim_residual_epoch_10.pth"
)

# SwinIR checkpoints
OUTPUT_DIR = "checkpoints_pop/edsr_swin"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# Training settings
# ------------------------------------------------------------

EPOCHS = 10

BATCH_SIZE = 1

LEARNING_RATE = 1e-4

# Small patch = much lower RAM usage
PATCH_SIZE = 64

# Print progress every N batches
PRINT_EVERY = 5


# ============================================================
# DATASET
# ============================================================

class SRDataset(Dataset):

    def __init__(
        self,
        input_dir,
        target_dir,
        patch_size=64
    ):

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.patch_size = patch_size

        self.input_images = sorted([
            f
            for f in os.listdir(input_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

        self.target_images = sorted([
            f
            for f in os.listdir(target_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

        if len(self.input_images) != len(
            self.target_images
        ):

            raise RuntimeError(
                f"Input images: {len(self.input_images)}\n"
                f"Target images: {len(self.target_images)}"
            )

        self.transform = transforms.ToTensor()

        print()
        print("Dataset loaded.")
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

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        input_image = self.transform(
            input_image
        )

        target_image = self.transform(
            target_image
        )

        # ----------------------------------------------------
        # Check same size
        # ----------------------------------------------------

        if input_image.shape != target_image.shape:

            raise RuntimeError(
                "\nInput and target sizes do not match.\n"
                f"Input : {input_image.shape}\n"
                f"Target: {target_image.shape}\n"
                f"Files:\n"
                f"{self.input_images[index]}\n"
                f"{self.target_images[index]}"
            )

        _, height, width = input_image.shape

        if (
            height < self.patch_size
            or
            width < self.patch_size
        ):

            raise RuntimeError(
                "Image is smaller than patch size."
            )

        # ----------------------------------------------------
        # RANDOM ALIGNED CROP
        #
        # IMPORTANT:
        # Input and target use EXACTLY the same
        # crop coordinates.
        # ----------------------------------------------------

        top = random.randint(
            0,
            height - self.patch_size
        )

        left = random.randint(
            0,
            width - self.patch_size
        )

        input_patch = input_image[
            :,
            top:top + self.patch_size,
            left:left + self.patch_size
        ]

        target_patch = target_image[
            :,
            top:top + self.patch_size,
            left:left + self.patch_size
        ]

        return (
            input_patch,
            target_patch
        )


# ============================================================
# START
# ============================================================

print("=" * 60)
print("EDSR -> SWINIR PATCH TRAINING")
print("=" * 60)

print()
print("Device:", DEVICE)

print()
print("Patch size:", PATCH_SIZE)
print("Batch size:", BATCH_SIZE)


# ============================================================
# DATASET
# ============================================================

dataset = SRDataset(
    INPUT_DIR,
    TARGET_DIR,
    patch_size=PATCH_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print()
print(
    "Training samples:",
    len(dataset)
)

print(
    "Batches per epoch:",
    len(loader)
)


# ============================================================
# EDSR
# ============================================================

print()
print("Loading EDSR:")
print(EDSR_CHECKPOINT)


edsr = EDSR(
    in_channels=3,
    out_channels=3,
    channels=64,
    num_blocks=8
).to(DEVICE)


checkpoint = torch.load(
    EDSR_CHECKPOINT,
    map_location=DEVICE
)

edsr.load_state_dict(
    checkpoint
)

print(
    "EDSR loaded successfully."
)


# ============================================================
# FREEZE EDSR
# ============================================================

for parameter in edsr.parameters():

    parameter.requires_grad = False


edsr.eval()

print(
    "EDSR frozen."
)


# ============================================================
# SWINIR
# ============================================================

print()
print("Creating SwinIR refinement network...")


swin = SwinIRRefinement(
    in_channels=3,
    embed_dim=48,
    num_groups=4,
    depth=4,
    num_heads=4,
    window_size=8
).to(DEVICE)


print(
    "SwinIR created."
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    swin.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LOSSES
# ============================================================

l1_loss = nn.L1Loss()


# ============================================================
# TRAINING
# ============================================================

loss_history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    swin.train()

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0


    print()
    print("=" * 60)
    print(
        f"Epoch [{epoch}/{EPOCHS}]"
    )
    print("=" * 60)


    # ========================================================
    # BATCH LOOP
    # ========================================================

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


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # EDSR
        #
        # Frozen.
        # ----------------------------------------------------

        with torch.no_grad():

            edsr_output = edsr(
                input_image
            )


        # ----------------------------------------------------
        # SWINIR
        # ----------------------------------------------------

        prediction = swin(
            edsr_output
        )


        # ----------------------------------------------------
        # Make absolutely sure dimensions match
        # ----------------------------------------------------

        if prediction.shape != target_image.shape:

            raise RuntimeError(
                "\nSIZE MISMATCH\n"
                f"Prediction: {prediction.shape}\n"
                f"Target:     {target_image.shape}"
            )


        # ----------------------------------------------------
        # L1 LOSS
        # ----------------------------------------------------

        loss_l1 = l1_loss(
            prediction,
            target_image
        )


        # ----------------------------------------------------
        # SSIM LOSS
        # ----------------------------------------------------

        prediction_for_ssim = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        ssim = (
            structural_similarity_index_measure(
                prediction_for_ssim,
                target_image,
                data_range=1.0
            )
        )

        loss_ssim = 1.0 - ssim


        # ----------------------------------------------------
        # COMBINED LOSS
        #
        # L1 is dominant.
        # SSIM encourages structural similarity.
        # ----------------------------------------------------

        loss = (
            0.9 * loss_l1
            +
            0.1 * loss_ssim
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        total_loss += loss.item()

        total_l1 += loss_l1.item()

        total_ssim_loss += (
            loss_ssim.item()
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % PRINT_EVERY == 0
            or
            (batch_idx + 1) == len(loader)
        ):

            print(
                f"Batch "
                f"[{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.6f} "
                f"L1: {loss_l1.item():.6f} "
                f"SSIM Loss: "
                f"{loss_ssim.item():.6f}"
            )


    # ========================================================
    # EPOCH RESULTS
    # ========================================================

    avg_loss = (
        total_loss /
        len(loader)
    )

    avg_l1 = (
        total_l1 /
        len(loader)
    )

    avg_ssim_loss = (
        total_ssim_loss /
        len(loader)
    )

    loss_history.append(
        avg_loss
    )


    print()
    print(
        f"Epoch {epoch} completed."
    )

    print(
        f"Average Loss     : "
        f"{avg_loss:.6f}"
    )

    print(
        f"Average L1       : "
        f"{avg_l1:.6f}"
    )

    print(
        f"Average SSIM Loss: "
        f"{avg_ssim_loss:.6f}"
    )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"edsr_swin_epoch_{epoch}.pth"
    )

    torch.save(
        swin.state_dict(),
        checkpoint_path
    )

    print()
    print(
        "Checkpoint saved:"
    )

    print(
        checkpoint_path
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 60)
print("EDSR -> SWINIR TRAINING COMPLETE")
print("=" * 60)

print()
print("Loss history:")

print(
    loss_history
)

print()
print("Checkpoints saved in:")

print(
    OUTPUT_DIR
)