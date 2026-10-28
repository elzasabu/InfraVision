import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from torchmetrics.functional.image import structural_similarity_index_measure

from models.edsr import EDSR


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_DIR = "dataset_pop/train/input"
TARGET_DIR = "dataset_pop/train/target"

# ------------------------------------------------------------
# NEW EXPERIMENT
# Start from scratch
# ------------------------------------------------------------

CHECKPOINT = None

# Save this experiment separately
OUTPUT_DIR = "checkpoints_pop/ssim_residual"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Training
START_EPOCH = 1
EPOCHS = 10

BATCH_SIZE = 1

LEARNING_RATE = 1e-4


# ============================================================
# Dataset
# ============================================================

class SRDataset(Dataset):

    def __init__(
        self,
        input_dir,
        target_dir
    ):

        self.input_dir = input_dir
        self.target_dir = target_dir

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

        # ----------------------------------------------------
        # Check number of images
        # ----------------------------------------------------

        if len(self.input_images) != len(
            self.target_images
        ):

            raise RuntimeError(
                f"Input images: {len(self.input_images)}\n"
                f"Target images: {len(self.target_images)}"
            )

        self.transform = transforms.ToTensor()

        print()
        print("SR dataset loaded.")
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

        # ----------------------------------------------------
        # Input path
        # ----------------------------------------------------

        input_path = os.path.join(
            self.input_dir,
            self.input_images[index]
        )

        # ----------------------------------------------------
        # Target path
        # ----------------------------------------------------

        target_path = os.path.join(
            self.target_dir,
            self.target_images[index]
        )

        # ----------------------------------------------------
        # Load images
        # ----------------------------------------------------

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

        return (
            input_image,
            target_image
        )


# ============================================================
# Start
# ============================================================

print("=" * 60)
print("EDSR L1 + SSIM RESIDUAL TRAINING")
print("=" * 60)

print()
print("Device:", DEVICE)

print(
    "Starting from Epoch:",
    START_EPOCH
)

print(
    "Ending at Epoch:",
    EPOCHS
)


# ============================================================
# Dataset
# ============================================================

dataset = SRDataset(
    INPUT_DIR,
    TARGET_DIR
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
    "Number of batches:",
    len(loader)
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
# Optional checkpoint
# ============================================================

print()

if CHECKPOINT is not None:

    print(
        "Loading existing EDSR checkpoint:"
    )

    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint
    )

    print(
        "Checkpoint loaded successfully."
    )

else:

    print(
        "No checkpoint specified."
    )

    print(
        "Starting EDSR training from scratch."
    )


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# L1 Loss
# ============================================================

l1_loss = nn.L1Loss()


# ============================================================
# Training
# ============================================================

loss_history = []

model.train()


for epoch in range(
    START_EPOCH,
    EPOCHS + 1
):

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
    # Batches
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
        # EDSR prediction
        # ----------------------------------------------------

        prediction = model(
            input_image
        )


        # ----------------------------------------------------
        # Check image sizes
        # ----------------------------------------------------

        if prediction.shape != target_image.shape:

            raise RuntimeError(
                "\nImage size mismatch!\n"
                f"Prediction: {prediction.shape}\n"
                f"Target:     {target_image.shape}\n\n"
                "The current EDSR model produces the "
                "same spatial size as the input.\n"
                "Input and target must therefore have "
                "the same H x W for this training setup."
            )


        # ----------------------------------------------------
        # L1 Loss
        # ----------------------------------------------------

        loss_l1 = l1_loss(
            prediction,
            target_image
        )


        # ----------------------------------------------------
        # SSIM
        # ----------------------------------------------------

        prediction_for_ssim = torch.clamp(
            prediction,
            0.0,
            1.0
        )

        ssim = structural_similarity_index_measure(
            prediction_for_ssim,
            target_image,
            data_range=1.0
        )

        loss_ssim = 1.0 - ssim


        # ----------------------------------------------------
        # Combined Loss
        #
        # 90% L1
        # 10% SSIM
        # ----------------------------------------------------

        loss = (
            0.9 * loss_l1
            +
            0.1 * loss_ssim
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Update weights
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_l1 += loss_l1.item()

        total_ssim_loss += (
            loss_ssim.item()
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (batch_idx + 1) % 50 == 0
            or
            (batch_idx + 1) == len(loader)
        ):

            print(
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.6f} "
                f"L1: {loss_l1.item():.6f} "
                f"SSIM Loss: {loss_ssim.item():.6f}"
            )


    # ========================================================
    # Epoch Statistics
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
    # Save checkpoint
    # ========================================================

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"edsr_ssim_residual_epoch_{epoch}.pth"
    )

    torch.save(
        model.state_dict(),
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
# Complete
# ============================================================

print()
print("=" * 60)
print("L1 + SSIM RESIDUAL TRAINING COMPLETE")
print("=" * 60)

print()

print(
    "Loss history:"
)

print(
    loss_history
)

print()

print(
    "Checkpoints saved in:"
)

print(
    OUTPUT_DIR
)

print()

print(
    "Training finished successfully."
)