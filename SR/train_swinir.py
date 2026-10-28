import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from torchmetrics.functional.image import structural_similarity_index_measure

from models.swinir_refine import SwinIRRefinement


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_DIR = "dataset_pop/train/input"
TARGET_DIR = "dataset_pop/train/target"

OUTPUT_DIR = "checkpoints_pop/swinir_direct"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCHS = 20

BATCH_SIZE = 1

LEARNING_RATE = 2e-4

# Training crop.
# The original images remain 256x256.
# During training we use 128x128 patches to reduce CPU load.
PATCH_SIZE = 128


# ============================================================
# DATASET
# ============================================================

class SRDataset(Dataset):

    def __init__(
        self,
        input_dir,
        target_dir,
        patch_size=128
    ):

        self.input_dir = input_dir
        self.target_dir = target_dir
        self.patch_size = patch_size

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

        print("Dataset loaded.")
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

        # ----------------------------------------------------
        # Make sure both images have exactly the same size
        # ----------------------------------------------------

        if input_image.size != target_image.size:

            raise RuntimeError(
                f"Size mismatch:\n"
                f"Input : {input_image.size}\n"
                f"Target: {target_image.size}\n"
                f"File  : {self.input_images[index]}"
            )

        # ----------------------------------------------------
        # Random crop
        # ----------------------------------------------------

        width, height = input_image.size

        if width < self.patch_size or height < self.patch_size:

            raise RuntimeError(
                f"Image is smaller than patch size: "
                f"{input_image.size}"
            )

        left = torch.randint(
            0,
            width - self.patch_size + 1,
            (1,)
        ).item()

        top = torch.randint(
            0,
            height - self.patch_size + 1,
            (1,)
        ).item()

        input_image = input_image.crop(
            (
                left,
                top,
                left + self.patch_size,
                top + self.patch_size
            )
        )

        target_image = target_image.crop(
            (
                left,
                top,
                left + self.patch_size,
                top + self.patch_size
            )
        )

        input_image = self.transform(input_image)

        target_image = self.transform(target_image)

        return input_image, target_image


# ============================================================
# START
# ============================================================

print("=" * 65)
print("DIRECT SWINIR 256x256 RESTORATION")
print("Input -> SwinIR -> Target")
print("=" * 65)

print()
print("Device:", DEVICE)
print("Input size : 256 x 256")
print("Target size: 256 x 256")
print("Output size: 256 x 256")
print("Training patch:", PATCH_SIZE)


# ============================================================
# DATASET
# ============================================================

dataset = SRDataset(
    INPUT_DIR,
    TARGET_DIR,
    PATCH_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print()
print("Training samples:", len(dataset))
print("Batches per epoch:", len(loader))


# ============================================================
# SWINIR MODEL
# ============================================================

print()
print("Creating SwinIR...")


model = SwinIRRefinement(
    in_channels=3,
    embed_dim=48,
    num_groups=4,
    depth=4,
    num_heads=4,
    window_size=8
).to(DEVICE)


print("SwinIR created successfully.")


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999)
)


# ============================================================
# LOSS
# ============================================================

l1_loss = nn.L1Loss()


# ============================================================
# TRAINING
# ============================================================

loss_history = []

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0
    total_l1 = 0.0
    total_ssim_loss = 0.0

    print()
    print("=" * 65)
    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )
    print("=" * 65)

    for batch_idx, (input_image, target_image) in enumerate(loader):

        input_image = input_image.to(DEVICE)

        target_image = target_image.to(DEVICE)

        optimizer.zero_grad()

        # ----------------------------------------------------
        # SwinIR prediction
        # ----------------------------------------------------

        prediction = model(
            input_image
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if prediction.shape != target_image.shape:

            raise RuntimeError(
                f"Output size mismatch!\n"
                f"Prediction: {prediction.shape}\n"
                f"Target    : {target_image.shape}"
            )

        # ----------------------------------------------------
        # L1
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
        # COMBINED LOSS
        # ----------------------------------------------------

        loss = (
            0.8 * loss_l1
            +
            0.2 * loss_ssim
        )

        # ----------------------------------------------------
        # BACKPROP
        # ----------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_l1 += loss_l1.item()

        total_ssim_loss += loss_ssim.item()

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Batch [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {loss.item():.6f} "
                f"L1: {loss_l1.item():.6f} "
                f"SSIM Loss: {loss_ssim.item():.6f}"
            )

    # ========================================================
    # EPOCH RESULTS
    # ========================================================

    avg_loss = (
        total_loss / len(loader)
    )

    avg_l1 = (
        total_l1 / len(loader)
    )

    avg_ssim_loss = (
        total_ssim_loss / len(loader)
    )

    loss_history.append(
        avg_loss
    )

    print()
    print(
        f"Epoch {epoch + 1} completed."
    )

    print(
        f"Average Loss     : {avg_loss:.6f}"
    )

    print(
        f"Average L1       : {avg_l1:.6f}"
    )

    print(
        f"Average SSIM Loss: {avg_ssim_loss:.6f}"
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"swinir_direct_epoch_{epoch + 1}.pth"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print()
    print("Checkpoint saved:")
    print(checkpoint_path)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 65)
print("DIRECT SWINIR TRAINING COMPLETE")
print("=" * 65)

print()
print("Loss history:")
print(loss_history)

print()
print("Checkpoints saved in:")
print(OUTPUT_DIR)