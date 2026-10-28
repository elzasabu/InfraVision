import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from models.edsr import EDSR


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TRAIN_INPUT = "dataset_pop/train/input"
TRAIN_TARGET = "dataset_pop/train/target"

BATCH_SIZE = 1

# Continue training
NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = "checkpoints_pop"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT = "checkpoints_pop/edsr_epoch_3.pth"


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

        print("SR dataset loaded.")
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
# Main
# ============================================================

print("=" * 60)
print("EDSR CONTINUED TRAINING")
print("=" * 60)

print("Device:", DEVICE)


# ============================================================
# Dataset
# ============================================================

dataset = SRDataset(
    TRAIN_INPUT,
    TRAIN_TARGET
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training samples:", len(dataset))
print("Number of batches:", len(train_loader))


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
# Load existing checkpoint
# ============================================================

if os.path.exists(CHECKPOINT):

    print()
    print("Existing checkpoint found.")

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=DEVICE
        )
    )

    print(
        "Loaded:",
        CHECKPOINT
    )

else:

    print()
    print(
        "No checkpoint found."
    )

    print(
        "Training will start from scratch."
    )


# ============================================================
# Loss
# ============================================================

criterion = nn.L1Loss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

loss_history = []


for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0

    for batch_idx, (
        input_image,
        target_image
    ) in enumerate(train_loader):

        input_image = input_image.to(DEVICE)
        target_image = target_image.to(DEVICE)

        # Forward
        prediction = model(
            input_image
        )

        # Loss
        loss = criterion(
            prediction,
            target_image
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 25 == 0:

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.6f}"
            )

    epoch_loss = (
        running_loss /
        len(train_loader)
    )

    loss_history.append(
        epoch_loss
    )

    print()
    print(
        f"Epoch {epoch + 1} completed."
    )

    print(
        f"Average Loss: {epoch_loss:.6f}"
    )

    # ========================================================
    # Save after EVERY epoch
    # ========================================================

    epoch_checkpoint = (
        f"checkpoints_pop/"
        f"edsr_epoch_{epoch + 1}.pth"
    )

    torch.save(
        model.state_dict(),
        epoch_checkpoint
    )

    # Also update the main checkpoint
    torch.save(
        model.state_dict(),
        "checkpoints_pop/edsr_model_latest.pth"
    )

    print(
        "Checkpoint saved:"
    )

    print(
        epoch_checkpoint
    )

    print()


# ============================================================
# Final checkpoint
# ============================================================

torch.save(
    model.state_dict(),
    "checkpoints_pop/edsr_model.pth"
)

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Final model saved to:"
)

print(
    "checkpoints_pop/edsr_model.pth"
)

print()
print(
    "Loss history:"
)

print(
    loss_history
)