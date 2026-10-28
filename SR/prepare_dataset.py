import os
import random
from PIL import Image, ImageFilter

# ============================================================
# Configuration
# ============================================================

SOURCE_DIR = "dataset/DIV2K_train_HR"

TRAIN_INPUT_DIR = "dataset/train/input"
TRAIN_TARGET_DIR = "dataset/train/target"

TEST_INPUT_DIR = "dataset/test/input"
TEST_TARGET_DIR = "dataset/test/target"

IMAGE_SIZE = 256

# 800 DIV2K images
TRAIN_COUNT = 700
TEST_COUNT = 100

SEED = 42

random.seed(SEED)


# ============================================================
# Create directories
# ============================================================

os.makedirs(TRAIN_INPUT_DIR, exist_ok=True)
os.makedirs(TRAIN_TARGET_DIR, exist_ok=True)

os.makedirs(TEST_INPUT_DIR, exist_ok=True)
os.makedirs(TEST_TARGET_DIR, exist_ok=True)


# ============================================================
# Get source images
# ============================================================

image_files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

image_files.sort()

print("Found images:", len(image_files))

if len(image_files) < TRAIN_COUNT + TEST_COUNT:
    raise RuntimeError(
        f"Not enough images. Found {len(image_files)}, "
        f"but need {TRAIN_COUNT + TEST_COUNT}."
    )


# ============================================================
# Shuffle deterministically
# ============================================================

random.shuffle(image_files)

train_files = image_files[:TRAIN_COUNT]
test_files = image_files[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]


# ============================================================
# Degradation function
# ============================================================


def create_degraded_image(image):
    """
    Creates a stronger degraded 256x256 image.

    Degradation:
      - Gaussian blur
      - 256 -> 64 -> 256 down/up sampling
      - JPEG compression
      - mild Gaussian noise

    Output remains 256x256 RGB.
    """

    import numpy as np

    # --------------------------------------------------------
    # 1. Stronger blur
    # --------------------------------------------------------

    image = image.filter(
        ImageFilter.GaussianBlur(radius=1.2)
    )

    # --------------------------------------------------------
    # 2. Simulate stronger resolution loss
    # --------------------------------------------------------

    image = image.resize(
        (64, 64),
        Image.Resampling.BICUBIC
    )

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BICUBIC
    )

    # --------------------------------------------------------
    # 3. Convert to numpy
    # --------------------------------------------------------

    image_array = np.asarray(image).astype(np.float32)

    # --------------------------------------------------------
    # 4. Add mild sensor-like noise
    # --------------------------------------------------------

    noise = np.random.normal(
        loc=0,
        scale=3.0,
        size=image_array.shape
    )

    image_array = image_array + noise

    image_array = np.clip(
        image_array,
        0,
        255
    ).astype(np.uint8)

    image = Image.fromarray(
        image_array,
        mode="RGB"
    )

    # --------------------------------------------------------
    # 5. JPEG compression
    # --------------------------------------------------------

    temp_path = "temp_degraded.jpg"

    image.save(
        temp_path,
        format="JPEG",
        quality=50
    )

    image = Image.open(
        temp_path
    ).convert("RGB")

    os.remove(temp_path)

    return image


# ============================================================
# Process dataset
# ============================================================

def process_images(files, input_dir, target_dir, split_name):

    print()
    print("=" * 50)
    print(f"Creating {split_name} dataset")
    print("=" * 50)

    for index, filename in enumerate(files):

        source_path = os.path.join(
            SOURCE_DIR,
            filename
        )

        # ----------------------------------------------------
        # Open original
        # ----------------------------------------------------

        image = Image.open(source_path).convert("RGB")

        # ----------------------------------------------------
        # Resize source while maintaining a simple pipeline
        # ----------------------------------------------------

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        # ----------------------------------------------------
        # TARGET = original high-quality image
        # ----------------------------------------------------

        target = image.copy()

        # ----------------------------------------------------
        # INPUT = degraded version
        # ----------------------------------------------------

        degraded = create_degraded_image(image)

        # ----------------------------------------------------
        # Save using matching filenames
        # ----------------------------------------------------

        output_name = f"{index + 1:04d}.png"

        target_path = os.path.join(
            target_dir,
            output_name
        )

        input_path = os.path.join(
            input_dir,
            output_name
        )

        target.save(
            target_path,
            format="PNG"
        )

        degraded.save(
            input_path,
            format="PNG"
        )

        if (index + 1) % 50 == 0:
            print(
                f"{index + 1}/{len(files)} images processed"
            )

    print(f"{split_name} dataset completed.")


# ============================================================
# Train
# ============================================================

process_images(
    train_files,
    TRAIN_INPUT_DIR,
    TRAIN_TARGET_DIR,
    "TRAIN"
)


# ============================================================
# Test
# ============================================================

process_images(
    test_files,
    TEST_INPUT_DIR,
    TEST_TARGET_DIR,
    "TEST"
)


# ============================================================
# Finished
# ============================================================

print()
print("=" * 50)
print("DATASET PREPARATION COMPLETE")
print("=" * 50)

print()
print("Training:")
print("Input :", TRAIN_INPUT_DIR)
print("Target:", TRAIN_TARGET_DIR)

print()
print("Testing:")
print("Input :", TEST_INPUT_DIR)
print("Target:", TEST_TARGET_DIR)