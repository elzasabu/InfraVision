"""Build the SR dataset from POP thermal images (replaces DIV2K).

Target = random native-resolution 256x256 crop of a POP image.
Input  = degraded copy (blur, 4x down/up, noise, JPEG), same recipe as prepare_dataset.py.
Train crops come from POP/train, test crops from POP/test, so no image leaks across splits.
"""
import io
import os
import random

import numpy as np
from PIL import Image, ImageFilter

POP = "../POP"
OUT = "dataset_pop"
IMAGE_SIZE = 256
TRAIN_COUNT = 2000
TEST_COUNT = 200
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def degrade(image):
    image = image.filter(ImageFilter.GaussianBlur(radius=1.2))
    image = image.resize((64, 64), Image.Resampling.BICUBIC)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(image).astype(np.float32) + np.random.normal(0, 3.0, (IMAGE_SIZE, IMAGE_SIZE, 3))
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build(split, count):
    src_dir = os.path.join(POP, split, "JPEGImages")
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    random.shuffle(files)
    in_dir = os.path.join(OUT, "train" if split == "train" else "test", "input")
    tg_dir = os.path.join(OUT, "train" if split == "train" else "test", "target")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(tg_dir, exist_ok=True)
    for i in range(count):
        img = Image.open(os.path.join(src_dir, files[i % len(files)])).convert("RGB")  # R=G=B thermal
        w, h = img.size
        x, y = random.randint(0, w - IMAGE_SIZE), random.randint(0, h - IMAGE_SIZE)
        target = img.crop((x, y, x + IMAGE_SIZE, y + IMAGE_SIZE))
        name = f"{i + 1:04d}.png"
        target.save(os.path.join(tg_dir, name))
        degrade(target).save(os.path.join(in_dir, name))
        if (i + 1) % 200 == 0:
            print(f"{split}: {i + 1}/{count}", flush=True)


build("train", TRAIN_COUNT)
build("test", TEST_COUNT)
print("Done ->", OUT)
