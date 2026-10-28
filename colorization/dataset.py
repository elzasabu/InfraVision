import csv
import glob
import os
import random

from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POP_DIR = os.path.join(ROOT, "POP")
FLIR_DIR = os.path.join(ROOT, "human_detection", "datasets", "flir_align")
DISASTER_DIR = os.path.join(ROOT, "data", "paired_disaster")  # built by paired_data/build_paired_dataset.py

IMG_EXT = ("*.jpg", "*.JPG", "*.jpeg", "*.png")


def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(folder, ext))
    return sorted(set(files))


def make_tf(size, train, channels):
    # POP is 1280x1024, FLIR is 640x512: both are resized to the same working scale.
    tfs = [T.Resize((int(size * 1.25), int(size * 1.25 * 1.25)))]
    tfs.append(T.RandomCrop(size) if train else T.CenterCrop(size))
    if train:
        tfs.append(T.RandomHorizontalFlip())
    tfs += [T.ToTensor(), T.Normalize([0.5] * channels, [0.5] * channels)]
    return T.Compose(tfs)


class UnpairedDataset(Dataset):
    """A = POP thermal (1ch), B = FLIR visible RGB (3ch). Randomly paired each step."""

    def __init__(self, split="train", size=256):
        self.a = list_images(os.path.join(POP_DIR, split, "JPEGImages"))
        self.b = list_images(os.path.join(FLIR_DIR, "visible", "train" if split != "test" else "test"))
        self.tf_a = make_tf(size, split == "train", 1)
        self.tf_b = make_tf(size, split == "train", 3)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        a = Image.open(self.a[i]).convert("L")
        b = Image.open(random.choice(self.b)).convert("RGB")
        return self.tf_a(a), self.tf_b(b)


class FlirPairedDataset(Dataset):
    """Aligned FLIR (thermal, visible) pairs for optional supervised loss."""

    def __init__(self, split="train", size=256):
        self.t = list_images(os.path.join(FLIR_DIR, "thermal", split))
        self.v = list_images(os.path.join(FLIR_DIR, "visible", split))
        assert len(self.t) == len(self.v), "thermal/visible count mismatch"
        self.size, self.train = size, split == "train"

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        t = Image.open(self.t[i]).convert("L")
        v = Image.open(self.v[i]).convert("RGB")
        return paired_tf(t, v, self.size, self.train)


class DisasterPairedDataset(Dataset):
    """Aligned (thermal, RGB) pairs from data/paired_disaster.

    The RGB target is the clean image by default; degraded=True returns the
    synthetic smoke/fog/dust/dark versions instead (one sample per variant).
    """

    def __init__(self, split="train", size=256, root=DISASTER_DIR, degraded=False):
        with open(os.path.join(root, "manifest.csv"), newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
        if not degraded:
            rows = list({r["id"]: r for r in rows}.values())
        key = "rgb_degraded" if degraded else "rgb"
        self.pairs = [(os.path.join(root, r["thermal"]), os.path.join(root, r[key])) for r in rows]
        assert self.pairs, f"no {split} pairs in {root}"
        self.size, self.train = size, split == "train"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        t, v = self.pairs[i]
        return paired_tf(Image.open(t).convert("L"), Image.open(v).convert("RGB"), self.size, self.train)


def paired_tf(t, v, size, train):
    """Same resize / crop / flip on both halves so the pair stays aligned."""
    h, w = int(size * 1.25), int(size * 1.25 * 1.25)
    t, v = T.functional.resize(t, (h, w)), T.functional.resize(v, (h, w))
    if train:
        y, x, ch, cw = T.RandomCrop.get_params(t, (size, size))
        t, v = T.functional.crop(t, y, x, ch, cw), T.functional.crop(v, y, x, ch, cw)
        if random.random() < 0.5:
            t, v = T.functional.hflip(t), T.functional.hflip(v)
    else:
        t, v = T.functional.center_crop(t, size), T.functional.center_crop(v, size)
    t = T.functional.normalize(T.functional.to_tensor(t), [0.5], [0.5])
    v = T.functional.normalize(T.functional.to_tensor(v), [0.5] * 3, [0.5] * 3)
    return t, v
