import glob
import os
import random

from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POP_DIR = os.path.join(ROOT, "POP")
FLIR_DIR = os.path.join(ROOT, "human_detection", "datasets", "flir_align")

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
        self.rgb_norm = T.Normalize([0.5] * 3, [0.5] * 3)
        self.l_norm = T.Normalize([0.5], [0.5])

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        t = Image.open(self.t[i]).convert("L")
        v = Image.open(self.v[i]).convert("RGB")
        h, w = int(self.size * 1.25), int(self.size * 1.25 * 1.25)
        t, v = T.functional.resize(t, (h, w)), T.functional.resize(v, (h, w))
        if self.train:
            y, x, ch, cw = T.RandomCrop.get_params(t, (self.size, self.size))
            t, v = T.functional.crop(t, y, x, ch, cw), T.functional.crop(v, y, x, ch, cw)
            if random.random() < 0.5:
                t, v = T.functional.hflip(t), T.functional.hflip(v)
        else:
            t, v = T.functional.center_crop(t, self.size), T.functional.center_crop(v, self.size)
        return self.l_norm(T.functional.to_tensor(t)), self.rgb_norm(T.functional.to_tensor(v))
