"""Aligned thermal / RGB pairs for the colorizer (data/paired_disaster and FLIR).

The thermal side goes through the same pre-enhancement as inference
(infravision.enhance.pre_enhance), so the colorizer sees CLAHE'd, denoised
thermal at training time too. The target is the clean RGB frame. The
synthetic smoke/fog/dust/dark RGB copies are not targets: the colorizer should
show the scene as it would look in clear daylight, which is the point.
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ROOT
from ..enhance import pre_enhance
from ..io import IMAGE_EXTS

DISASTER_DIR = ROOT / "data" / "paired_disaster"
FLIR_DIR = ROOT / "human_detection" / "datasets" / "flir_align"


def disaster_pairs(split, root=DISASTER_DIR):
    """(thermal path, clean RGB path, id) per pair of one split, from manifest.csv."""
    root = Path(root)
    with open(root / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    seen, out = set(), []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        t, v = (root / r[k].replace("\\", "/") for k in ("thermal", "rgb"))
        out.append((t, v, r["id"]))
    return out


def flir_pairs(split, root=FLIR_DIR):
    """FLIR ADAS aligned: thermal/<split>/x and visible/<split>/x with the same stem."""
    root = Path(root)
    t = {p.stem: p for p in (root / "thermal" / split).glob("*") if p.suffix.lower() in IMAGE_EXTS}
    v = {p.stem: p for p in (root / "visible" / split).glob("*") if p.suffix.lower() in IMAGE_EXTS}
    return [(t[k], v[k], f"flir_{k}") for k in sorted(t.keys() & v.keys())]


def _read(path, gray):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"unreadable image: {path}")
    return img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class PairedColor(Dataset):
    """-> (thermal 1xSxS in [-1, 1], RGB 3xSxS in [-1, 1]).

    Frames are resized so the longer side is `work_side` (the size the
    pipeline colorizes at), then cropped to `crop` (random in training, centre
    in validation) with the same window on both halves.
    """

    def __init__(self, pairs, crop=256, work_side=640, train=True, pre=None):
        if not pairs:
            raise ValueError("PairedColor: no pairs (build data/paired_disaster first)")
        self.pairs, self.crop, self.work_side, self.train = pairs, crop, work_side, train
        self.pre = pre if pre is not None else {}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        tp, vp, _ = self.pairs[i]
        t, v = _read(tp, True), _read(vp, False)
        h, w = t.shape
        s = self.work_side / max(h, w)
        size = (max(self.crop, round(w * s)), max(self.crop, round(h * s)))
        t = cv2.resize(t, size, interpolation=cv2.INTER_AREA)
        v = cv2.resize(v, size, interpolation=cv2.INTER_AREA)
        t = pre_enhance(t, **self.pre)
        H, W = t.shape
        c = self.crop
        if self.train:
            y, x = random.randint(0, H - c), random.randint(0, W - c)
        else:
            y, x = (H - c) // 2, (W - c) // 2
        t, v = t[y:y + c, x:x + c], v[y:y + c, x:x + c]
        if self.train and random.random() < 0.5:
            t, v = t[:, ::-1], v[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(t)).float().div_(127.5).sub_(1)[None]
        v = torch.from_numpy(np.ascontiguousarray(v)).float().div_(127.5).sub_(1).permute(2, 0, 1)
        return t, v
