"""POP drone thermal dataset: loading, flight-level splits and leakage checks.

Layout (unchanged from what the rest of the repo uses):
    POP/{train,val,test}/JPEGImages/<flight>_<frame>_<altitude>.JPG
    POP/{train,val,test}/Annotations/{split}.json      COCO, one class "person"

POP is thermal only (1280x1024, stored as 3 identical channels). There is no
RGB, so the colorizer cannot be trained or scored on POP; see README.

Frames from one flight are near-duplicates, so splitting by random frame leaks
almost-identical images between train and test. POP's own splits are already
by flight (train F1-F12, val F13-F16, test T1-T3), and `pop_splits` keeps them.
`pop_splits(resplit=True)` re-draws train/val by flight with a fixed seed;
the test flights are never touched.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ROOT
from ..io import read_thermal

POP_ROOT = ROOT / "POP"
SPLITS = ("train", "val", "test")
_FLIGHT = re.compile(r"^([A-Za-z]+\d+)_")


def flight_of(file_name):
    """'F13_0145_50.101.JPG' -> 'F13'. One flight = one scene/sequence."""
    m = _FLIGHT.match(Path(file_name).name)
    return m.group(1) if m else Path(file_name).stem


def altitude_of(file_name):
    """'F13_0145_50.101.JPG' -> 50.101 (metres), or None."""
    parts = Path(file_name).stem.split("_")
    try:
        return float(parts[-1])
    except (ValueError, IndexError):
        return None


def load_split(root, split):
    """Records for one official split: path, size, person boxes (xyxy, pixels), flight."""
    root = Path(root)
    ann = root / split / "Annotations" / f"{split}.json"
    if not ann.exists():
        raise FileNotFoundError(f"POP annotations not found: {ann}")
    with open(ann, encoding="utf-8") as f:
        coco = json.load(f)
    boxes = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        boxes[a["image_id"]].append([x, y, x + w, y + h])
    recs = []
    for im in coco["images"]:
        name = Path(im["file_name"]).name
        W, H = im["width"], im["height"]
        b = np.array(boxes.get(im["id"], []), dtype=np.float32).reshape(-1, 4)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
        b = b[(b[:, 2] - b[:, 0] >= 1) & (b[:, 3] - b[:, 1] >= 1)]
        recs.append(dict(path=root / split / "JPEGImages" / name, name=name, split=split,
                         flight=flight_of(name), altitude=altitude_of(name),
                         width=W, height=H, boxes=b))
    return recs


def pop_splits(root=POP_ROOT, resplit=False, val_flights=4, seed=0, existing_only=False):
    """{'train': [...], 'val': [...], 'test': [...]} split by flight, never by frame.

    resplit=False: POP's official splits. resplit=True: pool train+val and pick
    `val_flights` whole flights for val with random.Random(seed). Test is fixed.
    existing_only drops records whose image file is missing (partial copies).
    """
    data = {s: load_split(root, s) for s in SPLITS}
    if resplit:
        pool = data["train"] + data["val"]
        flights = sorted({r["flight"] for r in pool})
        val = set(random.Random(seed).sample(flights, val_flights))
        data["train"] = [r for r in pool if r["flight"] not in val]
        data["val"] = [r for r in pool if r["flight"] in val]
        for r in data["train"] + data["val"]:
            r["assigned"] = "val" if r["flight"] in val else "train"
    if existing_only:
        data = {s: [r for r in recs if r["path"].exists()] for s, recs in data.items()}
    check_no_leakage(data)
    return data


def check_no_leakage(data):
    """Raise if a flight appears in more than one split."""
    owner = {}
    for split, recs in data.items():
        for r in recs:
            other = owner.setdefault(r["flight"], split)
            if other != split:
                raise AssertionError(f"flight {r['flight']} is in both {other} and {split}")
    return True


def summarize(data):
    out = {}
    for split, recs in data.items():
        boxes = np.concatenate([r["boxes"] for r in recs]) if recs else np.zeros((0, 4))
        side = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if len(boxes) else boxes
        out[split] = dict(
            images=len(recs), people=int(len(boxes)),
            flights=sorted({r["flight"] for r in recs}, key=lambda f: (f[0], int(re.sub(r"\D", "", f) or 0))),
            sizes=sorted({(r["width"], r["height"]) for r in recs}),
            images_without_people=sum(1 for r in recs if len(r["boxes"]) == 0),
            box_side_px=dict(median=float(np.median(side)), p10=float(np.percentile(side, 10)),
                             min=float(side.min())) if len(side) else None,
        )
    return out


class POPDetection(Dataset):
    """(thermal uint8 HxW tensor, boxes Nx4 xyxy) per frame, for evaluation and tests."""

    def __init__(self, split="val", root=POP_ROOT, records=None, existing_only=True):
        recs = records if records is not None else pop_splits(root, existing_only=existing_only)[split]
        self.recs = recs

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        gray, _ = read_thermal(r["path"])
        return torch.from_numpy(gray), torch.from_numpy(r["boxes"]), r["name"]


def degrade_for_sr(clean, scale, rng):
    """Training input for SR: blur, `scale`x down, bicubic back up, noise, JPEG.

    Same size as `clean`, so the network learns to restore a bicubic-upscaled
    frame; at inference the frame is upscaled by `scale` first and restored.
    """
    h, w = clean.shape
    img = cv2.GaussianBlur(clean, (0, 0), sigmaX=float(rng.uniform(0.6, 1.4)))
    small = cv2.resize(img, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    img += rng.normal(0, rng.uniform(0.0, 4.0), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    q = int(rng.integers(40, 95))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else img


class POPSRCrops(Dataset):
    """Random crops of POP thermal frames with an on-the-fly degradation, for SR training.

    target = clean crop (1 x size x size, 0-1), input = blurred, `scale`x
    down-and-up-sampled, noisy, JPEG-compressed copy of the same crop, the
    recipe of SR/prepare_pop_dataset.py. Val crops are fixed per index.
    """

    def __init__(self, records, size=128, scale=2, per_image=4, train=True, seed=0):
        if not records:
            raise ValueError("POPSRCrops: no images")
        self.recs, self.size, self.scale = records, size, scale
        self.per_image, self.train, self.seed = per_image, train, seed

    def __len__(self):
        return len(self.recs) * self.per_image

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else self.seed * 100003 + i)
        gray, _ = read_thermal(self.recs[i // self.per_image]["path"])
        H, W = gray.shape
        s = self.size
        y, x = rng.integers(0, H - s + 1), rng.integers(0, W - s + 1)
        target = gray[y:y + s, x:x + s]
        if self.train and rng.random() < 0.5:
            target = target[:, ::-1]
        target = np.ascontiguousarray(target)
        inp = degrade_for_sr(target, self.scale, rng)
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0)[None]
        return to_t(inp), to_t(target)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Summarise POP and check the flight splits")
    p.add_argument("--root", default=str(POP_ROOT))
    p.add_argument("--resplit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    data = pop_splits(args.root, resplit=args.resplit, seed=args.seed)
    print(json.dumps(summarize(data), indent=2, default=str))
    print("No flight appears in two splits.")


if __name__ == "__main__":
    main()
