"""Add pseudo-thermal frames to colour-only sets with the reverse model (RGB -> thermal).

    python scripts/make_pseudo_pairs.py --model <out>/rgb2thermal_best.pth --src <dir with floodnet/ rescuenet/> \
        --out_dir /tmp/colorizer_data

<src>/<name>/{train,test}/rgb/*.jpg (scripts/pack_rgb_views.py) -> <out_dir>/<name>/{train,test}/{ir,rgb}/,
the paired layout infravision.train.thermovision reads. The reverse model (trained with
`infravision.train.thermovision --reverse` on the real aligned pairs) predicts an 8-bit thermal frame
for every colour view; the colorizer then learns flood / disaster colours from these pseudo pairs.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infravision.config import pick_device  # noqa: E402
from infravision.models.thermovision_colorizer import ThermoVisionUNet  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--src", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    dev = pick_device(a.device)
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    size = int(ck.get("meta", {}).get("input_size", 320))
    m = ThermoVisionUNet(in_ch=3, out_ch=1)
    m.load_state_dict(ck["state_dict"])
    m.to(dev).eval()
    # <src>/<name>/{train,test}/rgb; iterate the folders directly (rglob does not follow symlinks)
    sets = sorted(d.name for d in Path(a.src).iterdir() if (d / "train" / "rgb").is_dir())
    if not sets:
        raise SystemExit(f"no <name>/train/rgb folders in {a.src}")
    total = 0
    for name in sets:
        for split in ("train", "test"):
            src = Path(a.src) / name / split / "rgb"
            if not src.is_dir():
                continue
            files = sorted(src.glob("*.jpg"))
            ir_dir, rgb_dir = Path(a.out_dir) / name / split / "ir", Path(a.out_dir) / name / split / "rgb"
            ir_dir.mkdir(parents=True, exist_ok=True)
            rgb_dir.mkdir(parents=True, exist_ok=True)
            for i in range(0, len(files), a.batch):
                chunk = files[i:i + a.batch]
                imgs = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in chunk]
                x = torch.stack([torch.from_numpy(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA))
                                 .float().div(255).permute(2, 0, 1) for im in imgs]).to(dev)
                with torch.inference_mode():
                    y = m(x).float().clamp(0, 1).cpu().numpy()[:, 0]
                for f, im, t in zip(chunk, imgs, y):
                    t = cv2.resize((t * 255).round().astype(np.uint8), (im.shape[1], im.shape[0]),
                                   interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(ir_dir / f.name), t, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    shutil.copy(f, rgb_dir / f.name)
            total += len(files)
            print(f"{name}/{split}: {len(files)} pseudo pairs -> {ir_dir.parent}", flush=True)
    if not total:
        raise SystemExit("no colour views found: nothing to pair")


if __name__ == "__main__":
    main()
