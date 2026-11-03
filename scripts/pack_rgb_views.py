"""Pack colour-only drone photos (FloodNet, RescueNet) into views for pseudo-paired colorizer data.

    .venv/bin/python scripts/pack_rgb_views.py floodnet ../datasets/FloodNet/val ../datasets/FloodNet/test
    .venv/bin/python scripts/pack_rgb_views.py rescuenet ../datasets/RescueNet/images
    (several calls append to the same zip, default ../datasets/flood_rgb.zip)

Each photo (4000x3000 nadir UAV frames) gives `--views` views at --side x 3/4 side: the whole frame
plus random 40-70 % crops (seed 0), so the colorizer sees both wide and close-up flood scenes.
Every --test_every-th photo (all its views) goes to test. Layout: <name>/{train,test}/rgb/<stem>_vK.jpg.
colab_training/train_colorizer_v3_kaggle.ipynb adds the matching thermal with the reverse model
(scripts/make_pseudo_pairs.py).
"""
import argparse
import random
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png"}


def views(img, n, side, rng):
    h, w = img.shape[:2]
    out = [img]
    for _ in range(n - 1):
        f = rng.uniform(0.4, 0.7)
        cw, ch = int(w * f), int(w * f * 3 / 4)
        ch = min(ch, h)
        x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
        out.append(img[y:y + ch, x:x + cw])
    return [cv2.resize(v, (side, side * 3 // 4), interpolation=cv2.INTER_AREA) for v in out]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", default=str(ROOT.parent / "datasets" / "flood_rgb.zip"))
    p.add_argument("--side", type=int, default=400)
    p.add_argument("--views", type=int, default=3)
    p.add_argument("--test_every", type=int, default=8)
    p.add_argument("--out_dir", default="", help="write <out_dir>/<name>/... folders instead of the zip (Kaggle)")
    a = p.parse_args()
    files = sorted(f for d in a.dirs for f in Path(d).rglob("*") if f.suffix.lower() in EXTS)
    if not files:
        raise SystemExit("no images found")
    rng = random.Random(0)
    mode = "a" if Path(a.out).exists() else "w"
    n = {"train": 0, "test": 0}

    class Dir:   # same writestr interface as ZipFile
        def __init__(self, root):
            self.root = Path(root)

        def namelist(self):
            return []

        def writestr(self, name, data):
            f = self.root / name
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    with (Dir(a.out_dir) if a.out_dir else zipfile.ZipFile(a.out, mode, zipfile.ZIP_STORED)) as z:
        if not a.out_dir and mode == "a" and any(x.startswith(a.name + "/") for x in z.namelist()):
            raise SystemExit(f"{a.out} already has '{a.name}'")
        for k, f in enumerate(files):
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                continue
            split = "test" if k % a.test_every == a.test_every - 1 else "train"
            for j, v in enumerate(views(img, a.views, a.side, rng)):
                ok, buf = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 92])
                z.writestr(f"{a.name}/{split}/rgb/{f.stem}_v{j}.jpg", buf.tobytes())
                n[split] += 1
    where = Path(a.out_dir) / a.name if a.out_dir else f"{a.out} ({Path(a.out).stat().st_size / 1e6:.0f} MB)"
    print(f"{a.name}: {len(files)} photos -> train {n['train']}, test {n['test']} views -> {where}")


if __name__ == "__main__":
    main()
