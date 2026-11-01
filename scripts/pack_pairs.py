"""Pack extra co-registered thermal/colour pairs for the colorizer v2 notebook (colorizer_extra.zip).

    .venv/bin/python scripts/pack_pairs.py m3fd ../datasets/M3FD/Ir ../datasets/M3FD/Vis
    .venv/bin/python scripts/pack_pairs.py flame <ir dir> <rgb dir> --step 10 --test_every 8
    (several calls append to the same zip; --out to change it)

Pairs are matched by file stem (same name in both folders). Every --step-th pair is kept
(video frames are near-duplicates); every --test_every-th kept pair goes to test, unless
--test_prefixes names whole sequences / scenes to hold out (file names starting with them).
Layout: <name>/{train,test}/{ir,rgb}/<stem>.jpg at --side x (side * h / w), the layout
train_colorizer_v2_kaggle.ipynb (and infravision.train.thermovision) reads.
Checked sets: M3FD fusion (300 ground-level pairs, mostly day: streets incl. wet roads, beach).
"""
import argparse
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="set name in the zip, e.g. m3fd / flame")
    p.add_argument("ir")
    p.add_argument("rgb")
    p.add_argument("--out", default=str(ROOT.parent / "datasets" / "colorizer_extra.zip"))
    p.add_argument("--side", type=int, default=320)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--test_every", type=int, default=6)
    p.add_argument("--test_prefixes", nargs="*", default=[])
    p.add_argument("--test_block", type=int, default=1,
                   help="hold out test pairs in blocks of this many consecutive kept pairs (video frames)")
    p.add_argument("--min_brightness", type=float, default=0,
                   help="skip pairs whose colour image is darker than this mean (0-255): night frames")
    a = p.parse_args()
    ir = {f.stem: f for f in Path(a.ir).iterdir() if f.suffix.lower() in EXTS}
    rgb = {f.stem: f for f in Path(a.rgb).iterdir() if f.suffix.lower() in EXTS}
    stems = sorted(ir.keys() & rgb.keys())[:: a.step]
    if not stems:
        raise SystemExit("no pairs with matching names")
    mode = "a" if Path(a.out).exists() else "w"
    with zipfile.ZipFile(a.out, mode, zipfile.ZIP_STORED) as z:
        old = {n for n in z.namelist() if n.startswith(a.name + "/")} if mode == "a" else set()
        if old:
            raise SystemExit(f"{a.out} already has a '{a.name}' set; use another name or a new --out")
        n = {"train": 0, "test": 0}
        for k, s in enumerate(stems):
            if a.test_prefixes:
                split = "test" if any(s.startswith(t) for t in a.test_prefixes) else "train"
            else:
                b = k // max(1, a.test_block)
                split = "test" if b % a.test_every == a.test_every - 1 else "train"
            t = cv2.imread(str(ir[s]), cv2.IMREAD_GRAYSCALE)
            c = cv2.imread(str(rgb[s]), cv2.IMREAD_COLOR)
            if t is None or c is None:
                continue
            if a.min_brightness and cv2.resize(c, (64, 48)).mean() < a.min_brightness:
                n["dark"] = n.get("dark", 0) + 1
                continue
            size = (a.side, round(a.side * t.shape[0] / t.shape[1]))
            for kind, img in (("ir", t), ("rgb", c)):
                ok, buf = cv2.imencode(".jpg", cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                                       [cv2.IMWRITE_JPEG_QUALITY, 95])
                z.writestr(f"{a.name}/{split}/{kind}/{s}.jpg", buf.tobytes())
            n[split] += 1
    print(f"{a.name}: train {n['train']}, test {n['test']} pairs, skipped dark {n.get('dark', 0)} -> {a.out} "
          f"({Path(a.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
