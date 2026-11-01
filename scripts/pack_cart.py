"""Pack the Caltech Aerial RGB-Thermal pairs (CART) for the Kaggle colorizer notebook.

    .venv/bin/python scripts/pack_cart.py                        # -> ../datasets/cart_colorizer.zip
    .venv/bin/python scripts/pack_cart.py --check results/cart_pairs_check.jpg   # alignment check only

CART (Lee et al., ECCV 2024, CC BY-NC 4.0, data.caltech.edu/records/cks6g-ps927): daytime drone
flights over rivers, lakes, coastlines, deserts and forests; labeled_rgbt_pairs.zip has 2282
co-registered colour (`color/<flight>_eo-N.jpg`) and 8-bit thermal (`thermal8/<flight>_thermal-N.jpg`)
frames. LLVIP is night street scenes only, so the LLVIP colorizer paints a dark night look;
CART gives daytime aerial colour.

Split by flight (whole flights held out, seed 0, ~15% of frames) so no near-duplicate frame is in
both. Output layout = the LLVIP pack, so colab_training/train_colorizer_kaggle.ipynb works unchanged:
cart/{train,test}/{ir,rgb}/<flight>-<frame>.jpg, resized to --side x (side * h / w).
"""
import argparse
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "datasets" / "CART" / "labeled_rgbt_pairs"
PAT = re.compile(r"^(?P<flight>.+)_(?:eo|thermal)-(?P<frame>\d+)\.\w+$")


def pairs(src):
    col = {PAT.match(f.name).group("flight", "frame"): f for f in (src / "color").iterdir() if PAT.match(f.name)}
    th = {PAT.match(f.name).group("flight", "frame"): f for f in (src / "thermal8").iterdir() if PAT.match(f.name)}
    keys = sorted(col.keys() & th.keys())
    by_flight = defaultdict(list)
    for k in keys:
        by_flight[k[0]].append((k[1], th[k], col[k]))
    return by_flight


def split(by_flight, test_frac=0.15, seed=0):
    flights = sorted(by_flight)
    rng = np.random.default_rng(seed)
    order = [flights[i] for i in rng.permutation(len(flights))]
    total, test = sum(len(v) for v in by_flight.values()), set()
    for f in order:
        if sum(len(by_flight[t]) for t in test) >= test_frac * total:
            break
        test.add(f)
    return test


def check(by_flight, out, n=4):
    """Colour | thermal | thermal edges (red) on colour: the edges must sit on the same objects."""
    rows = []
    flights = sorted(by_flight)
    for f in flights[:: max(1, len(flights) // n)][:n]:
        _, tp, cp = by_flight[f][len(by_flight[f]) // 2]
        t, c = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE), cv2.imread(str(cp))
        c = cv2.resize(c, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
        e = cv2.Canny(cv2.GaussianBlur(t, (3, 3), 0), 40, 120)
        ov = c.copy()
        ov[e > 0] = (0, 0, 255)
        row = np.hstack([c, cv2.cvtColor(t, cv2.COLOR_GRAY2BGR), ov])
        cv2.putText(row, f, (8, 24), 0, 0.7, (0, 255, 255), 2)
        rows.append(cv2.resize(row, (1440, round(1440 * row.shape[0] / row.shape[1]))))
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("alignment check ->", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=str(SRC))
    p.add_argument("--side", type=int, default=320)
    p.add_argument("--out", default=str(ROOT.parent / "datasets" / "cart_colorizer.zip"))
    p.add_argument("--check", default="", help="only write an alignment check image")
    a = p.parse_args()
    src = Path(a.src)
    if not (src / "color").is_dir():
        sys.exit(f"{src}/color not found: unzip labeled_rgbt_pairs.zip into {src.parent}")
    by_flight = pairs(src)
    print(f"{sum(len(v) for v in by_flight.values())} pairs in {len(by_flight)} flights")
    if a.check:
        check(by_flight, Path(a.check))
        return
    test = split(by_flight)
    n = {"train": 0, "test": 0}
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
        for f, frames in sorted(by_flight.items()):
            s = "test" if f in test else "train"
            for frame, tp, cp in frames:
                t = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
                c = cv2.imread(str(cp))
                size = (a.side, round(a.side * t.shape[0] / t.shape[1]))
                for kind, img in (("ir", t), ("rgb", c)):
                    ok, buf = cv2.imencode(".jpg", cv2.resize(img, size, interpolation=cv2.INTER_AREA),
                                           [cv2.IMWRITE_JPEG_QUALITY, 95])
                    z.writestr(f"cart/{s}/{kind}/{f}-{frame}.jpg", buf.tobytes())
                n[s] += 1
    print(f"train {n['train']} pairs, test {n['test']} pairs ({len(test)} held-out flights: {sorted(test)})")
    print(f"Wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
