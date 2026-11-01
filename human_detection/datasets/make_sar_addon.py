"""Split sar_v3_yolo.zip into "already on Kaggle (v2 upload)" + a small add-on zip.

    .venv/bin/python human_detection/datasets/make_sar_addon.py
    .venv/bin/python human_detection/datasets/make_sar_addon.py --v2 human_detection/sar_yolo.zip \
        --v3 human_detection/sar_v3_yolo.zip --out human_detection/sar_v3_addon.zip

Kaggle uploads are slow; most of v3 (POP, flood v2, AIResQ, WiSARD) is byte-identical to the
v2 upload. The add-on holds only the files v2 lacks (RGBTDronePerson, HIT-UAV, LLVIP) plus:
  addon/train_v3.txt, addon/val_v3.txt   lines "v2/images/train/x.jpg" or "addon/images/train/y.jpg"
  addon/tests.json                       test split -> "v2" | "addon"
  addon/splits_v3.json                   the v3 zip's splits.json
colab_training/train_sar_v3_addon_kaggle.ipynb reads both datasets and resolves the lists.
"""
import argparse
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v2", default=str(REPO / "human_detection" / "sar_yolo.zip"))
    p.add_argument("--v3", default=str(REPO / "human_detection" / "sar_v3_yolo.zip"))
    p.add_argument("--out", default=str(REPO / "human_detection" / "sar_v3_addon.zip"))
    a = p.parse_args()
    z2, z3 = zipfile.ZipFile(a.v2), zipfile.ZipFile(a.v3)
    have = {i.filename: i.CRC for i in z2.infolist()}

    def src(path):   # "sar/images/x/y.jpg" -> "v2" if the v2 upload has the identical file
        crc = have.get(path)
        if crc is None:
            return "addon"
        if crc != z3.getinfo(path).CRC:
            raise SystemExit(f"{path} differs between the v2 and v3 zips: rebuild the add-on from scratch")
        return "v2"

    def listed(rel):  # "./images/train/x.jpg" -> "v2/images/train/x.jpg"
        rel = rel[2:] if rel.startswith("./") else rel
        return f"{src('sar/' + rel)}/{rel}"

    train = [listed(l) for l in z3.read("sar/train.txt").decode().split("\n") if l.strip()]
    val = [listed(n[len("sar/"):]) for n in z3.namelist() if n.startswith("sar/images/val/")]
    tests = {}
    for n in z3.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[1] == "images" and parts[2].startswith("test_") and parts[3]:
            tests.setdefault(parts[2], set()).add(src(n))
    if any(len(v) > 1 for v in tests.values()):
        raise SystemExit(f"a test split is split across the two zips: {tests}")
    tests = {k: v.pop() for k, v in tests.items()}

    n_add = size = 0
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as out:
        for i in z3.infolist():
            if i.filename.startswith("sar/") and (i.filename.startswith(("sar/images/", "sar/labels/"))) \
                    and i.filename not in have:
                out.writestr("addon/" + i.filename[len("sar/"):], z3.read(i))
                n_add += 1
                size += i.file_size
        out.writestr("addon/train_v3.txt", "\n".join(train) + "\n")
        out.writestr("addon/val_v3.txt", "\n".join(val) + "\n")
        out.writestr("addon/tests.json", json.dumps(tests, indent=1))
        out.writestr("addon/splits_v3.json", z3.read("sar/splits.json"))
    count = lambda xs, s: sum(x.startswith(s + "/") for x in xs)
    print(f"train {len(train)} (v2 {count(train, 'v2')}, addon {count(train, 'addon')}); "
          f"val {len(val)} (v2 {count(val, 'v2')}, addon {count(val, 'addon')})")
    print("tests:", tests)
    print(f"Wrote {a.out}: {n_add} files, {size / 1e9:.2f} GB "
          f"(instead of {sum(i.file_size for i in z3.infolist()) / 1e9:.2f} GB for the full v3 zip)")


if __name__ == "__main__":
    main()
