"""Pack datasets/pop (from convert_pop.py) into pop_yolo_colab.zip for the Colab/Kaggle notebooks.

Zip layout:
  pop/data.yaml, pop/images/{train,val,test}, pop/labels/{train,val,test}
  pop/data_flood.yaml, pop/{images,labels}/test_flood   (if prepare_flood_aug.py ran;
                                                         its flood_* train copies sit in train/)
  weights/thermal_best.pt   (optional FLIR thermal model to start from)

Usage:
  python make_pop_zip.py                      # uses runs/thermal/colab_resumed/weights/best.pt if it exists
  python make_pop_zip.py path/to/best.pt      # use another starting model
  python make_pop_zip.py none                 # no starting model: notebooks start from yolo11s.pt
"""
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
POP = HERE / "pop"
OUT = HERE.parent / "pop_yolo_colab.zip"
DEFAULT_WEIGHTS = HERE.parent / "runs" / "thermal" / "colab_resumed" / "weights" / "best.pt"

if not (POP / "images" / "train").is_dir():
    sys.exit(f"{POP} not found. Run convert_pop.py first.")

arg = sys.argv[1] if len(sys.argv) > 1 else None
weights = None if arg == "none" else Path(arg) if arg else DEFAULT_WEIGHTS
if weights is not None and not weights.is_file():
    if arg:
        sys.exit(f"Weights not found: {weights}")
    print(f"No FLIR model at {weights}; the notebooks will start from yolo11s.pt.")
    weights = None

# The local data.yaml holds a laptop path; the Colab notebook unzips to /content/pop.
data_yaml = "path: /content/pop\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: person\n"

# JPEGs are already compressed, so store them as-is (much faster).
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    z.writestr("pop/data.yaml", data_yaml)
    splits = ["train", "val", "test"]
    if (POP / "images" / "test_flood").is_dir():
        z.writestr("pop/data_flood.yaml", data_yaml.replace("test: images/test", "test: images/test_flood"))
        splits.append("test_flood")
    for split in splits:
        for kind in ("images", "labels"):
            files = sorted((POP / kind / split).iterdir())
            for f in files:
                z.write(f, f"pop/{kind}/{split}/{f.name}")
            print(f"{kind}/{split}: {len(files)} files")
    if weights is not None:
        z.write(weights, "weights/thermal_best.pt")
        print("Start weights:", weights)

print(f"Wrote {OUT} ({OUT.stat().st_size / 1e9:.2f} GB)")
