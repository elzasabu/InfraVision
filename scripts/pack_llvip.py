"""Pack LLVIP into a small zip for the Kaggle colorizer notebook.

    python scripts/pack_llvip.py                       # configs datasets.llvip -> ../datasets/llvip_colorizer.zip
    python scripts/pack_llvip.py --side 320 --out x.zip

Every pair is resized to `side` x (side * 4/5) (LLVIP is 1280x1024) and stored as
JPEG (quality 95): llvip/{train,test}/{ir,rgb}/<name>.jpg. The colorizer sees
256x256, so nothing it can use is lost, and the upload is a fraction of the original.
"""
import argparse
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infravision.config import ROOT, find_dir, load_config  # noqa: E402
from infravision.train.thermovision import llvip_pairs  # noqa: E402


def encode(job):
    path, gray, size = job
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"unreadable image: {path}")
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, size, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="LLVIP folder (default: configs datasets.llvip)")
    p.add_argument("--side", type=int, default=320)
    p.add_argument("--out", default=str(ROOT.parent / "datasets" / "llvip_colorizer.zip"))
    args = p.parse_args(argv)
    data = args.data or find_dir(load_config()["datasets"]["llvip"])
    if data is None:
        sys.exit("LLVIP not found: pass --data or set datasets.llvip in configs/infravision.yaml")
    size = (args.side, round(args.side * 4 / 5))

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z, ProcessPoolExecutor() as pool:
        for split in ("train", "test"):
            pairs = llvip_pairs(data, split)
            for kind, gray, col in (("ir", True, 0), ("rgb", False, 1)):
                jobs = [(pr[col], gray, size) for pr in pairs]
                for (t, _), buf in zip(pairs, pool.map(encode, jobs, chunksize=64)):
                    z.writestr(f"llvip/{split}/{kind}/{t.name}", buf)
            print(f"{split}: {len(pairs)} pairs", flush=True)
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
