"""Pack data/paired_disaster into a smaller zip for Google Drive / Colab.

    python paired_data/pack_for_colab.py                     # -> data/paired_disaster_colab.zip
    python paired_data/pack_for_colab.py --side 640 --splits train val --with_degraded

The full build is ~25 GB at 1280x1024. The colorizer trains at 640 px on the
longer side, so frames are resized to --side first (thermal PNG, RGB JPEG),
which brings the full LLVIP build to about 4 GB (160 pairs measured 40 MB).
Labels need no change (they are normalised). The zip unpacks to paired_disaster/ with a manifest.csv that
lists only the packed rows, in the same layout the trainers read.
"""
import argparse
import csv
import io
import os
import zipfile

from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resized(path, side, fmt, quality=92):
    im = Image.open(path)
    im = im.convert("L" if fmt == "PNG" else "RGB")
    s = side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format=fmt, **({} if fmt == "PNG" else {"quality": quality}))
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(ROOT, "data", "paired_disaster"))
    p.add_argument("--out", default=os.path.join(ROOT, "data", "paired_disaster_colab.zip"))
    p.add_argument("--side", type=int, default=640)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--with_degraded", action="store_true", help="also pack the synthetic degraded RGB")
    p.add_argument("--max_pairs", type=int, default=0, help="per split, 0 = all")
    args = p.parse_args()

    with open(os.path.join(args.root, "manifest.csv"), newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if r["split"] in args.splits]
    for r in rows:
        for k in ("thermal", "rgb", "rgb_degraded", "labels"):
            r[k] = r[k].replace("\\", "/")
    if args.max_pairs:
        keep = {}
        for r in rows:
            ids = keep.setdefault(r["split"], [])
            if r["id"] not in ids and len(ids) < args.max_pairs:
                ids.append(r["id"])
        allowed = {i for ids in keep.values() for i in ids}
        rows = [r for r in rows if r["id"] in allowed]

    done = set()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for n, r in enumerate(rows, 1):
            jobs = [("thermal", "PNG"), ("rgb", "JPEG")] + ([("rgb_degraded", "JPEG")] if args.with_degraded else [])
            for key, fmt in jobs:
                rel = r[key]
                if rel in done:
                    continue
                z.writestr(f"paired_disaster/{rel}", resized(os.path.join(args.root, rel), args.side, fmt))
                done.add(rel)
            if r["labels"] and r["labels"] not in done:
                z.write(os.path.join(args.root, r["labels"]), f"paired_disaster/{r['labels']}")
                done.add(r["labels"])
            if n % 2000 == 0:
                print(f"{n}/{len(rows)} rows", flush=True)
        if not args.with_degraded:
            for r in rows:
                r["rgb_degraded"] = ""
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
        z.writestr("paired_disaster/manifest.csv", buf.getvalue())
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB, {len(done)} files)")


if __name__ == "__main__":
    main()
