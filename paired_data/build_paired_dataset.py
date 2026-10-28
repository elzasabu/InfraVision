"""Build one paired thermal/RGB disaster dataset from several public sources.

Usage:
    python paired_data/build_paired_dataset.py --config paired_data/sources.json
    python paired_data/build_paired_dataset.py --config paired_data/sources.json --variants 3 --preview 12

Output (default data/paired_disaster/):
    {train,val,test}/thermal/<id>.png        grayscale thermal
    {train,val,test}/rgb/<id>.jpg            clean RGB, resized to the thermal frame
    {train,val,test}/rgb_degraded/<id>__<condition>_<severity>.jpg
    {train,val,test}/labels/<id>.txt         YOLO person boxes (class 0), when the source has them
    manifest.csv                             one row per degraded variant
    preview.jpg                              thermal | clean RGB | degraded variants

The config is a JSON list of sources; see sources.example.json.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

from degrade import CONDITIONS, SEVERITY, degrade

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
PERSON_NAMES = {"person", "people", "pedestrian", "human", "persons"}


# ---------------------------------------------------------------- sources

def list_images(folder, recursive=True):
    pattern = os.path.join(folder, "**", "*") if recursive else os.path.join(folder, "*")
    return sorted(p for p in glob.glob(pattern, recursive=recursive) if p.lower().endswith(IMG_EXT))


def match_by_stem(thermal_dir, rgb_dir, strip=None):
    """Pair files whose names match once modality tokens (strip regex) are removed."""
    rx = re.compile(strip, re.I) if strip else None

    def key(p, base):
        stem = os.path.splitext(os.path.relpath(p, base))[0].replace("\\", "/")
        return rx.sub("", stem) if rx else stem

    rgb = {key(p, rgb_dir): p for p in list_images(rgb_dir)}
    pairs = []
    for t in list_images(thermal_dir):
        k = key(t, thermal_dir)
        if k in rgb:
            pairs.append((t, rgb[k], k))
    return pairs


def llvip(src):
    # LLVIP/{infrared,visible}/{train,test}/<name>.jpg, Annotations/<name>.xml (VOC, "person")
    root = src["root"]
    out = []
    for split in ("train", "test"):
        for t, v, stem in match_by_stem(os.path.join(root, "infrared", split), os.path.join(root, "visible", split)):
            xml = os.path.join(root, "Annotations", os.path.basename(stem) + ".xml")
            out.append(dict(thermal=t, rgb=v, stem=stem, split=split, labels=xml if os.path.exists(xml) else None))
    return out


def dronevehicle(src):
    # DroneVehicle/{train,val,test}/{<s>img,<s>imgr}/<name>.jpg; images carry a 100px white border.
    root = src["root"]
    out = []
    for split in ("train", "val", "test"):
        for t, v, stem in match_by_stem(os.path.join(root, split, split + "imgr"), os.path.join(root, split, split + "img")):
            out.append(dict(thermal=t, rgb=v, stem=stem, split=split, labels=None))
    return out


def folders(src):
    # Any dataset laid out as a thermal folder and an RGB folder with matching file names.
    out = []
    for t, v, stem in match_by_stem(src["thermal"], src["rgb"], src.get("strip")):
        lab = None
        if src.get("labels"):
            for ext in (".txt", ".xml"):
                p = os.path.join(src["labels"], stem + ext)
                if os.path.exists(p):
                    lab = p
        out.append(dict(thermal=t, rgb=v, stem=stem, split=src.get("split"), labels=lab))
    return out


SOURCES = {"llvip": llvip, "dronevehicle": dronevehicle, "folders": folders}
DEFAULT_CROP = {"dronevehicle": 100}


# ---------------------------------------------------------------- labels

def read_labels(path, crop, size, person_ids):
    """Return YOLO person boxes normalised to the cropped thermal frame."""
    w, h = size
    boxes = []
    if path.endswith(".xml"):
        for obj in ET.parse(path).getroot().iter("object"):
            if obj.findtext("name", "").strip().lower() not in PERSON_NAMES:
                continue
            b = obj.find("bndbox")
            if b is None:
                continue
            x1, y1, x2, y2 = (float(b.findtext(k)) - crop for k in ("xmin", "ymin", "xmax", "ymax"))
            x1, x2 = max(0.0, x1), min(w, x2)
            y1, y2 = max(0.0, y1), min(h, y2)
            if x2 > x1 and y2 > y1:
                boxes.append(((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    else:
        # YOLO txt: already normalised to the source frame (assumed uncropped).
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5 and int(float(parts[0])) in person_ids:
                    boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


# ---------------------------------------------------------------- per pair

def split_for(item, val_pct):
    split = item.get("split") or "train"
    if split == "val":
        return "val"
    if split == "train" and int(hashlib.md5(item["id"].encode()).hexdigest(), 16) % 100 < val_pct:
        return "val"
    return split if split in ("train", "test") else "train"


def process(job):
    item, args = job
    out_root, split, pid = args["out"], item["out_split"], item["id"]
    try:
        t = Image.open(item["thermal"]).convert("L")
        v = Image.open(item["rgb"]).convert("RGB")
    except OSError as e:
        return [], f"skip {pid}: {e}"

    crop = item["crop"]
    if crop:
        t = t.crop((crop, crop, t.width - crop, t.height - crop))
        v = v.crop((crop, crop, v.width - crop, v.height - crop))
    label_size = t.size
    if args["size"]:
        t = t.resize(tuple(args["size"]), Image.BICUBIC)
    if v.size != t.size:
        v = v.resize(t.size, Image.BICUBIC)

    paths = {k: os.path.join(out_root, split, k) for k in ("thermal", "rgb", "rgb_degraded", "labels")}
    t_path = os.path.join(paths["thermal"], pid + ".png")
    v_path = os.path.join(paths["rgb"], pid + ".jpg")
    t.save(t_path)
    v.save(v_path, quality=95)

    lab_path = ""
    if item.get("labels"):
        boxes = read_labels(item["labels"], crop, label_size, set(args["person_ids"]))
        lab_path = os.path.join(paths["labels"], pid + ".txt")
        with open(lab_path, "w") as f:
            f.writelines(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in boxes)

    seed = int(hashlib.md5(f"{args['seed']}:{pid}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    rgb = np.asarray(v)
    rows = []
    # Distinct condition/severity combos per pair, so variants never overwrite each other.
    combos = [(c, s) for c in args["conditions"] for s in args["severities"]]
    for k in rng.permutation(len(combos))[: args["variants"]]:
        cond, sev = combos[k]
        d_path = os.path.join(paths["rgb_degraded"], f"{pid}__{cond.replace('+', '-')}_{sev}.jpg")
        Image.fromarray(degrade(rgb, cond, sev, rng)).save(d_path, quality=92)
        rows.append(dict(id=pid, source=item["source"], split=split,
                         thermal=os.path.relpath(t_path, out_root), rgb=os.path.relpath(v_path, out_root),
                         rgb_degraded=os.path.relpath(d_path, out_root), condition=cond, severity=sev,
                         labels=os.path.relpath(lab_path, out_root) if lab_path else ""))
    return rows, None


def save_preview(rows, out_root, n, cell=256):
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    picks = list(by_id.values())[:: max(1, len(by_id) // n)][:n]
    if not picks:
        return
    cols = 2 + max(len(p) for p in picks)
    sheet = Image.new("RGB", (cols * cell, len(picks) * cell * 4 // 5), "black")
    for i, group in enumerate(picks):
        tiles = [group[0]["thermal"], group[0]["rgb"]] + [r["rgb_degraded"] for r in group]
        for j, rel in enumerate(tiles):
            im = Image.open(os.path.join(out_root, rel)).convert("RGB").resize((cell, cell * 4 // 5))
            sheet.paste(im, (j * cell, i * cell * 4 // 5))
    sheet.save(os.path.join(out_root, "preview.jpg"), quality=90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="JSON list of sources (see sources.example.json)")
    p.add_argument("--out", default=os.path.join(ROOT, "data", "paired_disaster"))
    p.add_argument("--variants", type=int, default=2, help="degraded RGB copies per pair")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument("--severities", nargs="+", default=list(SEVERITY), choices=list(SEVERITY))
    p.add_argument("--size", type=int, nargs=2, metavar=("W", "H"), help="resize every pair (default: keep thermal size)")
    p.add_argument("--val_pct", type=int, default=5, help="percent of train held out as val")
    p.add_argument("--person_ids", type=int, nargs="+", default=[0], help="YOLO class ids that mean person")
    p.add_argument("--preview", type=int, default=8, help="rows in preview.jpg (0 = none)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.config) as f:
        sources = json.load(f)

    items = []
    for src in sources:
        kind = src["type"]
        name = src.get("name", kind)
        found = SOURCES[kind](src)
        if src.get("max"):
            found = found[: src["max"]]
        print(f"{name}: {len(found)} pairs")
        for it in found:
            it["source"] = name
            it["id"] = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{name}_{it['stem']}")
            it["crop"] = src.get("crop", DEFAULT_CROP.get(kind, 0))
            it["out_split"] = split_for(it, args.val_pct)
        items += found
    if not items:
        raise SystemExit("No pairs found; check the paths in the config.")

    for split in ("train", "val", "test"):
        for k in ("thermal", "rgb", "rgb_degraded", "labels"):
            os.makedirs(os.path.join(args.out, split, k), exist_ok=True)

    job_args = dict(out=args.out, variants=args.variants, conditions=args.conditions,
                    severities=args.severities, size=args.size, person_ids=args.person_ids, seed=args.seed)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (r, err) in enumerate(ex.map(process, [(it, job_args) for it in items], chunksize=16)):
            rows += r
            if err:
                print(err)
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{len(items)} pairs")

    fields = ["id", "source", "split", "thermal", "rgb", "rgb_degraded", "condition", "severity", "labels"]
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    if args.preview:
        save_preview(rows, args.out, args.preview)

    pairs = {r["id"]: r["split"] for r in rows}
    counts = {s: sum(1 for v in pairs.values() if v == s) for s in ("train", "val", "test")}
    print(f"Done: {len(pairs)} pairs, {len(rows)} degraded RGB images -> {args.out}")
    print("Pairs per split:", counts)


if __name__ == "__main__":
    main()
