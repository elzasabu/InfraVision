"""Check a built paired dataset and draw a spot-check grid.

Usage:
    python paired_data/verify_dataset.py                       # data/paired_disaster
    python paired_data/verify_dataset.py --grid docs/paired_dataset_samples.jpg --samples 12
    python paired_data/verify_dataset.py --quick               # sizes only, no image decoding

Checks, per manifest row:
  * every thermal / RGB / degraded RGB / label file exists and is non-empty
  * images decode (skipped with --quick) and thermal and RGB are the same size
  * label lines are "0 cx cy w h" with values in [0, 1]
  * no leftover *.tmp files from an interrupted build
Then prints pair counts per split and per condition, and writes a JSON report.
Exit code is 1 when anything is broken.
"""
import argparse
import collections
import csv
import glob
import json
import os
import random

from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def read_labels(path):
    boxes, bad = [], []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            try:
                cls, *xywh = int(parts[0]), *map(float, parts[1:5])
            except ValueError:
                bad.append(n)
                continue
            if len(parts) != 5 or cls != 0 or not all(0.0 <= v <= 1.0 for v in xywh):
                bad.append(n)
                continue
            boxes.append(xywh)
    return boxes, bad


def check_image(path, quick):
    if not os.path.exists(path):
        return None, "missing"
    if os.path.getsize(path) == 0:
        return None, "zero-byte"
    if quick:
        return None, None
    try:
        with Image.open(path) as im:
            im.load()
            return im.size, None
    except Exception as e:  # truncated or corrupt
        return None, f"unreadable ({e.__class__.__name__})"


def draw_boxes(img, boxes, color, width=3):
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for cx, cy, bw, bh in boxes:
        d.rectangle([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
                    outline=color, width=width)
    return img


def save_grid(root, groups, out, n, seed, cell=(320, 256)):
    """n random pairs, one row each: thermal | degraded RGB | clean RGB with person boxes."""
    picks = random.Random(seed).sample(sorted(groups), min(n, len(groups)))
    cw, ch = cell
    header = 28
    sheet = Image.new("RGB", (3 * cw, header + len(picks) * ch), "black")
    d = ImageDraw.Draw(sheet)
    for j, title in enumerate(("thermal", "degraded RGB (synthetic)", "RGB + person boxes")):
        d.text((j * cw + 8, 8), title, fill="white")
    for i, pid in enumerate(picks):
        row = random.Random(f"{seed}:{pid}").choice(groups[pid])
        boxes = read_labels(os.path.join(root, row["labels"]))[0] if row["labels"] else []
        tiles = [
            Image.open(os.path.join(root, row["thermal"])),
            Image.open(os.path.join(root, row["rgb_degraded"])),
            draw_boxes(Image.open(os.path.join(root, row["rgb"])), boxes, (0, 255, 0)),
        ]
        y = header + i * ch
        for j, im in enumerate(tiles):
            sheet.paste(im.convert("RGB").resize(cell), (j * cw, y))
        d.text((8, y + 6), f"{pid}  {row['condition']}/{row['severity']}  people={len(boxes)}",
               fill="yellow")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    sheet.save(out, quality=90)
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(ROOT, "data", "paired_disaster"))
    p.add_argument("--quick", action="store_true", help="check existence and size only")
    p.add_argument("--expect_pairs", type=int, default=0, help="fail if the pair count differs")
    p.add_argument("--expect_variants", type=int, default=0, help="fail if a pair has a different count")
    p.add_argument("--grid", default="", help="write a spot-check grid to this path")
    p.add_argument("--samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", default="", help="JSON report path (default <root>/verify_report.json)")
    args = p.parse_args()

    root = args.root
    # Manifest paths use the separator of the OS that built them.
    with open(os.path.join(root, "manifest.csv"), newline="") as f:
        rows = [{k: (v.replace("\\", "/") if k in ("thermal", "rgb", "rgb_degraded", "labels") else v)
                 for k, v in r.items()} for r in csv.DictReader(f)]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["id"]].append(r)

    problems = collections.defaultdict(list)
    seen = set()
    for n, (pid, group) in enumerate(groups.items(), 1):
        first = group[0]
        t_size, err = check_image(os.path.join(root, first["thermal"]), args.quick)
        if err:
            problems[f"thermal {err}"].append(first["thermal"])
        v_size, err = check_image(os.path.join(root, first["rgb"]), args.quick)
        if err:
            problems[f"rgb {err}"].append(first["rgb"])
        if t_size and v_size and t_size != v_size:
            problems["thermal/rgb size mismatch"].append(pid)
        for r in group:
            if r["rgb_degraded"] in seen:
                problems["duplicate degraded path"].append(r["rgb_degraded"])
            seen.add(r["rgb_degraded"])
            d_size, err = check_image(os.path.join(root, r["rgb_degraded"]), args.quick)
            if err:
                problems[f"rgb_degraded {err}"].append(r["rgb_degraded"])
            elif d_size and t_size and d_size != t_size:
                problems["degraded size mismatch"].append(r["rgb_degraded"])
        if first["labels"]:
            lab = os.path.join(root, first["labels"])
            if not os.path.exists(lab):
                problems["label missing"].append(first["labels"])
            else:
                _, bad = read_labels(lab)
                if bad:
                    problems["label bad lines"].append(first["labels"])
        if args.expect_variants and len(group) != args.expect_variants:
            problems[f"pair without {args.expect_variants} variants"].append(pid)
        if n % 2000 == 0:
            print(f"checked {n}/{len(groups)} pairs", flush=True)

    tmp = glob.glob(os.path.join(root, "*", "*", "*.tmp"))
    if tmp:
        problems["leftover .tmp files"] = [os.path.relpath(t, root) for t in tmp]

    # Files on disk that the manifest does not know about (e.g. from an older build).
    on_disk = {os.path.relpath(f, root).replace("\\", "/")
               for f in glob.glob(os.path.join(root, "*", "*", "*")) if not f.endswith(".tmp")}
    known = {r[k] for r in rows for k in ("thermal", "rgb", "rgb_degraded", "labels") if r[k]}
    extra = sorted(on_disk - known)

    pairs_per_split = collections.Counter(g[0]["split"] for g in groups.values())
    labelled = sum(1 for g in groups.values() if g[0]["labels"])
    people = 0
    if not args.quick:
        for g in groups.values():
            if g[0]["labels"] and os.path.exists(os.path.join(root, g[0]["labels"])):
                people += len(read_labels(os.path.join(root, g[0]["labels"]))[0])
    report = dict(
        root=root, pairs=len(groups), degraded_images=len(rows),
        pairs_per_split=dict(pairs_per_split),
        degraded_per_split=dict(collections.Counter(r["split"] for r in rows)),
        per_condition=dict(collections.Counter(r["condition"] for r in rows)),
        per_severity=dict(collections.Counter(r["severity"] for r in rows)),
        per_source=dict(collections.Counter(g[0]["source"] for g in groups.values())),
        variants_per_pair=dict(collections.Counter(len(g) for g in groups.values())),
        pairs_with_labels=labelled, person_boxes=people if not args.quick else None,
        files_not_in_manifest=len(extra), files_not_in_manifest_examples=extra[:10],
        problems={k: dict(count=len(v), examples=v[:10]) for k, v in problems.items()},
        quick=args.quick,
    )
    if args.expect_pairs and len(groups) != args.expect_pairs:
        report["problems"]["pair count"] = dict(count=1, examples=[f"{len(groups)} != {args.expect_pairs}"])

    if args.grid:
        report["grid"] = args.grid
        report["grid_ids"] = save_grid(root, groups, args.grid, args.samples, args.seed)

    out = args.report or os.path.join(root, "verify_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "grid_ids"}, indent=2))
    print("OK" if not report["problems"] else "PROBLEMS FOUND", "-> report:", out)
    raise SystemExit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
