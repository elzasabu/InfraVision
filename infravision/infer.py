"""Batch inference: every thermal image in a folder -> fused images + detections.

    python -m infravision.infer --input POP/test/JPEGImages --out outputs/pop_test
    python -m infravision.infer --input frame.png --out outputs/ --all --device cpu

Writes, per image, <name>_fused.png (colorized + boxes + people count) and,
with --all, <name>_enhanced.png and <name>_sr.png. Also writes
detections.json (boxes in fused-image pixels, scores, timings) and
summary.csv. Unreadable files are reported and skipped, not fatal.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from .io import IMAGE_EXTS, ThermalImageError, list_images, save_rgb


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="image file or folder (searched recursively)")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--config", default=None, help="YAML config (default configs/infravision.yaml)")
    p.add_argument("--device", default=None, help="auto | cpu | cuda (overrides the config)")
    p.add_argument("--scale", type=int, default=None, help="SR scale (overrides the config)")
    p.add_argument("--limit", type=int, default=0, help="only the first N images")
    p.add_argument("--all", action="store_true", help="also save the enhanced and SR images")
    p.add_argument("--sequential", action="store_true", help="run the two branches one after the other")
    args = p.parse_args(argv)

    src = Path(args.input)
    if src.is_file():
        files = [src]
    elif src.is_dir():
        files = list_images(src)
    else:
        sys.exit(f"Input not found: {src}")
    if args.limit:
        files = files[: args.limit]
    if not files:
        sys.exit(f"No images ({', '.join(sorted(IMAGE_EXTS))}) in {src}")

    over = {}
    if args.device:
        over["device"] = args.device
    if args.scale:
        over["sr"] = {"scale": args.scale}
    from .pipeline import Pipeline

    pipe = Pipeline(args.config, over)
    info = pipe.describe()
    print("SR       :", info["sr"])
    print("Colorizer:", info["colorizer"])
    print("Detector :", info["detector"])
    for n in info["notes"]:
        print("NOTE     :", n)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results, failed = [], []
    for i, f in enumerate(files, 1):
        try:
            r = pipe.run(f, parallel=not args.sequential)
        except ThermalImageError as e:
            print(f"[{i}/{len(files)}] SKIP {f.name}: {e}")
            failed.append(dict(file=str(f), error=str(e)))
            continue
        stem = f.stem
        save_rgb(out / f"{stem}_fused.png", r["fused"])
        if args.all:
            save_rgb(out / f"{stem}_enhanced.png", r["enhanced_rgb"])
            save_rgb(out / f"{stem}_sr.png", r["sr"][..., None].repeat(3, 2))
        d = r["detections"]
        t = {k: round(v, 3) for k, v in r["timings"].items()}
        results.append(dict(file=str(f), people=d["count"], output_size=d["output_size"],
                            input_size=d["input_size"], boxes=d["boxes"].round(1).tolist(),
                            scores=d["scores"].round(3).tolist(), timings=t, notes=r["notes"]))
        print(f"[{i}/{len(files)}] {f.name}: {d['count']} people, {t['total']:.2f}s "
              f"(A {t['branch_a']:.2f}s | B {t['branch_b']:.2f}s)", flush=True)
    pipe.close()

    with open(out / "detections.json", "w") as fh:
        json.dump(dict(models=info, results=results, failed=failed), fh, indent=2)
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "people", "total_s", "branch_a_s", "branch_b_s"])
        for r in results:
            w.writerow([Path(r["file"]).name, r["people"], r["timings"]["total"], r["timings"]["branch_a"],
                        r["timings"]["branch_b"]])
    print(f"Done: {len(results)} images -> {out}" + (f", {len(failed)} skipped" if failed else ""))
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
