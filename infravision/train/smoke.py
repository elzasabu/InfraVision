"""Quick check of all three training code paths on a CPU, before using Colab.

    python -m infravision.train.smoke            # SR -> colorizer -> YOLO, 2 tiny epochs each
    python -m infravision.train.smoke --only sr colorizer

Uses a handful of images from POP/ and data/paired_disaster/ and writes to
checkpoints/smoke/, which the demo never reads. Takes a few minutes.
"""
import argparse
import time
import traceback

from ..data.paired import DISASTER_DIR
from ..data.pop import POP_ROOT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", default=["sr", "colorizer", "yolo"], choices=["sr", "colorizer", "yolo"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--pop", default=str(POP_ROOT))
    p.add_argument("--data", default=str(DISASTER_DIR))
    args = p.parse_args(argv)

    from . import colorizer, sr, yolo

    jobs = dict(
        sr=lambda: sr.main(["--smoke", "--device", args.device, "--pop", args.pop, "--workers", "0"]),
        colorizer=lambda: colorizer.main(["--smoke", "--device", args.device, "--data", args.data,
                                          "--workers", "0"]),
        yolo=lambda: yolo.main(["train", "--smoke", "--device", args.device, "--pop", args.pop,
                                "--workers", "0"]),
    )
    results = {}
    for name in args.only:
        print(f"\n===== smoke: {name} =====", flush=True)
        t = time.time()
        try:
            jobs[name]()
            results[name] = f"OK ({time.time() - t:.0f}s)"
        except Exception as e:
            traceback.print_exc()
            results[name] = f"FAILED: {e.__class__.__name__}: {e}"
    print("\n===== smoke summary =====")
    for k, v in results.items():
        print(f"{k:10s} {v}")
    raise SystemExit(0 if all(v.startswith("OK") for v in results.values()) else 1)


if __name__ == "__main__":
    main()
