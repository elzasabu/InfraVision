"""Step 2.6 check: with the ThermoVision colorizer, boxes found on the thermal frame land at the right place on
the (2x SR) fused frame. Runs the default pipeline on frames given on the command line and compares the fused
boxes with the input boxes scaled by output/input size; also saves a crop sheet (thermal box | fused box).

usage: python -B box_check.py <frame> [<frame> ...] --out docs/colorizer_integration
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from infravision.io import save_rgb  # noqa: E402
from infravision.pipeline import Pipeline  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("frames", nargs="+")
p.add_argument("--out", required=True)
a = p.parse_args()

pipe = Pipeline(overrides={"device": "cpu"})
res, crops = {}, []
for f in a.frames:
    r = pipe.run(f)
    d = r["detections"]
    (wi, hi), (wo, ho) = d["input_size"], d["output_size"]
    expect = d["boxes_input"] * np.array([wo / wi, ho / hi, wo / wi, ho / hi], np.float32)
    err = float(np.abs(d["boxes"] - expect).max()) if d["count"] else 0.0
    res[Path(f).name] = dict(count=d["count"], input_size=[wi, hi], output_size=[wo, ho], max_abs_error_px=err,
                             fused_shape=list(r["fused"].shape), colorizer=pipe.describe()["colorizer"])
    th = cv2.cvtColor(r["thermal"], cv2.COLOR_GRAY2RGB)
    for (x1, y1, x2, y2), (X1, Y1, X2, Y2) in list(zip(d["boxes_input"], d["boxes"]))[:4]:
        pad = 12
        c1 = th[max(0, int(y1) - pad):int(y2) + pad, max(0, int(x1) - pad):int(x2) + pad]
        c2 = r["fused"][max(0, int(Y1) - 2 * pad):int(Y2) + 2 * pad, max(0, int(X1) - 2 * pad):int(X2) + 2 * pad]
        h = 160
        crops.append(np.hstack([cv2.resize(c1, (round(h * c1.shape[1] / c1.shape[0]), h)),
                                np.full((h, 6, 3), 255, np.uint8),
                                cv2.resize(c2, (round(h * c2.shape[1] / c2.shape[0]), h))]))
pipe.close()
if crops:
    w = max(c.shape[1] for c in crops)
    sheet = np.vstack([np.pad(c, ((0, 6), (0, w - c.shape[1]), (0, 0)), constant_values=255) for c in crops])
    save_rgb(Path(a.out) / "box_check.png", sheet)
Path(a.out, "box_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
