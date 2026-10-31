"""InfraVision pipeline: branch A (enhance -> SR -> colorize) and branch B (YOLO)
run concurrently on one thermal frame, then fuse into one output image.

    from infravision.pipeline import Pipeline
    pipe = Pipeline()                      # configs/infravision.yaml
    out = pipe.run("POP/test/JPEGImages/T1_0016_29.562.JPG")
    out["fused"]          # HxWx3 uint8: colorized frame with people boxed
    out["detections"]     # boxes in fused-image pixels, scores, count
    out["timings"]        # seconds per stage / branch

`run(image)` (module level) does the same with a cached default pipeline.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .config import load_config
from .detect import Detector
from .enhance import Enhancer, pre_enhance
from .fusion import draw_detections, rescale_boxes
from .io import prepare_thermal, read_thermal


class Pipeline:
    def __init__(self, config=None, overrides=None):
        self.cfg = config if isinstance(config, dict) else load_config(config, overrides)
        self.enhancer = Enhancer(self.cfg)
        self.detector = Detector(device=self.cfg.get("device", "auto"), **self.cfg["detection"])
        # Two workers, one per branch; each branch is deterministic on its own,
        # so running them side by side gives the same result as running in turn.
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="infravision")

    @property
    def notes(self):
        return self.enhancer.notes + [n for n in (self.detector.note,) if n]

    def describe(self):
        e = self.enhancer
        return dict(sr=e.sr.name, colorizer=e.colorizer.name, detector=self.detector.name,
                    pre_enhance=self.cfg["pre_enhance"], notes=self.notes)

    def _branch_a(self, gray):
        t0 = time.perf_counter()
        stages = {}
        out = self.enhancer(gray, stages)
        stages["branch_a"] = time.perf_counter() - t0
        return out, stages

    def _branch_b(self, gray):
        t0 = time.perf_counter()
        src = gray
        if self.detector.run_on == "enhanced":
            src = pre_enhance(gray, **self.cfg["pre_enhance"])
        det = self.detector(src)
        return det, {"branch_b": time.perf_counter() - t0}

    def run(self, image, parallel=True):
        """image: path or numpy array (8/16-bit, grey or 3-channel)."""
        t0 = time.perf_counter()
        if isinstance(image, (str, Path)):
            gray, notes = read_thermal(image)
        else:
            gray, notes = prepare_thermal(image)
        t_load = time.perf_counter() - t0

        if parallel:
            fa = self.pool.submit(self._branch_a, gray)
            fb = self.pool.submit(self._branch_b, gray)
            (a, ta), (det, tb) = fa.result(), fb.result()
        else:
            (a, ta), (det, tb) = self._branch_a(gray), self._branch_b(gray)

        t1 = time.perf_counter()
        rgb = a["enhanced_rgb"]
        out_size = (rgb.shape[1], rgb.shape[0])
        boxes = rescale_boxes(det["boxes"], det["size"], out_size)
        fused = draw_detections(rgb, boxes, det["scores"], **self.cfg["fusion"])
        t_fuse = time.perf_counter() - t1

        timings = dict(load=t_load, **ta, **tb, fusion=t_fuse, total=time.perf_counter() - t0)
        return dict(
            thermal=gray,
            pre_enhanced=a["pre_enhanced"],
            sr=a["sr"],
            colorized=a["colorized"],   # colorizer output before colour boost / hotspots
            enhanced_rgb=rgb,
            fused=fused,
            detections=dict(boxes=boxes, scores=det["scores"], count=int(len(det["scores"])),
                            boxes_input=det["boxes"], input_size=det["size"], output_size=out_size),
            timings=timings,
            notes=notes + self.notes,
        )

    def close(self):
        self.pool.shutdown(wait=True)


_default = None


def run(image, **kw):
    """Run the default pipeline on one image -> {enhanced_rgb, fused, detections, timings, ...}."""
    global _default
    if _default is None:
        _default = Pipeline()
    return _default.run(image, **kw)


def as_uint8(a):
    return np.clip(a, 0, 255).astype(np.uint8)
