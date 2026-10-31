"""Branch B: YOLO person detection on the thermal frame, optionally tiled."""
import numpy as np
import torch
from torchvision.ops import nms

from .config import ROOT, find_weights, pick_device, short


def tile_grid(h, w, n, overlap):
    """n x n overlapping tiles covering an h x w frame -> list of (x0, y0, x1, y1)."""
    if n <= 1:
        return [(0, 0, w, h)]
    th = int(np.ceil(h / (n - (n - 1) * overlap)))
    tw = int(np.ceil(w / (n - (n - 1) * overlap)))
    ys = np.linspace(0, h - th, n).round().astype(int)
    xs = np.linspace(0, w - tw, n).round().astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


class Detector:
    """Ultralytics YOLO, single class person. Returns boxes in input-frame pixels."""

    def __init__(self, weights=(), imgsz=1024, conf=0.25, iou=0.5, tile=0, tile_overlap=0.2,
                 device="auto", run_on="original", **_):
        from ultralytics import YOLO

        self.path = find_weights(weights)
        self.note = None
        if self.path is None:
            raise FileNotFoundError(
                "No YOLO weights found. Looked for:\n" + "\n".join(str(ROOT / w) for w in weights)
                + "\nTrain one (python -m infravision.train.yolo) or copy best.pt to checkpoints/yolo/.")
        if "runs/thermal" in self.path.as_posix():
            self.note = "detector is the FLIR street-scene model, not trained on POP drone views yet"
        self.model = YOLO(str(self.path))
        self.imgsz, self.conf, self.iou = int(imgsz), float(conf), float(iou)
        self.tile, self.overlap = int(tile), float(tile_overlap)
        self.run_on = run_on
        self.device = pick_device(device)

    @property
    def name(self):
        return f"YOLO ({short(self.path)}) imgsz={self.imgsz} conf={self.conf} iou={self.iou}" + (
            f" tiles={self.tile}x{self.tile}" if self.tile > 1 else "")

    def _predict(self, bgr, imgsz):
        r = self.model.predict(bgr, imgsz=imgsz, conf=self.conf, iou=self.iou, classes=[0],
                               device=str(self.device), quantize=16 if self.device.type == "cuda" else None,
                               verbose=False)[0]
        b = r.boxes
        return b.xyxy.cpu().numpy().astype(np.float32), b.conf.cpu().numpy().astype(np.float32)

    def __call__(self, gray):
        """uint8 HxW thermal -> dict(boxes Nx4 xyxy float32, scores N, size (w, h))."""
        h, w = gray.shape
        bgr = np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2))
        if self.tile <= 1:
            boxes, scores = self._predict(bgr, self.imgsz)
        else:
            all_b, all_s = [], []
            # Whole frame plus tiles, so big (near) people are not cut in half.
            for x0, y0, x1, y1 in [(0, 0, w, h)] + tile_grid(h, w, self.tile, self.overlap):
                b, s = self._predict(np.ascontiguousarray(bgr[y0:y1, x0:x1]), self.imgsz)
                all_b.append(b + np.array([x0, y0, x0, y0], np.float32))
                all_s.append(s)
            boxes, scores = np.concatenate(all_b), np.concatenate(all_s)
            if len(boxes):
                keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores), self.iou).numpy()
                boxes, scores = boxes[keep], scores[keep]
        order = np.argsort(-scores)
        return dict(boxes=boxes[order].reshape(-1, 4), scores=scores[order], size=(w, h))
