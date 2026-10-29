"""Fixtures: a tiny POP-style dataset and tiny random-weight models, so the
tests run on a CPU in seconds without the real datasets or checkpoints."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def thermal_frame(h=128, w=160, seed=0, people=((40, 30, 12, 20),)):
    """Smooth background with warm blobs where the people are."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(40, 90, (h, w)).astype(np.uint8), (0, 0), 3)
    for x, y, bw, bh in people:
        cv2.ellipse(img, (x + bw // 2, y + bh // 2), (bw // 2, bh // 2), 0, 0, 360, 230, -1)
    return img


@pytest.fixture(scope="session")
def pop_root(tmp_path_factory):
    """POP layout with 2 flights per split and COCO annotations."""
    root = tmp_path_factory.mktemp("POP")
    flights = {"train": ["F1", "F2"], "val": ["F13", "F14"], "test": ["T1", "T2"]}
    for split, fl in flights.items():
        (root / split / "JPEGImages").mkdir(parents=True)
        (root / split / "Annotations").mkdir(parents=True)
        images, anns = [], []
        for f in fl:
            for k in range(3):
                name = f"{f}_{k:04d}_50.0.JPG"
                img = thermal_frame(256, 320, seed=len(images), people=((60 + 10 * k, 50, 24, 36),))
                cv2.imwrite(str(root / split / "JPEGImages" / name), cv2.merge([img] * 3))
                iid = len(images) + 1
                images.append(dict(file_name=name, id=iid, width=320, height=256))
                anns.append(dict(id=len(anns) + 1, image_id=iid, category_id=1, bbox=[60 + 10 * k, 50, 24, 36]))
        with open(root / split / "Annotations" / f"{split}.json", "w") as fh:
            json.dump(dict(images=images, annotations=anns, categories=[dict(id=1, name="person")]), fh)
    return root


@pytest.fixture(scope="session")
def tiny_weights(tmp_path_factory):
    """Random-weight EDSR, colorizer and YOLO checkpoints in the formats the pipeline loads."""
    from infravision.enhance import build_colorizer, build_edsr

    d = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    edsr = build_edsr(1, 8, 1, True)
    torch.save({"state_dict": edsr.state_dict(),
                "meta": dict(in_channels=1, channels=8, num_blocks=1, global_residual=True)}, d / "edsr_best.pth")
    col = build_colorizer(8, 1)
    torch.save({"state_dict": col.state_dict(), "meta": dict(ngf=8, n_blocks=1)}, d / "colorizer_best.pth")

    from ultralytics import YOLO

    yolo = YOLO("yolo11n.yaml")   # built from the architecture file: no download
    yolo.model.names = {0: "person"}
    yolo.save(str(d / "yolo.pt"))
    return d


@pytest.fixture(scope="session")
def tiny_cfg(tiny_weights):
    from infravision.config import load_config

    w = tiny_weights
    return load_config(overrides={
        "device": "cpu",
        "sr": {"weights": [str(w / "edsr_best.pth")], "scale": 2, "max_input_side": 96},
        "colorizer": {"weights": [str(w / "colorizer_best.pth")], "work_side": 128},
        "detection": {"weights": [str(w / "yolo.pt")], "imgsz": 160, "conf": 0.25, "iou": 0.5},
    })


@pytest.fixture(scope="session")
def pipeline(tiny_cfg):
    from infravision.pipeline import Pipeline

    p = Pipeline(tiny_cfg)
    yield p
    p.close()
