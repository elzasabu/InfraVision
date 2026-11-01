"""Synthetic flood occlusion: covers the lower part of each box, leaves the rest of the frame alone."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "prepare_flood_aug", Path(__file__).resolve().parents[1] / "scripts" / "prepare_flood_aug.py")
flood_aug = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flood_aug)


def test_read_boxes(tmp_path):
    lbl = tmp_path / "a.txt"
    lbl.write_text("0 0.5 0.5 0.2 0.4\n\n")
    assert flood_aug.read_boxes(lbl, 100, 50) == [(40, 15, 60, 35)]
    assert flood_aug.read_boxes(tmp_path / "missing.txt", 100, 50) == []


def test_flood_covers_lower_part_of_warm_person():
    img = np.full((200, 200), 80, np.uint8)
    img[60:140, 90:110] = 230                                   # warm person
    out = flood_aug.flood(img, [(90, 60, 110, 140)], np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[125:140, 95:105].mean() < 120                    # feet under cool water
    assert out[60:70, 95:105].mean() > 200                      # head still warm
    assert np.array_equal(out[:, :20], img[:, :20])             # far away untouched


def test_decoys_same_size_and_clear_of_people():
    rng = np.random.default_rng(0)
    boxes = [(100, 100, 120, 150), (400, 300, 430, 360)]
    dec = flood_aug.decoy_boxes(boxes, 640, 512, rng)
    assert len(dec) == 2
    for (x0, y0, x1, y1), (a0, b0, a1, b1) in zip(boxes, dec):
        assert (x1 - x0, y1 - y0) == (a1 - a0, b1 - b0)               # same size as its person
        assert 0 <= a0 and a1 <= 640 and 0 <= b0 and b1 <= 512
        for p in boxes:                                               # no overlap with any person
            assert a1 <= p[0] or p[2] <= a0 or b1 <= p[1] or p[3] <= b0
