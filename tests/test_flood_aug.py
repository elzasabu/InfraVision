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
