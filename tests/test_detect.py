import numpy as np

from infravision.detect import tile_grid
from infravision.train.yolo import average_precision, iou_matrix, match, pr_at


def test_tiles_cover_frame_with_overlap():
    tiles = tile_grid(1024, 1280, 2, 0.2)
    assert len(tiles) == 4
    assert min(t[0] for t in tiles) == 0 and max(t[2] for t in tiles) == 1280
    assert min(t[1] for t in tiles) == 0 and max(t[3] for t in tiles) == 1024
    (x0, _, x1, _), (x2, _, _, _) = tiles[0], tiles[1]
    assert x2 < x1   # neighbouring tiles overlap
    assert tile_grid(100, 100, 0, 0.2) == [(0, 0, 100, 100)]


def test_iou_and_matching():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    pred = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert abs(iou_matrix(pred, gt)[0, 0] - 1) < 1e-6
    s, tp, used = match(pred, np.array([0.9, 0.8, 0.7]), gt)
    assert tp.tolist() == [True, False, False]       # duplicate and stray are false positives
    assert used.tolist() == [True, False]


def test_average_precision_perfect_and_half():
    assert abs(average_precision(np.array([0.9, 0.8]), np.array([True, True]), 2) - 1) < 1e-9
    assert abs(average_precision(np.array([0.9]), np.array([True]), 2) - 0.5) < 1e-9
    row = pr_at(np.array([0.9, 0.3]), np.array([True, False]), 2, [0.5])[0]
    assert row["precision"] == 1 and row["recall"] == 0.5
