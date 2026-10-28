import numpy as np
import pytest

from infravision.fusion import draw_detections, rescale_boxes


def test_rescale_doubles_with_2x_sr():
    b = rescale_boxes([[10, 20, 30, 40]], (100, 80), (200, 160))
    np.testing.assert_allclose(b, [[20, 40, 60, 80]])


def test_rescale_non_uniform_and_odd_sizes():
    # 101x77 input, 3x SR -> 303x231: x and y scale separately and exactly.
    b = rescale_boxes([[0, 0, 101, 77], [50.5, 38.5, 60, 70]], (101, 77), (303, 231))
    np.testing.assert_allclose(b, [[0, 0, 303, 231], [151.5, 115.5, 180, 210]], rtol=1e-6)
    b = rescale_boxes([[10, 10, 20, 20]], (100, 50), (150, 200))
    np.testing.assert_allclose(b, [[15, 40, 30, 80]])


def test_rescale_identity_clip_and_empty():
    np.testing.assert_allclose(rescale_boxes([[1, 2, 3, 4]], (64, 64), (64, 64)), [[1, 2, 3, 4]])
    np.testing.assert_allclose(rescale_boxes([[-5, -5, 70, 70]], (64, 64), (128, 128)), [[0, 0, 128, 128]])
    assert rescale_boxes(np.zeros((0, 4)), (64, 64), (128, 128)).shape == (0, 4)
    with pytest.raises(ValueError):
        rescale_boxes([[0, 0, 1, 1]], (0, 10), (10, 10))


def test_draw_keeps_shape_and_marks_boxes():
    img = np.full((120, 160, 3), 50, np.uint8)
    out = draw_detections(img, [[40, 40, 100, 100]], [0.9], box_color=(0, 255, 0))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert (out == [0, 255, 0]).all(axis=2).any()   # box colour present
    assert (img == 50).all()                          # input untouched
    empty = draw_detections(img, np.zeros((0, 4)), np.zeros(0))
    assert empty.shape == img.shape
