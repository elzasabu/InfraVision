import cv2
import numpy as np
import pytest

from infravision.io import ThermalImageError, prepare_thermal, read_thermal


def test_grey_and_three_channel_grey(tmp_path):
    g = (np.arange(64 * 80) % 256).astype(np.uint8).reshape(64, 80)
    cv2.imwrite(str(tmp_path / "g.png"), g)
    cv2.imwrite(str(tmp_path / "c.png"), cv2.merge([g, g, g]))
    a, na = read_thermal(tmp_path / "g.png")
    b, nb = read_thermal(tmp_path / "c.png")
    assert a.shape == b.shape == (64, 80) and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert na == nb == []


def test_16bit_is_stretched(tmp_path):
    raw = np.linspace(20000, 30000, 64 * 64).astype(np.uint16).reshape(64, 64)
    cv2.imwrite(str(tmp_path / "r.png"), raw)
    g, notes = read_thermal(tmp_path / "r.png")
    assert g.dtype == np.uint8 and g.min() == 0 and g.max() == 255
    assert any("uint16" in n for n in notes)


def test_colour_input_converted_with_note():
    rgb = np.random.default_rng(0).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    g, notes = prepare_thermal(rgb)
    assert g.shape == (40, 50) and notes


def test_rgba_and_float():
    rgba = np.zeros((40, 40, 4), np.uint8)
    assert prepare_thermal(rgba)[0].shape == (40, 40)
    f = np.random.default_rng(0).normal(300, 5, (40, 40)).astype(np.float32)
    assert prepare_thermal(f)[0].dtype == np.uint8


@pytest.mark.parametrize("content,msg", [(b"", "empty"), (b"not an image at all", "not a readable image")])
def test_bad_files(tmp_path, content, msg):
    p = tmp_path / "bad.jpg"
    p.write_bytes(content)
    with pytest.raises(ThermalImageError, match=msg):
        read_thermal(p)


def test_missing_and_tiny(tmp_path):
    with pytest.raises(ThermalImageError, match="not found"):
        read_thermal(tmp_path / "nope.png")
    with pytest.raises(ThermalImageError, match="at least"):
        prepare_thermal(np.zeros((10, 200), np.uint8))
