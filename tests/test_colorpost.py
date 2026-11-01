"""Colorizer post-processing: LAB chroma boost and thermal hotspot fusion."""
import cv2
import numpy as np

from infravision.colorpost import chroma_boost, colour_view, hotspot_blend, luma_transfer, postprocess


def saturation(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[..., 1].mean() / 255


def muted_image():
    rng = np.random.default_rng(0)
    base = np.full((64, 64, 3), (120, 128, 118), np.uint8)   # washed-out grey-green
    return np.clip(base + rng.integers(-6, 7, base.shape), 0, 255).astype(np.uint8)


def test_chroma_boost_raises_saturation_and_is_identity_at_one():
    img = muted_image()
    assert saturation(chroma_boost(img, 1.8, 0)) > 1.5 * saturation(img)
    same = chroma_boost(img, 1.0, 0)
    assert np.abs(same.astype(int) - img).max() <= 3          # LAB round trip only


def test_hotspot_colours_only_the_hot_pixels():
    rgb = np.full((100, 100, 3), 90, np.uint8)
    thermal = np.full((50, 50), 60, np.uint8)                 # half size: gets resized
    thermal[20:25, 20:25] = 250                               # one warm person
    out = hotspot_blend(rgb, thermal, alpha=0.35)
    assert out.shape == rgb.shape
    assert np.abs(out[40:50, 40:50].astype(int) - 90).max() > 20   # person is tinted
    assert np.array_equal(out[:20, :20], rgb[:20, :20])            # cold background untouched
    assert np.array_equal(hotspot_blend(rgb, thermal, alpha=0), rgb)
    assert np.array_equal(hotspot_blend(rgb, np.full((50, 50), 7, np.uint8)), rgb)   # flat frame


def test_postprocess_switches():
    rgb, thermal = muted_image(), np.zeros((64, 64), np.uint8)
    assert np.array_equal(postprocess(rgb, thermal, enabled=False), rgb)
    assert np.array_equal(postprocess(rgb, thermal, learned=False, hotspot_alpha=0), rgb)   # fallback map kept
    assert saturation(postprocess(rgb, thermal, learned=True, hotspot_alpha=0, view="ai")) > saturation(rgb)


def test_luma_view_keeps_thermal_detail_and_ai_colour():
    dark = np.full((32, 32, 3), (20, 30, 15), np.uint8)       # "night photo" colorizer output
    gray = np.full((64, 64), 100, np.uint8)
    gray[20:30, 20:30] = 240                                   # warm person
    out = luma_transfer(dark, gray, 1.8)
    assert out.shape == (64, 64, 3)
    L = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)[..., 0].astype(int)
    assert abs(L[0, 0] - 100) <= 3 and L[25, 25] > 220         # brightness follows the thermal frame
    assert out[0, 0, 1] > out[0, 0, 0] and out[0, 0, 1] > out[0, 0, 2]   # keeps the green tint


def test_colour_views():
    rgb, gray = muted_image(), np.tile(np.arange(64, dtype=np.uint8) * 4, (64, 1))
    assert colour_view(rgb, gray, "palette").shape == (64, 64, 3)
    assert not np.array_equal(colour_view(rgb, gray, "palette"), colour_view(rgb, gray, "luma"))
    assert np.array_equal(colour_view(rgb, gray, "luma", learned=False), rgb)   # fallback map untouched
    out, view = postprocess(rgb, gray, view="palette", return_view=True)
    assert np.array_equal(out, view)                           # no hotspot LUT on top of a palette
