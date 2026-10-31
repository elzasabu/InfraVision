"""Colorizer post-processing: LAB chroma boost and thermal hotspot fusion."""
import cv2
import numpy as np

from infravision.colorpost import chroma_boost, hotspot_blend, postprocess


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
    assert saturation(postprocess(rgb, thermal, learned=True, hotspot_alpha=0)) > saturation(rgb)
