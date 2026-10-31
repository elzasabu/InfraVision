import numpy as np

from conftest import thermal_frame
from infravision.enhance import Colorizer, SuperResolver, pre_enhance


def test_pre_enhance_keeps_shape_and_can_be_disabled():
    g = thermal_frame(97, 131)
    out = pre_enhance(g)
    assert out.shape == g.shape and out.dtype == np.uint8
    assert pre_enhance(g, enabled=False) is g


def test_sr_scale_and_tiling(tiny_weights):
    g = thermal_frame(97, 131)   # odd sizes
    sr = SuperResolver([str(tiny_weights / "edsr_best.pth")], scale=2, max_input_side=64, device="cpu")
    out = sr(g)
    assert out.shape == (194, 262) and out.dtype == np.uint8
    bic = SuperResolver([], scale=3, device="cpu")
    assert bic(g).shape == (291, 393) and bic.note


def test_colorizer_one_to_three_channels(tiny_weights):
    g = thermal_frame(97, 131)
    c = Colorizer([str(tiny_weights / "colorizer_best.pth")], work_side=64, device="cpu")
    out = c(g)
    assert out.shape == (97, 131, 3) and out.dtype == np.uint8
    fallback = Colorizer([], device="cpu")
    assert fallback(g).shape == (97, 131, 3) and fallback.note


def test_tiled_sr_has_no_seams(tiny_weights):
    g = thermal_frame(150, 200, seed=7)
    w = [str(tiny_weights / "edsr_best.pth")]
    whole = SuperResolver(w, scale=2, max_input_side=4096, device="cpu")(g)
    tiled = SuperResolver(w, scale=2, max_input_side=96, device="cpu")(g)
    assert np.abs(whole.astype(int) - tiled).max() <= 1


def test_bilateral_denoise_and_bicubic_backend(tiny_weights):
    g = thermal_frame(97, 131)
    out = pre_enhance(g, denoise_method="bilateral", clahe=False)
    assert out.shape == g.shape and out.dtype == np.uint8 and not np.array_equal(out, g)
    bic = SuperResolver([str(tiny_weights / "edsr_best.pth")], scale=2, device="cpu", backend="bicubic")
    assert bic.model is None and bic.note is None and bic.name == "bicubic x2"


def test_clahe_runs_after_sr_when_post():
    from infravision.config import load_config
    from infravision.enhance import Enhancer, apply_clahe

    cfg = load_config(overrides={"device": "cpu", "sr": {"backend": "bicubic"}, "colorizer": {"enabled": False},
                                 "pre_enhance": {"clahe_stage": "post", "denoise": False}})
    g = thermal_frame(64, 80)
    t = {}
    out = Enhancer(cfg)(g, t)
    assert np.array_equal(out["pre_enhanced"], g)          # no CLAHE before SR
    assert out["sr"].shape == (128, 160) and "post_clahe" in t
    up = SuperResolver([], scale=2, device="cpu", backend="bicubic")(g)
    assert np.array_equal(out["sr"], apply_clahe(up, cfg["pre_enhance"]["clahe_clip"]))
