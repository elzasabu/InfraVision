"""ThermoVision colorizer adapter: input formats, determinism, strict loading, backend switch."""
import numpy as np
import pytest
import torch

from conftest import ROOT, thermal_frame
from infravision.colorizer import ColorizerAdapter, ColorizerCheckpointError, load_thermovision, resolve_device
from infravision.config import load_config
from infravision.enhance import Colorizer, Enhancer
from infravision.models.thermovision_colorizer import ThermoVisionUNet

REAL = ROOT / "checkpoints" / "colorizer" / "thermovision_exp7_last.pth"


@pytest.fixture(scope="module")
def adapter(tiny_weights):
    return ColorizerAdapter(tiny_weights / "thermovision.pth", device="cpu")


@pytest.mark.parametrize("make", [
    lambda: thermal_frame(97, 131),                                         # 8-bit, odd size
    lambda: thermal_frame(96, 128).astype(np.uint16) * 250 + 3000,          # 16-bit
    lambda: thermal_frame(97, 131)[..., None],                              # HxWx1
    lambda: np.dstack([thermal_frame(97, 131)] * 3),                        # grey stored as 3 channels
    lambda: np.dstack([thermal_frame(64, 300, seed=1), thermal_frame(64, 300, seed=2),
                       thermal_frame(64, 300, seed=3)]),                    # real 3-channel colour
    lambda: thermal_frame(33, 517).astype(np.float32) / 255.0,              # float, extreme aspect
])
def test_output_shape_dtype_range(adapter, make):
    x = make()
    out = adapter.colorize(x)
    assert out.shape == (x.shape[0], x.shape[1], 3) and out.dtype == np.uint8
    assert 0 <= out.min() <= out.max() <= 255
    assert adapter.colorize(x, size=(200, 150)).shape == (150, 200, 3)


def test_deterministic_and_batch_matches_single(adapter):
    a, b = thermal_frame(97, 131, seed=3), thermal_frame(120, 90, seed=4)
    np.testing.assert_array_equal(adapter.colorize(a), adapter.colorize(a))
    batch = adapter.colorize_batch([a, b])
    np.testing.assert_array_equal(batch[0], adapter.colorize(a))
    np.testing.assert_array_equal(batch[1], adapter.colorize(b))


def test_model_input_range_and_eval_mode(adapter):
    x = adapter._input(thermal_frame(97, 131))
    assert x.shape == (256, 256) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1
    assert not adapter.model.training


def _ck(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def test_strict_loading_reports_key_diff(tiny_weights, tmp_path):
    ck = _ck(tiny_weights / "thermovision.pth")
    sd = dict(ck["model_state_dict"])
    sd.pop("head.weight")
    sd["extra.layer"] = torch.zeros(1)
    torch.save({**ck, "model_state_dict": sd}, tmp_path / "bad.pth")
    with pytest.raises(ColorizerCheckpointError) as e:
        load_thermovision(tmp_path / "bad.pth")
    assert "missing keys (1): head.weight" in str(e.value) and "unexpected keys (1): extra.layer" in str(e.value)

    sd = dict(ck["model_state_dict"])
    sd["head.weight"] = torch.zeros(3, 16, 1, 1)
    torch.save({**ck, "model_state_dict": sd}, tmp_path / "shape.pth")
    with pytest.raises(ColorizerCheckpointError, match="shape mismatches \\(1\\): head.weight"):
        load_thermovision(tmp_path / "shape.pth")


def test_prefixes_and_ema(tiny_weights, tmp_path):
    ck = _ck(tiny_weights / "thermovision.pth")
    sd = ck["model_state_dict"]
    torch.save({"netG": {"module." + k: v for k, v in sd.items()}}, tmp_path / "gan.pth")
    m, info = load_thermovision(tmp_path / "gan.pth", verbose=False)
    assert info["weights"] == "netG"
    torch.save({f"generator.{k}": v for k, v in sd.items()}, tmp_path / "raw.pth")
    assert load_thermovision(tmp_path / "raw.pth", verbose=False)[1]["weights"] == "raw state_dict"

    ema = {k: (v + 1 if v.is_floating_point() else v) for k, v in sd.items()}
    torch.save({**ck, "ema_state_dict": ema}, tmp_path / "ema.pth")
    m, info = load_thermovision(tmp_path / "ema.pth", verbose=False)
    assert info["weights"].startswith("EMA")
    torch.testing.assert_close(m.head.bias, ema["head.bias"])
    m, info = load_thermovision(tmp_path / "ema.pth", prefer_ema=False, verbose=False)
    torch.testing.assert_close(m.head.bias, sd["head.bias"])


def test_wrong_architecture_and_missing_file(tmp_path, tiny_weights):
    torch.save({"model_config": {"decoder_type": "transposed"}, "model_state_dict": {"a": torch.zeros(1)}},
               tmp_path / "exp3.pth")
    with pytest.raises(ColorizerCheckpointError, match="not a ThermoVision U-Net"):
        load_thermovision(tmp_path / "exp3.pth")
    with pytest.raises(ColorizerCheckpointError) as e:
        load_thermovision(tmp_path / "nope.pth")
    assert "nope.pth" in str(e.value) and "Fix:" in str(e.value)
    # The pipeline stage keeps running on a false-colour map and says why.
    c = Colorizer(backend="thermovision", checkpoint=str(tmp_path / "nope.pth"), device="cpu")
    assert c.adapter is None and "nope.pth" in c.note and "Fix:" in c.note
    assert c(thermal_frame(64, 80)).shape == (64, 80, 3)
    # A file that exists but does not match fails loudly in the stage too.
    with pytest.raises(ColorizerCheckpointError):
        Colorizer(backend="thermovision", checkpoint=str(tmp_path / "exp3.pth"), device="cpu")


def test_device_fallback(capsys):
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("not-a-device").type == "cpu"
    if not torch.cuda.is_available():
        assert resolve_device("cuda").type == "cpu"
    assert "WARNING" in capsys.readouterr().out


def test_config_switch_picks_backend(tiny_cfg):
    from infravision.config import merge

    tv = Enhancer(tiny_cfg).colorizer
    assert tv.backend == "thermovision" and tv.adapter is not None and tv.input == "raw"
    old = Enhancer(merge(tiny_cfg, {"colorizer": {"backend": "infravision_tiny"}})).colorizer
    assert old.backend == "infravision_tiny" and old.adapter is None and old.model is not None
    assert old.input == "enhanced"
    assert load_config()["colorizer"]["backend"] == "thermovision"   # repo default
    with pytest.raises(ValueError, match="backend"):
        Colorizer(backend="nope")


def test_pipeline_end_to_end_three_images(pipeline):
    frames = [thermal_frame(96, 128, seed=1), np.dstack([thermal_frame(99, 133, seed=2)] * 3),
              thermal_frame(80, 100, seed=3).astype(np.uint16) * 200]
    assert pipeline.enhancer.colorizer.backend == "thermovision"
    for f in frames:
        r = pipeline.run(f)
        h, w = f.shape[:2]
        assert r["enhanced_rgb"].shape == (2 * h, 2 * w, 3) and r["enhanced_rgb"].dtype == np.uint8
        assert r["fused"].shape == r["enhanced_rgb"].shape
        d = r["detections"]
        assert d["output_size"] == (2 * w, 2 * h) and d["boxes"].shape == (d["count"], 4)


@pytest.mark.skipif(not REAL.is_file(), reason="real ThermoVision checkpoint not copied (see README)")
def test_real_checkpoint_loads_strictly():
    m, info = load_thermovision(REAL, verbose=False)
    assert isinstance(m, ThermoVisionUNet) and info["parameters"] == 24_435_203
    out = ColorizerAdapter(REAL, device="cpu", verbose=False).colorize(thermal_frame(128, 160))
    assert out.shape == (128, 160, 3) and out.std() > 1   # not a constant image
