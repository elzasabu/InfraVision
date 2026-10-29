import cv2
import numpy as np

from conftest import thermal_frame


def test_pipeline_output_shapes(pipeline):
    g = thermal_frame(101, 141)
    r = pipeline.run(g)
    assert r["enhanced_rgb"].shape == (202, 282, 3)
    assert r["fused"].shape == r["enhanced_rgb"].shape
    d = r["detections"]
    assert d["input_size"] == (141, 101) and d["output_size"] == (282, 202)
    assert d["boxes"].shape == (d["count"], 4)
    for k in ("pre_enhance", "super_resolution", "colorize", "branch_a", "branch_b", "fusion", "total"):
        assert r["timings"][k] >= 0


def test_parallel_matches_sequential(pipeline):
    g = thermal_frame(96, 128, seed=4)
    a = pipeline.run(g, parallel=True)
    b = pipeline.run(g, parallel=False)
    np.testing.assert_array_equal(a["fused"], b["fused"])
    np.testing.assert_array_equal(a["detections"]["boxes"], b["detections"]["boxes"])


def test_boxes_follow_sr(pipeline, monkeypatch):
    """A fixed detection on the input frame lands at 2x on the 2x SR output."""
    def fake(gray):
        return dict(boxes=np.array([[10, 20, 30, 50]], np.float32), scores=np.array([0.9], np.float32),
                    size=(gray.shape[1], gray.shape[0]))
    monkeypatch.setattr(type(pipeline.detector), "__call__", lambda self, g: fake(g))
    r = pipeline.run(thermal_frame(100, 120))
    np.testing.assert_allclose(r["detections"]["boxes"], [[20, 40, 60, 100]])
    assert r["detections"]["count"] == 1


def test_16bit_and_three_channel_input(pipeline):
    raw = (thermal_frame(80, 96).astype(np.uint16) * 100 + 20000)
    assert pipeline.run(raw)["enhanced_rgb"].shape == (160, 192, 3)
    assert pipeline.run(cv2.merge([thermal_frame(80, 96)] * 3))["fused"].shape == (160, 192, 3)
