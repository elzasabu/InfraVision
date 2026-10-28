import json

import cv2
import numpy as np

from conftest import thermal_frame


def test_cli_on_three_images(tmp_path, tiny_cfg, monkeypatch):
    import yaml

    from infravision import infer

    src = tmp_path / "in"
    src.mkdir()
    cv2.imwrite(str(src / "a.png"), thermal_frame(96, 128, seed=1))
    cv2.imwrite(str(src / "b.jpg"), cv2.merge([thermal_frame(99, 133, seed=2)] * 3))
    cv2.imwrite(str(src / "c.png"), (thermal_frame(80, 100, seed=3).astype(np.uint16) * 200))
    (src / "broken.jpg").write_bytes(b"\x00\x01garbage")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(tiny_cfg))
    out = tmp_path / "out"
    assert infer.main(["--input", str(src), "--out", str(out), "--config", str(cfg), "--all"]) == 0
    for stem, (h, w) in (("a", (192, 256)), ("b", (198, 266)), ("c", (160, 200))):
        img = cv2.imread(str(out / f"{stem}_fused.png"))
        assert img is not None and img.shape == (h, w, 3)
        assert (out / f"{stem}_enhanced.png").exists() and (out / f"{stem}_sr.png").exists()
    rep = json.loads((out / "detections.json").read_text())
    assert len(rep["results"]) == 3 and len(rep["failed"]) == 1
    assert "not a readable image" in rep["failed"][0]["error"]
    assert (out / "summary.csv").read_text().count("\n") == 4
