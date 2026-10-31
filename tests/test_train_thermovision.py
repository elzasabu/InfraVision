"""ThermoVision colorizer training: smoke run on a fake LLVIP folder, output loads in the pipeline."""
import cv2
import numpy as np

from infravision.colorizer import ColorizerAdapter
from infravision.train.thermovision import llvip_pairs, main


def test_smoke_training_writes_loadable_checkpoint(tmp_path):
    rng = np.random.default_rng(0)
    data = tmp_path / "llvip"
    for split in ("train", "test"):
        for kind in ("ir", "rgb"):
            (data / split / kind).mkdir(parents=True)
        for i in range(4):
            cv2.imwrite(str(data / split / "ir" / f"{i}.jpg"), rng.integers(0, 255, (80, 100), np.uint8))
            cv2.imwrite(str(data / split / "rgb" / f"{i}.jpg"), rng.integers(0, 255, (80, 100, 3), np.uint8))
    assert len(llvip_pairs(data, "train")) == 4

    out = tmp_path / "out"
    main(["--smoke", "--device", "cpu", "--workers", "0", "--data", str(data), "--out", str(out)])
    adapter = ColorizerAdapter(out / "thermovision_best.pth", device="cpu", input_size=64, verbose=False)
    assert adapter.colorize(np.zeros((50, 70), np.uint8)).shape == (50, 70, 3)
