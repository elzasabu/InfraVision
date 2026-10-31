"""Pipeline configuration: defaults, YAML overrides and weight lookup."""
import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "infravision.yaml"

DEFAULTS = {
    "device": "auto",
    "datasets": {"pop": ["POP", "../datasets/POP"]},
    "pre_enhance": {"enabled": True, "clahe": True, "clahe_clip": 2.0, "clahe_tiles": 8,
                    "denoise": True, "denoise_strength": 5},
    "sr": {"enabled": True, "scale": 2, "max_input_side": 1280, "weights": []},
    "colorizer": {"enabled": True, "backend": "thermovision",
                  "checkpoint": "checkpoints/colorizer/thermovision_exp7_last.pth", "device": "auto",
                  "input_size": 256, "input": "auto", "work_side": 640, "detail": 0.5, "weights": [],
                  "fallback_colormap": "inferno"},
    "detection": {"run_on": "original", "imgsz": 1024, "conf": 0.25, "iou": 0.5,
                  "tile": 0, "tile_overlap": 0.2, "weights": []},
    "fusion": {"box_color": [0, 255, 0], "outline_color": [0, 0, 0], "thickness": 0,
               "show_conf": True, "label_people_count": True},
}


def merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path=None, overrides=None):
    """Defaults, then the YAML file (configs/infravision.yaml by default), then overrides."""
    cfg = DEFAULTS
    path = Path(path) if path else DEFAULT_CONFIG
    if path.exists():
        with open(path) as f:
            cfg = merge(cfg, yaml.safe_load(f) or {})
    elif path != DEFAULT_CONFIG:
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = merge(cfg, overrides)

    # Thresholds tuned on POP val by infravision.train.yolo --tune live next to
    # the weights they were tuned for.
    det = cfg["detection"]
    weights = find_weights(det["weights"])
    if weights:
        tuned = weights.parent / "thresholds.json"
        if tuned.exists() and not (overrides or {}).get("detection", {}).keys() & {"conf", "iou"}:
            with open(tuned) as f:
                t = json.load(f)
            det["conf"], det["iou"] = t.get("conf", det["conf"]), t.get("iou", det["iou"])
            det["tuned_from"] = str(tuned)
    return cfg


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def find_dir(candidates):
    """First existing folder in a list of candidate paths, or None."""
    for c in [candidates] if isinstance(candidates, (str, Path)) else candidates:
        p = resolve(c)
        if p.is_dir():
            return p
    return None


def find_weights(candidates):
    """First existing file in a list of candidate paths, or None."""
    if isinstance(candidates, (str, Path)):
        candidates = [candidates]
    for c in candidates:
        p = resolve(c)
        if p.is_file():
            return p
    return None


def pick_device(name="auto"):
    import torch

    if name in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: {name} requested but CUDA is not available; using CPU")
        return torch.device("cpu")
    return dev


def short(path):
    """Path relative to the repo when it is inside it, else as given (for messages)."""
    p = Path(path)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)
