"""ColorizerAdapter: the ThermoVision U-Net behind one numpy interface.

    from infravision.colorizer import ColorizerAdapter
    adapter = ColorizerAdapter("checkpoints/colorizer/thermovision_exp7_last.pth")   # device="auto"
    rgb = adapter.colorize(thermal)                  # uint8 HxWx3 RGB, same HxW as `thermal`
    rgb = adapter.colorize(thermal, size=(w, h))     # or any requested output size
    rgbs = adapter.colorize_batch([a, b, c])

`thermal` may be uint8 or uint16/float (stretched to 8 bits with the same
0.5-99.5 percentile rule as infravision.io), HxW, HxWx1 or HxWx3/4. The model
sees what it was trained on: one grey channel, the whole frame resized to
`input_size` x `input_size` (256), values in [0, 1]. Its sigmoid output is
resized back with area (down) / bicubic (up) interpolation and rounded to uint8.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import resolve, short
from .io import to_gray, to_uint8
from .models.thermovision_colorizer import ARCHITECTURE, MULTIPLE, ThermoVisionUNet

PREFIXES = ("module.", "generator.", "netG.")
STATE_KEYS = ("model_state_dict", "state_dict", "generator", "netG", "G", "model")
EMA_KEYS = ("ema_state_dict", "model_ema", "state_dict_ema", "ema")


class ColorizerCheckpointError(RuntimeError):
    """The colorizer checkpoint is missing, unreadable or does not match the model."""


def resolve_device(name="auto"):
    """auto -> cuda when available, else cpu. A bad or unavailable device falls back to CPU with a warning."""
    if name in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        dev = torch.device(name)
    except (RuntimeError, ValueError):
        print(f"WARNING: unknown device {name!r} for the colorizer; using CPU")
        return torch.device("cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: {name} requested for the colorizer but CUDA is not available; using CPU")
        return torch.device("cpu")
    if dev.type not in ("cpu", "cuda"):
        print(f"WARNING: device {name!r} is not supported by the colorizer; using CPU")
        return torch.device("cpu")
    return dev


def strip_prefixes(sd):
    """Drop wrapper prefixes (DataParallel 'module.', GAN 'generator.' / 'netG.'), repeatedly."""
    out = {}
    for k, v in sd.items():
        while k.startswith(PREFIXES):
            k = k[k.index(".") + 1:]
        out[k] = v
    return out


def _state_dict(ck, prefer_ema):
    """(state_dict, which) from a ThermoVision, GAN or raw checkpoint."""
    if not isinstance(ck, dict):
        raise ColorizerCheckpointError(f"checkpoint is a {type(ck).__name__}, expected a dict")
    if prefer_ema:
        for k in EMA_KEYS:
            if isinstance(ck.get(k), dict) and ck[k]:
                return ck[k], f"EMA ({k})"
    for k in STATE_KEYS:
        if isinstance(ck.get(k), dict) and ck[k]:
            return ck[k], k
    if ck and all(torch.is_tensor(v) for v in ck.values()):
        return ck, "raw state_dict"
    raise ColorizerCheckpointError(f"no weights found; top-level keys are {sorted(ck)[:20]}")


def missing_checkpoint_message(path):
    return (f"ThermoVision colorizer checkpoint not found: {path}\n"
            f"  Fix: copy the ThermoVision exp7 weights there, e.g.\n"
            f"    copy <ThermoVision>\\experiments\\exp7_resnet34_unet_continue\\last.pth {path}\n"
            f"  (see README 'Model weights'), point colorizer.checkpoint in configs/infravision.yaml at it,\n"
            f"  or run with --colorizer infravision_tiny.")


def load_thermovision(path, device="cpu", prefer_ema=True, verbose=True):
    """Build the U-Net and load a checkpoint strictly. Raises ColorizerCheckpointError with the key diff."""
    path = resolve(path)
    if not path.is_file():
        raise ColorizerCheckpointError(missing_checkpoint_message(short(path)))
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:   # corrupt file, pickled objects, ...
        raise ColorizerCheckpointError(f"could not read colorizer checkpoint {path}: {e}") from e
    # A ThermoVision checkpoint names its architecture; one without model_config (raw / GAN state)
    # is checked against the U-Net keys below. exp1-4 checkpoints have model_config without it.
    cfg = ck.get("model_config") if isinstance(ck, dict) and isinstance(ck.get("model_config"), dict) else None
    arch = ARCHITECTURE if cfg is None else cfg.get("architecture")
    cfg = cfg or {}
    if arch != ARCHITECTURE:
        raise ColorizerCheckpointError(
            f"{path} is a {cfg} checkpoint, not a ThermoVision U-Net ({ARCHITECTURE}). "
            f"Only the exp6/exp7 U-Net checkpoints are supported by this backend.")
    sd, which = _state_dict(ck, prefer_ema)
    sd = strip_prefixes(sd)
    try:
        model = ThermoVisionUNet(cfg.get("encoder", "resnet34"))
    except ValueError as e:
        raise ColorizerCheckpointError(f"{path}: {e}") from e

    expected = model.state_dict()
    missing = sorted(set(expected) - set(sd))
    unexpected = sorted(set(sd) - set(expected))
    shapes = sorted(f"{k}: checkpoint {tuple(sd[k].shape)} vs model {tuple(v.shape)}"
                    for k, v in expected.items() if k in sd and tuple(sd[k].shape) != tuple(v.shape))
    if missing or unexpected or shapes:
        def show(keys):
            return ", ".join(keys[:15]) + (f", ... (+{len(keys) - 15})" if len(keys) > 15 else "")
        raise ColorizerCheckpointError(
            f"colorizer checkpoint {path} does not match the ThermoVision U-Net ({which} weights):\n"
            f"  missing keys ({len(missing)}): {show(missing)}\n"
            f"  unexpected keys ({len(unexpected)}): {show(unexpected)}\n"
            f"  shape mismatches ({len(shapes)}): {show(shapes)}")
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    run = ck.get("run_args", {}) if isinstance(ck.get("run_args"), dict) else {}
    info = dict(path=str(path), weights=which, step=ck.get("step"), epoch=ck.get("epoch"),
                val_total=ck.get("val_total"), run_name=run.get("name"),
                parameters=sum(p.numel() for p in model.parameters()))
    if verbose:
        when = ", ".join(f"{k} {info[k]}" for k in ("epoch", "step") if info[k] is not None) or "no epoch/step stored"
        print(f"Colorizer: loaded {short(path)} ({which} weights, run {info['run_name']}, {when}) on {device}")
    return model, info


class ColorizerAdapter:
    """Thermal frame(s) -> uint8 RGB with the ThermoVision U-Net."""

    def __init__(self, checkpoint, device="auto", input_size=256, prefer_ema=True, verbose=True):
        self.device = resolve_device(device)
        size = int(input_size)
        if size < MULTIPLE or size % MULTIPLE:
            size = max(MULTIPLE, round(size / MULTIPLE) * MULTIPLE)
            print(f"WARNING: colorizer input_size {input_size} is not a multiple of {MULTIPLE}; using {size}")
        self.input_size = size
        self.model, self.info = load_thermovision(checkpoint, self.device, prefer_ema, verbose)
        self.path = Path(self.info["path"])

    @property
    def name(self):
        return f"ThermoVision U-Net ({short(self.path)}, {self.input_size}px)"

    @staticmethod
    def to_gray8(thermal):
        """Any thermal array -> uint8 HxW (grey; 16-bit/float stretched as in infravision.io)."""
        a = np.asarray(thermal)
        if a.size == 0:
            raise ValueError("empty thermal image")
        if a.ndim == 3 and a.shape[2] == 4:
            a = a[..., :3]
        if a.ndim == 3 and a.shape[2] == 3 and a.dtype != np.uint8:
            a = a.mean(axis=2)          # cv2 colour conversion needs 8-bit / float32; thermal grey = channel mean
        gray, _ = to_gray(a)
        return np.ascontiguousarray(to_uint8(gray))

    def _input(self, gray):
        s = self.input_size
        h, w = gray.shape
        interp = cv2.INTER_AREA if (h > s or w > s) else cv2.INTER_LINEAR
        return cv2.resize(gray, (s, s), interpolation=interp).astype(np.float32) / 255.0

    @staticmethod
    def _output(y, size):
        """float32 sxsx3 in [0, 1] -> uint8 at size (w, h)."""
        w, h = size
        s = y.shape[0]
        if (h, w) != y.shape[:2]:
            interp = cv2.INTER_AREA if (h < s and w < s) else cv2.INTER_CUBIC
            y = cv2.resize(y, (w, h), interpolation=interp)
        return (np.clip(y, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    def colorize(self, thermal, size=None):
        """One frame -> uint8 HxWx3 RGB at `size` (w, h), default the input's own size."""
        return self.colorize_batch([thermal], None if size is None else [size])[0]

    def colorize_batch(self, frames, sizes=None, batch_size=8):
        """List of frames (any sizes / depths) -> list of uint8 RGB images."""
        grays = [self.to_gray8(f) for f in frames]
        sizes = list(sizes) if sizes is not None else [(g.shape[1], g.shape[0]) for g in grays]
        if len(sizes) != len(grays):
            raise ValueError(f"{len(grays)} frames but {len(sizes)} sizes")
        out = []
        self.model.eval()
        for i in range(0, len(grays), batch_size):
            x = torch.from_numpy(np.stack([self._input(g) for g in grays[i:i + batch_size]]))[:, None]
            with torch.inference_mode():
                y = self.model(x.to(self.device)).float().clamp_(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            out += [self._output(np.ascontiguousarray(yi), sz) for yi, sz in zip(y, sizes[i:i + batch_size])]
        return out
