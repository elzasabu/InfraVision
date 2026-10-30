"""Branch A: pre-enhancement -> super-resolution (EDSR) -> colorization.

Every stage takes and returns numpy uint8 images so stages can be swapped,
switched off from the config or tested on their own:

    pre_enhance(gray)              HxW   -> HxW     CLAHE + mild denoise
    SuperResolver(...)(gray)       HxW   -> sH x sW EDSR on a bicubic-upscaled frame
    Colorizer(...)(gray)           HxW   -> HxWx3   thermal -> RGB (ThermoVision U-Net or tiny generator)
"""
import importlib.util
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import ROOT, find_weights, pick_device, short


def load_module(rel_path, name):
    """Import a model file by path (SR/models and colorization/models are both
    called `models`, so they cannot both be imported as packages)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def edsr_module():
    return load_module("SR/models/edsr.py", "infravision_edsr")


def cyclegan_module():
    return load_module("colorization/models/cyclegan.py", "infravision_cyclegan")


def state_dict_of(ck):
    """Raw state_dict from any of the checkpoint formats used in this repo."""
    meta = {}
    if isinstance(ck, dict):
        for key in ("state_dict", "model_state_dict", "model", "G_AB"):
            if key in ck and isinstance(ck[key], dict):
                meta = {k: v for k, v in ck.items() if k != key and not isinstance(v, dict)}
                meta.update(ck.get("meta", {}) if isinstance(ck.get("meta"), dict) else {})
                ck = ck[key]
                break
    return {(k[7:] if k.startswith("module.") else k): v for k, v in ck.items()}, meta


# ---------------------------------------------------------------- pre-enhancement

def pre_enhance(gray, clahe=True, clahe_clip=2.0, clahe_tiles=8, denoise=True, denoise_strength=5,
                enabled=True, **_):
    """Local contrast (CLAHE) and mild non-local-means denoise for a uint8 thermal frame."""
    if not enabled:
        return gray
    out = gray
    if denoise and denoise_strength > 0:
        # Denoise first so CLAHE does not amplify sensor noise.
        out = cv2.fastNlMeansDenoising(out, None, h=float(denoise_strength), templateWindowSize=7,
                                       searchWindowSize=21)
    if clahe:
        tiles = max(1, int(clahe_tiles))
        out = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(tiles, tiles)).apply(out)
    return out


# ---------------------------------------------------------------- super-resolution

def build_edsr(in_channels=1, channels=64, num_blocks=8, global_residual=True):
    return edsr_module().EDSR(in_channels=in_channels, out_channels=in_channels, channels=channels,
                              num_blocks=num_blocks, global_residual=global_residual)


def load_edsr(path, device="cpu"):
    """EDSR from a checkpoint saved by infravision.train.sr or the older SR/ scripts."""
    sd, meta = state_dict_of(torch.load(path, map_location="cpu", weights_only=False))
    in_ch = meta.get("in_channels") or sd["head.weight"].shape[1]
    channels = meta.get("channels") or sd["head.weight"].shape[0]
    blocks = meta.get("num_blocks") or len({k.split(".")[1] for k in sd if k.startswith("body.")})
    # Older checkpoints without metadata: ssim_residual ones use the global residual.
    residual = meta.get("global_residual", "residual" in str(path) or "edsr_best" in str(path))
    model = build_edsr(in_ch, channels, blocks, residual)
    model.load_state_dict(sd)
    return model.to(device).eval(), {**meta, "in_channels": in_ch, "global_residual": residual}


class SuperResolver:
    """Bicubic x`scale` followed by EDSR restoration; bicubic only when no weights exist."""

    def __init__(self, weights=(), scale=2, max_input_side=1280, device="auto", enabled=True, **_):
        self.scale = max(1, int(scale))
        self.enabled = enabled
        self.device = pick_device(device)
        self.max_side = int(max_input_side)
        self.path = find_weights(weights) if enabled else None
        self.model, self.meta = (load_edsr(self.path, self.device) if self.path else (None, {}))
        self.note = None
        if enabled and self.model is None:
            self.note = "no SR weights found: bicubic upscaling only"

    @property
    def name(self):
        return f"EDSR ({short(self.path) if self.path else 'none'}) x{self.scale}"

    def __call__(self, gray):
        if not self.enabled:
            return gray
        h, w = gray.shape
        up = gray if self.scale == 1 else cv2.resize(gray, (w * self.scale, h * self.scale),
                                                      interpolation=cv2.INTER_CUBIC)
        if self.model is None:
            return up
        return self._run_tiled(up)

    @torch.inference_mode()
    def _run_tiled(self, img, overlap=24):
        """Run EDSR in as few tiles as fit max_input_side (bounded memory on CPU).

        Tiles overlap by `overlap` pixels on each side and only their centres
        are kept, so there are no seams.
        """
        H, W = img.shape
        x = torch.from_numpy(img).float().div_(255.0)[None, None]
        in_ch = self.meta["in_channels"]
        out = torch.empty(1, 1, H, W)
        ny, nx = -(-H // self.max_side), -(-W // self.max_side)
        ys = np.linspace(0, H, ny + 1).round().astype(int)
        xs = np.linspace(0, W, nx + 1).round().astype(int)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            for x0, x1 in zip(xs[:-1], xs[1:]):
                py0, px0 = max(0, y0 - overlap), max(0, x0 - overlap)
                py1, px1 = min(H, y1 + overlap), min(W, x1 + overlap)
                patch = x[..., py0:py1, px0:px1].repeat(1, in_ch, 1, 1).to(self.device)
                pred = self.model(patch).float().mean(1, keepdim=True).cpu()
                out[..., y0:y1, x0:x1] = pred[..., y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        out.clamp_(0, 1)
        return (out[0, 0].numpy() * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------- colorization

def build_colorizer(ngf=64, n_blocks=9):
    return cyclegan_module().ResnetGenerator(1, 3, ngf=ngf, n_blocks=n_blocks)


def load_colorizer(path, device="cpu"):
    sd, meta = state_dict_of(torch.load(path, map_location="cpu", weights_only=False))
    ngf = meta.get("ngf") or sd["net.1.weight"].shape[0]
    n_blocks = meta.get("n_blocks") or len({k.split(".")[1] for k in sd if ".body." in k})
    model = build_colorizer(ngf, n_blocks)
    model.load_state_dict(sd)
    return model.to(device).eval(), meta


COLORMAPS = {"inferno": cv2.COLORMAP_INFERNO, "jet": cv2.COLORMAP_JET, "turbo": cv2.COLORMAP_TURBO,
             "magma": cv2.COLORMAP_MAGMA, "hot": cv2.COLORMAP_HOT}


def colorize_tensor(model, gray, device):
    """uint8 HxW -> uint8 HxWx3 through the generator (pads to a multiple of 4)."""
    h, w = gray.shape
    x = torch.from_numpy(gray).float().div_(127.5).sub_(1.0)[None, None].to(device)
    ph, pw = (-h) % 4, (-w) % 4
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.inference_mode():
        y = model(x)[..., :h, :w]
    y = (y.float().clamp(-1, 1) + 1.0) * 127.5
    return y[0].permute(1, 2, 0).cpu().numpy().round().astype(np.uint8)


BACKENDS = ("thermovision", "infravision_tiny")


def add_detail(rgb, gray, scale, amount):
    """Add the high frequencies of `gray` that a model run at `scale` x the frame size cannot produce."""
    if amount <= 0 or scale >= 1.0:
        return rgb
    g = gray.astype(np.float32)
    hf = g - cv2.GaussianBlur(g, (0, 0), sigmaX=1.0 / scale)
    return np.clip(rgb.astype(np.float32) + amount * hf[..., None], 0, 255).astype(np.uint8)


class Colorizer:
    """Thermal (1 channel) -> RGB (3 channels), with one of two backends:

    thermovision      ThermoVision U-Net (infravision.colorizer.ColorizerAdapter):
                      the whole frame at input_size x input_size, as it was trained,
                      fed the raw thermal frame by default (`input: auto`)
    infravision_tiny  the InfraVision ResnetGenerator at `work_side` (longer side),
                      fed the CLAHE'd + SR frame it was trained on

    The colour is resized to the SR frame; `detail` adds the SR frame's
    high-frequency detail back so the super-resolution is not lost. Without
    weights a false-colour map is used and the reason (with the fix) is in `note`.
    """

    # The class keeps its old positional signature (weights first) and old default
    # backend; the pipeline's default (thermovision) comes from the config.
    def __init__(self, weights=(), work_side=640, detail=0.5, fallback_colormap="inferno", device="auto",
                 enabled=True, backend="infravision_tiny", checkpoint=None, input_size=256, input="auto", **_):
        if backend not in BACKENDS:
            raise ValueError(f"colorizer.backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend
        self.enabled = enabled
        self.device = pick_device(device)
        self.work_side = int(work_side)
        self.input_size = int(input_size)
        self.detail = float(detail)
        self.input = ("raw" if backend == "thermovision" else "enhanced") if input in (None, "auto") else input
        if self.input not in ("raw", "enhanced"):
            raise ValueError(f"colorizer.input must be auto, raw or enhanced, got {input!r}")
        self.colormap = COLORMAPS.get(fallback_colormap, cv2.COLORMAP_INFERNO)
        self.model, self.meta, self.adapter, self.path, self.note = None, {}, None, None, None
        if not enabled:
            return
        if backend == "thermovision":
            from .colorizer import ColorizerAdapter, missing_checkpoint_message

            path = find_weights(checkpoint) if checkpoint else None
            if path is None:
                # Missing file: keep the demo usable, but say exactly what is missing and how to fix it.
                self.note = (missing_checkpoint_message(checkpoint or "<colorizer.checkpoint not set>")
                             + f"\n  Showing a '{fallback_colormap}' false-colour map instead.")
                print("WARNING:", self.note)
            else:
                # A file that exists but does not load is an error, never a silent fallback.
                self.adapter = ColorizerAdapter(path, self.device, self.input_size)
                self.path, self.meta = self.adapter.path, self.adapter.info
        else:
            self.path = find_weights(weights)
            if self.path:
                self.model, self.meta = load_colorizer(self.path, self.device)
            else:
                self.note = f"no colorizer weights found: showing a '{fallback_colormap}' false-colour map instead"

    @property
    def name(self):
        if self.adapter is not None:
            return self.adapter.name
        return f"InfraVision tiny colorizer ({short(self.path)})" if self.path else "false-colour map"

    def __call__(self, gray, source=None):
        """gray: the (SR) frame that sets the output size and detail; source: what the model sees (default gray)."""
        if not self.enabled:
            return np.repeat(gray[..., None], 3, axis=2)
        src = gray if source is None else source
        h, w = gray.shape
        if self.adapter is not None:
            rgb = self.adapter.colorize(src, size=(w, h))
            return add_detail(rgb, gray, self.adapter.input_size / max(h, w), self.detail)
        if self.model is None:
            if src.shape != gray.shape:
                src = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC)
            return cv2.cvtColor(cv2.applyColorMap(src, self.colormap), cv2.COLOR_BGR2RGB)
        if src.shape != gray.shape:
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_CUBIC)
        s = min(1.0, self.work_side / max(h, w)) if self.work_side > 0 else 1.0
        small = src if s == 1.0 else cv2.resize(src, (max(4, round(w * s)), max(4, round(h * s))),
                                                interpolation=cv2.INTER_AREA)
        rgb = colorize_tensor(self.model, small, self.device)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_CUBIC)
            rgb = add_detail(rgb, gray, s, self.detail)
        return rgb


# ---------------------------------------------------------------- branch A

class Enhancer:
    """pre_enhance -> SR -> colorize, with per-stage timings."""

    def __init__(self, cfg):
        dev = cfg.get("device", "auto")
        self.pre_cfg = cfg["pre_enhance"]
        self.sr = SuperResolver(device=dev, **cfg["sr"])
        col = dict(cfg["colorizer"])
        # colorizer.device: auto (or unset) follows the pipeline device; cpu/cuda overrides it.
        if col.get("device") in (None, "", "auto"):
            col["device"] = dev
        self.colorizer = Colorizer(**col)

    @property
    def notes(self):
        return [n for n in (self.sr.note, self.colorizer.note) if n]

    def __call__(self, gray, timings=None):
        import time

        t = {}
        t0 = time.perf_counter()
        pre = pre_enhance(gray, **self.pre_cfg)
        t["pre_enhance"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sr = self.sr(pre)
        t["super_resolution"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        # input: raw -> the model sees the original frame (ThermoVision was trained on raw previews),
        # but the colour is produced at the SR size and gets the SR detail added back.
        rgb = self.colorizer(sr, source=gray if self.colorizer.input == "raw" else None)
        t["colorize"] = time.perf_counter() - t0
        if timings is not None:
            timings.update(t)
        return dict(pre_enhanced=pre, sr=sr, enhanced_rgb=rgb)
