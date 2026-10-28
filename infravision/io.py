"""Reading thermal frames: 8/16-bit, grayscale or 3-channel, with clear errors."""
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MIN_SIDE = 32


class ThermalImageError(ValueError):
    """The input cannot be used as a thermal frame (unreadable, empty, too small...)."""


def to_uint8(img, low_pct=0.5, high_pct=99.5):
    """Map any-depth thermal data to 0-255. 8-bit input is returned unchanged.

    16-bit and float radiometric frames are stretched between two percentiles,
    which keeps people visible when a few very hot pixels (fire) dominate.
    """
    if img.dtype == np.uint8:
        return img
    a = img.astype(np.float32)
    finite = np.isfinite(a)
    if not finite.any():
        raise ThermalImageError("image contains no finite pixel values")
    lo, hi = np.percentile(a[finite], [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1.0
    a = np.where(finite, a, lo)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).round().astype(np.uint8)


def to_gray(img):
    """HxW, HxWx1, HxWx3 or HxWx4 -> HxW (thermal is single channel)."""
    if img.ndim == 2:
        return img, None
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ThermalImageError(f"unsupported image shape {img.shape}")
    if img.shape[2] == 1:
        return img[..., 0], None
    rgb = img[..., :3]
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
        return rgb[..., 0], None   # grey stored as 3 channels (POP, LLVIP)
    note = "input has colour channels; converted to grey (is this a false-colour thermal image?)"
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY), note


def prepare_thermal(img):
    """Any numpy thermal frame -> (uint8 HxW grey, list of notes)."""
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if img.size == 0:
        raise ThermalImageError("image is empty")
    gray, note = to_gray(img)
    notes = [note] if note else []
    if gray.dtype != np.uint8:
        notes.append(f"{gray.dtype} input stretched to 8-bit (0.5-99.5 percentile)")
        gray = to_uint8(gray)
    h, w = gray.shape
    if min(h, w) < MIN_SIDE:
        raise ThermalImageError(f"image is {w}x{h}; need at least {MIN_SIDE}x{MIN_SIDE} pixels")
    return np.ascontiguousarray(gray), notes


def read_thermal(path):
    """Read a thermal image file -> (uint8 HxW grey, notes). Raises ThermalImageError."""
    path = Path(path)
    if not path.is_file():
        raise ThermalImageError(f"file not found: {path}")
    if path.stat().st_size == 0:
        raise ThermalImageError(f"file is empty (0 bytes): {path}")
    # imdecode handles non-ASCII Windows paths; UNCHANGED keeps 16-bit data.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ThermalImageError(f"not a readable image (corrupt or unsupported format): {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    return prepare_thermal(img)


def read_rgb(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ThermalImageError(f"not a readable image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"could not encode {path}")
    buf.tofile(str(path))


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
