"""Synthetic disaster conditions for the RGB half of a thermal/RGB pair.

Each effect takes a uint8 HxWx3 RGB array and returns a degraded copy. The
thermal half is never touched: long-wave IR passes through smoke, dust and
darkness, which is the whole point of the pairs.

Fog, smoke and dust use the atmospheric scattering model
    I = J * t + A * (1 - t),   t = exp(-beta * density)
with a different airlight colour A and density map for each. Darkness
simulates a short exposure in linear light with shot and read noise.
"""
import numpy as np
from PIL import Image, ImageFilter

SEVERITY = {"light": 0.35, "medium": 0.65, "heavy": 0.9}
CONDITIONS = ("fog", "smoke", "dust", "dark", "smoke+dark", "dust+dark")


def fbm(h, w, rng, octaves=5, base=3, persistence=0.55, stretch=1.0):
    """Fractal noise in [0, 1]. stretch > 1 elongates blobs vertically (rising plumes)."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        rows = max(2, int(base * 2 ** o / stretch))
        cols = max(2, int(base * 2 ** o * w / h))
        grid = rng.random((rows, cols)).astype(np.float32)
        out += amp * np.asarray(Image.fromarray(grid).resize((w, h), Image.BICUBIC))
        total += amp
        amp *= persistence
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def blur(img, radius):
    if radius <= 0:
        return img
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))


def to_f(img):
    return img.astype(np.float32) / 255.0


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def depth_ramp(h, w, rng):
    # Drone / oblique views: the top of the frame is usually further away.
    y = np.linspace(1.0, 0.35, h, dtype=np.float32)[:, None]
    tilt = rng.uniform(-0.15, 0.15) * np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    return np.clip(y + tilt, 0.2, 1.0)


def scatter(j, density, airlight):
    t = np.exp(-density)[..., None]
    return j * t + np.asarray(airlight, np.float32) * (1 - t)


def fog(img, s, rng):
    h, w = img.shape[:2]
    density = depth_ramp(h, w, rng) * (0.7 + 0.3 * fbm(h, w, rng, octaves=3))
    a = rng.uniform(0.78, 0.92)
    out = scatter(to_f(img), density * (0.6 + 3.0 * s), (a, a + 0.01, a + 0.03))
    return blur(to_u8(out), 1.5 * s)


def smoke(img, s, rng):
    h, w = img.shape[:2]
    # Patchy plumes: thick in places, thinner gaps where the scene shows through.
    plume = fbm(h, w, rng, octaves=6, base=2, stretch=1.8)
    plume = np.clip((plume - rng.uniform(0.2, 0.4)) / 0.6, 0, 1) ** 1.3
    density = (0.08 + 0.92 * plume) * (0.5 + 0.5 * depth_ramp(h, w, rng))
    grey = rng.uniform(0.28, 0.55)
    tint = np.array([1.0, 0.96, 0.9]) * grey
    out = scatter(to_f(img), density * (1.0 + 4.5 * s), tint)
    if rng.random() < 0.4:
        # Warm glow from fire below the smoke.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = rng.uniform(0, w), rng.uniform(0.6 * h, 1.2 * h)
        r = rng.uniform(0.3, 0.7) * max(h, w)
        glow = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)) * s * 0.5
        out = out + glow[..., None] * np.array([1.0, 0.45, 0.1], np.float32)
    return blur(to_u8(out), 2.0 * s)


def dust(img, s, rng):
    h, w = img.shape[:2]
    density = 0.5 + 0.5 * fbm(h, w, rng, octaves=5)
    tan = np.array([0.78, 0.66, 0.48]) * rng.uniform(0.85, 1.05)
    out = scatter(to_f(img), density * (0.6 + 2.8 * s), tan)
    grey = out.mean(axis=2, keepdims=True)
    out = out * (1 - 0.4 * s) + grey * 0.4 * s
    # Airborne particles: sparse specks, slightly blurred.
    n = int(h * w * 0.002 * s)
    ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
    specks = np.zeros((h, w), np.float32)
    specks[ys, xs] = rng.uniform(0.3, 1.0, n)
    specks = to_f(blur(to_u8(specks), 0.8))
    out = out + specks[..., None] * tan * 0.8
    return blur(to_u8(out), 1.2 * s)


def dark(img, s, rng):
    lin = to_f(img) ** 2.2
    exposure = 0.4 * (0.07 / 0.4) ** s * rng.uniform(0.7, 1.3)
    photons = 900.0
    lin = rng.poisson(lin * exposure * photons).astype(np.float32) / photons
    lin += rng.normal(0, 0.004 + 0.01 * s, lin.shape).astype(np.float32)
    lin = np.clip(lin, 0, 1)
    # Low light: colour fades and shifts cool.
    grey = lin.mean(axis=2, keepdims=True)
    lin = lin * (1 - 0.5 * s) + grey * 0.5 * s
    lin = lin * np.array([0.85, 0.95, 1.1], np.float32)
    return to_u8(np.clip(lin, 0, 1) ** (1 / 2.2))


EFFECTS = {"fog": fog, "smoke": smoke, "dust": dust, "dark": dark}


def degrade(img, condition, severity, rng):
    """Apply a condition such as "smoke" or "smoke+dark" at "light"/"medium"/"heavy"."""
    s = float(np.clip(SEVERITY[severity] + rng.uniform(-0.08, 0.08), 0.05, 1.0))
    out = img
    for name in condition.split("+"):
        # A second effect in a combo is applied a bit lighter so the frame isn't pure black.
        out = EFFECTS[name](out, s if out is img else 0.8 * s, rng)
    return out
