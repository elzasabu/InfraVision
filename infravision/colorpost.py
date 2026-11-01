"""Colorizer post-processing (RUN_ORDER 3): LAB chroma boost and thermal hotspot fusion.

The U-Net colorizer regresses towards washed-out average colours, and warm people
can end up the same grey-green as the background. Two fixes, both config switches
(`color_post` in configs/infravision.yaml):

  chroma_boost(rgb)               LAB a/b scaled by `boost`, optional CLAHE on L
  hotspot_blend(rgb, thermal)     hottest pixels (top 5% by default) get a magma
                                  false colour, alpha-blended so they stand out

The LLVIP-trained colorizer outputs dark "night photo" colour (mean brightness ~20 of 255
on POP vs ~100 for the thermal frame), which hides warm people. `colour_view` therefore
picks what the operator sees (`color_post.view`):

  luma     (default) lightness = the enhanced thermal frame, colour (LAB a/b) = the
           colorizer, boosted by `chroma`. All thermal detail stays; people stay bright.
  palette  thermal false-colour LUT (inferno), the standard thermal-camera display
  ai       the colorizer output itself (chroma boost + CLAHE on L), the old behaviour
"""
import cv2
import numpy as np

COLORMAPS = {"magma": cv2.COLORMAP_MAGMA, "inferno": cv2.COLORMAP_INFERNO, "hot": cv2.COLORMAP_HOT,
             "jet": cv2.COLORMAP_JET, "turbo": cv2.COLORMAP_TURBO}


def chroma_boost(rgb, boost=1.8, l_clahe_clip=1.5, l_clahe_tiles=8):
    """RGB uint8 -> RGB uint8 with LAB a/b pushed away from grey by `boost` (1 = unchanged)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    if l_clahe_clip and l_clahe_clip > 0:
        lab[..., 0] = cv2.createCLAHE(clipLimit=float(l_clahe_clip),
                                      tileGridSize=(int(l_clahe_tiles),) * 2).apply(lab[..., 0])
    ab = lab[..., 1:].astype(np.float32)
    lab[..., 1:] = np.clip((ab - 128) * float(boost) + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


VIEWS = ("luma", "palette", "ai")


def luma_transfer(rgb, gray, boost=1.8):
    """LAB lightness from `gray` (the thermal detail), a/b colour from `rgb`, scaled by `boost`."""
    h, w = gray.shape[:2]
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = gray
    ab = lab[..., 1:].astype(np.float32)
    lab[..., 1:] = np.clip((ab - 128) * float(boost) + 128, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def palette(gray, colormap="inferno"):
    return cv2.cvtColor(cv2.applyColorMap(gray, COLORMAPS.get(colormap, cv2.COLORMAP_INFERNO)), cv2.COLOR_BGR2RGB)


def colour_view(rgb, luma, view="luma", chroma=1.8, l_clahe_clip=1.5, palette_colormap="inferno", learned=True):
    """What stage 3 shows (see module doc). `luma`: enhanced thermal (uint8 HxW) at output size."""
    if not learned:            # fallback false-colour map: already a thermal palette
        return rgb
    if view == "palette":
        return palette(luma, palette_colormap)
    if view == "luma":
        return luma_transfer(rgb, luma, chroma)
    if chroma != 1 or l_clahe_clip:
        return chroma_boost(rgb, chroma, l_clahe_clip)
    return rgb


def hotspot_mask(thermal, percentile=95):
    """Soft 0-1 weight: 0 below the `percentile` intensity, rising to 1 at the hottest pixel."""
    t = thermal.astype(np.float32)
    lo, hi = np.percentile(t, percentile), t.max()
    if hi <= lo:  # flat frame: nothing stands out
        return np.zeros_like(t), lo, hi
    w = np.clip((t - lo) / (hi - lo), 0, 1)
    # Strictly above: when most of the frame shares one value, that value is the percentile itself.
    w = np.where(t > lo, np.maximum(w, 0.25), 0).astype(np.float32)   # every hot pixel is visible
    return cv2.GaussianBlur(w, (0, 0), 1.0), lo, hi                     # soft edges, no jaggies


def hotspot_blend(rgb, thermal, alpha=0.35, percentile=95, colormap="magma"):
    """Blend a false-colour LUT onto the hottest pixels of `thermal` (uint8, any size)."""
    if alpha <= 0:
        return rgb
    h, w = rgb.shape[:2]
    if thermal.shape[:2] != (h, w):
        thermal = cv2.resize(thermal, (w, h), interpolation=cv2.INTER_LINEAR)
    m, lo, hi = hotspot_mask(thermal, percentile)
    if hi <= lo:
        return rgb
    # Stretch only the hot range over the LUT so hotspots span its full colour range.
    hot = np.clip((thermal.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    lut = cv2.cvtColor(cv2.applyColorMap(hot, COLORMAPS.get(colormap, cv2.COLORMAP_MAGMA)), cv2.COLOR_BGR2RGB)
    a = (float(alpha) * m)[..., None]
    return np.clip(rgb.astype(np.float32) * (1 - a) + lut.astype(np.float32) * a, 0, 255).astype(np.uint8)


def postprocess(rgb, thermal, learned=True, chroma=1.8, l_clahe_clip=1.5, hotspot_alpha=0.35,
                hotspot_percentile=95, hotspot_colormap="magma", enabled=True, view="ai", luma=None,
                palette_colormap="inferno", return_view=False, **_):
    """Colour view then hotspots. `thermal`: SR frame before CLAHE (tracks temperature, for hotspots);
    `luma`: the displayed thermal (after CLAHE) for view=luma/palette (defaults to `thermal`).
    `learned` = a real colorizer made `rgb` (the fallback map is already vivid)."""
    if not enabled:
        return (rgb, rgb) if return_view else rgb
    if luma is None:
        luma = thermal
    if luma.shape[:2] != rgb.shape[:2] and view in ("luma", "palette"):
        luma = cv2.resize(luma, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    shown = colour_view(rgb, luma, view, chroma, l_clahe_clip, palette_colormap, learned)
    # A heat palette already colours the hot spots; a second LUT on top only muddies them.
    out = shown if (view == "palette" and learned) else hotspot_blend(shown, thermal, hotspot_alpha,
                                                                       hotspot_percentile, hotspot_colormap)
    return (out, shown) if return_view else out
