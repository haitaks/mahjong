"""Preprocessing: PIL crop -> upscaled grayscale + binary mask + color mask.

Three robustness layers, in increasing strength:

  1. **Color mask** (preferred on quality color photos): markings on suited
     tiles are colored (red/green/blue dots and sticks) on a white face, so
     saturation segmentation is far more reliable than luminance thresholding.
     Returns None on grayscale images so callers can fall back.
  2. **Tile-face detection**: the largest light rectangular region inside the
     crop. Removes the tile's own frame/edge, which otherwise dominates
     connected-components. Falls back to a fixed inset when detection fails.
  3. **Adaptive threshold** (fallback): luminance binarization, used when color
     is unavailable.

cv2 is imported lazily; a numpy/PIL fallback keeps the module usable in a
minimal environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .types import ClassifyParams


@dataclass
class PreparedTile:
    """Output of :func:`prepare`."""

    gray: np.ndarray  # uint8 grayscale of the analyzed region, upscaled
    binary: np.ndarray  # uint8 0/255; 255 = foreground (ink). May come from
    # adaptive-threshold OR from the color mask when color_tile is used.
    color_mask: Optional[np.ndarray]  # uint8 0/255 colored-pixel mask, or None
    face_bbox: Optional[tuple[int, int, int, int]]  # (x0,y0,x1,y1) in upscaled
    # coords of the detected tile face; None if not detected.
    scale: float  # upscale factor applied
    has_cv2: bool
    method: str  # "color" | "threshold" — which path produced `binary`


def prepare(
    crop: Image.Image,
    params: ClassifyParams = ClassifyParams(),
) -> PreparedTile:
    """Upscale, locate the tile face, and produce grayscale + binary + color masks."""
    rgb = crop.convert("RGB")

    # Upscale small crops so detection/thresholding have signal to work with.
    w, h = rgb.size
    short = min(w, h)
    if short < params.upscale_short_side:
        scale = params.upscale_short_side / short
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        rgb = rgb.resize(new_size, Image.LANCZOS)
        scale_factor = scale
    else:
        scale_factor = 1.0

    # Restrict analysis to the tile face (drop the frame/edge).
    face_bbox = None
    if params.detect_face:
        face_bbox = _detect_face(rgb, params)
    if face_bbox is None:
        x0, y0, x1, y1 = _inset_bbox(rgb.size, params.inset_fraction)
    else:
        x0, y0, x1, y1 = face_bbox
    region = rgb.crop((x0, y0, x1, y1))

    gray_region = np.asarray(region.convert("L"), dtype=np.uint8)

    # Color mask (preferred path). None on grayscale images.
    color_mask = _color_mask(region, params)

    cv2 = _try_cv2()
    if params.color_tile and color_mask is not None:
        binary = color_mask
        method = "color"
    elif cv2 is not None:
        block = max(3, params.binarize_block | 1)
        binary = cv2.adaptiveThreshold(
            gray_region,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            block,
            params.binarize_c,
        )
        binary = _ensure_ink_is_foreground(binary)
        method = "threshold"
    else:
        binary = _ensure_ink_is_foreground(_numpy_threshold(gray_region))
        method = "threshold"

    return PreparedTile(
        gray=gray_region,
        binary=binary,
        color_mask=color_mask,
        face_bbox=face_bbox,
        scale=scale_factor,
        has_cv2=cv2 is not None,
        method=method,
    )


# --------------------------------------------------------------------------- #
# tile-face detection                                                         #
# --------------------------------------------------------------------------- #


def _inset_bbox(size: tuple[int, int], inset_fraction: float) -> tuple[int, int, int, int]:
    w, h = size
    if w < 8 or h < 8 or inset_fraction <= 0:
        return (0, 0, w, h)
    lo, hi = inset_fraction, 1.0 - inset_fraction
    return (max(1, int(w * lo)), max(1, int(h * lo)), min(w, int(w * hi)), min(h, int(h * hi)))


def _detect_face(rgb: Image.Image, params: ClassifyParams) -> Optional[tuple[int, int, int, int]]:
    """Find the tile face as the largest light rectangular region.

    Uses cv2 contours if available; returns None on failure (caller falls back
    to the fixed inset).
    """
    cv2 = _try_cv2()
    if cv2 is None:
        return None
    arr = np.asarray(rgb.convert("L"), dtype=np.uint8)
    h, w = arr.shape
    if w < 16 or h < 16:
        return None
    # The tile face is light; threshold to find candidate bright regions.
    _, bright = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Close gaps so the face becomes one solid blob.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Pick the largest contour whose area is a substantial fraction of the crop
    # (the face should cover most of the tile) but not the whole crop.
    img_area = float(w * h)
    best = None
    best_score = 0.0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = float(cw * ch)
        ratio = area / img_area
        # Prefer large, roughly tile-shaped regions; avoid the full-image box.
        if ratio < 0.25 or ratio > 0.98:
            continue
        # Score by area; tie-break toward squarish shapes.
        score = ratio * min(cw, ch) / max(cw, ch)
        if score > best_score:
            best_score = score
            best = (x, y, x + cw, y + ch)
    return best


# --------------------------------------------------------------------------- #
# color mask                                                                  #
# --------------------------------------------------------------------------- #


def _color_mask(region: Image.Image, params: ClassifyParams) -> Optional[np.ndarray]:
    """Mask of colored (saturated) pixels in the region; None if grayscale.

    Distinguish "genuinely color photo" from "grayscale saved as RGB" by the
    *fraction* of pixels with notable saturation, not the mean — a tile with a
    few small colored dots on a large white face has a low mean saturation but
    is still a color photo.
    """
    arr = np.asarray(region, dtype=np.int16)  # HxWx3
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    sat = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    mask_bool = sat >= params.color_saturation_min
    # Grayscale-saved images have essentially no saturated pixels.
    if float(mask_bool.mean()) < 0.01:
        return None
    return mask_bool.astype(np.uint8) * 255


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _try_cv2():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def _numpy_threshold(arr: np.ndarray) -> np.ndarray:
    """Simple global threshold fallback (median split) when cv2 is absent."""
    thr = int(np.median(arr))
    return np.where(arr < thr, 0, 255).astype(np.uint8)


def _ensure_ink_is_foreground(binary: np.ndarray) -> np.ndarray:
    """Make the minority shade the foreground (255)."""
    fg_ratio = float(np.mean(binary > 0))
    if fg_ratio > 0.5:
        binary = 255 - binary
    return binary


def component_masks(binary: np.ndarray, params: ClassifyParams):
    """Return labeled components + stats, via cv2 if available.

    Returns (num_labels, labels, stats) where stats columns are
    [x, y, w, h, area]. If cv2 is unavailable, returns None.
    """
    cv2 = _try_cv2()
    if cv2 is None:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    return num_labels, labels, stats
