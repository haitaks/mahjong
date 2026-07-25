"""OCR engine wrapper around RapidOCR.

The heavy dependency (rapidocr_onnxruntime + onnxruntime) is imported lazily
and cached at module level so the model loads once per process. Every call is
wrapped so that a missing/broken engine degrades to an empty result instead of
raising — this keeps the rest of the pipeline (and the unit tests) working
without the OCR stack installed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .types import ClassifyParams

# Module-level cache for the engine instance (None = not yet tried).
_ENGINE = None
_ENGINE_TRIED = False


def _get_engine():
    """Return a cached RapidOCR instance, or None if unavailable."""
    global _ENGINE, _ENGINE_TRIED
    if _ENGINE_TRIED:
        return _ENGINE
    _ENGINE_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        return None
    try:
        _ENGINE = RapidOCR()
    except Exception:
        _ENGINE = None
    return _ENGINE


def reset_engine_cache() -> None:
    """Clear the cached engine (useful for tests / reconfiguration)."""
    global _ENGINE, _ENGINE_TRIED
    _ENGINE = None
    _ENGINE_TRIED = False


def is_available() -> bool:
    """True if the OCR engine can be instantiated in this environment."""
    return _get_engine() is not None


def ocr(
    image: Image.Image | np.ndarray,
    params: ClassifyParams = ClassifyParams(),
) -> list[tuple[str, float]]:
    """Run OCR on an image; return list of (text, score).

    Returns [] if the engine is unavailable or fails. Never raises.
    """
    engine = _get_engine()
    if engine is None:
        return []
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    try:
        result, _elapsed = engine(arr)
    except Exception:
        return []
    out: list[tuple[str, float]] = []
    if not result:
        return out
    for item in result:
        # RapidOCR item format: [box, text, score]
        try:
            text = item[1]
            score = float(item[2]) if len(item) > 2 else 0.5
        except Exception:
            continue
        if text:
            out.append((str(text).strip(), score))
    return out


def find_text(
    image: Image.Image | np.ndarray,
    params: ClassifyParams = ClassifyParams(),
) -> list[str]:
    """Convenience: return just the recognized text strings."""
    return [t for t, _ in ocr(image, params)]


def ocr_multi(
    image: Image.Image,
    variants: tuple[str, ...] = ("raw", "x2"),
    params: ClassifyParams = ClassifyParams(),
) -> list[tuple[str, float]]:
    """Run OCR over several image variants and merge results.

    `variants` may include "raw" (as-is), "x2"/"x3"/... (upscaled), "gray"
    (grayscale), "gray_x2" (grayscale + x2). Results from all variants are
    concatenated; callers decide how to pick the best (e.g. first that matches
    a known character set).

    Rationale: upscaling helps small crops but can HURT large photos — e.g. a
    north-wind tile reads 北 raw and 'K' at x2. Trying multiple variants and
    letting the caller pick a known-good match is more robust than a single
    fixed preprocessing.
    """
    w, h = image.size
    results: list[tuple[str, float]] = []
    seen: set[str] = set()
    for variant in variants:
        prepared = _make_variant(image, variant)
        for text, score in ocr(prepared, params):
            # Merge: keep the max score per text string.
            key = text.strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            results.append((key, score))
    return results


def _make_variant(image: Image.Image, variant: str) -> Image.Image:
    """Build one image variant from its name."""
    gray = variant.startswith("gray")
    base = image.convert("L") if gray else image.convert("RGB")
    # parse optional scale suffix
    scale = 1
    rest = variant.split("_", 1)[1] if "_" in variant else variant
    if rest == "raw":
        scale = 1
    elif rest.startswith("x"):
        try:
            scale = int(rest[1:])
        except ValueError:
            scale = 1
    if scale != 1:
        w, h = base.size
        base = base.resize((max(1, w * scale), max(1, h * scale)), Image.LANCZOS)
    return base
