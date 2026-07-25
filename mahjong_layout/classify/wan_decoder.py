"""Wan (characters / 万) decoder: OCR the numeral and map it to 1..9.

A wan tile shows a Chinese numeral (一/二/.../九) on top of the 萬 character.
We OCR the top portion of the tile and translate the recognized numeral via
:class:`CN_NUMERAL_TO_INT`.
"""

from __future__ import annotations

from typing import Optional

from PIL import Image

from .constants import CN_NUMERAL_TO_INT, make_label
from .ocr_engine import ocr
from .types import ClassifyParams, Suit, TileClassification


def _top_crop(crop: Image.Image, params: ClassifyParams) -> Image.Image:
    """Crop the upper fraction of the tile where the numeral lives."""
    w, h = crop.size
    top_h = max(1, int(round(h * params.wan_top_fraction)))
    return crop.crop((0, 0, w, top_h))


def _first_known_numeral(texts: list[tuple[str, float]]) -> Optional[tuple[int, float]]:
    """Scan OCR tokens (char-by-char) for the first mapped numeral."""
    best: Optional[tuple[int, float]] = None
    for text, score in texts:
        for ch in text:
            if ch in CN_NUMERAL_TO_INT:
                val = CN_NUMERAL_TO_INT[ch]
                if val == 0:
                    continue
                if best is None or score > best[1]:
                    best = (val, score)
    return best


def decode_wan(
    crop: Image.Image,
    suit_confidence: float,
    params: ClassifyParams = ClassifyParams(),
) -> TileClassification:
    """OCR the wan numeral and return a TileClassification."""
    # Restrict to the tile face first (drop the frame), then take the top
    # portion where the numeral lives. This removes the 萬 character from the
    # OCR frame, which otherwise dominates recognition.
    face = _face_region(crop, params)
    region = _top_crop(face, params)
    w, h = region.size
    if min(w, h) < params.upscale_short_side:
        scale = params.upscale_short_side / max(1, min(w, h))
        region = region.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
        )

    texts = ocr(region, params)
    found = _first_known_numeral(texts)

    if found is None:
        raw = {"ocr_texts": [t for t, _ in texts]}
        return TileClassification(
            suit=Suit.WAN,
            value=None,
            label="unknown",
            confidence=0.0,
            suit_confidence=suit_confidence,
            method="ocr",
            raw=raw,
        )

    value, score = found
    return TileClassification(
        suit=Suit.WAN,
        value=value,
        label=make_label(Suit.WAN.value, value),
        confidence=min(1.0, suit_confidence * 0.5 + score * 0.5),
        suit_confidence=suit_confidence,
        method="ocr",
        raw={"ocr_texts": [t for t, _ in texts], "score": round(score, 3)},
    )


def _face_region(crop: Image.Image, params: ClassifyParams) -> Image.Image:
    """Crop to the detected tile face, or return the crop unchanged."""
    if not params.detect_face:
        return crop
    try:
        from .preprocess import _detect_face
    except Exception:
        return crop
    face = _detect_face(crop, params)
    if face is None:
        return crop
    x0, y0, x1, y1 = face
    return crop.crop((x0, y0, x1, y1))
