"""Honor tiles (winds + dragons) decoder.

Winds (東南西北) and dragons (中發 + blank) are single-character honor tiles.
On quality photos, RapidOCR reads them well (東/南/西/中/發), so the primary
path is OCR over multiple image variants (raw first — upscaling can hurt), with
two fallbacks:

  * white_dragon: a near-blank tile face (just a frame) — detected by very low
    foreground density, not by OCR.
  * color fallback: red_dragon 中 is red, green_dragon 發 is green. When OCR
    can't read the character, the dominant ink color is a weak signal.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .constants import (
    HONOR_CHAR_TO_LABEL,
    HONOR_COLOR_HINTS,
    WHITE_DRAGON_LABEL,
)
from .preprocess import prepare
from .types import ClassifyParams, Suit, TileClassification


def decode_honor(
    crop: Image.Image,
    suit_confidence: float,
    params: ClassifyParams = ClassifyParams(),
) -> TileClassification:
    """Decode an honor tile (wind or dragon) from a crop."""
    prep = prepare(crop, params)
    fg_ratio = float(np.mean(prep.binary > 0))

    # --- 1. white_dragon: nearly blank face WITH a frame --------------------
    # A truly empty crop (no frame, no ink) is NOT a white dragon. Distinguish
    # by checking that the little ink present hugs the border (the tile frame).
    if fg_ratio < params.honor_fg_ratio_max and _has_border_frame(prep.binary):
        return TileClassification(
            suit=Suit.HONOR,
            value=None,
            label=WHITE_DRAGON_LABEL,
            confidence=0.8,
            suit_confidence=suit_confidence,
            method="blank_face",
            raw={"fg_ratio": round(fg_ratio, 4)},
        )

    # --- 2. OCR over multiple variants --------------------------------------
    ocr_texts = _ocr_honor_variants(crop, params)
    label, score = _match_honor_char(ocr_texts)

    if label is not None:
        return TileClassification(
            suit=Suit.HONOR,
            value=None,
            label=label,
            confidence=min(1.0, suit_confidence * 0.4 + score * 0.6),
            suit_confidence=suit_confidence,
            method="ocr",
            raw={"ocr_texts": [t for t, _ in ocr_texts], "score": round(score, 3)},
        )

    # --- 3. color fallback --------------------------------------------------
    if params.honor_color_fallback:
        color_label = _dominant_color_label(crop, params)
        if color_label is not None:
            return TileClassification(
                suit=Suit.HONOR,
                value=None,
                label=color_label,
                confidence=0.5,
                suit_confidence=suit_confidence,
                method="color_fallback",
                raw={"ocr_texts": [t for t, _ in ocr_texts]},
            )

    # --- 4. give up ---------------------------------------------------------
    return TileClassification(
        suit=Suit.HONOR,
        value=None,
        label="unknown_honor",
        confidence=0.0,
        suit_confidence=suit_confidence,
        method="stub",
        raw={"ocr_texts": [t for t, _ in ocr_texts], "fg_ratio": round(fg_ratio, 4)},
    )


def _ocr_honor_variants(crop: Image.Image, params: ClassifyParams):
    """Run OCR over the configured variants; return merged (text, score) list."""
    from .ocr_engine import ocr_multi

    return ocr_multi(crop, params.honor_ocr_variants, params)


def _has_border_frame(binary: np.ndarray) -> bool:
    """True if foreground ink concentrates at the border vs the center.

    A white-dragon tile shows a thin frame; a truly empty crop has no ink at
    all; a suited tile's markings sit in the center. This distinguishes them.
    """
    fg = binary > 0
    h, w = fg.shape
    if h < 8 or w < 8:
        return False
    if not fg.any():
        return False
    bw = max(2, w // 8)
    bh = max(2, h // 8)
    border = np.concatenate([
        fg[:bh, :].ravel(), fg[-bh:, :].ravel(),
        fg[:, :bw].ravel(), fg[:, -bw:].ravel(),
    ])
    center = fg[bh:-bh, bw:-bw]
    if center.size == 0:
        return False
    return float(border.mean()) > float(center.mean()) * 2.0


def _match_honor_char(texts):
    """Return (label, score) of the first recognized honor character, or (None, 0)."""
    best = None
    for text, score in texts:
        for ch in text:
            if ch in HONOR_CHAR_TO_LABEL:
                label = HONOR_CHAR_TO_LABEL[ch]
                if best is None or score > best[1]:
                    best = (label, score)
    return (best if best else (None, 0.0))


def _dominant_color_label(crop: Image.Image, params: ClassifyParams):
    """Return an honor label if a single color dominates the ink; else None."""
    arr = np.asarray(crop.convert("RGB"), dtype=np.int16)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # "Ink" = saturated pixels (the colored character on white face).
    sat = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(r - b))
    ink = sat >= params.color_saturation_min
    if ink.sum() == 0:
        return None
    frac = float(ink.mean())
    if frac < 0.005:
        return None  # too little ink to judge

    ir, ig, ib = r[ink], g[ink], b[ink]
    red_dominant = int(np.sum((ir > 120) & (ir > ig + 40) & (ir > ib + 40)))
    green_dominant = int(np.sum((ig > 100) & (ig > ir + 30) & (ig > ib + 30)))

    if red_dominant / ink.sum() >= params.honor_dominant_color_frac:
        return "red_dragon"
    if green_dominant / ink.sum() >= params.honor_dominant_color_frac:
        return "green_dragon"
    return None
