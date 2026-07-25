"""Suit router: decide which suit a tile belongs to from its appearance.

We rely on cheap shape statistics of connected components (no OCR here, except
an optional cheap check for the wan marker). The decision is intentionally
robust and degrades to UNKNOWN rather than guessing.

Heuristics:
  * many round-ish components  -> PIN (dots)
  * many tall vertical components -> TIAO (bamboo sticks)
  * a single large blob (the bird) -> TIAO (1 of bamboo)
  * presence of the 萬/万 marker (via a light OCR pass if available) -> WAN
  * thick horizontal-ish strokes concentrated in the lower half -> WAN fallback
  * otherwise -> HONOR (if enabled) / UNKNOWN
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image

from .count_decoder import _components_from_prepared
from .preprocess import prepare
from .types import ClassifyParams, Suit


@dataclass
class SuitGuess:
    suit: Suit
    confidence: float


def determine_suit(
    crop: Image.Image,
    params: ClassifyParams = ClassifyParams(),
) -> SuitGuess:
    """Route a tile crop to a Suit with a confidence score.

    Order of checks:
      1. WAN via the 萬/万 OCR marker — the only reliable wan signal on real
         (small, noisy) crops, since wan strokes are geometrically ambiguous
         with tiao sticks. Only runs when OCR is available.
      2. HONOR via an honor character (東南西北中發) detected by OCR, or a
         near-blank face (white dragon). Honor chars are single characters and
         are best recognized, like wan, via OCR rather than geometry.
      3. TIAO 1 bird (single large blob).
      4. PIN via circles, TIAO via sticks.
      5. WAN textual fallback, then UNKNOWN.
    """
    # --- 1. WAN via OCR marker (early, authoritative when present) ---------
    if params.enabled_suits and Suit.WAN in params.enabled_suits:
        wan = _check_wan_marker(crop, params)
        if wan is not None:
            return SuitGuess(Suit.WAN, wan)

    # --- 2. HONOR via OCR character or blank face --------------------------
    if params.enabled_suits and Suit.HONOR in params.enabled_suits:
        honor = _check_honor(crop, params)
        if honor is not None:
            return honor

    prep = prepare(crop, params)
    comps = _components_from_prepared(prep, params)
    tile_area = float(prep.binary.shape[0] * prep.binary.shape[1])

    # --- shape-based signals -------------------------------------------------
    circles = [
        c for c in comps
        if c.circularity >= params.circle_circularity_min and c.aspect <= params.stick_aspect_min
    ]
    sticks = [c for c in comps if c.aspect >= params.stick_aspect_min]
    # Use bbox coverage (not pixel area) for the bird check — outline drawings
    # and morphologically thinned strokes keep a large bbox with little fill.
    big = max(comps, key=lambda c: c.area) if comps else None
    big_bbox_ratio = (big.w * big.h) / tile_area if (big and tile_area) else 0.0

    # --- 2. bird: single dominant blob -> TIAO 1 ---------------------------
    if 1 <= len(comps) <= 2 and big is not None:
        if big_bbox_ratio >= params.bird_area_ratio_min and big.aspect >= 1.0:
            return SuitGuess(Suit.TIAO, 0.7)

    # --- 3. strong circle signal -> PIN ------------------------------------
    if len(circles) >= 1 and len(circles) >= len(sticks):
        conf = _scaled_confidence(len(circles), expected_range=(1, 9))
        return SuitGuess(Suit.PIN, conf)

    # --- 4. strong stick signal -> TIAO ------------------------------------
    if len(sticks) >= 2 and len(sticks) > len(circles):
        conf = _scaled_confidence(len(sticks), expected_range=(2, 9))
        return SuitGuess(Suit.TIAO, conf)

    # --- 5. WAN fallback: thick strokes in the lower half ------------------
    # wan tiles have dense dark ink; if the crop is text-like (many medium
    # components, none clearly circle/stick) and lower-half is denser, guess WAN.
    if _looks_textual(comps, prep):
        return SuitGuess(Suit.WAN, 0.4)

    # --- 6. give up --------------------------------------------------------
    # Truly unrecognized tiles (noise, non-tiles) end up UNKNOWN. The white
    # dragon was already handled by _check_honor above.
    return SuitGuess(Suit.UNKNOWN, 0.2)


def _scaled_confidence(n: int, expected_range: tuple[int, int]) -> float:
    """Confidence peaks when n is in the plausible 1..9 range."""
    lo, hi = expected_range
    if n < lo:
        return 0.4
    if n > hi:
        return 0.5
    # smooth ramp inside the range
    return 0.6 + 0.3 * (1.0 - abs(n - (lo + hi) / 2.0) / (hi - lo))


def _looks_textual(comps, prep) -> bool:
    """Heuristic: wan tiles show several medium, non-round, non-stick strokes."""
    if len(comps) < 2:
        return False
    # many components, none of them clearly geometric
    medium = [c for c in comps if 0.3 < c.aspect < 2.5 and c.circularity < 0.85]
    if len(medium) < 2:
        return False
    # lower half denser than upper half (where 萬 sits)? cheap check on ink mass.
    h = prep.binary.shape[0]
    upper = float((prep.binary[: h // 2] > 0).mean())
    lower = float((prep.binary[h // 2 :] > 0).mean())
    return lower >= upper * 0.8


def _check_wan_marker(crop: Image.Image, params: ClassifyParams) -> Optional[float]:
    """Run a light OCR pass; return confidence if the 万/萬 marker is present.

    Returns None if OCR is unavailable or the marker isn't found, so callers
    can fall through to cheaper heuristics. Crops in the dataset are tiny, so
    we upscale before OCR to give the engine a fighting chance.
    """
    try:
        from .ocr_engine import find_text
    except Exception:  # pragma: no cover - import guard
        return None

    probe = crop
    w, h = probe.size
    if min(w, h) < params.upscale_short_side:
        scale = params.upscale_short_side / max(1, min(w, h))
        probe = probe.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
        )

    text = find_text(probe, params)
    if not text:
        return None
    joined = "".join(text)
    from .constants import WAN_MARKERS

    if any(m in joined for m in WAN_MARKERS):
        return 0.85
    return None


def _check_honor(crop: Image.Image, params: ClassifyParams) -> Optional[SuitGuess]:
    """Detect honor tiles via an honor character (東南西北中發) or a white-dragon frame.

    Returns a SuitGuess(HONOR, ...) if detected, else None. Uses the same
    multi-variant OCR as the honor decoder so routing and decoding agree.

    The white dragon is a near-blank tile whose only marking is a thin frame
    along the edges. We detect it by checking that foreground ink concentrates
    near the crop border (not in the center, where a suited tile's markings sit).
    """
    from .constants import HONOR_CHAR_TO_LABEL

    # OCR path: look for any honor character.
    try:
        from .ocr_engine import ocr_multi
    except Exception:  # pragma: no cover - import guard
        return None
    texts = ocr_multi(crop, params.honor_ocr_variants, params)
    for text, score in texts:
        for ch in text:
            if ch in HONOR_CHAR_TO_LABEL:
                return SuitGuess(Suit.HONOR, max(0.7, score))

    # White-dragon path: low overall ink AND ink concentrated near the border.
    if _looks_like_white_dragon(crop, params):
        return SuitGuess(Suit.HONOR, 0.7)

    return None


def _looks_like_white_dragon(crop: Image.Image, params: ClassifyParams) -> bool:
    """A blank white dragon has little ink, and that ink hugs the border."""
    import numpy as np

    prep = prepare(crop, params)
    fg = prep.binary > 0
    fg_ratio = float(fg.mean())
    if fg_ratio >= params.honor_fg_ratio_max * 4:  # too much ink -> not blank
        return False
    if fg_ratio == 0.0:
        return False  # truly empty -> not a tile at all
    h, w = fg.shape
    if h < 8 or w < 8:
        return False
    # Compare border strip vs central region ink density.
    bw = max(2, w // 8)
    bh = max(2, h // 8)
    border = np.concatenate([
        fg[:bh, :].ravel(), fg[-bh:, :].ravel(),
        fg[:, :bw].ravel(), fg[:, -bw:].ravel(),
    ])
    center = fg[bh:-bh, bw:-bw]
    if center.size == 0:
        return False
    border_density = float(border.mean())
    center_density = float(center.mean())
    # White dragon: border noticeably denser than (near-empty) center.
    return border_density > center_density * 2.0 and center_density < 0.05
