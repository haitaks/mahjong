"""High-level tile classifier: crop -> TileClassification.

Routes by suit, then delegates to the matching decoder. Any failure degrades
to an UNKNOWN result rather than raising, so batch jobs over many tiles are
robust to the occasional bad crop.
"""

from __future__ import annotations

from PIL import Image

from .count_decoder import decode_count
from .honor import decode_honor
from .router import determine_suit
from .types import ClassifyParams, Suit, TileClassification
from .wan_decoder import decode_wan


def classify_tile(
    crop: Image.Image,
    params: ClassifyParams = ClassifyParams(),
) -> TileClassification:
    """Classify a single tile crop into (suit, value)."""
    if crop is None:
        return _unknown("null crop", 0.0)

    try:
        guess = determine_suit(crop, params)
    except Exception as exc:  # pragma: no cover - defensive
        return _unknown(f"router error: {exc!r}", 0.0)

    suit, suit_conf = guess.suit, guess.confidence

    try:
        if suit == Suit.WAN:
            return decode_wan(crop, suit_conf, params)
        if suit == Suit.PIN:
            return decode_count(crop, suit, suit_conf, "circle", params)
        if suit == Suit.TIAO:
            return decode_count(crop, suit, suit_conf, "stick", params)
        if suit == Suit.HONOR:
            if params.honor_as_unknown:
                return decode_honor(crop, suit_conf, params)
            return decode_honor(crop, suit_conf, params)
    except Exception as exc:  # pragma: no cover - defensive
        return _unknown(f"decoder error: {exc!r}", suit_conf, suit=suit)

    # UNKNOWN suit (or disabled).
    return TileClassification(
        suit=Suit.UNKNOWN,
        value=None,
        label="unknown",
        confidence=suit_conf * 0.5,
        suit_confidence=suit_conf,
        method="none",
        raw={"reason": "unknown suit"},
    )


def _unknown(reason: str, suit_conf: float, suit: Suit = Suit.UNKNOWN) -> TileClassification:
    return TileClassification(
        suit=suit,
        value=None,
        label="unknown",
        confidence=0.0,
        suit_confidence=suit_conf,
        method="none",
        raw={"reason": reason},
    )
