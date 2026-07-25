"""Tile classification subpackage.

Public API
----------
- :func:`classify_tile` — classify a PIL crop into (suit, value).
- :class:`TileClassification`, :class:`ClassifyParams`, :class:`Suit`.
- :func:`determine_suit` — the suit router (useful on its own / for tests).
- :func:`count_components` — the pin/tiao component counter.
- :func:`crop_tile` (re-exported from ..crop) — crop a TileBox from an image.
"""

from .types import ClassifyParams, Suit, TileClassification

__all__ = [
    "Suit",
    "TileClassification",
    "ClassifyParams",
    "classify_tile",
    "determine_suit",
    "count_components",
    "crop_tile",
]


def __getattr__(name: str):
    if name == "classify_tile":
        from .classifier import classify_tile

        return classify_tile
    if name == "determine_suit":
        from .router import determine_suit

        return determine_suit
    if name == "count_components":
        from .count_decoder import count_components

        return count_components
    if name == "crop_tile":
        from ..crop import crop_tile

        return crop_tile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
