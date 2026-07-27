"""mahjong_layout — classify, cluster and zone mahjong tiles.

Public API
----------
- :func:`classify_layout` — full pipeline: detects → classifies → zones.
- :class:`TileBox`, :class:`ClassifiedTile`, :class:`LayoutParams`, :class:`LayoutResult`
  — the data types you'll handle.
"""

from .types import ClassifiedTile, LayoutParams, LayoutResult, TileBox

__all__ = [
    "TileBox",
    "ClassifiedTile",
    "LayoutParams",
    "LayoutResult",
    "classify_layout",
    "classify_tile",
    "crop_tile",
]


def __getattr__(name: str):
    if name == "classify_layout":
        from .pipeline import classify_layout

        return classify_layout
    if name == "classify_tile":
        from .classify.classifier import classify_tile

        return classify_tile
    if name == "crop_tile":
        from .crop import crop_tile

        return crop_tile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
