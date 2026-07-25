"""High-level pipeline: detections -> LayoutResult.

This is the main entry point most callers use. It glues clustering and role
heuristics together and partitions clusters into the LayoutResult buckets.
"""

from __future__ import annotations

from typing import Optional

from .clustering import dbscan_cluster
from .heuristics import assign_roles
from .types import LayoutParams, LayoutResult, TileBox


def cluster_layout(
    boxes,
    image_size: Optional[tuple[int, int]] = None,
    params: Optional[LayoutParams] = None,
) -> LayoutResult:
    """Cluster detections and assign roles.

    Parameters
    ----------
    boxes:
        Anything :func:`mahjong_layout.io_readers.from_raw` accepts: a list of
        TileBox, dicts, or 4/5-tuples.
    image_size:
        Optional (width, height) in pixels. Only stored for the viz layer.
    params:
        Tunable thresholds. Defaults to :class:`LayoutParams`.
    """
    if params is None:
        params = LayoutParams()

    tiles = _coerce_tiles(boxes)
    clusters = dbscan_cluster(tiles, params)
    assign_roles(clusters, image_size=image_size, params=params)

    hand = next((c for c in clusters if c.role == "hand"), None)
    discards = [c for c in clusters if c.role == "discard"]
    walls = [c for c in clusters if c.role == "wall"]
    others = [c for c in clusters if c.role == "other"]

    return LayoutResult(
        clusters=clusters,
        hand=hand,
        discards=discards,
        walls=walls,
        others=others,
        image_size=image_size,
    )


def _coerce_tiles(boxes) -> list[TileBox]:
    """Accept a list[TileBox] directly, or delegate to from_raw for mixed input."""
    if not boxes:
        return []
    if all(isinstance(b, TileBox) for b in boxes):
        return list(boxes)
    from .io_readers import from_raw

    return from_raw(boxes)
