"""mahjong_layout — spatial clustering of mahjong tiles into play zones.

Public API
----------
- :func:`cluster_layout` — high-level entry point: detections -> LayoutResult.
- :class:`TileBox`, :class:`Cluster`, :class:`LayoutParams`, :class:`LayoutResult`
  — the data types you'll handle.
- :func:`dbscan_cluster` — the low-level clustering routine.
- :func:`assign_roles` — the hand/discard/wall/other heuristic.

See ``README.md`` for usage examples and the CLI.
"""

from .types import Cluster, LayoutParams, LayoutResult, TileBox

__all__ = [
    "TileBox",
    "Cluster",
    "LayoutParams",
    "LayoutResult",
    "cluster_layout",
    "dbscan_cluster",
    "assign_roles",
]


def __getattr__(name: str):
    # Lazy imports to keep import-time cheap and avoid hard deps at import.
    if name == "cluster_layout":
        from .pipeline import cluster_layout

        return cluster_layout
    if name == "dbscan_cluster":
        from .clustering import dbscan_cluster

        return dbscan_cluster
    if name == "assign_roles":
        from .heuristics import assign_roles

        return assign_roles
    if name == "classify_tile":
        from .classify.classifier import classify_tile

        return classify_tile
    if name == "crop_tile":
        from .crop import crop_tile

        return crop_tile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
