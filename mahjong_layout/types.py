"""Core data types for the mahjong layout module.

All spatial coordinates are stored normalized to the [0, 1] image extent,
matching the YOLO label format (cx, cy, w, h). Pixel conversions only happen
in the viz layer, where the actual image size is known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TileBox:
    """A single detected tile.

    Attributes mirror the YOLO label format. Coordinates are normalized.
    """

    cx: float
    cy: float
    w: float
    h: float
    class_id: Optional[int] = None

    @property
    def size(self) -> float:
        """Average extent of the box in normalized units."""
        return (self.w + self.h) / 2.0

    @property
    def is_horizontal(self) -> bool:
        """True if the tile lies flat (wider than tall)."""
        return self.w >= self.h

    @property
    def aspect(self) -> float:
        """Height-to-width ratio; > 1 means the tile stands upright."""
        return self.h / self.w if self.w > 0 else 0.0

    def corners(self) -> tuple[float, float, float, float]:
        """Return (xmin, ymin, xmax, ymax) in normalized coordinates."""
        return (
            self.cx - self.w / 2.0,
            self.cy - self.h / 2.0,
            self.cx + self.w / 2.0,
            self.cy + self.h / 2.0,
        )


@dataclass
class ClassifiedTile:
    """A detected tile with its classification result.

    ``label`` is e.g. "wan5", "pin3", "east", or "unknown" for empty/wall tiles.
    ``is_empty`` is True when the classifier could not identify the tile
    (empty tile face, wall tile face, noise).
    """

    box: TileBox
    label: str = "unknown"
    is_empty: bool = True


@dataclass
class LayoutResult:
    """Output of :func:`mahjong_layout.classify_layout`.

    All tiles are classified first. Then:
    - Empty tiles are candidates for the wall (unless they belong to the hand).
    - Non-empty tiles are clustered by proximity.
    - Clusters in the lower half of the frame → hand.
    - Clusters in the upper half → discard.
    - Empty tiles near no meaningful tiles → wall.
    """

    hand: list[ClassifiedTile]
    discard: list[ClassifiedTile]
    wall: list[ClassifiedTile]
    unknown: list[ClassifiedTile]

    def summary(self) -> str:
        """One-line human summary."""
        return (
            f"hand={len(self.hand)} "
            f"discard={len(self.discard)} "
            f"wall={len(self.wall)} "
            f"unknown={len(self.unknown)}"
        )


@dataclass
class LayoutParams:
    """Tunable thresholds for layout detection.

    All spatial thresholds are in normalized image units.
    """

    # --- classification ---
    classify_params: Optional[dict] = None
    """Optional dict forwarded as kwargs to classify_tile(). Uses defaults if None."""

    # --- hand detection by anchor ---
    hand_eps_k: float = 4.0
    """Tiles within `hand_eps_k * median_tile_size` of the lowest (max cy)
    meaningful tile are considered part of the hand. The rest → discard."""

    hand_max_tiles: int = 14
    """Maximum tiles in a hand (13 + up to 1 winning tile). Increases by 1
    per exposed meld (kan/pon/chii)."""

    # --- clustering ---
    eps_k: float = 1.5
    """eps = eps_k * median_tile_size for the DBSCAN on non-empty tiles."""

    min_samples: int = 2
    """Minimum neighbors (incl. the point itself) for a core tile."""

    # --- wall ---
    wall_neighbor_eps: float = 0.08
    """Max distance (normalized) from an empty tile to the nearest non-empty
    tile for it to be considered 'nearby' (and therefore NOT wall). Empty tiles
    with no non-empty neighbour within this radius become wall."""

    wall_min_tiles: int = 3
    """A contiguous group of empty tiles must have at least this many to be
    considered a wall; smaller empty groups stay unknown."""
