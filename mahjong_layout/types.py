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
class Cluster:
    """A spatial group of tiles produced by DBSCAN-style clustering."""

    tiles: list[TileBox]
    centroid: tuple[float, float]
    bbox: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (norm)
    n_tiles: int
    median_tile_size: float
    dominant_orientation: str  # 'h' | 'v'
    n_rows: int
    n_cols: int
    regularity: float  # 0..1, 1 = perfectly regular row
    role: str = "other"  # hand | discard | wall | other
    confidence: float = 0.0
    label: str = ""  # optional human-readable id, e.g. "hand", "discard_2"


@dataclass
class LayoutParams:
    """Tunable thresholds for clustering and role assignment.

    All spatial thresholds are in normalized image units and are scaled by the
    median tile size, so the module adapts to both close-up and wide shots.
    """

    # --- clustering ---
    eps_k: float = 2.5
    """eps = eps_k * median_tile_size. Tiles whose centers are within `eps`
    are neighbors. 2.5 covers tightly packed tiles plus a small gap."""

    min_samples: int = 2
    """Minimum neighbors (including the point itself) for a core tile. Tiles
    that end up unclustered become singleton clusters tagged `other`."""

    # --- role heuristics ---
    hand_y_min: float = 0.60
    """Lower part of the frame is the hand. A cluster's centroid must be below
    this y to be considered a hand candidate."""

    hand_max_tiles: int = 18
    """Typical hand is 13-14 tiles; allow slack for exposed melds."""

    discard_min_tiles: int = 2
    """Clusters with at least this many tiles (after hand/wall assignment) are
    treated as discards; smaller leftovers stay 'other' as noise."""

    hand_max_rows: int = 3
    """Hands lie in a few rows; tall stacks are not hands."""

    wall_aspect: float = 1.5
    """A cluster whose median h/w exceeds this is treated as standing wall
    tiles."""

    hand_y_weight: float = 0.6
    hand_reg_weight: float = 0.4
    """Confidence weighting between 'how low is the cluster' and 'how regular
    is the row' for picking the hand cluster."""


@dataclass
class LayoutResult:
    """Output of :func:`mahjong_layout.cluster_layout`."""

    clusters: list[Cluster]
    hand: Optional[Cluster]
    discards: list[Cluster] = field(default_factory=list)
    walls: list[Cluster] = field(default_factory=list)
    others: list[Cluster] = field(default_factory=list)
    image_size: Optional[tuple[int, int]] = None  # (width, height) in px

    def summary(self) -> str:
        """One-line human summary, e.g. 'hand=14 discard=2c(6) wall=0 other=1'."""
        n_discard_tiles = sum(c.n_tiles for c in self.discards)
        n_wall_tiles = sum(c.n_tiles for c in self.walls)
        n_other_tiles = sum(c.n_tiles for c in self.others)
        hand_n = self.hand.n_tiles if self.hand else 0
        return (
            f"hand={hand_n} "
            f"discard={len(self.discards)}c({n_discard_tiles}) "
            f"wall={len(self.walls)}c({n_wall_tiles}) "
            f"other={n_other_tiles}"
        )
