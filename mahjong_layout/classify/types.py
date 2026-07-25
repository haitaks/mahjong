"""Public types for the tile classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Suit(str, Enum):
    """Mahjong tile suits / groups."""

    WAN = "wan"        # characters / 万
    PIN = "pin"        # dots / circles
    TIAO = "tiao"      # bamboo / sticks
    HONOR = "honor"    # winds + dragons (not implemented this iteration)
    UNKNOWN = "unknown"


@dataclass
class TileClassification:
    """Result of classifying a single tile crop."""

    suit: Suit
    value: Optional[int]  # 1..9 for wan/pin/tiao; None for honor/unknown
    label: str  # e.g. "wan5", "pin9", "east", "unknown"
    confidence: float  # overall 0..1
    suit_confidence: float  # how sure the router was about the suit
    method: str  # "ocr" | "count_components" | "stub" | "none"
    raw: Optional[dict] = None  # raw decoder output (ocr text, component count, ...)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.label} (conf={self.confidence:.2f}, suit={self.suit.value}, method={self.method})"


@dataclass
class ClassifyParams:
    """Tunable knobs for classification.

    Kept separate from LayoutParams so the layout module stays dependency-light.
    """

    # --- preprocessing ---
    upscale_short_side: int = 128
    """Crops are tiny (32-76px in the dataset); upscale via Lanczos so OCR and
    connected-components have something to work with."""

    binarize_block: int = 15
    """adaptiveThreshold block size (must be odd >= 3)."""

    binarize_c: int = 5
    """adaptiveThreshold constant subtracted from the mean."""

    # --- connected components (pin/tiao) ---
    min_component_area: int = 8
    """Drop connected components smaller than this (noise)."""

    # component shape filters (relative to the median component size of the tile)
    circle_circularity_min: float = 0.55  # area / (bbox area); 1.0 = perfect square
    stick_aspect_min: float = 2.0  # h/w above this counts as a vertical stick

    # --- real-tile robustness ---
    inset_fraction: float = 0.18
    """Fallback fraction of each side to crop away before analysis. Real tile
    crops include the tile's own frame/edge, which dominates connected-components;
    insetting removes it so only the face markings remain."""

    detect_face: bool = True
    """Auto-detect the tile face (largest light rectangular region) instead of
    the fixed inset. Falls back to inset_fraction if detection fails."""

    use_modal_area_filter: bool = True
    """Group components by area and keep only those near the modal (most common)
    area. Tiles draw repeated, same-sized markings (dots/sticks), so the true
    count is the size of the largest area-cluster, not the raw component count."""

    modal_area_tolerance: float = 0.5
    """Components within +/- this fraction of the modal area are kept."""

    # --- color segmentation (preferred path on quality color photos) ---
    color_tile: bool = True
    """When the crop has color (not grayscale), segment markings by color
    saturation rather than luminance thresholding. Dots/bamboo are colored, the
    tile face is white, so color is the most reliable signal on good photos."""

    color_saturation_min: float = 30.0
    """Minimum saturation (max channel difference, 0-255) for a pixel to count
    as 'colored'. Crops whose mean saturation is below this are treated as
    grayscale and the color path is skipped."""

    # tiao1 (bird) heuristic: a single large component covering this fraction
    # of the tile area is treated as the bird = 1 of bamboo. Kept modest
    # because the bird, while the largest blob, typically fills only ~20% of
    # the tile; normal sticks instead produce many small components.
    bird_area_ratio_min: float = 0.18

    # --- OCR (wan) ---
    ocr_dpi: int = 150
    wan_top_fraction: float = 0.45
    """The numeral sits in the top ~45% of a wan tile; crop that for OCR."""

    # --- behavior ---
    enabled_suits: set = field(
        default_factory=lambda: {Suit.WAN, Suit.PIN, Suit.TIAO, Suit.HONOR}
    )
    """Suits the router is allowed to emit. Honor is now on by default since
    the honor decoder is implemented for quality photos."""

    honor_as_unknown: bool = False
    """If False, honor tiles are decoded (winds/dragons via OCR). If True,
    honor tiles return UNKNOWN (useful for pipelines that only care about
    suited tiles)."""

    # --- honor decoder ---
    honor_fg_ratio_max: float = 0.20
    """A tile face whose foreground covers less than this fraction is treated
    as the (blank) white dragon."""

    honor_ocr_variants: tuple = ("raw", "x2")
    """Image variants to try for honor OCR, in order. Upscaling large photos
    can HURT recognition (north_wind reads 北 raw, K at x2), so we try raw
    first and take the first variant that yields a known honor character."""

    honor_color_fallback: bool = True
    """When OCR can't read the honor character, fall back to the dominant ink
    color (red -> red_dragon, green -> green_dragon)."""

    honor_dominant_color_frac: float = 0.25
    """Minimum fraction of foreground pixels that must be a single color for
    the color fallback to fire."""
