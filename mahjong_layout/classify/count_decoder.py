"""Count-based decoder for pin (dots) and tiao (bamboo) tiles.

The value of a pin/tiao tile equals the number of like-shaped components on its
face (N circles -> N of dots, N sticks -> N of bamboo). We extract connected
components from the binarized crop, filter by size and shape, and count them.

Special case: the 1 of bamboo is a bird (one large blob), not a single stick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from PIL import Image

from .constants import MAX_VALUE, MIN_VALUE, make_label
from .preprocess import PreparedTile, component_masks, prepare
from .types import ClassifyParams, Suit, TileClassification


Prefer = Literal["circle", "stick"]


@dataclass
class ComponentInfo:
    """Geometry of one connected component."""

    area: int
    w: int
    h: int
    circularity: float  # area / (w*h); ~1 for filled squares/circles
    aspect: float  # h / w; >1 = vertical


def _components_from_prepared(prep: PreparedTile, params: ClassifyParams) -> list[ComponentInfo]:
    """Extract size-filtered component geometries from a prepared tile."""
    res = component_masks(prep.binary, params)
    if res is None:
        comps = _scipy_components(prep.binary)
    else:
        num_labels, _labels, stats = res
        # stats[0] is the background label.
        comps = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            comps.append((area, int(w), int(h)))
    return _to_component_infos(comps, params)


def _scipy_components(binary: np.ndarray) -> list[tuple[int, int, int]]:
    """Fallback connected components using scipy.ndimage (no cv2)."""
    try:
        from scipy import ndimage  # type: ignore
    except Exception:
        return []
    # binary: 255 = ink foreground
    fg = binary > 0
    labeled, num = ndimage.label(fg)
    if num == 0:
        return []
    slices = ndimage.find_objects(labeled)
    out = []
    for s in slices:
        if s is None:
            continue
        sub = labeled[s]
        area = int((sub != 0).sum())
        h = s[0].stop - s[0].start
        w = s[1].stop - s[1].start
        out.append((area, w, h))
    return out


def _to_component_infos(
    comps: list[tuple[int, int, int]], params: ClassifyParams
) -> list[ComponentInfo]:
    infos: list[ComponentInfo] = []
    for area, w, h in comps:
        if area < params.min_component_area:
            continue
        if w <= 0 or h <= 0:
            continue
        circularity = area / float(w * h)
        aspect = h / float(w)
        infos.append(ComponentInfo(area=area, w=w, h=h, circularity=circularity, aspect=aspect))
    return infos


def _shape_filter(comps: list[ComponentInfo], prefer: Prefer, params: ClassifyParams) -> list[ComponentInfo]:
    """Keep only components matching the requested shape.

    circles: roughly filled blobs, not too tall (aspect near 1).
    sticks: tall vertical blobs (aspect >= stick_aspect_min).
    """
    if prefer == "circle":
        return [
            c
            for c in comps
            if c.circularity >= params.circle_circularity_min and c.aspect <= params.stick_aspect_min
        ]
    return [c for c in comps if c.aspect >= params.stick_aspect_min]


def _modal_area_filter(comps: list[ComponentInfo], tolerance: float) -> list[ComponentInfo]:
    """Keep components whose area is within +/- tolerance of the modal area.

    A tile's markings (dots/sticks) are drawn at a consistent size, so the true
    count is the size of the largest cluster of same-sized components. This
    discards both tiny speckle and large frame/edge fragments.
    """
    if len(comps) < 2:
        return comps
    areas = sorted(c.area for c in comps)
    # Build a histogram by binning areas into bins of width = tolerance * median.
    import numpy as np

    med = float(np.median(areas))
    if med <= 0:
        return comps
    bin_w = max(1.0, med * tolerance)
    counts: dict[int, int] = {}
    for a in areas:
        b = int(a // bin_w)
        counts[b] = counts.get(b, 0) + 1
    modal_bin = max(counts, key=counts.get)
    lo, hi = modal_bin * bin_w * (1.0 - tolerance), (modal_bin + 1) * bin_w * (1.0 + tolerance)
    return [c for c in comps if lo <= c.area <= hi]


def count_components(
    crop: Image.Image,
    prefer: Prefer,
    params: ClassifyParams = ClassifyParams(),
) -> tuple[Optional[int], dict]:
    """Count shape-matched components on a crop; return (value, raw).

    Handles the tiao1 bird: a single large blob covering a big fraction of the
    tile is interpreted as value 1 for the stick suit.
    """
    prep = prepare(crop, params)
    comps = _components_from_prepared(prep, params)

    tile_area = float(prep.binary.shape[0] * prep.binary.shape[1])

    # tiao1 bird special-case (single large blob). We judge size by the
    # bounding-box coverage, not raw pixel area: a bird drawn as an outline
    # (or thinned by morphology) keeps a large bbox but a small fill, which
    # would defeat an area-based threshold.
    if prefer == "stick" and 1 <= len(comps) <= 2:
        big = max(comps, key=lambda c: c.area)
        bbox_ratio = (big.w * big.h) / tile_area
        if bbox_ratio >= params.bird_area_ratio_min and big.aspect >= 1.0:
            return 1, {
                "n_components": len(comps),
                "bird": True,
                "area_ratio": round(big.area / tile_area, 3),
                "bbox_ratio": round(bbox_ratio, 3),
            }

    # Modal-area filtering: real tiles draw same-sized markings, so cluster by
    # area first to reject both speckle and large frame fragments.
    analysis = comps
    if params.use_modal_area_filter:
        analysis = _modal_area_filter(comps, params.modal_area_tolerance)

    matched = _shape_filter(analysis, prefer, params)
    n = len(matched)
    raw = {
        "n_components_total": len(comps),
        "n_components_matched": n,
        "components": [
            {"area": c.area, "w": c.w, "h": c.h, "circularity": round(c.circularity, 3), "aspect": round(c.aspect, 3)}
            for c in matched[:12]
        ],
    }

    if MIN_VALUE <= n <= MAX_VALUE:
        return n, raw
    return None, raw


def decode_count(
    crop: Image.Image,
    suit: Suit,
    suit_confidence: float,
    prefer: Prefer,
    params: ClassifyParams = ClassifyParams(),
) -> TileClassification:
    """High-level: count components and build a TileClassification."""
    value, raw = count_components(crop, prefer, params)
    if value is None:
        return TileClassification(
            suit=suit,
            value=None,
            label="unknown",
            confidence=0.0,
            suit_confidence=suit_confidence,
            method="count_components",
            raw=raw,
        )
    return TileClassification(
        suit=suit,
        value=value,
        label=make_label(suit.value, value),
        confidence=min(1.0, suit_confidence * 0.7 + 0.3),
        suit_confidence=suit_confidence,
        method="count_components",
        raw=raw,
    )
