"""Role heuristics: tag each cluster as hand / discard / wall / other.

The headline simplification (from the product brief): the lower part of the
frame is the hand. So the hand is selected primarily by vertical position,
refined by orientation/regularity. Everything else is classified by geometry.
No tile-class (dot/tiao/wan) information is used here.
"""

from __future__ import annotations

from typing import Optional

from .types import Cluster, LayoutParams

ROLES = ("hand", "discard", "wall", "other")


def _hand_score(cluster: Cluster, params: LayoutParams) -> Optional[float]:
    """Return a confidence score for this cluster being the hand, or None.

    A cluster is a hand candidate if:
      * its centroid is in the lower zone (cy > hand_y_min),
      * it is mostly horizontal (hands are laid flat),
      * it occupies few rows (hand_max_rows).

    Confidence blends 'how low' with 'how regular'.
    """
    _, cy = cluster.centroid
    if cy < params.hand_y_min:
        return None
    if cluster.dominant_orientation != "h":
        return None
    if cluster.n_rows > params.hand_max_rows:
        return None
    if cluster.n_tiles > params.hand_max_tiles:
        # An overly large lower cluster is probably the wall base, not a hand.
        return None

    # 'low' component: how far below the threshold, normalized to [0,1] over
    # the [hand_y_min, 1] band.
    low = (cy - params.hand_y_min) / max(1.0 - params.hand_y_min, 1e-6)
    low = max(0.0, min(1.0, low))
    score = params.hand_y_weight * low + params.hand_reg_weight * cluster.regularity
    return score


def assign_roles(
    clusters: list[Cluster],
    image_size: Optional[tuple[int, int]] = None,
    params: LayoutParams = LayoutParams(),
) -> list[Cluster]:
    """Tag each cluster's ``role``/``confidence`` in place and return them.

    Order of decisions:
      1. Pick the single best hand candidate (highest score) among lower-zone,
         horizontal, shallow clusters.
      2. Remaining vertical tall clusters -> wall.
      3. Remaining lower-zone clusters -> discard.
      4. Everything else -> other.
    """
    # Reset roles for idempotency.
    for c in clusters:
        c.role = "other"
        c.confidence = 0.0

    # --- 1. Hand ----------------------------------------------------------
    hand: Optional[Cluster] = None
    hand_score: Optional[float] = None
    for c in clusters:
        s = _hand_score(c, params)
        if s is None:
            continue
        if hand_score is None or s > hand_score:
            hand, hand_score = c, s
    if hand is not None:
        hand.role = "hand"
        hand.confidence = hand_score or 0.0
        hand.label = "hand"

    # --- 2. Wall ----------------------------------------------------------
    for c in clusters:
        if c.role != "other":
            continue
        aspect = _median_aspect(c)
        if c.dominant_orientation == "v" and aspect > params.wall_aspect:
            c.role = "wall"
            c.confidence = min(1.0, (aspect - params.wall_aspect) / 2.0 + 0.5)

    # --- 3. Discard (remaining meaningful clusters of >= discard_min_tiles)
    # The discard pile can sit anywhere in the frame (often to the side of the
    # hand or scattered), so we don't constrain it by zone. Singletons stay
    # 'other' as noise unless the caller lowers discard_min_tiles.
    for c in clusters:
        if c.role != "other":
            continue
        if c.n_tiles >= params.discard_min_tiles:
            c.role = "discard"
            c.confidence = 0.5

    # --- labels for discard/wall/other ------------------------------------
    disc_i = wall_i = 0
    for c in clusters:
        if c.role == "discard":
            disc_i += 1
            c.label = "discard" if disc_i == 1 else f"discard_{disc_i}"
        elif c.role == "wall":
            wall_i += 1
            c.label = "wall" if wall_i == 1 else f"wall_{wall_i}"
        elif c.role == "other":
            c.label = "other"

    return clusters


def _median_aspect(cluster: Cluster) -> float:
    """Median h/w of the tiles in the cluster (1.0 = square, >1 = standing)."""
    if not cluster.tiles:
        return 0.0
    vals = []
    for t in cluster.tiles:
        vals.append(t.aspect if t.w > 0 else 0.0)
    vals.sort()
    return vals[len(vals) // 2]
