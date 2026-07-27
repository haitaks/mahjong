"""New pipeline: classify tiles, find hand by lowest tile, identify wall.

Flow:
  1. Classify every detected tile (crop from the image).
  2. Split: empty tiles (label=="unknown") → wall candidates.
     Non-empty tiles → cluster by proximity.
  3. The lowest (max cy) meaningful tile is the anchor. Its cluster = hand.
     All other clusters → discard.
  4. Empty tiles far from any meaningful (non-empty) tile → wall.
     The rest stay unknown.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .classify import ClassifyParams, classify_tile
from .crop import crop_tile
from .types import ClassifiedTile, LayoutParams, LayoutResult, TileBox


def classify_layout(
    boxes: list[TileBox],
    image: Image.Image,
    params: Optional[LayoutParams] = None,
) -> LayoutResult:
    """Full pipeline: classify all tiles → find hand by lowest tile → wall detection.

    Parameters
    ----------
    boxes:
        Detected tile bounding boxes.
    image:
        The full image (PIL RGB) from which crops are taken for classification.
    params:
        Tunable thresholds. Defaults to :class:`LayoutParams`.
    """
    if params is None:
        params = LayoutParams()

    # --- 1. Classify every tile -------------------------------------------
    clf_params = ClassifyParams(**(params.classify_params or {}))
    classified: list[ClassifiedTile] = []
    for box in boxes:
        crop = crop_tile(image, box)
        result = classify_tile(crop, clf_params)
        is_empty = result.label == "unknown"
        classified.append(ClassifiedTile(box=box, label=result.label, is_empty=is_empty))

    # --- 2. Split empty vs non-empty --------------------------------------
    empty = [ct for ct in classified if ct.is_empty]
    meaningful = [ct for ct in classified if not ct.is_empty]

    if not meaningful:
        return LayoutResult(hand=[], discard=[], wall=empty, unknown=[])

    # --- 3. Find lowest meaningful tile as hand anchor --------------------
    anchor = max(meaningful, key=lambda ct: ct.box.cy)
    med_size = float(np.median([ct.box.size for ct in meaningful]))
    hand_radius = params.hand_eps_k * med_size

    # Sort all meaningful tiles by distance to anchor, take up to hand_max_tiles
    with_dist = [(ct, np.sqrt((ct.box.cx - anchor.box.cx)**2 + (ct.box.cy - anchor.box.cy)**2))
                 for ct in meaningful]
    with_dist.sort(key=lambda x: x[1])

    hand_tiles = [ct for ct, d in with_dist[:params.hand_max_tiles] if d <= hand_radius]
    discard_tiles = [ct for ct, d in with_dist if d > hand_radius]
    # If the radius cap left some within-radius tiles unused (e.g. all 20 are
    # within radius but hand_max_tiles=14), the extra go to discard too.
    extra_within = [ct for ct, d in with_dist[params.hand_max_tiles:] if d <= hand_radius]
    discard_tiles.extend(extra_within)

    # --- 4. Wall detection ------------------------------------------------
    if empty:
        wall_tiles, remaining_empty = _find_wall(empty, meaningful, params)
    else:
        wall_tiles, remaining_empty = [], []

    return LayoutResult(
        hand=hand_tiles,
        discard=discard_tiles,
        wall=wall_tiles,
        unknown=remaining_empty,
    )


def _find_wall(
    empty: list[ClassifiedTile],
    meaningful: list[ClassifiedTile],
    params: LayoutParams,
) -> tuple[list[ClassifiedTile], list[ClassifiedTile]]:
    """Partition empty tiles into wall (far from meaningful) and unknown."""
    if not empty or not meaningful:
        return [], empty

    empty_centers = np.array([[e.box.cx, e.box.cy] for e in empty], dtype=float)
    meaningful_centers = np.array([[m.box.cx, m.box.cy] for m in meaningful], dtype=float)

    diff = empty_centers[:, None, :] - meaningful_centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    min_dist_to_meaningful = dist.min(axis=1)
    far = min_dist_to_meaningful > params.wall_neighbor_eps

    wall: list[ClassifiedTile] = []
    unknown: list[ClassifiedTile] = []

    if far.any():
        far_indices = np.nonzero(far)[0]
        far_centers = empty_centers[far_indices]
        far_diff = far_centers[:, None, :] - far_centers[None, :, :]
        far_dist = np.sqrt((far_diff ** 2).sum(axis=-1))
        far_adj = far_dist < params.wall_neighbor_eps * 1.5
        np.fill_diagonal(far_adj, True)

        far_visited = np.full(len(far_indices), -1, dtype=int)
        gid = -1
        for i in range(len(far_indices)):
            if far_visited[i] != -1:
                continue
            gid += 1
            far_visited[i] = gid
            stack = [i]
            while stack:
                j = stack.pop()
                for k in np.nonzero(far_adj[j])[0]:
                    if far_visited[k] != -1:
                        continue
                    far_visited[k] = gid
                    stack.append(int(k))

        wall_groups: dict[int, list[int]] = {}
        for orig_idx, label in zip(far_indices, far_visited.tolist()):
            wall_groups.setdefault(label, []).append(orig_idx)

        for group in wall_groups.values():
            if len(group) >= params.wall_min_tiles:
                wall.extend(empty[i] for i in group)
            else:
                unknown.extend(empty[i] for i in group)
    else:
        for i, is_far in enumerate(far):
            if is_far:
                unknown.append(empty[i])

    near_indices = np.nonzero(~far)[0]
    unknown.extend(empty[i] for i in near_indices)

    return wall, unknown
