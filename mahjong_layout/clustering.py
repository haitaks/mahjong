"""Scale-aware DBSCAN clustering of tile detections into spatial groups.

We do NOT use sklearn on purpose: the neighbor metric is anchored to the
median tile size (so the same eps works for a close-up hand or a wide table
shot), which is awkward to express with a stock estimator. The
implementation is a textbook DBSCAN over a precomputed adjacency matrix and is
about 80 lines.

All coordinates are normalized to [0, 1].
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .types import Cluster, LayoutParams, TileBox


def _median_tile_size(tiles: list[TileBox]) -> Optional[float]:
    if not tiles:
        return None
    return float(np.median([t.size for t in tiles]))


def _adjacency(centers: np.ndarray, eps: float) -> np.ndarray:
    """Boolean adjacency matrix: True where pairwise distance < eps."""
    # centers: (N, 2)
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return dist < eps


def dbscan_cluster(
    tiles: list[TileBox], params: LayoutParams = LayoutParams()
) -> list[Cluster]:
    """Cluster tiles into spatial groups.

    Tiles whose centers are within ``eps_k * median_tile_size`` of each other
    (transitively) end up in the same cluster. Isolated tiles become singleton
    clusters tagged ``role="other"``.

    Returns a list of :class:`Cluster`, each enriched with geometric
    descriptors (centroid, bbox, orientation, n_rows/n_cols, regularity).
    """
    if not tiles:
        return []

    med = _median_tile_size(tiles)
    assert med is not None and med > 0
    eps = params.eps_k * med

    centers = np.array([[t.cx, t.cy] for t in tiles], dtype=float)
    adj = _adjacency(centers, eps)
    np.fill_diagonal(adj, True)

    # Standard DBSCAN expansion. visited: assigned cluster id per index, -1 = noise.
    n = len(tiles)
    visited = np.full(n, -1, dtype=int)
    neighbor_counts = adj.sum(axis=1)
    cid = -1
    for i in range(n):
        if visited[i] != -1:
            continue
        # Core points have >= min_samples neighbors (incl. themselves).
        if neighbor_counts[i] < params.min_samples:
            visited[i] = -2  # noise; becomes singleton later
            continue
        cid += 1
        visited[i] = cid
        # BFS-style expansion over the adjacency graph.
        frontier = [i]
        while frontier:
            j = frontier.pop()
            for k in np.nonzero(adj[j])[0]:
                if visited[k] != -1:
                    continue
                visited[k] = cid
                if neighbor_counts[k] >= params.min_samples:
                    frontier.append(int(k))

    # Group indices by cluster id; noise points become their own singleton.
    groups: dict[int, list[int]] = {}
    singletons: list[int] = []
    for idx, label in enumerate(visited.tolist()):
        if label < 0:
            singletons.append(idx)
        else:
            groups.setdefault(label, []).append(idx)

    clusters: list[Cluster] = []
    for idxs in groups.values():
        clusters.append(_build_cluster([tiles[i] for i in idxs], med))
    for idx in singletons:
        clusters.append(_build_cluster([tiles[idx]], med))

    # Largest first — a stable order helps tests and heuristics.
    clusters.sort(key=lambda c: c.n_tiles, reverse=True)
    return clusters


def _build_cluster(tiles: list[TileBox], median_size: float) -> Cluster:
    """Enrich a group of tiles with geometric descriptors."""
    xs = np.array([t.cx for t in tiles])
    ys = np.array([t.cy for t in tiles])
    ws = np.array([t.w for t in tiles])
    hs = np.array([t.h for t in tiles])

    cx = float(xs.mean())
    cy = float(ys.mean())

    xmin = float(min(t.cx - t.w / 2 for t in tiles))
    ymin = float(min(t.cy - t.h / 2 for t in tiles))
    xmax = float(max(t.cx + t.w / 2 for t in tiles))
    ymax = float(max(t.cy + t.h / 2 for t in tiles))

    n_h = int(sum(1 for t in tiles if t.is_horizontal))
    dominant = "h" if n_h >= len(tiles) / 2 else "v"

    n_rows, n_cols = _estimate_grid(ys, xs, median_size)
    regularity = _regularity(tiles, dominant)

    return Cluster(
        tiles=tiles,
        centroid=(cx, cy),
        bbox=(xmin, ymin, xmax, ymax),
        n_tiles=len(tiles),
        median_tile_size=float(np.median([t.size for t in tiles])),
        dominant_orientation=dominant,
        n_rows=n_rows,
        n_cols=n_cols,
        regularity=regularity,
    )


def _estimate_grid(ys: np.ndarray, xs: np.ndarray, size: float) -> tuple[int, int]:
    """Estimate (rows, cols) via histograms along each axis.

    Bin width is ~0.6 * median tile size, so adjacent tiles in a row land in
    the same y-bin while different rows separate.
    """
    bin_w = max(0.6 * size, 1e-3)

    y_edges = np.arange(ys.min() - bin_w / 2, ys.max() + bin_w, bin_w)
    x_edges = np.arange(xs.min() - bin_w / 2, xs.max() + bin_w, bin_w)
    n_rows = max(len(np.histogram(ys, bins=y_edges)[0].nonzero()[0]), 1)
    n_cols = max(len(np.histogram(xs, bins=x_edges)[0].nonzero()[0]), 1)
    return n_rows, n_cols


def _regularity(tiles: list[TileBox], dominant: str) -> float:
    """How uniform the horizontal spacing between tiles is, 0..1.

    We always measure spacing along the **horizontal (cx)** axis, because
    any cluster laid out as a row (hand, discard row) is primarily
    horizontal. The ``dominant`` parameter (whether individual tiles are
    horizontal or vertical) affects the coordinate used only for wall
    clusters, which are stacked vertically.

    Projects tile centers onto the x axis, sorts them, and measures the
    coefficient of variation of consecutive gaps. Returns ``1 - cv``
    clamped to [0, 1]. Singletons and pairs default to regular.
    """
    if len(tiles) < 3:
        return 1.0
    # For wall-like clusters (vertical stacks), measure vertical regularity.
    # For everything else (hand, discard rows), measure horizontal regularity.
    if dominant == "v" and len(tiles) >= 4:
        # Check if this is a genuinely stacked vertical cluster (wall):
        # tiles stacked vertically will have low y-variance and high x-variance,
        # the opposite of a row.
        xs = np.array([t.cx for t in tiles])
        ys = np.array([t.cy for t in tiles])
        if ys.std() > xs.std():
            coord = np.sort(ys)
        else:
            coord = np.sort(xs)
    else:
        coord = np.sort(np.array([t.cx for t in tiles]))
    gaps = np.diff(coord)
    if gaps.mean() <= 0:
        return 1.0
    cv = float(gaps.std() / gaps.mean())
    return float(max(0.0, min(1.0, 1.0 - cv)))
