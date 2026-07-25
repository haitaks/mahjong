"""Tests for the scale-aware DBSCAN clustering routine.

Synthetic layouts only — the real dataset is intentionally not used.
"""

from mahjong_layout import LayoutParams, dbscan_cluster
from mahjong_layout.types import TileBox


def _row(y: float, x0: float, w: float, h: float, n: int, step: float, cls: int = 0):
    """Build `n` tiles in a horizontal row starting at x0, spacing `step`."""
    return [TileBox(x0 + i * step, y, w, h, class_id=cls) for i in range(n)]


def test_empty_returns_empty():
    assert dbscan_cluster([]) == []


def test_dense_row_is_one_cluster():
    tiles = _row(y=0.8, x0=0.1, w=0.05, h=0.05, n=5, step=0.06)
    clusters = dbscan_cluster(tiles)
    assert len(clusters) == 1
    assert clusters[0].n_tiles == 5
    assert clusters[0].dominant_orientation == "h"


def test_dense_row_plus_scatter_noise():
    # A tight row of 5 ...
    tiles = _row(y=0.8, x0=0.1, w=0.05, h=0.05, n=5, step=0.06)
    # ... plus 3 far-away scattered tiles (noise -> singletons).
    tiles += [
        TileBox(0.9, 0.1, 0.05, 0.05),
        TileBox(0.05, 0.4, 0.05, 0.05),
        TileBox(0.95, 0.45, 0.05, 0.05),
    ]
    clusters = dbscan_cluster(tiles)
    big = [c for c in clusters if c.n_tiles > 1]
    singletons = [c for c in clusters if c.n_tiles == 1]
    assert len(big) == 1
    assert big[0].n_tiles == 5
    assert len(singletons) == 3
    # All singletons should start as 'other'.
    assert all(c.role == "other" for c in singletons)


def test_eps_scales_with_tile_size():
    # Big tiles far apart should still cluster (eps scales up).
    big = _row(y=0.5, x0=0.1, w=0.22, h=0.22, n=4, step=0.27)
    clusters = dbscan_cluster(big)
    assert len(clusters) == 1
    assert clusters[0].n_tiles == 4


def test_eps_does_not_overmerge_small():
    # Small tiles at the same spacing as the big-tile case must NOT merge into
    # one blob — confirms eps tracks tile size, not a fixed absolute value.
    small = [
        TileBox(0.1 + i * 0.27, 0.5, 0.05, 0.05) for i in range(4)
    ]
    clusters = dbscan_cluster(small)
    assert len(clusters) == 4  # all singletons: too far apart at this scale


def test_two_separated_rows_are_two_clusters():
    row_a = _row(y=0.8, x0=0.1, w=0.05, h=0.05, n=4, step=0.06)
    row_b = _row(y=0.2, x0=0.1, w=0.05, h=0.05, n=4, step=0.06)
    clusters = dbscan_cluster(row_a + row_b)
    big = [c for c in clusters if c.n_tiles > 1]
    assert len(big) == 2
    assert all(c.n_tiles == 4 for c in big)


def test_grid_estimation_rows_cols():
    # Two rows of three tiles.
    row_a = _row(y=0.80, x0=0.1, w=0.05, h=0.05, n=3, step=0.07)
    row_b = _row(y=0.88, x0=0.1, w=0.05, h=0.05, n=3, step=0.07)
    clusters = dbscan_cluster(row_a + row_b)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.n_rows >= 2
    assert c.n_cols >= 3


def test_regularity_high_for_uniform_row_low_for_irregular():
    uniform = _row(y=0.8, x0=0.1, w=0.05, h=0.05, n=6, step=0.06)
    # One cluster (all gaps < eps) but with alternating big/small spacing.
    irregular = [
        TileBox(x, 0.8, 0.05, 0.05) for x in (0.10, 0.20, 0.22, 0.32, 0.34)
    ]
    cu = dbscan_cluster(uniform)
    ci = dbscan_cluster(irregular)
    assert len(ci) == 1  # sanity: stays a single cluster
    reg_u = cu[0].regularity
    reg_i = ci[0].regularity
    assert reg_u > reg_i
    assert reg_u > 0.8
    assert reg_i < 0.7
