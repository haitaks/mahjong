"""Tests for role heuristics (hand/discard/wall/other)."""

from mahjong_layout import LayoutParams, assign_roles, dbscan_cluster
from mahjong_layout.types import TileBox


def _row(y, x0, w, h, n, step):
    return [TileBox(x0 + i * step, y, w, h) for i in range(n)]


def test_hand_picked_from_lower_horizontal_cluster():
    # Hand: lower zone, horizontal, ~13 tiles.
    hand = _row(y=0.85, x0=0.1, w=0.05, h=0.05, n=13, step=0.06)
    # Discard: upper scatter, a few tiles.
    discard = _row(y=0.25, x0=0.4, w=0.05, h=0.05, n=4, step=0.06)
    clusters = dbscan_cluster(hand + discard)
    assign_roles(clusters)
    hands = [c for c in clusters if c.role == "hand"]
    assert len(hands) == 1
    assert hands[0].n_tiles == 13
    assert hands[0].label == "hand"


def test_upper_scatter_becomes_discard():
    clusters = dbscan_cluster(_row(y=0.25, x0=0.4, w=0.05, h=0.05, n=4, step=0.06))
    assign_roles(clusters)
    assert all(c.role == "discard" for c in clusters)


def test_vertical_standing_row_is_wall():
    # Standing wall: tall tiles (h >> w), vertical orientation.
    wall = [TileBox(0.3, 0.1 + i * 0.09, 0.06, 0.12) for i in range(5)]
    clusters = dbscan_cluster(wall)
    assign_roles(clusters)
    walls = [c for c in clusters if c.role == "wall"]
    assert len(walls) == 1


def test_large_lower_cluster_is_not_hand():
    # 30 tiles in the lower zone: too big for a hand -> must not be tagged hand.
    big = _row(y=0.85, x0=0.05, w=0.025, h=0.025, n=30, step=0.03)
    clusters = dbscan_cluster(big)
    assign_roles(clusters)
    assert all(c.role != "hand" for c in clusters)


def test_singleton_far_from_hand_zone_is_other():
    clusters = dbscan_cluster([TileBox(0.5, 0.2, 0.05, 0.05)])
    assign_roles(clusters)
    # y=0.2 is above hand_y_min (0.4) -> not hand; n=1 < discard_min_tiles -> other.
    assert clusters[0].role == "other"


def test_only_one_hand_assigned():
    # Two lower-zone horizontal clusters; exactly one should win.
    a = _row(y=0.85, x0=0.1, w=0.05, h=0.05, n=13, step=0.06)
    b = _row(y=0.65, x0=0.1, w=0.05, h=0.05, n=8, step=0.06)
    clusters = dbscan_cluster(a + b)
    assign_roles(clusters)
    hands = [c for c in clusters if c.role == "hand"]
    assert len(hands) == 1
    # The lower (y=0.85) cluster should win.
    assert abs(hands[0].centroid[1] - 0.85) < 0.02


def test_params_override_hand_zone():
    # Move hand_y_min very high so nothing qualifies as hand.
    params = LayoutParams(hand_y_min=0.95)
    clusters = dbscan_cluster(_row(y=0.85, x0=0.1, w=0.05, h=0.05, n=13, step=0.06))
    assign_roles(clusters, params=params)
    assert all(c.role != "hand" for c in clusters)
