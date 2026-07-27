"""Tests for the classify_layout pipeline (anchor-based hand detection).

Uses PIL to create synthetic photos with drawn tiles, and monkeypatches
classify_tile to return predictable results.
"""

from __future__ import annotations

from typing import Optional

import pytest
from PIL import Image, ImageDraw

from mahjong_layout import LayoutParams, classify_layout
from mahjong_layout.classify import TileClassification
from mahjong_layout.classify.types import Suit
from mahjong_layout.types import ClassifiedTile, LayoutResult, TileBox


def _fake_classify(label: str, suit: Suit = Suit.PIN, value: Optional[int] = 1) -> TileClassification:
    is_empty = label == "unknown"
    return TileClassification(
        suit=suit if not is_empty else Suit.UNKNOWN,
        value=value if not is_empty else None,
        label=label,
        confidence=0.0 if is_empty else 0.9,
        suit_confidence=0.0 if is_empty else 0.9,
        method="stub",
    )


def _synthetic_image(boxes: list[TileBox], size=(400, 600)) -> Image.Image:
    img = Image.new("RGB", size, (200, 200, 200))
    d = ImageDraw.Draw(img)
    for b in boxes:
        xmin = int((b.cx - b.w / 2) * size[0])
        ymin = int((b.cy - b.h / 2) * size[1])
        xmax = int((b.cx + b.w / 2) * size[0])
        ymax = int((b.cy + b.h / 2) * size[1])
        d.rectangle([xmin, ymin, xmax, ymax], fill=(220, 220, 220))
    return img


def _monkey_classify(monkeypatch, stub):
    import mahjong_layout.pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "classify_tile", stub)


# --------------------------------------------------------------------------- #
# tests                                                                        #
# --------------------------------------------------------------------------- #


def test_empty_input():
    img = _synthetic_image([])
    result = classify_layout([], img)
    assert result.hand == []
    assert result.discard == []
    assert result.wall == []
    assert result.unknown == []


def test_all_empty_means_no_hand(monkeypatch):
    _monkey_classify(monkeypatch, lambda crop, params=None: _fake_classify("unknown"))
    boxes = [TileBox(0.2, 0.8, 0.05, 0.05) for _ in range(5)]
    img = _synthetic_image(boxes)
    result = classify_layout(boxes, img)
    assert result.hand == []
    assert result.discard == []
    assert len(result.wall) == 5
    assert result.unknown == []


def test_lowest_tile_anchor_picks_hand(monkeypatch):
    """Tiles near the lowest meaningful tile → hand; far ones → discard."""
    _monkey_classify(monkeypatch, lambda crop, params=None: _fake_classify("pin3"))
    # Compact cluster at bottom (step=0.04, so all 5 fit in radius 0.15)
    hand_group = [TileBox(0.2 + i * 0.04, 0.85, 0.05, 0.05) for i in range(5)]
    far_discard = [TileBox(0.5, 0.15, 0.05, 0.05)]
    boxes = hand_group + far_discard
    img = _synthetic_image(boxes)
    result = classify_layout(boxes, img)
    assert len(result.hand) == 5
    assert len(result.discard) == 1
    assert result.wall == []


def test_two_separate_groups_lowest_group_is_hand(monkeypatch):
    """Two groups: lower one should be hand, upper one discard."""
    _monkey_classify(monkeypatch, lambda crop, params=None: _fake_classify("pin3"))
    lower = [TileBox(0.1 + i * 0.04, 0.85, 0.05, 0.05) for i in range(5)]
    upper = [TileBox(0.1 + i * 0.04, 0.15, 0.05, 0.05) for i in range(5)]
    boxes = lower + upper
    img = _synthetic_image(boxes)
    result = classify_layout(boxes, img)
    assert len(result.hand) == 5
    assert len(result.discard) == 5


def test_single_tile_is_hand(monkeypatch):
    """Only one meaningful tile → it's the hand."""
    _monkey_classify(monkeypatch, lambda crop, params=None: _fake_classify("wan5"))
    boxes = [TileBox(0.5, 0.5, 0.05, 0.05)]
    img = _synthetic_image(boxes)
    result = classify_layout(boxes, img)
    assert len(result.hand) == 1
    assert result.discard == []


def test_hand_radius_controls_grouping(monkeypatch):
    """Small hand_eps_k → only very close tiles join the hand."""
    _monkey_classify(monkeypatch, lambda crop, params=None: _fake_classify("pin3"))
    anchor = TileBox(0.5, 0.9, 0.05, 0.05)
    close = TileBox(0.52, 0.87, 0.05, 0.05)   # dist ≈ 0.036
    far = TileBox(0.7, 0.85, 0.05, 0.05)       # dist ≈ 0.206
    boxes = [anchor, close, far]
    img = _synthetic_image(boxes)
    result = classify_layout(boxes, img, LayoutParams(hand_eps_k=1.0))
    assert len(result.hand) == 2
    assert len(result.discard) == 1


def test_empty_tiles_near_meaningful_are_unknown(monkeypatch):
    """Empty tiles close to meaningful tiles → unknown, not wall."""
    call_count = 0

    def _stub(crop, params=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 5:
            return _fake_classify("pin3")
        return _fake_classify("unknown")

    _monkey_classify(monkeypatch, _stub)
    meaningful = [TileBox(0.2 + i * 0.04, 0.85, 0.05, 0.05) for i in range(5)]
    empty_near = [TileBox(0.22 + i * 0.04, 0.80, 0.05, 0.05) for i in range(3)]
    boxes = meaningful + empty_near
    img = _synthetic_image(boxes)
    result = classify_layout(boxes, img, LayoutParams(wall_neighbor_eps=0.02))
    assert len(result.hand) == 5
    assert len(result.unknown) == 3
    assert result.wall == []


def test_far_empty_tiles_become_wall(monkeypatch):
    """Empty tiles far from meaningful tiles → wall."""
    call_count = 0

    def _stub(crop, params=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 5:
            return _fake_classify("wan3")
        return _fake_classify("unknown")

    _monkey_classify(monkeypatch, _stub)
    meaningful = [TileBox(0.2 + i * 0.04, 0.85, 0.05, 0.05) for i in range(5)]
    far_empty = [TileBox(0.8, 0.1 + i * 0.07, 0.04, 0.04) for i in range(5)]
    boxes = meaningful + far_empty
    img = _synthetic_image(boxes)
    result = classify_layout(boxes, img, LayoutParams(wall_neighbor_eps=0.05))
    assert len(result.hand) == 5
    assert len(result.wall) == 5
    assert result.discard == []


def test_small_far_empty_group_stays_unknown(monkeypatch):
    """Far empty tiles with < wall_min_tiles → unknown."""
    call_count = 0

    def _stub(crop, params=None):
        nonlocal call_count
        call_count += 1
        if call_count <= 5:
            return _fake_classify("wan3")
        return _fake_classify("unknown")

    _monkey_classify(monkeypatch, _stub)
    meaningful = [TileBox(0.2, 0.85, 0.05, 0.05) for _ in range(5)]
    far_empty = [TileBox(0.9, 0.1, 0.04, 0.04)]
    boxes = meaningful + far_empty
    img = _synthetic_image(boxes)
    result = classify_layout(boxes, img, LayoutParams(wall_neighbor_eps=0.05, wall_min_tiles=2))
    assert len(result.wall) == 0
    assert len(result.unknown) == 1
    assert len(result.hand) == 5


def test_summary_string():
    t = ClassifiedTile(TileBox(0.5, 0.5, 0.1, 0.1), "pin3", is_empty=False)
    result = LayoutResult(hand=[t], discard=[t], wall=[t], unknown=[t])
    s = result.summary()
    assert "hand=1" in s
    assert "discard=1" in s
    assert "wall=1" in s
    assert "unknown=1" in s
