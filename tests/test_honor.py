"""Tests for the honor decoder (winds + dragons).

OCR is mocked so these run without the heavy RapidOCR model. The blank-face
and color-fallback paths use synthetic PIL tiles, no OCR at all.
"""

from __future__ import annotations

import pytest
from PIL import Image, ImageDraw

from mahjong_layout.classify import ClassifyParams
from mahjong_layout.classify.honor import decode_honor
from mahjong_layout.classify.types import Suit


# --------------------------------------------------------------------------- #
# white dragon (blank face) — no OCR needed                                  #
# --------------------------------------------------------------------------- #


def _white_dragon_tile(size=(200, 240), frame=8, frame_color=(40, 40, 40)) -> Image.Image:
    """A near-blank tile: only a thin dark frame around a white face."""
    img = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size[0] - 1, frame], fill=frame_color)
    d.rectangle([0, size[1] - 1 - frame, size[0] - 1, size[1] - 1], fill=frame_color)
    d.rectangle([0, 0, frame, size[1] - 1], fill=frame_color)
    d.rectangle([size[0] - 1 - frame, 0, size[0] - 1, size[1] - 1], fill=frame_color)
    return img


def _marked_tile(mark_color=(20, 20, 20), size=(200, 240)) -> Image.Image:
    """A tile with a substantial central mark — enough foreground to NOT be
    flagged as a blank white dragon, so the OCR/color paths are exercised."""
    img = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([50, 70, size[0] - 50, size[1] - 70], fill=mark_color)
    return img


def test_white_dragon_detected_from_blank_face():
    res = decode_honor(_white_dragon_tile(), suit_confidence=0.7)
    assert res.suit == Suit.HONOR
    assert res.label == "white_dragon"
    assert res.method == "blank_face"
    assert res.confidence > 0.5


def test_truly_empty_crop_not_white_dragon(monkeypatch):
    # No frame, no ink at all -> not a white dragon, not an honor tile.
    import mahjong_layout.classify.honor as honor_mod

    monkeypatch.setattr(honor_mod, "_ocr_honor_variants", lambda crop, params: [])
    res = decode_honor(Image.new("RGB", (100, 100), (245, 245, 245)), suit_confidence=0.5)
    assert res.label == "unknown_honor"


# --------------------------------------------------------------------------- #
# OCR path (mocked)                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_honor_ocr(monkeypatch):
    """Replace multi-variant OCR with a fixed return value."""
    import mahjong_layout.classify.honor as honor_mod

    def _fake(crop, params):
        return _fake.value  # type: ignore[attr-defined]

    _fake.value = []
    monkeypatch.setattr(honor_mod, "_ocr_honor_variants", _fake)
    return _fake


@pytest.mark.parametrize(
    "char,label",
    [("東", "east"), ("南", "south"), ("西", "west"), ("北", "north"),
     ("中", "red_dragon"), ("發", "green_dragon"), ("东", "east")],
)
def test_honor_ocr_chars(mock_honor_ocr, char, label):
    mock_honor_ocr.value = [(char, 0.9)]
    # Use a marked (non-blank) tile so the OCR path is reached.
    res = decode_honor(_marked_tile(), suit_confidence=0.8)
    assert res.suit == Suit.HONOR
    assert res.label == label
    assert res.method == "ocr"


def test_honor_multi_variant_picks_known_char(mock_honor_ocr):
    # Simulate north_wind behavior: 'K' from one variant + 北 from another.
    mock_honor_ocr.value = [("K", 0.6), ("北", 0.85)]
    res = decode_honor(_marked_tile(), suit_confidence=0.8)
    assert res.label == "north"


def test_honor_ocr_unrecognized_falls_through(mock_honor_ocr):
    mock_honor_ocr.value = [("x", 0.9)]
    # Black mark -> no dominant red/green color -> unknown honor.
    res = decode_honor(_marked_tile(mark_color=(20, 20, 20)), suit_confidence=0.8)
    assert res.label == "unknown_honor"


# --------------------------------------------------------------------------- #
# color fallback (no OCR needed, but mocked to be empty)                     #
# --------------------------------------------------------------------------- #


def _colored_tile(color, size=(200, 240)) -> Image.Image:
    """A white tile with a large central blob in `color`."""
    img = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 60, size[0] - 40, size[1] - 60], fill=color)
    return img


def test_red_dragon_color_fallback(mock_honor_ocr):
    mock_honor_ocr.value = []  # OCR finds nothing
    res = decode_honor(_colored_tile((200, 20, 20)), suit_confidence=0.6)
    assert res.label == "red_dragon"
    assert res.method == "color_fallback"


def test_green_dragon_color_fallback(mock_honor_ocr):
    mock_honor_ocr.value = []
    res = decode_honor(_colored_tile((20, 180, 30)), suit_confidence=0.6)
    assert res.label == "green_dragon"
    assert res.method == "color_fallback"


def test_no_color_no_ocr_is_unknown(mock_honor_ocr):
    mock_honor_ocr.value = []
    # Black ink only -> not a dominant red/green -> unknown.
    res = decode_honor(_colored_tile((20, 20, 20)), suit_confidence=0.6)
    assert res.label == "unknown_honor"
