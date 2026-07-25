"""End-to-end classifier tests, plus a wan scenario with a mocked OCR engine.

We don't require the real RapidOCR model: we monkeypatch the OCR helpers so the
wan path is exercised deterministically.
"""

import pytest
from PIL import Image

from mahjong_layout.classify import ClassifyParams, classify_tile
from mahjong_layout.classify.types import Suit

from test_decoders import _pin_tile, _tiao_tile


# --------------------------------------------------------------------------- #
# count-based suits (no OCR needed)                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9])
def test_classify_pin(n):
    res = classify_tile(_pin_tile(n))
    assert res.suit == Suit.PIN
    assert res.value == n
    assert res.label == f"pin{n}"
    assert res.method == "count_components"


@pytest.mark.parametrize("n", [1, 2, 3, 5, 9])
def test_classify_tiao(n):
    res = classify_tile(_tiao_tile(n))
    assert res.suit == Suit.TIAO
    assert res.value == n
    assert res.label == f"tiao{n}"


def test_classify_tiao1_is_bird():
    res = classify_tile(_tiao_tile(1))
    assert res.suit == Suit.TIAO
    assert res.value == 1


# --------------------------------------------------------------------------- #
# wan via mocked OCR                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def patched_wan_router(monkeypatch):
    """Force the router to return WAN, and OCR to return a chosen numeral."""
    import mahjong_layout.classify.router as router_mod
    import mahjong_layout.classify.wan_decoder as wan_mod

    def fake_determine_suit(crop, params=None):
        from mahjong_layout.classify.router import SuitGuess
        from mahjong_layout.classify.types import Suit
        return SuitGuess(Suit.WAN, 0.85)

    monkeypatch.setattr(router_mod, "determine_suit", fake_determine_suit)
    # classifier imports determine_suit by name, so patch its reference too.
    import mahjong_layout.classify.classifier as clf_mod
    monkeypatch.setattr(clf_mod, "determine_suit", fake_determine_suit)
    return wan_mod


@pytest.mark.parametrize(
    "ocr_text,expected",
    [("一", 1), ("二", 2), ("三", 3), ("五", 5), ("九", 9), ("5", 5), ("玖", 9)],
)
def test_classify_wan_with_mocked_ocr(patched_wan_router, monkeypatch, ocr_text, expected):
    import mahjong_layout.classify.wan_decoder as wan_mod

    monkeypatch.setattr(
        wan_mod, "ocr", lambda region, params=None: [(ocr_text, 0.9)]
    )
    res = classify_tile(Image.new("RGB", (100, 200), 245))
    assert res.suit == Suit.WAN
    assert res.value == expected
    assert res.label == f"wan{expected}"
    assert res.method == "ocr"


def test_classify_wan_unrecognized_ocr_returns_unknown(patched_wan_router, monkeypatch):
    import mahjong_layout.classify.wan_decoder as wan_mod

    monkeypatch.setattr(wan_mod, "ocr", lambda region, params=None: [("x", 0.9)])
    res = classify_tile(Image.new("RGB", (100, 200), 245))
    assert res.suit == Suit.WAN
    assert res.value is None
    assert res.label == "unknown"


# --------------------------------------------------------------------------- #
# graceful degradation                                                        #
# --------------------------------------------------------------------------- #


def test_classify_empty_crop_is_unknown():
    # NB: a Python int fill to Image.new("RGB", ...) only fills the first
    # channel, so 245 -> (245,0,0) red. Pass a proper white tuple for a blank.
    res = classify_tile(Image.new("RGB", (100, 100), (245, 245, 245)))
    assert res.suit == Suit.UNKNOWN
    assert res.value is None


def test_classify_null_crop_is_unknown():
    res = classify_tile(None)  # type: ignore[arg-type]
    assert res.suit == Suit.UNKNOWN


def test_classify_never_raises_on_bad_input():
    # An all-noise image must not raise.
    import numpy as np
    arr = (np.random.rand(50, 50, 3) * 255).astype("uint8")
    res = classify_tile(Image.fromarray(arr))
    assert isinstance(res.suit, Suit)
