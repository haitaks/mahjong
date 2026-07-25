"""Tests for count_decoder and the wan OCR map (no real OCR model needed).

Synthetic tiles are drawn on the fly with PIL so we don't touch the dataset.
"""

import pytest
from PIL import Image, ImageDraw

from mahjong_layout.classify import ClassifyParams, count_components
from mahjong_layout.classify.constants import CN_NUMERAL_TO_INT


# --------------------------------------------------------------------------- #
# synthetic tile generators                                                    #
# --------------------------------------------------------------------------- #


def _blank(size=(160, 200), bg=245):
    img = Image.new("L", size, bg)
    return img


def _pin_tile(n: int) -> Image.Image:
    """A dot/pin tile: n filled circles arranged in a grid, light bg."""
    img = _blank().convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = img.size
    # arrange in up to 3 columns
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    r = 10
    margin_x, margin_y = 30, 30
    cell_w = (W - 2 * margin_x) / cols
    cell_h = (H - 2 * margin_y) / rows
    for i in range(n):
        c, ridx = i % cols, i // cols
        cx = margin_x + cell_w * (c + 0.5)
        cy = margin_y + cell_h * (ridx + 0.5)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 20, 20))
    return img


def _tiao_tile(n: int) -> Image.Image:
    """A bamboo/tiao tile: n thin vertical sticks, light bg. n=1 -> a bird blob."""
    img = _blank().convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = img.size
    if n == 1:
        # bird: one large blob
        d.ellipse([W * 0.25, H * 0.3, W * 0.75, H * 0.85], fill=(20, 20, 20))
        return img
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    sw, sh = 8, 34
    margin_x, margin_y = 30, 30
    cell_w = (W - 2 * margin_x) / cols
    cell_h = (H - 2 * margin_y) / rows
    for i in range(n):
        c, ridx = i % cols, i // cols
        cx = margin_x + cell_w * (c + 0.5)
        cy = margin_y + cell_h * (ridx + 0.5)
        d.rectangle([cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2], fill=(20, 20, 20))
    return img


# --------------------------------------------------------------------------- #
# count decoder                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_pin_count(n):
    value, raw = count_components(_pin_tile(n), "circle")
    assert value == n, f"pin {n}: counted {value}, raw={raw}"


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_tiao_count(n):
    value, raw = count_components(_tiao_tile(n), "stick")
    assert value == n, f"tiao {n}: counted {value}, raw={raw}"


def test_tiao1_detected_as_bird():
    value, raw = count_components(_tiao_tile(1), "stick")
    assert value == 1
    assert raw.get("bird") is True


def test_empty_crop_returns_none():
    value, _ = count_components(_blank().convert("RGB"), "circle")
    assert value is None


# --------------------------------------------------------------------------- #
# wan numeral map (pure, no OCR)                                              #
# --------------------------------------------------------------------------- #


def test_cn_numeral_map_basics():
    assert CN_NUMERAL_TO_INT["一"] == 1
    assert CN_NUMERAL_TO_INT["九"] == 9
    for ch, val in zip("一二三四五六七八九", range(1, 10)):
        assert CN_NUMERAL_TO_INT[ch] == val


def test_cn_numeral_map_formal_and_digits():
    assert CN_NUMERAL_TO_INT["壹"] == 1
    assert CN_NUMERAL_TO_INT["玖"] == 9
    assert CN_NUMERAL_TO_INT["5"] == 5


# --------------------------------------------------------------------------- #
# color path (quality photo simulation)                                       #
# --------------------------------------------------------------------------- #


def _color_pin_tile(n):
    """A large, quality pin tile: white face with colored dots."""
    img = Image.new("RGB", (400, 500), (245, 245, 245))
    d = ImageDraw.Draw(img)
    W, H = img.size
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    mx, my = 80, 80
    cw = (W - 2 * mx) / cols
    ch = (H - 2 * my) / rows
    r = 30
    colors = [(200, 30, 30), (30, 160, 30), (30, 30, 200)]
    for i in range(n):
        c, ri = i % cols, i // cols
        cx = mx + cw * (c + 0.5)
        cy = my + ch * (ri + 0.5)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colors[i % len(colors)])
    return img


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8, 9])
def test_pin_count_via_color(n):
    value, raw = count_components(_color_pin_tile(n), "circle")
    assert value == n, f"color pin {n}: counted {value}, raw={raw}"


def test_color_pin_uses_color_method():
    from mahjong_layout.classify.preprocess import prepare

    prep = prepare(_color_pin_tile(5))
    assert prep.method == "color"
