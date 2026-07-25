"""Tests for the upgraded preprocess: tile-face detection and color masking."""

import numpy as np
from PIL import Image, ImageDraw

from mahjong_layout.classify import ClassifyParams
from mahjong_layout.classify.preprocess import _color_mask, _detect_face, prepare


def _framed_tile(face_color=(245, 245, 245), frame_color=(40, 40, 40), size=(200, 240), frame=30):
    """A light tile face surrounded by a dark frame (mimics a dense crop)."""
    img = Image.new("RGB", size, frame_color)
    d = ImageDraw.Draw(img)
    d.rectangle([frame, frame, size[0] - frame, size[1] - frame], fill=face_color)
    return img


def _color_pin(n, size=(400, 500), dot_colors=None):
    img = Image.new("RGB", size, (245, 245, 245))
    d = ImageDraw.Draw(img)
    W, H = size
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    mx, my = 80, 80
    cw = (W - 2 * mx) / cols
    ch = (H - 2 * my) / rows
    r = 30
    if dot_colors is None:
        dot_colors = [(200, 30, 30), (30, 160, 30), (30, 30, 200)]
    for i in range(n):
        c, ri = i % cols, i // cols
        cx = mx + cw * (c + 0.5)
        cy = my + ch * (ri + 0.5)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=dot_colors[i % len(dot_colors)])
    return img


# --------------------------------------------------------------------------- #
# tile-face detection                                                         #
# --------------------------------------------------------------------------- #


def test_detect_face_finds_inner_region():
    img = _framed_tile()
    face = _detect_face(img, ClassifyParams())
    assert face is not None
    x0, y0, x1, y1 = face
    # Detected face should sit inside the dark frame, not cover the whole crop.
    assert x0 > 10 and y0 > 10
    assert (x1 - x0) < img.size[0]
    assert (y1 - y0) < img.size[1]


def test_prepare_uses_face_when_detected():
    img = _framed_tile()
    prep = prepare(img)
    assert prep.face_bbox is not None


def test_prepare_falls_back_to_inset_when_face_disabled():
    img = _framed_tile()
    params = ClassifyParams(detect_face=False)
    prep = prepare(img, params)
    assert prep.face_bbox is None  # inset fallback used instead


# --------------------------------------------------------------------------- #
# color mask                                                                  #
# --------------------------------------------------------------------------- #


def test_color_mask_present_on_colored_tile():
    img = _color_pin(5)
    mask = _color_mask(img, ClassifyParams())
    assert mask is not None
    assert mask.sum() > 0  # colored dots produce foreground


def test_color_mask_none_on_grayscale():
    gray = Image.new("RGB", (100, 100), (200, 200, 200))
    assert _color_mask(gray, ClassifyParams()) is None


def test_prepare_color_path_when_color_available():
    img = _color_pin(5)
    prep = prepare(img)
    assert prep.method == "color"
    assert prep.color_mask is not None


def test_prepare_threshold_path_when_grayscale():
    gray = Image.new("RGB", (200, 240), (200, 200, 200))
    prep = prepare(gray)
    assert prep.method == "threshold"
    assert prep.color_mask is None
