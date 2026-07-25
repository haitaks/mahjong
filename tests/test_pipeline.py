"""End-to-end pipeline + IO reader tests.

Synthetic layouts only — no real dataset.
"""

import json
from pathlib import Path

import pytest

from mahjong_layout import LayoutParams, cluster_layout
from mahjong_layout.io_readers import from_raw, from_yolo_label, iter_yolo_dir
from mahjong_layout.types import TileBox


def _row(y, x0, w, h, n, step):
    return [TileBox(x0 + i * step, y, w, h) for i in range(n)]


# --------------------------------------------------------------------------- #
# pipeline                                                                     #
# --------------------------------------------------------------------------- #


def test_pipeline_hand_plus_discard_layout():
    hand = _row(0.85, 0.1, 0.05, 0.05, 13, 0.06)
    discard = _row(0.25, 0.5, 0.05, 0.05, 5, 0.06)
    result = cluster_layout(hand + discard)
    assert result.hand is not None
    assert result.hand.n_tiles == 13
    assert len(result.discards) == 1
    assert result.discards[0].n_tiles == 5
    assert result.walls == []
    assert result.others == []


def test_pipeline_accepts_tuples_and_dicts():
    boxes = [
        (0.1, 0.85, 0.05, 0.05),
        {"cx": 0.16, "cy": 0.85, "w": 0.05, "h": 0.05},
        {"cx": 0.22, "cy": 0.85, "w": 0.05, "h": 0.05, "class_id": 7},
    ]
    result = cluster_layout(boxes)
    assert result.hand is not None
    assert result.hand.n_tiles == 3


def test_pipeline_empty_input():
    result = cluster_layout([])
    assert result.hand is None
    assert result.clusters == []


def test_pipeline_summary_string():
    result = cluster_layout(_row(0.85, 0.1, 0.05, 0.05, 13, 0.06))
    s = result.summary()
    assert s.startswith("hand=13")


# --------------------------------------------------------------------------- #
# io_readers                                                                   #
# --------------------------------------------------------------------------- #


def test_from_raw_tilebox_passthrough():
    t = TileBox(0.5, 0.5, 0.1, 0.1)
    assert from_raw([t]) == [t]


def test_from_raw_tuple_variants():
    out = from_raw([(0.1, 0.2, 0.3, 0.4), (9, 0.1, 0.2, 0.3, 0.4)])
    assert out[0].class_id is None
    assert out[1].class_id == 9
    assert out[1].cx == 0.1


def test_from_raw_bad_length():
    with pytest.raises(ValueError):
        from_raw([(0.1, 0.2, 0.3)])


def test_from_yolo_label_roundtrip(tmp_path: Path):
    lbl = tmp_path / "x.txt"
    lbl.write_text("47 0.5 0.5 0.1 0.1\n36 0.2 0.2 0.05 0.05\n")
    tiles = from_yolo_label(lbl)
    assert len(tiles) == 2
    assert tiles[0].class_id == 47
    assert tiles[1].class_id == 36


def test_iter_yolo_dir(tmp_path: Path):
    (tmp_path / "a.txt").write_text("0 0.1 0.9 0.05 0.05\n")
    (tmp_path / "b.txt").write_text("0 0.2 0.9 0.05 0.05\n0 0.3 0.9 0.05 0.05\n")
    items = dict(iter_yolo_dir(tmp_path))
    assert set(items) == {"a", "b"}
    assert len(items["a"]) == 1
    assert len(items["b"]) == 2


def test_cli_runs(tmp_path: Path):
    # Minimal end-to-end: labels dir + JSON output.
    lbl_dir = tmp_path / "labels"
    lbl_dir.mkdir()
    # Lower row of 13 = hand.
    (lbl_dir / "p1.txt").write_text(
        "".join(f"0 {0.1 + i*0.06:.4f} 0.85 0.05 0.05\n" for i in range(13))
    )
    from mahjong_layout.cli import main

    out_dir = tmp_path / "out"
    rc = main([str(lbl_dir), "--out", str(out_dir), "--json", "--quiet"])
    assert rc == 0
    data = json.loads((out_dir / "layout.json").read_text())
    assert data["p1"]["hand"]["n_tiles"] == 13
