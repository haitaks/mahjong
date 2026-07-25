"""Input readers: convert external detection formats into list[TileBox].

Supported sources:
  * YOLO ``.txt`` label files (``class cx cy w h``, normalized) — the format
    the Roboflow dataset ships in.
  * A directory of such ``.txt`` files.
  * JSON: ``[{"class_id":..,"cx":..,"cy":..,"w":..,"h":..}, ...]``.
  * Raw Python lists (tuples, dicts, or TileBox) — handy for tests and as the
    integration boundary with a future YOLO inference adapter.

Coordinates are always stored normalized, matching YOLO conventions. Pixel
coordinates are only needed by the viz layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Sequence

from .types import TileBox


def from_yolo_label(path: str | os.PathLike) -> list[TileBox]:
    """Parse a single YOLO ``.txt`` label file."""
    tiles: list[TileBox] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"{path}:{lineno}: expected 'class cx cy w h', got {line!r}"
                )
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
            tiles.append(TileBox(cx=cx, cy=cy, w=w, h=h, class_id=cls))
    return tiles


def iter_yolo_dir(directory: str | os.PathLike) -> Iterator[tuple[str, list[TileBox]]]:
    """Yield ``(stem, tiles)`` for every ``.txt`` file in a directory.

    `stem` is the filename without extension — used to match against images
    in the viz/CLI layer.
    """
    base = Path(directory)
    for p in sorted(base.glob("*.txt")):
        yield p.stem, from_yolo_label(p)


def from_json(path: str | os.PathLike) -> list[TileBox]:
    """Parse a JSON file: a list of objects with cx/cy/w/h (and optional class_id)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return from_raw(data)


def from_raw(boxes: Sequence) -> list[TileBox]:
    """Coerce a sequence of detections into TileBox objects.

    Accepts:
      * :class:`TileBox` (passed through),
      * dict with keys cx/cy/w/h (class_id optional),
      * 4- or 5-tuple/list: ``(cx, cy, w, h)`` or ``(class_id, cx, cy, w, h)``.
    """
    out: list[TileBox] = []
    for b in boxes:
        if isinstance(b, TileBox):
            out.append(b)
        elif isinstance(b, dict):
            out.append(
                TileBox(
                    cx=float(b["cx"]),
                    cy=float(b["cy"]),
                    w=float(b["w"]),
                    h=float(b["h"]),
                    class_id=int(b["class_id"]) if b.get("class_id") is not None else None,
                )
            )
        elif isinstance(b, (tuple, list)):
            if len(b) == 4:
                cx, cy, w, h = (float(x) for x in b)
                out.append(TileBox(cx=cx, cy=cy, w=w, h=h))
            elif len(b) == 5:
                cls, cx, cy, w, h = b
                out.append(
                    TileBox(cx=float(cx), cy=float(cy), w=float(w), h=float(h), class_id=int(cls))
                )
            else:
                raise ValueError(f"Cannot parse detection {b!r}: need 4 or 5 elements")
        else:
            raise TypeError(f"Unsupported detection type: {type(b).__name__}")
    return out
