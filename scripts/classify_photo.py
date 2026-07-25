"""Classify tiles from your own photos.

Two modes:
  * Single tile:  --tile photo.jpg --box "cx cy w h"   (normalized coords)
  * Whole photo:  --photo photo.jpg
                   (auto-detect tiles needs a YOLO model; if absent, every
                    connected bright region is tried — crude, for quick checks)

Prints the classification and, with --debug, writes an annotated crop to --out.

This is a helper for calibrating photo quality and tuning ClassifyParams, not
a production pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly (python scripts/classify_photo.py) without
# installing the package: prepend the project root to sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from mahjong_layout import TileBox, classify_tile, crop_tile
from mahjong_layout.classify import ClassifyParams
from mahjong_layout.classify.ocr_engine import is_available as ocr_available


def _parse_box(s: str) -> TileBox:
    parts = [float(x) for x in s.replace(",", " ").split()]
    if len(parts) == 4:
        cx, cy, w, h = parts
        return TileBox(cx, cy, w, h)
    if len(parts) == 5:
        cls, cx, cy, w, h = parts
        return TileBox(cx, cy, w, h, class_id=int(cls))
    raise SystemExit(f"--box expects 4 or 5 numbers, got: {s!r}")


def _annotate(crop: Image.Image, label: str, out: Path) -> None:
    from PIL import ImageDraw, ImageFont

    img = crop.copy()
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if font is not None:
        d.rectangle([0, 0, 220, 20], fill=(0, 0, 0))
        d.text((4, 3), label, fill=(255, 255, 255), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"  debug: wrote {out}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Classify mahjong tiles from a photo.")
    p.add_argument("--photo", help="Path to photo (whole-photo mode).")
    p.add_argument("--tile", help="Path to photo (single-tile mode, use with --box).")
    p.add_argument("--box", help='Normalized box "cx cy w h" for --tile.')
    p.add_argument("--out", default="out", help="Output dir for debug crops.")
    p.add_argument("--debug", action="store_true", help="Save annotated crop.")
    p.add_argument("--no-color", action="store_true", help="Disable color path.")
    p.add_argument("--no-face-detect", action="store_true", help="Disable tile-face detection.")
    args = p.parse_args(argv)

    if not ocr_available():
        print("warning: RapidOCR not available — wan numerals won't decode.", file=sys.stderr)

    params = ClassifyParams(
        color_tile=not args.no_color,
        detect_face=not args.no_face_detect,
    )

    if args.tile:
        if not args.box:
            print("error: --tile requires --box", file=sys.stderr)
            return 2
        img = Image.open(args.tile).convert("RGB")
        tb = _parse_box(args.box)
        crop = crop_tile(img, tb)
        res = classify_tile(crop, params)
        print(f"{Path(args.tile).name} {args.box} -> {res}")
        if args.debug:
            _annotate(crop, res.label, Path(args.out) / f"{Path(args.tile).stem}_classified.png")
        return 0

    if args.photo:
        print("error: whole-photo auto-detection needs a YOLO model, which is not "
              "wired up yet. Use --tile + --box with normalized coords for now.",
              file=sys.stderr)
        return 2

    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
