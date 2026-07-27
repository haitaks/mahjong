"""Command-line entry point: ``mahjong-layout``.

Classify and zone mahjong tile detections. Reads YOLO-format label files
and their corresponding images, runs the full pipeline, and optionally
writes JSON and/or overlay visualisations.

Examples
--------
    # Just print a summary line per photo:
    mahjong-layout valid/labels --images-dir valid/images

    # Full run with JSON + overlays:
    mahjong-layout valid/labels --images-dir valid/images --out out --json --viz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from PIL import Image

from .io_readers import iter_yolo_dir
from .pipeline import classify_layout
from .types import ClassifiedTile, LayoutParams, LayoutResult

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _to_dict(result: LayoutResult) -> dict:
    def _ct(ct: ClassifiedTile) -> dict:
        return {
            "label": ct.label,
            "cx": round(ct.box.cx, 4),
            "cy": round(ct.box.cy, 4),
            "w": round(ct.box.w, 4),
            "h": round(ct.box.h, 4),
        }

    return {
        "summary": result.summary(),
        "hand": [_ct(t) for t in result.hand],
        "discard": [_ct(t) for t in result.discard],
        "wall": [_ct(t) for t in result.wall],
        "unknown": [_ct(t) for t in result.unknown],
    }


def _find_image(images_dir: Optional[Path], stem: str) -> Optional[Path]:
    if images_dir is None:
        return None
    for ext in _IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mahjong-layout",
        description="Classify and zone mahjong tiles from YOLO detections.",
    )
    parser.add_argument("input_dir", help="Directory with YOLO .txt label files.")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory with source photos. Matched by stem.",
    )
    parser.add_argument("--out", default="out", help="Output directory.")
    parser.add_argument("--json", action="store_true", help="Write layout.json.")
    parser.add_argument("--viz", action="store_true", help="Render overlay images.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-photo output.")

    parser.add_argument("--hand-y-min", type=float, default=LayoutParams().hand_y_min)
    parser.add_argument("--eps-k", type=float, default=LayoutParams().eps_k)
    parser.add_argument("--min-samples", type=int, default=LayoutParams().min_samples)
    parser.add_argument(
        "--wall-neighbor-eps", type=float, default=LayoutParams().wall_neighbor_eps
    )
    parser.add_argument(
        "--wall-min-tiles", type=int, default=LayoutParams().wall_min_tiles
    )

    args = parser.parse_args(argv)

    params = LayoutParams(
        eps_k=args.eps_k,
        min_samples=args.min_samples,
        hand_y_min=args.hand_y_min,
        wall_neighbor_eps=args.wall_neighbor_eps,
        wall_min_tiles=args.wall_min_tiles,
    )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"error: {input_dir} is not a directory", file=sys.stderr)
        return 2

    images_dir = Path(args.images_dir) if args.images_dir else None
    out_dir = Path(args.out)
    if args.json or args.viz:
        out_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for stem, tiles in iter_yolo_dir(input_dir):
        img_path = _find_image(images_dir, stem) if images_dir else None
        if img_path is None:
            # Without an image we can't classify. Print a warning and skip.
            if not args.quiet:
                print(f"{stem}: no image found, skipping", file=sys.stderr)
            continue

        img = Image.open(img_path).convert("RGB")
        result = classify_layout(tiles, img, params=params)

        if not args.quiet:
            print(f"{stem}: {result.summary()}")

        if args.json:
            all_results[stem] = _to_dict(result)

        if args.viz:
            _render_overlay(img_path, result, out_dir / f"{stem}_layout.jpg")

    if args.json:
        json_path = out_dir / "layout.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, ensure_ascii=False, indent=2)
        print(f"wrote {json_path}")

    return 0


def _render_overlay(
    img_path: Path, result: LayoutResult, out_path: Path
) -> None:
    """Draw a simple overlay: colour-code each tile by its zone."""
    from PIL import ImageDraw

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    colors = {
        "hand": (0, 200, 0),
        "discard": (220, 30, 30),
        "wall": (30, 120, 220),
        "unknown": (160, 160, 160),
    }

    for zone_name, zone_tiles in [
        ("hand", result.hand),
        ("discard", result.discard),
        ("wall", result.wall),
        ("unknown", result.unknown),
    ]:
        color = colors[zone_name]
        for ct in zone_tiles:
            b = ct.box
            xmin = int((b.cx - b.w / 2) * W)
            ymin = int((b.cy - b.h / 2) * H)
            xmax = int((b.cx + b.w / 2) * W)
            ymax = int((b.cy + b.h / 2) * H)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=90)


if __name__ == "__main__":
    raise SystemExit(main())
