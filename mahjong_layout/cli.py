"""Command-line entry point: ``mahjong-layout``.

Run clustering over a directory of YOLO label files, print a per-photo
summary, and optionally dump JSON and rendered overlay images.

Examples
--------
    # Just print a summary line per photo:
    mahjong-layout valid/labels

    # Also write JSON + overlay JPEGs over matching photos:
    mahjong-layout valid/labels --images-dir ../images --out out --json --viz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .io_readers import iter_yolo_dir
from .pipeline import cluster_layout
from .types import LayoutParams, LayoutResult
from .viz import render

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _cluster_to_dict(result: LayoutResult) -> dict:
    def cl(c):
        return {
            "role": c.role,
            "label": c.label,
            "n_tiles": c.n_tiles,
            "centroid": [round(c.centroid[0], 4), round(c.centroid[1], 4)],
            "bbox": [round(v, 4) for v in c.bbox],
            "dominant_orientation": c.dominant_orientation,
            "n_rows": c.n_rows,
            "n_cols": c.n_cols,
            "regularity": round(c.regularity, 3),
            "confidence": round(c.confidence, 3),
        }

    return {
        "summary": result.summary(),
        "hand": cl(result.hand) if result.hand else None,
        "discards": [cl(c) for c in result.discards],
        "walls": [cl(c) for c in result.walls],
        "others": [cl(c) for c in result.others],
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
        description="Cluster mahjong tile detections into hand/discard/wall zones.",
    )
    parser.add_argument("input_dir", help="Directory with YOLO .txt label files.")
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Directory with source photos (for --viz). Matched by stem.",
    )
    parser.add_argument("--out", default="out", help="Output directory.")
    parser.add_argument("--json", action="store_true", help="Write layout.json.")
    parser.add_argument("--viz", action="store_true", help="Render overlay images.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-photo output.")

    parser.add_argument("--hand-y-min", type=float, default=LayoutParams().hand_y_min)
    parser.add_argument("--eps-k", type=float, default=LayoutParams().eps_k)
    parser.add_argument("--min-samples", type=int, default=LayoutParams().min_samples)
    parser.add_argument("--hand-max-tiles", type=int, default=LayoutParams().hand_max_tiles)
    parser.add_argument("--discard-min-tiles", type=int, default=LayoutParams().discard_min_tiles)
    parser.add_argument("--wall-aspect", type=float, default=LayoutParams().wall_aspect)

    args = parser.parse_args(argv)

    params = LayoutParams(
        eps_k=args.eps_k,
        min_samples=args.min_samples,
        hand_y_min=args.hand_y_min,
        hand_max_tiles=args.hand_max_tiles,
        discard_min_tiles=args.discard_min_tiles,
        wall_aspect=args.wall_aspect,
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
        result = cluster_layout(tiles, params=params)
        if not args.quiet:
            print(f"{stem}: {result.summary()}")

        if args.json:
            all_results[stem] = _cluster_to_dict(result)

        if args.viz:
            img_path = _find_image(images_dir, stem)
            if img_path is None:
                if not args.quiet:
                    print(f"  (no image for {stem}, skipping viz)", file=sys.stderr)
            else:
                with open(img_path, "rb") as fh:
                    pass  # ensure readable
                render(img_path, result, out_path=out_dir / f"{stem}_layout.jpg")

    if args.json:
        json_path = out_dir / "layout.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, ensure_ascii=False, indent=2)
        print(f"wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
