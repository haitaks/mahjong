"""Run YOLO detection then classify_layout on test photos.

Usage:
    python scripts/yolo_pipeline.py                  # uses yolo/model.pt
    python scripts/yolo_pipeline.py --weights path/to/best.pt

Expects:
    yolo/model.pt        — trained YOLO weights (copy from target machine)
    tests/*.jpg          — photos to run on
    mahjong_layout/      — classification + clustering package
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import argparse

import cv2
import numpy as np
from PIL import Image

from mahjong_layout import classify_layout
from mahjong_layout.types import TileBox

OUT = _ROOT / "out" / "yolo_pipeline"
OUT.mkdir(parents=True, exist_ok=True)


def detect_tiles_yolo(model, img_bgr):
    """Run YOLO inference, return list[TileBox] in normalized coords."""
    h, w = img_bgr.shape[:2]
    results = model(img_bgr, verbose=False)
    boxes = []
    if results[0].boxes is not None:
        for b in results[0].boxes.xywhn:  # [x_center, y_center, width, height] normalized
            cx, cy, bw, bh = b.tolist()
            boxes.append(TileBox(cx=cx, cy=cy, w=bw, h=bh))
    return boxes


def draw_overlay(img_bgr, layout_result):
    """Draw zone-colored boxes with labels on image."""
    h, w = img_bgr.shape[:2]
    vis = img_bgr.copy()
    colors = {"hand": (0, 200, 0), "discard": (0, 0, 220),
              "wall": (200, 120, 0), "unknown": (128, 128, 128)}
    for zname in ("hand", "discard", "wall", "unknown"):
        color = colors[zname]
        for ct in getattr(layout_result, zname):
            b = ct.box
            xmin = int((b.cx - b.w / 2) * w)
            ymin = int((b.cy - b.h / 2) * h)
            xmax = int((b.cx + b.w / 2) * w)
            ymax = int((b.cy + b.h / 2) * h)
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(vis, ct.label, (xmin, ymin - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def main():
    parser = argparse.ArgumentParser(description="YOLO -> classify_layout pipeline")
    parser.add_argument("--weights", default=str(_ROOT / "yolo" / "model.pt"),
                        help="Path to YOLO weights")
    parser.add_argument("--photos", nargs="+",
                        default=sorted(str(p) for p in (_ROOT / "tests").glob("*.jpg")),
                        help="Photos to process")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found at {weights}")
        print("Train YOLO on another machine and copy model.pt here, or provide --weights")
        return 1

    print("Loading YOLO model...")
    from ultralytics import YOLO
    model = YOLO(str(weights))
    print(f"  model: {weights.name}")

    for photo_path in args.photos:
        p = Path(photo_path)
        if not p.exists():
            print(f"  skip {p.name}: not found")
            continue

        print(f"\n=== {p.name} ===")
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            print(f"  ERROR: cannot read {p}")
            continue
        h, w = img_bgr.shape[:2]
        print(f"  size: {w}x{h}")

        # 1. YOLO detection
        tile_boxes = detect_tiles_yolo(model, img_bgr)
        print(f"  YOLO detected: {len(tile_boxes)} tiles")
        if not tile_boxes:
            print("  SKIP: no tiles detected")
            continue

        # 2. classify_layout
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        result = classify_layout(tile_boxes, pil_img)

        # 3. Save overlay
        overlay = draw_overlay(img_bgr, result)
        out_path = OUT / f"{p.stem}_yolo_pipeline.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"  Overlay saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    raise SystemExit(main())
