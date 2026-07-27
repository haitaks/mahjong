"""YOLO detection -> classify_layout on test photos.

Runs end-to-end on the target machine (where YOLO model lives).
Saves overlays to out/yolo_pipeline/.

Usage:
    python scripts/run_pipeline.py                                      # use yolo/model.pt
    python scripts/run_pipeline.py --weights path/to/best.pt           # custom weights
    python scripts/run_pipeline.py --photos photo1.jpg photo2.jpg      # specific photos
    python scripts/run_pipeline.py --no-save --no-classify             # YOLO detection only
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse

import cv2
from PIL import Image

from mahjong_layout import classify_layout
from mahjong_layout.types import TileBox

OUT = ROOT / "out" / "yolo_pipeline"


def detect_tiles_yolo(model, img_bgr):
    h, w = img_bgr.shape[:2]
    results = model(img_bgr, verbose=False)
    boxes = []
    if results[0].boxes is not None:
        for b in results[0].boxes.xywhn:
            cx, cy, bw, bh = b.tolist()
            boxes.append(TileBox(cx=cx, cy=cy, w=bw, h=bh))
    return boxes


def draw_overlay(img_bgr, layout_result):
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
    parser.add_argument("--weights", default=str(ROOT / "yolo" / "model.pt"),
                        help="Path to YOLO weights")
    parser.add_argument("--photos", nargs="+",
                        default=sorted(str(p) for p in (ROOT / "tests").glob("*.jpg")),
                        help="Photos to process")
    parser.add_argument("--out", default=str(OUT), help="Output directory for overlays")
    parser.add_argument("--no-save", action="store_true", help="Don't save overlays (just print)")
    parser.add_argument("--no-classify", action="store_true",
                        help="YOLO detection only, skip classify_layout")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        # Fallback: search for latest training run
        runs_dir = ROOT / "yolo" / "runs" / "detect" / "train" / "weights" / "best.pt"
        if runs_dir.exists():
            weights = runs_dir
            print(f"  (found training weights: {weights})")
        else:
            print(f"ERROR: weights not found: {weights}")
            print("Train first: python scripts/train_yolo.py")
            print("Or provide --weights path/to/best.pt")
            return 1

    out_dir = Path(args.out)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

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

        tile_boxes = detect_tiles_yolo(model, img_bgr)
        print(f"  YOLO detected: {len(tile_boxes)} tiles")
        if not tile_boxes:
            print("  SKIP: no tiles detected")
            continue

        if args.no_classify:
            print("  (classification skipped)")
            continue

        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        result = classify_layout(tile_boxes, pil_img)
        print(f"  Layout: {result.summary()}")

        if not args.no_save:
            overlay = draw_overlay(img_bgr, result)
            out_path = out_dir / f"{p.stem}_pipeline.jpg"
            cv2.imwrite(str(out_path), overlay)
            print(f"  Overlay: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    raise SystemExit(main())
