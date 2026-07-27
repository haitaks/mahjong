"""Quick test: run classify_layout on a few valid photos and print results."""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from mahjong_layout import LayoutParams, classify_layout
from mahjong_layout.io_readers import from_yolo_label

DATA = Path("data")
VALID_LABELS = DATA / "valid" / "labels"
VALID_IMAGES = DATA / "valid" / "images"

# Just do first 3 photos
label_files = sorted(VALID_LABELS.glob("*.txt"))[:3]

for lbl_path in label_files:
    stem = lbl_path.stem.replace("_jpg.rf.", "_jpg.rf.")
    # Find matching image
    img_path = VALID_IMAGES / f"{lbl_path.stem}.jpg"
    if not img_path.exists():
        print(f"{lbl_path.stem}: no image found, skipping")
        continue

    tiles = from_yolo_label(lbl_path)
    img = Image.open(img_path).convert("RGB")
    result = classify_layout(tiles, img)

    print(f"\n=== {lbl_path.stem} ===")
    print(f"  Summary: {result.summary()}")
    print(f"  Hand ({len(result.hand)}): {[t.label for t in result.hand]}")
    print(f"  Discard ({len(result.discard)}): {[t.label for t in result.discard]}")
    print(f"  Wall ({len(result.wall)}): {[t.label for t in result.wall]}")
    print(f"  Unknown ({len(result.unknown)}): {[t.label for t in result.unknown]}")
