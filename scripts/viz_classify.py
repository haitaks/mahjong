"""Визуализация: overlay с классификацией на исходных фото.
Сохраняет .jpg с цветными рамками + подписями тайлов.

Usage:
    python scripts/viz_classify.py [--photo N] [--all]

Defaults to first 2 photos from data/valid/.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw, ImageFont

from mahjong_layout.io_readers import from_yolo_label
from mahjong_layout.pipeline import classify_layout

LABELS_DIR = _ROOT / "yolo" / "valid" / "labels"
IMAGES_DIR = _ROOT / "yolo" / "valid" / "images"
OUT_DIR = _ROOT / "out" / "viz"

# Цвета зон
ZONE_COLORS = {
    "hand":    (0, 200, 0),    # зелёный
    "discard": (220, 30, 30),  # красный
    "wall":    (30, 120, 220), # синий
    "unknown": (160, 160, 160),# серый
}
ZONE_LABEL_COLORS = {
    "hand":    (0, 80, 0),
    "discard": (80, 0, 0),
    "wall":    (0, 0, 80),
    "unknown": (60, 60, 60),
}
BG_COLOR = (255, 255, 200)  # светло-жёлтый фон для текста


def draw_overlay(img: Image.Image, result) -> Image.Image:
    """Рисует рамки + подписи на копии изображения."""
    out = img.copy()
    W, H = out.size
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        small_font = font
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            small_font = font
        except Exception:
            font = ImageFont.load_default()
            small_font = font

    for zone_name in ("hand", "discard", "wall", "unknown"):
        color = ZONE_COLORS[zone_name]
        label_bg = ZONE_LABEL_COLORS[zone_name]
        for ct in getattr(result, zone_name):
            b = ct.box
            xmin = int((b.cx - b.w / 2) * W)
            ymin = int((b.cy - b.h / 2) * H)
            xmax = int((b.cx + b.w / 2) * W)
            ymax = int((b.cy + b.h / 2) * H)

            # Рамка
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

            # Подпись — label + зона
            label_text = f"{ct.label} [{zone_name}]"
            bbox = draw.textbbox((0, 0), label_text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            # Фон под текст — сверху рамки
            draw.rectangle([xmin, ymin - th - 4, xmin + tw + 6, ymin], fill=label_bg)
            draw.text((xmin + 3, ymin - th - 2), label_text, fill=(255, 255, 255), font=font)

    # Легенда
    legend_y = 8
    for zone_name, color in ZONE_COLORS.items():
        lx, ly = 8, legend_y
        draw.rectangle([lx, ly, lx + 18, ly + 12], fill=color)
        draw.text((lx + 22, ly - 2), zone_name, fill=(0, 0, 0), font=font)
        legend_y += 20

    return out


def main():
    label_files = sorted(LABELS_DIR.glob("*.txt"))[:5]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for lf in label_files:
        stem = lf.stem
        # Ищем картинку
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            c = IMAGES_DIR / f"{stem}{ext}"
            if c.exists():
                img_path = c
                break
        if img_path is None:
            print(f"  skip {stem}: no image")
            continue

        print(f"  Processing {stem}...")
        img = Image.open(img_path).convert("RGB")
        boxes = from_yolo_label(lf)
        result = classify_layout(boxes, img)

        overlay = draw_overlay(img, result)
        out_path = OUT_DIR / f"{stem}_viz.jpg"
        overlay.save(out_path, quality=92)
        print(f"    -> {out_path}  ({result.summary()})")

    print(f"\nDone. {len(list(OUT_DIR.glob('*_viz.jpg')))} overlays saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
