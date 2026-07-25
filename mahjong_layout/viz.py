"""Overlay clustering results on a photo using PIL.

Color coding (configurable):
  * hand    — green
  * discard — red
  * wall    — blue
  * other   — grey
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .types import Cluster, LayoutResult

ROLE_COLORS = {
    "hand": (0, 200, 0),
    "discard": (220, 30, 30),
    "wall": (30, 120, 220),
    "other": (160, 160, 160),
}


def render(
    image_path: str | Path,
    result: LayoutResult,
    out_path: Optional[str | Path] = None,
    colors: Optional[dict[str, tuple[int, int, int]]] = None,
) -> "Image.Image":
    """Draw clusters over an image; return the PIL image (and save if out_path).

    `image_path` is opened with PIL. If `result.image_size` is set it must match
    the image; otherwise we read size from the file.
    """
    colors = colors or ROLE_COLORS
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - env without default font
        font = None

    for cluster in result.clusters:
        color = colors.get(cluster.role, colors["other"])
        _draw_cluster(draw, cluster, color, W, H, font)

    # Top-left summary.
    summary = result.summary()
    if font is not None:
        draw.rectangle([0, 0, 360, 22], fill=(0, 0, 0, 180))
        draw.text((6, 4), summary, fill=(255, 255, 255), font=font)

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, quality=90)
    return img


def _draw_cluster(draw, cluster: Cluster, color, W, H, font):
    # Tile boxes.
    for t in cluster.tiles:
        xmin = int((t.cx - t.w / 2) * W)
        ymin = int((t.cy - t.h / 2) * H)
        xmax = int((t.cx + t.w / 2) * W)
        ymax = int((t.cy + t.h / 2) * H)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)

    # Cluster bounding box (dashed-ish: just a thinner contrasting rect).
    xmin, ymin, xmax, ymax = cluster.bbox
    draw.rectangle(
        [int(xmin * W), int(ymin * H), int(xmax * W), int(ymax * H)],
        outline=color,
        width=1,
    )

    # Label near top-left of the cluster.
    lx, ly = int(xmin * W), max(0, int(ymin * H) - 16)
    label = f"{cluster.label}({cluster.n_tiles})"
    if font is not None:
        draw.text((lx + 2, ly), label, fill=color, font=font)
