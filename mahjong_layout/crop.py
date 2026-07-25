"""Helper: crop a TileBox out of an image.

Bridges the layout module (TileBox in normalized coords) and the classifier
(works on a PIL crop).
"""

from __future__ import annotations

import os
from typing import Union

from PIL import Image

from .types import TileBox


def crop_tile(
    image: Union[Image.Image, str, "os.PathLike[str]"],
    tilebox: TileBox,
) -> Image.Image:
    """Crop `tilebox` out of `image`.

    `image` may be an open PIL Image or a path. `tilebox` uses normalized
    coordinates (0..1); pixel coords are derived from the image size.

    The crop is returned as an RGB PIL Image.
    """
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")

    W, H = img.size
    left = int(round((tilebox.cx - tilebox.w / 2.0) * W))
    upper = int(round((tilebox.cy - tilebox.h / 2.0) * H))
    right = int(round((tilebox.cx + tilebox.w / 2.0) * W))
    lower = int(round((tilebox.cy + tilebox.h / 2.0) * H))

    # Clamp to image bounds; a degenerate box yields a 1px crop rather than error.
    left = max(0, min(left, W - 1))
    right = max(left + 1, min(right, W))
    upper = max(0, min(upper, H - 1))
    lower = max(upper + 1, min(lower, H))

    return img.crop((left, upper, right, lower))
