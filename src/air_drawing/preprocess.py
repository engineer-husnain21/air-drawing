"""Conversion of drawings into the Quick, Draw! 28x28 tensor format.

Quick, Draw! bitmaps are 28x28, black background (0) and white strokes (255).
Pipeline: bounding box -> square with padding (aspect ratio kept) -> centre ->
thick strokes at high resolution -> resize to 28x28 -> scale to [0, 1].
"""

import cv2
import numpy as np
import torch

from air_drawing import config
from air_drawing.canvas import Stroke, draw_strokes


def bitmaps_to_tensor(bitmaps: np.ndarray) -> torch.Tensor:
    """Convert uint8 bitmaps (N, 784) or (N, 28, 28) to float (N, 1, 28, 28) in [0, 1].

    This is the single scaling step shared by training and inference.
    """
    arr = np.asarray(bitmaps, dtype=np.float32).reshape(-1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    return torch.from_numpy(arr / config.PIXEL_MAX)


def render_strokes_square(
    strokes: list[Stroke],
    size: int = config.RENDER_SIZE,
    padding_ratio: float = config.RENDER_PADDING_RATIO,
    thickness: int = config.RENDER_STROKE_THICKNESS,
) -> np.ndarray | None:
    """Render strokes white-on-black into a centred `size` x `size` uint8 image.

    The drawing is cropped to its bounding box, scaled uniformly so its longest side
    fills the area inside the padding, and centred. Returns None if nothing is drawn.
    """
    non_empty = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) > 0]
    if not non_empty:
        return None
    points = np.concatenate(non_empty)
    lo, hi = points.min(axis=0), points.max(axis=0)
    extent = max(float((hi - lo).max()), 1.0)
    # Leave room for padding and half the stroke width on each side so nothing is clipped.
    inner = size * (1.0 - 2.0 * padding_ratio) - thickness
    scale = inner / extent
    offset = (size - (hi - lo) * scale) / 2.0 - lo * scale
    scaled = [list(map(tuple, s * scale + offset)) for s in non_empty]
    image = np.zeros((size, size), dtype=np.uint8)
    draw_strokes(image, scaled, color=int(config.PIXEL_MAX), thickness=thickness)
    return image


def strokes_to_bitmap(strokes: list[Stroke]) -> np.ndarray | None:
    """Return a 28x28 uint8 Quick, Draw!-style bitmap, or None for an empty drawing."""
    image = render_strokes_square(strokes)
    if image is None:
        return None
    return cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def preprocess_strokes(strokes: list[Stroke]) -> torch.Tensor | None:
    """Return a (1, 1, 28, 28) float tensor in [0, 1], or None for an empty drawing."""
    bitmap = strokes_to_bitmap(strokes)
    if bitmap is None:
        return None
    return bitmaps_to_tensor(bitmap)


def render_for_llm(strokes: list[Stroke]) -> np.ndarray | None:
    """Render a larger black-on-white image for a vision LLM, or None if empty."""
    image = render_strokes_square(strokes, size=config.LLM_IMAGE_SIZE)
    if image is None:
        return None
    return cv2.bitwise_not(image)
