"""Tests for converting strokes into Quick, Draw!-format tensors."""

import math

import numpy as np
import pytest
import torch

from air_drawing import config
from air_drawing.preprocess import (
    bitmaps_to_tensor,
    preprocess_strokes,
    render_for_llm,
    render_text_for_llm,
    strokes_to_bitmap,
)


def circle(cx: float, cy: float, r: float, n: int = 60) -> list[tuple[float, float]]:
    return [
        (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n + 1)
    ]


def test_empty_drawing_returns_none() -> None:
    assert preprocess_strokes([]) is None
    assert preprocess_strokes([[]]) is None
    assert strokes_to_bitmap([]) is None
    assert render_for_llm([]) is None


def test_tensor_shape_dtype_and_range() -> None:
    tensor = preprocess_strokes([circle(300, 200, 80)])
    assert tensor is not None
    assert tensor.shape == (1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    assert 0.0 <= tensor.min() and tensor.max() <= 1.0
    assert tensor.max() > 0.9  # strokes are white


def test_black_background_white_strokes() -> None:
    bitmap = strokes_to_bitmap([circle(300, 200, 80)])
    assert bitmap is not None
    assert bitmap.dtype == np.uint8
    assert bitmap[0, 0] == 0  # corner is background
    assert bitmap[config.IMAGE_SIZE // 2, config.IMAGE_SIZE // 2] == 0  # circle is hollow


def test_drawing_is_centered() -> None:
    bitmap = strokes_to_bitmap([circle(900, 50, 30)])  # far off-centre on the canvas
    assert bitmap is not None
    ys, xs = np.nonzero(bitmap)
    weights = bitmap[ys, xs].astype(float)
    centre = (config.IMAGE_SIZE - 1) / 2
    assert abs(np.average(xs, weights=weights) - centre) < 1.0
    assert abs(np.average(ys, weights=weights) - centre) < 1.0


def test_position_and_scale_invariance() -> None:
    small = strokes_to_bitmap([circle(100, 100, 20)])
    large = strokes_to_bitmap([circle(600, 400, 200)])
    assert small is not None and large is not None
    diff = np.abs(small.astype(int) - large.astype(int)).mean()
    assert diff < 5


def test_aspect_ratio_is_kept() -> None:
    bitmap = strokes_to_bitmap([[(0, 0), (400, 0)]])  # a horizontal line
    assert bitmap is not None
    rows = np.nonzero(bitmap.max(axis=1))[0]
    cols = np.nonzero(bitmap.max(axis=0))[0]
    assert len(cols) > config.IMAGE_SIZE * 0.7
    assert len(rows) <= 4  # stays a thin line instead of being stretched


def test_strokes_survive_resizing() -> None:
    bitmap = strokes_to_bitmap([[(0, 0), (100, 100)]])
    assert bitmap is not None
    assert bitmap.max() > 200


def test_single_point_is_not_empty() -> None:
    tensor = preprocess_strokes([[(50, 50)]])
    assert tensor is not None
    assert tensor.max() > 0


def test_bitmaps_to_tensor_matches_training_scaling() -> None:
    raw = np.array([[0] * 783 + [255]], dtype=np.uint8)
    tensor = bitmaps_to_tensor(raw)
    assert tensor.shape == (1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert tensor.max().item() == 1.0 and tensor.min().item() == 0.0


def test_llm_image_is_black_on_white() -> None:
    image = render_for_llm([circle(200, 200, 50)])
    assert image is not None
    assert image.shape == (config.LLM_IMAGE_SIZE, config.LLM_IMAGE_SIZE)
    assert image[0, 0] == 255
    assert image.min() == 0


def test_text_image_keeps_aspect_ratio_and_is_black_on_white() -> None:
    word = [[(0, 0), (0, 100)], [(50, 0), (50, 100)], [(400, 0), (400, 100)]]  # wide
    image = render_text_for_llm(word)
    assert image is not None
    height, width = image.shape
    assert width == config.TEXT_IMAGE_LONG_SIDE
    inner_w = width - 2 * config.TEXT_IMAGE_PADDING
    inner_h = height - 2 * config.TEXT_IMAGE_PADDING
    assert inner_w / inner_h == pytest.approx(400 / 100, rel=0.02)
    assert image[0, 0] == 255 and image.min() == 0


@pytest.mark.parametrize("scale", [0.2, 1.0, 5.0])
def test_text_stroke_width_is_fixed_in_output_pixels(scale: float) -> None:
    stroke = [[(0, 0), (0, 300 * scale)]]  # a vertical bar
    image = render_text_for_llm(stroke)
    assert image is not None
    middle_row = image[image.shape[0] // 2]
    ink = int((middle_row < 128).sum())
    assert abs(ink - config.TEXT_STROKE_THICKNESS) <= 2


def test_text_image_empty_is_none() -> None:
    assert render_text_for_llm([]) is None
    assert render_text_for_llm([[]]) is None
