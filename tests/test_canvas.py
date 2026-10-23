"""Tests for stroke storage, smoothing and rendering."""

import numpy as np
import pytest

from air_drawing.canvas import Canvas, PointSmoother, draw_strokes
from air_drawing.preprocess import preprocess_strokes


def test_smoother_first_point_passthrough() -> None:
    assert PointSmoother(alpha=0.5).update((10.0, 20.0)) == (10.0, 20.0)


def test_smoother_exponential_average() -> None:
    smoother = PointSmoother(alpha=0.5)
    smoother.update((0.0, 0.0))
    assert smoother.update((10.0, 20.0)) == pytest.approx((5.0, 10.0))


def test_smoother_reset() -> None:
    smoother = PointSmoother(alpha=0.5)
    smoother.update((0.0, 0.0))
    smoother.reset()
    assert smoother.update((10.0, 10.0)) == (10.0, 10.0)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_smoother_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        PointSmoother(alpha=alpha)


def make_canvas() -> Canvas:
    return Canvas(smoother=PointSmoother(alpha=1.0), min_distance=0.0)


def test_canvas_starts_empty() -> None:
    canvas = make_canvas()
    assert canvas.is_empty
    assert canvas.strokes == []


def test_pen_up_splits_strokes() -> None:
    canvas = make_canvas()
    canvas.add_point((0, 0))
    canvas.add_point((10, 0))
    canvas.pen_up()
    canvas.add_point((50, 50))
    assert canvas.strokes == [[(0, 0), (10, 0)], [(50, 50)]]
    assert not canvas.is_empty


def test_repeated_pen_up_creates_no_empty_strokes() -> None:
    canvas = make_canvas()
    canvas.pen_up()
    canvas.pen_up()
    assert canvas.is_empty
    assert not canvas.is_drawing


def test_clear_removes_everything() -> None:
    canvas = make_canvas()
    canvas.add_point((1, 1))
    canvas.clear()
    assert canvas.is_empty
    assert not canvas.is_drawing


def test_min_distance_filters_jitter() -> None:
    canvas = Canvas(smoother=PointSmoother(alpha=1.0), min_distance=5.0)
    for point in [(0, 0), (1, 1), (2, 0), (10, 0)]:
        canvas.add_point(point)
    assert canvas.strokes == [[(0, 0), (10, 0)]]


def test_smoothing_resets_between_strokes() -> None:
    canvas = Canvas(smoother=PointSmoother(alpha=0.5), min_distance=0.0)
    canvas.add_point((0, 0))
    canvas.pen_up()
    canvas.add_point((100, 100))
    assert canvas.strokes[1][0] == (100, 100)


def test_render_draws_pixels() -> None:
    canvas = make_canvas()
    canvas.add_point((10, 10))
    canvas.add_point((90, 90))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas.render(frame, color=(255, 255, 255), thickness=3)
    assert frame[50, 50].sum() > 0
    assert frame[10, 90].sum() == 0


def test_draw_single_point_stroke_as_dot() -> None:
    image = np.zeros((50, 50), dtype=np.uint8)
    draw_strokes(image, [[(25, 25)]], color=255, thickness=6)
    assert image[25, 25] > 0


def test_undo_removes_last_stroke() -> None:
    canvas = make_canvas()
    canvas.add_point((0, 0))
    canvas.pen_up()
    canvas.add_point((5, 5))
    assert canvas.undo()
    assert canvas.strokes == [[(0, 0)]]
    assert canvas.undo()
    assert not canvas.undo()
    assert canvas.is_empty


def test_undo_ends_stroke_in_progress() -> None:
    canvas = make_canvas()
    canvas.add_point((0, 0))
    canvas.undo()
    assert not canvas.is_drawing
    canvas.add_point((9, 9))
    assert canvas.strokes == [[(9, 9)]]


def test_version_changes_on_every_edit() -> None:
    canvas = make_canvas()
    versions = [canvas.version]
    canvas.add_point((0, 0))
    versions.append(canvas.version)
    canvas.pen_up()
    versions.append(canvas.version)  # pen up alone is not an edit
    canvas.undo()
    versions.append(canvas.version)
    canvas.clear()
    versions.append(canvas.version)
    assert versions[1] > versions[0]
    assert versions[2] == versions[1]
    assert versions[4] > versions[3] > versions[2]


def test_each_stroke_keeps_its_own_style() -> None:
    canvas = make_canvas()
    canvas.set_style(color=(0, 0, 255), thickness=3)
    canvas.add_point((10, 10))
    canvas.add_point((40, 10))
    canvas.pen_up()
    canvas.set_style(color=(255, 0, 0))
    canvas.add_point((10, 60))
    canvas.add_point((40, 60))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    canvas.render(frame)
    assert tuple(frame[10, 25]) == (0, 0, 255)
    assert tuple(frame[60, 25]) == (255, 0, 0)
    assert canvas.style.thickness == 3


def test_style_does_not_change_points_or_preprocessing() -> None:
    plain, styled = make_canvas(), make_canvas()
    styled.set_style(color=(0, 255, 0), thickness=20)
    for canvas in (plain, styled):
        for point in [(0, 0), (50, 80), (100, 0)]:
            canvas.add_point(point)
    assert plain.strokes == styled.strokes
    a, b = preprocess_strokes(plain.strokes), preprocess_strokes(styled.strokes)
    assert a is not None and b is not None
    assert bool((a == b).all())
