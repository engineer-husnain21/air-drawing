"""Stroke storage, exponential smoothing and rendering of the air drawing."""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from air_drawing import config

Point = tuple[float, float]
Stroke = list[Point]


class PointSmoother:
    """Exponential moving average over 2D points."""

    def __init__(self, alpha: float = config.SMOOTHING_ALPHA) -> None:
        """Create a smoother; alpha is the weight of the newest point, in (0, 1]."""
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._state: Point | None = None

    def update(self, point: Point) -> Point:
        """Return the smoothed point after observing `point`."""
        if self._state is None:
            self._state = point
        else:
            a = self._alpha
            self._state = (
                a * point[0] + (1 - a) * self._state[0],
                a * point[1] + (1 - a) * self._state[1],
            )
        return self._state

    def reset(self) -> None:
        """Forget the previous point (e.g. when the pen is lifted)."""
        self._state = None


Color = tuple[int, int, int]


@dataclass(frozen=True)
class StrokeStyle:
    """Display-only look of a stroke; never used for recognition."""

    color: Color = config.CANVAS_STROKE_COLOR
    thickness: int = config.CANVAS_STROKE_THICKNESS


class Canvas:
    """Holds strokes drawn with the fingertip and renders them on frames.

    Each stroke keeps its own display style, but `strokes` returns only the raw
    points, so pen colour and thickness never affect preprocessing.
    """

    def __init__(
        self,
        smoother: PointSmoother | None = None,
        min_distance: float = config.MIN_POINT_DISTANCE_PX,
    ) -> None:
        """Create an empty canvas."""
        self._smoother = smoother or PointSmoother()
        self._min_distance = min_distance
        self._strokes: list[Stroke] = []
        self._styles: list[StrokeStyle] = []
        self._drawing = False
        self.style = StrokeStyle()
        self.version = 0  # bumped on every change; identifies "the same drawing"

    @property
    def strokes(self) -> list[Stroke]:
        """Copy of all non-empty strokes."""
        return [list(stroke) for stroke in self._strokes if stroke]

    @property
    def is_empty(self) -> bool:
        """True when nothing has been drawn."""
        return not any(self._strokes)

    @property
    def is_drawing(self) -> bool:
        """True while a stroke is in progress."""
        return self._drawing

    def set_style(self, color: Color | None = None, thickness: int | None = None) -> None:
        """Change the pen used for the next strokes (existing strokes keep theirs)."""
        self.style = StrokeStyle(
            color=color if color is not None else self.style.color,
            thickness=thickness if thickness is not None else self.style.thickness,
        )

    def add_point(self, point: Point) -> None:
        """Add a fingertip point, starting a new stroke if the pen was up."""
        if not self._drawing:
            self._smoother.reset()
            self._strokes.append([])
            self._styles.append(self.style)
            self._drawing = True
        smoothed = self._smoother.update(point)
        stroke = self._strokes[-1]
        if stroke and math.dist(stroke[-1], smoothed) < self._min_distance:
            return
        stroke.append(smoothed)
        self.version += 1

    def pen_up(self) -> None:
        """End the current stroke."""
        self._drawing = False
        self._smoother.reset()
        if self._strokes and not self._strokes[-1]:
            self._strokes.pop()
            self._styles.pop()

    def undo(self) -> bool:
        """Remove the last stroke; return False if there was nothing to remove."""
        self.pen_up()
        if not self._strokes:
            return False
        self._strokes.pop()
        self._styles.pop()
        self.version += 1
        return True

    def clear(self) -> None:
        """Remove all strokes."""
        self._strokes.clear()
        self._styles.clear()
        self.pen_up()
        self.version += 1

    def render(
        self,
        frame: np.ndarray,
        color: Color | None = None,
        thickness: int | None = None,
    ) -> np.ndarray:
        """Draw all strokes onto `frame` in place and return it.

        Each stroke uses its own style unless `color` / `thickness` override it.
        """
        for stroke, style in zip(self._strokes, self._styles, strict=True):
            draw_strokes(
                frame,
                [stroke],
                color if color is not None else style.color,
                thickness if thickness is not None else style.thickness,
            )
        return frame


def draw_strokes(
    image: np.ndarray,
    strokes: list[Stroke],
    color: int | tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw strokes as anti-aliased polylines (single points become dots)."""
    for stroke in strokes:
        if not stroke:
            continue
        pts = np.round(np.asarray(stroke, dtype=np.float64)).astype(np.int32)
        if len(pts) == 1:
            center = (int(pts[0][0]), int(pts[0][1]))
            cv2.circle(image, center, max(thickness // 2, 1), color, cv2.FILLED, cv2.LINE_AA)
        else:
            cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA)
