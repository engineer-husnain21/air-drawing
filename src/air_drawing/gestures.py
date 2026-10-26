"""Gesture recognition from hand landmarks: draw, pen up, clear, predict."""

import time
from collections.abc import Callable
from enum import Enum

import numpy as np

from air_drawing import config

# (tip, pip) landmark pairs for the four non-thumb fingers.
_FINGERS: tuple[tuple[int, int], ...] = (
    (config.INDEX_TIP, config.INDEX_PIP),
    (config.MIDDLE_TIP, config.MIDDLE_PIP),
    (config.RING_TIP, config.RING_PIP),
    (config.PINKY_TIP, config.PINKY_PIP),
)


class Gesture(Enum):
    """Recognised hand gestures."""

    NONE = "none"
    DRAW = "draw"
    PEN_UP = "pen up"
    CLEAR = "clear"
    PREDICT = "predict"


def fingers_up(landmarks: np.ndarray) -> tuple[bool, bool, bool, bool, bool]:
    """Return (thumb, index, middle, ring, pinky) extension flags.

    Uses distance ratios instead of y-comparisons so it works for tilted hands.
    """
    wrist = landmarks[config.WRIST]

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    others = [
        dist(landmarks[tip], wrist) > config.FINGER_EXTENDED_RATIO * dist(landmarks[pip], wrist)
        for tip, pip in _FINGERS
    ]
    pinky_mcp = landmarks[config.PINKY_MCP]
    thumb = dist(landmarks[config.THUMB_TIP], pinky_mcp) > config.THUMB_EXTENDED_RATIO * dist(
        landmarks[config.THUMB_IP], pinky_mcp
    )
    return thumb, others[0], others[1], others[2], others[3]


def classify_gesture(landmarks: np.ndarray | None) -> Gesture:
    """Map a single frame's landmarks to a gesture (no debouncing)."""
    if landmarks is None:
        return Gesture.NONE
    thumb, index, middle, ring, pinky = fingers_up(landmarks)
    if thumb and index and middle and ring and pinky:
        return Gesture.CLEAR
    if not (index or middle or ring or pinky):
        return Gesture.PREDICT
    if index and not (middle or ring or pinky):
        return Gesture.DRAW
    if index and middle and not (ring or pinky):
        return Gesture.PEN_UP
    return Gesture.NONE


class GestureDebouncer:
    """Only accepts a gesture after it has been seen for N consecutive frames."""

    def __init__(self, frames: int = config.GESTURE_DEBOUNCE_FRAMES) -> None:
        """Create a debouncer requiring `frames` identical frames to switch."""
        if frames < 1:
            raise ValueError("frames must be >= 1")
        self._frames = frames
        self._candidate = Gesture.NONE
        self._count = 0
        self.stable = Gesture.NONE
        self.changed = False

    def update(self, gesture: Gesture) -> Gesture:
        """Feed one frame's gesture; return the current stable gesture.

        `changed` is True only on the frame where the stable gesture switches,
        so one-shot actions (clear, predict) fire once per gesture.
        """
        if gesture == self._candidate:
            self._count += 1
        else:
            self._candidate = gesture
            self._count = 1
        self.changed = self._count >= self._frames and gesture != self.stable
        if self.changed:
            self.stable = gesture
        return self.stable

    def reset(self) -> None:
        """Forget all history."""
        self._candidate = Gesture.NONE
        self._count = 0
        self.stable = Gesture.NONE
        self.changed = False


class HoldTimer:
    """Fires once after a condition has stayed true for `seconds`.

    Used for the open-hand CLEAR gesture so a brief open hand while drawing does
    not wipe the canvas. The condition must become false before it can fire again.
    """

    def __init__(
        self,
        seconds: float = config.CLEAR_HOLD_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a timer; `clock` is injectable for tests."""
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self._seconds = seconds
        self._clock = clock
        self._start: float | None = None
        self._fired = False
        self.progress = 0.0

    def update(self, active: bool, now: float | None = None) -> bool:
        """Report whether the condition holds this frame; return True when it fires."""
        now = self._clock() if now is None else now
        if not active:
            self.reset()
            return False
        if self._start is None:
            self._start = now
        if self._fired:
            self.progress = 0.0
            return False
        self.progress = min((now - self._start) / self._seconds, 1.0)
        if self.progress >= 1.0:
            self._fired = True
            self.progress = 0.0
            return True
        return False

    def reset(self) -> None:
        """Cancel the hold."""
        self._start, self._fired, self.progress = None, False, 0.0
