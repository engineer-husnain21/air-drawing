"""Tests for gesture classification and debouncing (synthetic landmarks, no webcam)."""

import numpy as np
import pytest

from air_drawing import config
from air_drawing.gestures import (
    Gesture,
    GestureDebouncer,
    HoldTimer,
    classify_gesture,
    fingers_up,
)

WRIST_XY = (100.0, 300.0)
FINGER_X = {"index": 100.0, "middle": 120.0, "ring": 140.0, "pinky": 160.0}
FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}  # MCP index; PIP, DIP, TIP follow


def make_hand(
    thumb: bool = False,
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
) -> np.ndarray:
    """Build a (21, 2) upright hand with the chosen fingers extended."""
    pts = np.zeros((config.NUM_LANDMARKS, 2), dtype=np.float32)
    pts[config.WRIST] = WRIST_XY
    # Thumb: CMC, MCP, IP, TIP
    pts[1] = (85.0, 270.0)
    pts[2] = (70.0, 250.0)
    pts[config.THUMB_IP] = (60.0, 230.0)
    pts[config.THUMB_TIP] = (20.0, 200.0) if thumb else (130.0, 215.0)
    extended = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, base in FINGER_BASE.items():
        x = FINGER_X[name]
        pts[base] = (x, 200.0)
        pts[base + 1] = (x, 150.0)
        if extended[name]:
            pts[base + 2] = (x, 100.0)
            pts[base + 3] = (x, 50.0)
        else:
            pts[base + 2] = (x, 190.0)
            pts[base + 3] = (x, 230.0)
    return pts


@pytest.mark.parametrize(
    ("fingers", "expected"),
    [
        ({"index": True}, Gesture.DRAW),
        ({"index": True, "thumb": True}, Gesture.DRAW),
        ({"index": True, "middle": True}, Gesture.PEN_UP),
        (
            {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
            Gesture.CLEAR,
        ),
        ({}, Gesture.PREDICT),
        ({"index": True, "middle": True, "ring": True}, Gesture.NONE),
        ({"pinky": True}, Gesture.NONE),
    ],
)
def test_classify_gesture(fingers: dict[str, bool], expected: Gesture) -> None:
    assert classify_gesture(make_hand(**fingers)) == expected


def test_no_hand_is_none() -> None:
    assert classify_gesture(None) == Gesture.NONE


def test_fingers_up_flags() -> None:
    assert fingers_up(make_hand(thumb=True, middle=True)) == (True, False, True, False, False)


def test_classification_is_rotation_invariant() -> None:
    hand = make_hand(index=True)
    angle = np.deg2rad(60)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = (hand - hand[config.WRIST]) @ rot.T + hand[config.WRIST]
    assert classify_gesture(rotated) == Gesture.DRAW


def test_debouncer_requires_consecutive_frames() -> None:
    debouncer = GestureDebouncer(frames=3)
    assert debouncer.update(Gesture.DRAW) == Gesture.NONE
    assert debouncer.update(Gesture.DRAW) == Gesture.NONE
    assert debouncer.update(Gesture.DRAW) == Gesture.DRAW
    assert debouncer.changed


def test_debouncer_ignores_flicker() -> None:
    debouncer = GestureDebouncer(frames=3)
    for gesture in (Gesture.DRAW, Gesture.DRAW, Gesture.CLEAR, Gesture.DRAW, Gesture.DRAW):
        debouncer.update(gesture)
    assert debouncer.stable == Gesture.NONE


def test_debouncer_changed_fires_once() -> None:
    debouncer = GestureDebouncer(frames=2)
    flags = [debouncer.update(Gesture.PREDICT) and debouncer.changed for _ in range(5)]
    assert flags == [False, True, False, False, False]


def test_debouncer_reset() -> None:
    debouncer = GestureDebouncer(frames=1)
    debouncer.update(Gesture.CLEAR)
    debouncer.reset()
    assert debouncer.stable == Gesture.NONE
    assert not debouncer.changed


def test_debouncer_rejects_invalid_frames() -> None:
    with pytest.raises(ValueError):
        GestureDebouncer(frames=0)


# --- Clear hold timer -----------------------------------------------------------------


def test_hold_timer_fires_after_hold_time() -> None:
    timer = HoldTimer(seconds=1.0)
    assert not timer.update(True, now=0.0)
    assert not timer.update(True, now=0.5)
    assert timer.progress == pytest.approx(0.5)
    assert not timer.update(True, now=0.9)
    assert timer.update(True, now=1.0)


def test_hold_timer_fires_once_per_hold() -> None:
    timer = HoldTimer(seconds=1.0)
    timer.update(True, now=0.0)
    assert timer.update(True, now=1.0)
    assert not timer.update(True, now=3.0)
    assert timer.progress == 0.0
    timer.update(False, now=3.1)
    timer.update(True, now=3.2)
    assert timer.update(True, now=4.25)


def test_hold_timer_release_resets() -> None:
    timer = HoldTimer(seconds=1.0)
    timer.update(True, now=0.0)
    timer.update(True, now=0.8)
    timer.update(False, now=0.85)
    assert timer.progress == 0.0
    timer.update(True, now=0.9)
    assert not timer.update(True, now=1.5)  # restarted at 0.9
    assert timer.update(True, now=1.95)


def test_hold_timer_uses_injected_clock_and_config_default() -> None:
    now = [5.0]
    timer = HoldTimer(clock=lambda: now[0])
    timer.update(True)
    now[0] += config.CLEAR_HOLD_SECONDS
    assert timer.update(True)


def test_hold_timer_rejects_invalid_seconds() -> None:
    with pytest.raises(ValueError):
        HoldTimer(seconds=0)
