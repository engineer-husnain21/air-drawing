"""Tests for toolbar hit-testing, air-click dwell timing and UI rendering."""

import numpy as np
import pytest

from air_drawing import config
from air_drawing.ui import (
    ACTION_COLOR,
    ACTION_THICKNESS,
    Button,
    CardContent,
    DwellClicker,
    Toolbar,
    ToolbarState,
    draw_mode_badges,
    draw_prediction_card,
    draw_progress_ring,
    draw_toolbar,
)

# --- Button / toolbar hit-testing -------------------------------------------------


def test_button_contains_edges() -> None:
    b = Button("undo", "Undo", x=10, y=20, w=30, h=40, color=(0, 0, 0))
    assert b.contains((10, 20))
    assert b.contains((39.9, 59.9))
    assert not b.contains((40, 30))  # right edge excluded
    assert not b.contains((20, 60))  # bottom edge excluded
    assert not b.contains((9, 30))
    assert b.center == (25, 40)


def test_button_ids() -> None:
    assert Button("quit", "Quit", 0, 0, 1, 1, (0, 0, 0)).id == "quit"
    assert Button(ACTION_COLOR, "red", 0, 0, 1, 1, (0, 0, 0), index=5).id == "color:5"


def test_toolbar_has_all_buttons() -> None:
    ids = [b.id for b in Toolbar(config.UI_DESIGN_WIDTH).buttons]
    assert ids[:5] == ["clear", "predict", "undo", "mute", "quit"]
    assert sum(i.startswith(ACTION_COLOR) for i in ids) == len(config.PALETTE) >= 6
    assert sum(i.startswith(ACTION_THICKNESS) for i in ids) == len(config.PEN_THICKNESSES)
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("width", [640, 960, 1280, 1920])
def test_hit_test_finds_every_button_at_its_centre(width: int) -> None:
    toolbar = Toolbar(width)
    for button in toolbar.buttons:
        hit = toolbar.hit_test(button.center)
        assert hit is not None and hit.id == button.id


@pytest.mark.parametrize("width", [640, 960, 1280, 1920])
def test_toolbar_fits_frame_and_buttons_do_not_overlap(width: int) -> None:
    toolbar = Toolbar(width)
    buttons = sorted(toolbar.buttons, key=lambda b: b.x)
    assert buttons[-1].x + buttons[-1].w <= width
    for left, right in zip(buttons, buttons[1:], strict=False):
        assert left.x + left.w <= right.x
    assert all(b.y >= 0 and b.y + b.h <= toolbar.height for b in buttons)


def test_hit_test_misses_gaps_and_canvas() -> None:
    toolbar = Toolbar(config.UI_DESIGN_WIDTH)
    first, second = toolbar.buttons[0], toolbar.buttons[1]
    gap_x = (first.x + first.w + second.x) / 2
    assert toolbar.hit_test((gap_x, first.center[1])) is None
    assert toolbar.hit_test((first.center[0], toolbar.height + 50)) is None


def test_toolbar_contains_strip_only() -> None:
    toolbar = Toolbar(config.UI_DESIGN_WIDTH)
    assert toolbar.contains((5, 0))
    assert toolbar.contains((5, toolbar.height - 1))
    assert not toolbar.contains((5, toolbar.height))


# --- Dwell (air click) timing -----------------------------------------------------


def test_dwell_fires_after_hold_time() -> None:
    dwell = DwellClicker(seconds=1.0)
    assert dwell.update("undo", now=0.0) is None
    assert dwell.update("undo", now=0.5) is None
    assert dwell.progress == pytest.approx(0.5)
    assert dwell.update("undo", now=0.99) is None
    assert dwell.update("undo", now=1.0) == "undo"


def test_dwell_fires_once_until_pointer_leaves() -> None:
    dwell = DwellClicker(seconds=1.0)
    dwell.update("undo", now=0.0)
    assert dwell.update("undo", now=1.0) == "undo"
    assert dwell.update("undo", now=2.5) is None
    assert dwell.progress == 0.0
    dwell.update(None, now=2.6)
    dwell.update("undo", now=3.0)
    assert dwell.update("undo", now=4.0) == "undo"


def test_dwell_restarts_when_moving_to_another_button() -> None:
    dwell = DwellClicker(seconds=1.0)
    dwell.update("clear", now=0.0)
    dwell.update("clear", now=0.8)
    assert dwell.update("undo", now=0.9) is None
    assert dwell.progress == 0.0
    assert dwell.update("undo", now=1.5) is None
    assert dwell.update("undo", now=1.95) == "undo"


def test_dwell_leaving_resets_progress() -> None:
    dwell = DwellClicker(seconds=1.0)
    dwell.update("mute", now=0.0)
    dwell.update("mute", now=0.7)
    dwell.update(None, now=0.8)
    assert dwell.progress == 0.0
    dwell.update("mute", now=0.9)
    assert dwell.update("mute", now=1.5) is None  # timer restarted at 0.9


def test_dwell_uses_injected_clock() -> None:
    now = [10.0]
    dwell = DwellClicker(seconds=1.0, clock=lambda: now[0])
    dwell.update("quit")
    now[0] = 11.0
    assert dwell.update("quit") == "quit"


def test_dwell_reset_and_validation() -> None:
    dwell = DwellClicker(seconds=1.0)
    dwell.update("quit", now=0.0)
    dwell.reset()
    assert dwell.progress == 0.0
    with pytest.raises(ValueError):
        DwellClicker(seconds=0)


# --- Rendering smoke tests ----------------------------------------------------------


def frame(width: int = 1280, height: int = 720) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize("width", [320, 1280])
def test_widgets_render_without_errors(width: int) -> None:
    img = frame(width)
    toolbar = Toolbar(width)
    draw_toolbar(img, toolbar, ToolbarState(hover_id="undo", muted=True))
    draw_progress_ring(img, (100, 200), 0.5)
    draw_prediction_card(
        img,
        CardContent("cat", [("cat", 0.7), ("tree", 0.2), ("sun", 0.1)], "Gemini", "note"),
        top=toolbar.height + config.CARD_Y_OFFSET,
    )
    draw_mode_badges(img, "Draw", muted=True, top=toolbar.height + config.CARD_Y_OFFSET)
    assert img[toolbar.height // 2].any()  # toolbar was drawn


def test_progress_ring_zero_draws_nothing() -> None:
    img = frame()
    draw_progress_ring(img, (100, 100), 0.0)
    assert not img.any()


def test_empty_card_draws_nothing() -> None:
    img = frame()
    draw_prediction_card(img, CardContent(), top=100)
    assert not img.any()
