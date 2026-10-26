"""Tests for toolbar hit-testing, air-click dwell timing and UI rendering."""

import cv2
import numpy as np
import pytest

from air_drawing import config
from air_drawing.ui import (
    ACTION_COLOR,
    ACTION_MODE,
    ACTION_THICKNESS,
    ELLIPSIS,
    FONT,
    RECOGNITION_TEXT,
    Button,
    CardContent,
    DwellClicker,
    Toolbar,
    ToolbarState,
    draw_debug_view,
    draw_mode_badges,
    draw_prediction_card,
    draw_progress_ring,
    draw_toolbar,
    fit_title,
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


# --- Mode button, badges and long titles ------------------------------------------


def test_toolbar_has_mode_button_that_fits() -> None:
    for width in (640, 1280):
        toolbar = Toolbar(width)
        mode = [b for b in toolbar.buttons if b.action == ACTION_MODE]
        assert len(mode) == 1
        assert toolbar.hit_test(mode[0].center) == mode[0]
        assert max(b.x + b.w for b in toolbar.buttons) <= width


def test_mode_button_color_changes_in_text_mode() -> None:
    toolbar = Toolbar(config.UI_DESIGN_WIDTH)
    mode = next(b for b in toolbar.buttons if b.action == ACTION_MODE)
    shape_img, text_img = frame(), frame()
    draw_toolbar(shape_img, toolbar, ToolbarState(text_mode=False))
    draw_toolbar(text_img, toolbar, ToolbarState(text_mode=True))
    x, y = mode.x + 4, mode.center[1]
    assert tuple(shape_img[y, x]) == config.BUTTON_COLORS["mode"]
    assert tuple(text_img[y, x]) == config.BUTTON_COLORS["mode_text"]


def test_fit_title_keeps_short_text() -> None:
    assert fit_title("cat", 1000) == ("cat", config.CARD_TITLE_SCALE)


def test_fit_title_shrinks_then_shortens_long_text() -> None:
    long_text = "the quick brown fox jumps over the lazy dog again and again"
    text, scale = fit_title(long_text, 300)
    assert config.CARD_MIN_TITLE_SCALE <= scale < config.CARD_TITLE_SCALE
    assert text.endswith(ELLIPSIS)
    assert len(text) <= config.CARD_MAX_TITLE_CHARS
    assert cv2.getTextSize(text, FONT, scale, config.CARD_TITLE_THICKNESS)[0][0] <= 300


def test_recognition_badge_is_drawn() -> None:
    plain, with_badge = frame(), frame()
    draw_mode_badges(plain, "Draw", muted=False, top=100)
    draw_mode_badges(with_badge, "Draw", muted=False, top=100, recognition=RECOGNITION_TEXT)
    assert (with_badge != 0).sum() > (plain != 0).sum()


def test_long_text_card_stays_inside_frame() -> None:
    img = frame(640, 480)
    draw_prediction_card(img, CardContent(title="w" * 200, source="Gemini"), top=100)
    assert img[:, -1].sum() == 0  # nothing drawn on the right edge


# --- Debug view and clear ring ------------------------------------------------------


def test_debug_view_shows_exact_pixels_scaled_up() -> None:
    img = frame()
    bitmap = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
    bitmap[3, 5] = 255
    draw_debug_view(img, bitmap)
    s = config.DEBUG_VIEW_SCALE
    size = config.IMAGE_SIZE * s
    x = img.shape[1] - config.DEBUG_VIEW_MARGIN - size
    y = img.shape[0] - config.DEBUG_VIEW_MARGIN - size - config.DEBUG_VIEW_LABEL_HEIGHT
    block = img[y + 3 * s : y + 4 * s, x + 5 * s : x + 6 * s]
    assert (block == 255).all()  # one CNN pixel = an s x s block (nearest neighbour)
    assert (img[y + s, x + s] == 0).all()


def test_debug_view_skips_tiny_frames() -> None:
    img = frame(100, 100)
    draw_debug_view(img, np.full((config.IMAGE_SIZE, config.IMAGE_SIZE), 255, np.uint8))
    assert not img.any()


def test_progress_ring_custom_color() -> None:
    img = frame()
    draw_progress_ring(img, (300, 300), 1.0, color=config.CLEAR_RING_COLOR, radius=40)
    assert (img == np.array(config.CLEAR_RING_COLOR, np.uint8)).all(axis=2).any()
