"""On-screen UI: toolbar buttons, air-click dwell timer, prediction card and mode badge."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from air_drawing import config

Color = tuple[int, int, int]
Point = tuple[float, float]

ACTION_CLEAR = "clear"
ACTION_PREDICT = "predict"
ACTION_UNDO = "undo"
ACTION_MUTE = "mute"
ACTION_QUIT = "quit"
ACTION_COLOR = "color"
ACTION_THICKNESS = "thickness"
ACTION_MODE = "mode"

ACTION_BUTTONS: tuple[tuple[str, str], ...] = (
    (ACTION_CLEAR, "Clear"),
    (ACTION_PREDICT, "Predict"),
    (ACTION_UNDO, "Undo"),
    (ACTION_MUTE, "Mute"),
    (ACTION_QUIT, "Quit"),
    (ACTION_MODE, "Shape / Text"),
)
UNMUTE_LABEL = "Unmute"
MODE_DRAW = "Draw"
MODE_PEN_UP = "Pen up"
MODE_IDLE = "Idle"
MUTED_BADGE = "Muted"
RECOGNITION_SHAPE = "Shape"
RECOGNITION_TEXT = "Text"
RECOGNITION_BADGE_SUFFIX = " mode"
ELLIPSIS = "..."
SOURCE_CNN = "CNN"
SOURCE_GEMINI = "Gemini"
FONT = cv2.FONT_HERSHEY_SIMPLEX


# --- Buttons and layout -----------------------------------------------------------


@dataclass(frozen=True)
class Button:
    """A clickable rectangle. `index` selects a palette colour or a pen thickness."""

    action: str
    label: str
    x: int
    y: int
    w: int
    h: int
    color: Color
    index: int | None = None

    @property
    def id(self) -> str:
        """Unique identifier, e.g. "undo" or "color:3"."""
        return self.action if self.index is None else f"{self.action}:{self.index}"

    @property
    def center(self) -> tuple[int, int]:
        """Centre pixel of the button."""
        return self.x + self.w // 2, self.y + self.h // 2

    def contains(self, point: Point) -> bool:
        """True if the point lies inside the button (right/bottom edges excluded)."""
        px, py = point
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


class Toolbar:
    """A row of buttons along the top of the frame, scaled down to fit narrow frames."""

    def __init__(self, frame_width: int) -> None:
        """Lay out action buttons, palette swatches and thickness buttons."""
        self.frame_width = frame_width
        self.scale = min(1.0, frame_width / _design_width())
        self.height = self._s(config.TOOLBAR_HEIGHT)
        self.buttons: list[Button] = []
        x = config.TOOLBAR_PADDING
        for action, label in ACTION_BUTTONS:
            width = _button_width(action)
            self._add(action, label, x, width, config.BUTTON_HEIGHT, config.BUTTON_COLORS[action])
            x += width + config.TOOLBAR_GAP
        x += config.TOOLBAR_GROUP_GAP - config.TOOLBAR_GAP
        for i, (name, color) in enumerate(config.PALETTE):
            self._add(ACTION_COLOR, name, x, config.SWATCH_SIZE, config.SWATCH_SIZE, color, i)
            x += config.SWATCH_SIZE + config.TOOLBAR_GAP
        x += config.TOOLBAR_GROUP_GAP - config.TOOLBAR_GAP
        for i, thickness in enumerate(config.PEN_THICKNESSES):
            self._add(
                ACTION_THICKNESS,
                str(thickness),
                x,
                config.THICKNESS_BUTTON_WIDTH,
                config.SWATCH_SIZE,
                config.THICKNESS_BUTTON_COLOR,
                i,
            )
            x += config.THICKNESS_BUTTON_WIDTH + config.TOOLBAR_GAP

    def contains(self, point: Point) -> bool:
        """True if the point is anywhere on the toolbar strip."""
        return 0 <= point[1] < self.height

    def hit_test(self, point: Point) -> Button | None:
        """Return the button under the point, if any."""
        return next((b for b in self.buttons if b.contains(point)), None)

    def _s(self, value: float) -> int:
        """Scale a design-space length to frame pixels."""
        return int(round(value * self.scale))

    def _add(
        self,
        action: str,
        label: str,
        x: int,
        w: int,
        h: int,
        color: Color,
        index: int | None = None,
    ) -> None:
        """Append a vertically centred button given design-space geometry."""
        y = (config.TOOLBAR_HEIGHT - h) // 2
        self.buttons.append(
            Button(action, label, self._s(x), self._s(y), self._s(w), self._s(h), color, index)
        )


def _button_width(action: str) -> int:
    """Design width of an action button (the mode button is wider for its label)."""
    return config.MODE_BUTTON_WIDTH if action == ACTION_MODE else config.BUTTON_WIDTH


def _design_width() -> int:
    """Total toolbar width in design pixels."""
    actions = sum(_button_width(a) + config.TOOLBAR_GAP for a, _ in ACTION_BUTTONS)
    swatches = len(config.PALETTE) * (config.SWATCH_SIZE + config.TOOLBAR_GAP)
    thick = len(config.PEN_THICKNESSES) * (config.THICKNESS_BUTTON_WIDTH + config.TOOLBAR_GAP)
    gaps = 2 * (config.TOOLBAR_GROUP_GAP - config.TOOLBAR_GAP) - config.TOOLBAR_GAP
    return 2 * config.TOOLBAR_PADDING + actions + swatches + thick + gaps


# --- Air click ----------------------------------------------------------------------


class DwellClicker:
    """Fires a click after a target has been hovered continuously for `seconds`.

    After firing, the pointer must leave the target before it can fire again.
    """

    def __init__(
        self, seconds: float = config.DWELL_SECONDS, clock: Callable[[], float] = time.monotonic
    ) -> None:
        """Create a dwell timer; `clock` is injectable for tests."""
        if seconds <= 0:
            raise ValueError("seconds must be > 0")
        self._seconds = seconds
        self._clock = clock
        self._target: str | None = None
        self._start = 0.0
        self._fired = False
        self.progress = 0.0

    def update(self, target: str | None, now: float | None = None) -> str | None:
        """Report the hovered target id (or None); return the id when a click fires."""
        now = self._clock() if now is None else now
        if target != self._target:
            self._target, self._start, self._fired = target, now, False
        if self._target is None or self._fired:
            self.progress = 0.0
            return None
        self.progress = min((now - self._start) / self._seconds, 1.0)
        if self.progress >= 1.0:
            self._fired = True
            self.progress = 0.0
            return self._target
        return None

    def reset(self) -> None:
        """Cancel any dwell in progress."""
        self._target, self._fired, self.progress = None, False, 0.0


# --- Drawing primitives -----------------------------------------------------------


def lighten(color: Color, amount: float) -> Color:
    """Blend a colour towards white by `amount` in [0, 1]."""
    top = config.PIXEL_MAX
    return tuple(int(c + (top - c) * amount) for c in color)  # type: ignore[return-value]


def rounded_rect(
    img: np.ndarray, x: int, y: int, w: int, h: int, r: int, color: Color, thickness: int = -1
) -> None:
    """Draw a filled (thickness < 0) or outlined rounded rectangle in place."""
    r = max(0, min(r, w // 2, h // 2))
    corners = (
        (x + r, y + r, 180),
        (x + w - r - 1, y + r, 270),
        (x + w - r - 1, y + h - r - 1, 0),
        (x + r, y + h - r - 1, 90),
    )
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r - 1, y + h - 1), color, cv2.FILLED)
        cv2.rectangle(img, (x, y + r), (x + w - 1, y + h - r - 1), color, cv2.FILLED)
        for cx, cy, _ in corners:
            cv2.circle(img, (cx, cy), r, color, cv2.FILLED, cv2.LINE_AA)
        return
    cv2.line(img, (x + r, y), (x + w - r - 1, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + r, y + h - 1), (x + w - r - 1, y + h - 1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y + r), (x, y + h - r - 1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w - 1, y + r), (x + w - 1, y + h - r - 1), color, thickness, cv2.LINE_AA)
    quarter = config.FULL_CIRCLE_DEG / 4
    for cx, cy, angle in corners:
        cv2.ellipse(img, (cx, cy), (r, r), angle, 0, quarter, color, thickness, cv2.LINE_AA)


def translucent_rounded_rect(
    frame: np.ndarray, x: int, y: int, w: int, h: int, r: int, color: Color, alpha: float
) -> None:
    """Blend a filled rounded rectangle onto the frame (clipped to the frame)."""
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    rounded_rect(overlay, x - x0, y - y0, w, h, r, color)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def put_text_centered(
    img: np.ndarray, text: str, center: tuple[int, int], scale: float, color: Color, thickness: int
) -> None:
    """Draw text centred on a point."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    origin = (center[0] - tw // 2, center[1] + th // 2)
    cv2.putText(img, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def text_color_for(color: Color) -> Color:
    """Dark text on bright backgrounds, white text otherwise."""
    luminance = sum(w * c for w, c in zip(config.LUMA_WEIGHTS_BGR, color, strict=True))
    return (
        config.BUTTON_DARK_TEXT_COLOR
        if luminance > config.PIXEL_MAX / 2
        else config.BUTTON_TEXT_COLOR
    )


# --- Widgets ------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolbarState:
    """What the toolbar needs to highlight."""

    hover_id: str | None = None
    color_index: int = 0
    thickness_index: int = config.DEFAULT_THICKNESS_INDEX
    muted: bool = False
    text_mode: bool = False


def draw_toolbar(frame: np.ndarray, toolbar: Toolbar, state: ToolbarState) -> None:
    """Draw the translucent toolbar strip and all its buttons."""
    translucent_rounded_rect(
        frame,
        0,
        0,
        frame.shape[1],
        toolbar.height,
        0,
        config.TOOLBAR_BG_COLOR,
        config.TOOLBAR_BG_ALPHA,
    )
    radius = max(1, int(config.BUTTON_RADIUS * toolbar.scale))
    font_scale = config.BUTTON_FONT_SCALE * toolbar.scale
    for b in toolbar.buttons:
        hovered = b.id == state.hover_id
        base = b.color
        if b.action == ACTION_MODE and state.text_mode:
            base = config.BUTTON_COLORS["mode_text"]
        fill = lighten(base, config.BUTTON_HOVER_LIGHTEN) if hovered else base
        rounded_rect(frame, b.x, b.y, b.w, b.h, radius, fill)
        selected = (b.action == ACTION_COLOR and b.index == state.color_index) or (
            b.action == ACTION_THICKNESS and b.index == state.thickness_index
        )
        if selected:
            rounded_rect(
                frame,
                b.x,
                b.y,
                b.w,
                b.h,
                radius,
                config.BUTTON_SELECTED_COLOR,
                config.BUTTON_SELECTED_THICKNESS,
            )
        if b.action == ACTION_COLOR:
            continue
        if b.action == ACTION_THICKNESS and b.index is not None:
            dot = max(1, int(config.PEN_THICKNESSES[b.index] * toolbar.scale))
            cv2.circle(frame, b.center, dot, config.BUTTON_TEXT_COLOR, cv2.FILLED, cv2.LINE_AA)
            continue
        label = UNMUTE_LABEL if b.action == ACTION_MUTE and state.muted else b.label
        put_text_centered(
            frame, label, b.center, font_scale, text_color_for(fill), config.HUD_THICKNESS
        )


def draw_progress_ring(frame: np.ndarray, center: tuple[int, int], progress: float) -> None:
    """Draw a circular progress ring (0..1) around a point."""
    if progress <= 0:
        return
    r = config.PROGRESS_RING_RADIUS
    cv2.circle(
        frame,
        center,
        r,
        config.PROGRESS_RING_TRACK_COLOR,
        config.PROGRESS_RING_THICKNESS,
        cv2.LINE_AA,
    )
    end = config.RING_START_DEG + config.FULL_CIRCLE_DEG * min(progress, 1.0)
    cv2.ellipse(
        frame,
        center,
        (r, r),
        0,
        config.RING_START_DEG,
        end,
        config.PROGRESS_RING_COLOR,
        config.PROGRESS_RING_THICKNESS,
        cv2.LINE_AA,
    )


@dataclass(frozen=True)
class CardContent:
    """Text and bars for the prediction card."""

    title: str | None = None
    bars: list[tuple[str, float]] = field(default_factory=list)
    source: str | None = None
    message: str | None = None


def fit_title(text: str, max_width: int) -> tuple[str, float]:
    """Return (text, font scale) so the title fits `max_width` pixels.

    The font shrinks down to CARD_MIN_TITLE_SCALE first, then the text is shortened
    with an ellipsis. Very long text is capped at CARD_MAX_TITLE_CHARS.
    """
    if len(text) > config.CARD_MAX_TITLE_CHARS:
        text = text[: config.CARD_MAX_TITLE_CHARS - len(ELLIPSIS)] + ELLIPSIS

    def width(t: str, scale: float) -> int:
        return cv2.getTextSize(t, FONT, scale, config.CARD_TITLE_THICKNESS)[0][0]

    scale = config.CARD_TITLE_SCALE
    if width(text, scale) > max_width:
        scale = max(config.CARD_MIN_TITLE_SCALE, scale * max_width / width(text, scale))
    while width(text, scale) > max_width and len(text) > len(ELLIPSIS) + 1:
        text = text.removesuffix(ELLIPSIS)[:-1] + ELLIPSIS
    return text, scale


def draw_prediction_card(frame: np.ndarray, content: CardContent, top: int) -> None:
    """Draw a card: large top-1 label or text, confidence bars, source and message."""
    if content.title is None and not content.bars and content.message is None:
        return
    pad, row = config.CARD_PADDING, config.CARD_ROW_HEIGHT
    height = pad * 2 + len(content.bars) * row
    height += config.CARD_TITLE_HEIGHT if content.title else 0
    height += row if content.source else 0
    height += row if content.message else 0
    x = config.CARD_X
    max_w = max(config.CARD_WIDTH, frame.shape[1] - 2 * config.CARD_X)
    title, title_scale = fit_title(content.title or "", max_w - 2 * pad)
    title_w = cv2.getTextSize(title, FONT, title_scale, config.CARD_TITLE_THICKNESS)[0][0]
    w = min(max(config.CARD_WIDTH, title_w + 2 * pad), max_w)
    translucent_rounded_rect(
        frame, x, top, w, height, config.CARD_RADIUS, config.CARD_BG_COLOR, config.CARD_BG_ALPHA
    )
    y = top + pad
    if content.title:
        y += config.CARD_TITLE_HEIGHT
        cv2.putText(
            frame,
            title,
            (x + pad, y - pad // 2),
            FONT,
            title_scale,
            config.BUTTON_TEXT_COLOR,
            config.CARD_TITLE_THICKNESS,
            cv2.LINE_AA,
        )
    bar_w = w - 2 * pad
    for i, (label, conf) in enumerate(content.bars):
        color = config.CARD_BAR_COLORS[i % len(config.CARD_BAR_COLORS)]
        text = f"{label}  {conf * config.PERCENT:.0f}%"
        cv2.putText(
            frame,
            text,
            (x + pad, y + row // 2),
            FONT,
            config.CARD_TEXT_SCALE,
            config.BUTTON_TEXT_COLOR,
            config.CARD_TEXT_THICKNESS,
            cv2.LINE_AA,
        )
        bar_y = y + row - config.CARD_BAR_HEIGHT
        rounded_rect(
            frame,
            x + pad,
            bar_y,
            bar_w,
            config.CARD_BAR_HEIGHT,
            config.CARD_BAR_HEIGHT // 2,
            config.CARD_BAR_TRACK_COLOR,
        )
        filled = int(bar_w * min(max(conf, 0.0), 1.0))
        if filled > 0:
            rounded_rect(
                frame,
                x + pad,
                bar_y,
                filled,
                config.CARD_BAR_HEIGHT,
                config.CARD_BAR_HEIGHT // 2,
                color,
            )
        y += row
    if content.source:
        color = (
            config.CARD_GEMINI_COLOR if content.source == SOURCE_GEMINI else config.CARD_CNN_COLOR
        )
        y += row
        cv2.putText(
            frame,
            f"via {content.source}",
            (x + pad, y - pad // 2),
            FONT,
            config.CARD_TEXT_SCALE,
            color,
            config.HUD_THICKNESS,
            cv2.LINE_AA,
        )
    if content.message:
        y += row
        cv2.putText(
            frame,
            content.message,
            (x + pad, y - pad // 2),
            FONT,
            config.CARD_TEXT_SCALE,
            config.HUD_WARNING_COLOR,
            config.HUD_THICKNESS,
            cv2.LINE_AA,
        )


def draw_badge(frame: np.ndarray, text: str, color: Color, right: int, top: int) -> int:
    """Draw a pill badge whose right edge is at `right`; return its left edge."""
    (tw, _), _ = cv2.getTextSize(text, FONT, config.BADGE_FONT_SCALE, config.HUD_THICKNESS)
    w = tw + 2 * config.BADGE_PADDING_X
    x = right - w
    rounded_rect(frame, x, top, w, config.BADGE_HEIGHT, config.BADGE_RADIUS, color)
    put_text_centered(
        frame,
        text,
        (x + w // 2, top + config.BADGE_HEIGHT // 2),
        config.BADGE_FONT_SCALE,
        text_color_for(color),
        config.HUD_THICKNESS,
    )
    return x


def draw_mode_badges(
    frame: np.ndarray, mode: str, muted: bool, top: int, recognition: str | None = None
) -> None:
    """Draw the drawing-mode badge, the Shape/Text badge and a "Muted" badge (top right)."""
    right = frame.shape[1] - config.HUD_MARGIN
    color = config.BADGE_COLORS.get(mode, config.BADGE_COLORS[MODE_IDLE])
    left = draw_badge(frame, mode, color, right, top)
    if recognition is not None:
        rec_color = config.RECOGNITION_BADGE_COLORS[recognition]
        text = recognition + RECOGNITION_BADGE_SUFFIX
        left = draw_badge(frame, text, rec_color, left - config.TOOLBAR_GAP, top)
    if muted:
        draw_badge(frame, MUTED_BADGE, config.MUTED_BADGE_COLOR, left - config.TOOLBAR_GAP, top)
