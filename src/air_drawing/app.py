"""Main loop: webcam -> hand tracking -> gestures -> canvas -> CNN (+ Gemini fallback)."""

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field

import cv2
import numpy as np

from air_drawing import config
from air_drawing.canvas import Canvas
from air_drawing.classifier import Classifier, Prediction
from air_drawing.gestures import Gesture, GestureDebouncer, classify_gesture
from air_drawing.hand_tracker import HandResult, HandTracker, draw_landmarks
from air_drawing.llm_fallback import GeminiFallback, LLMResult
from air_drawing.preprocess import preprocess_strokes, render_for_llm
from air_drawing.speech import Speaker, make_sentence
from air_drawing.ui import (
    ACTION_CLEAR,
    ACTION_COLOR,
    ACTION_MUTE,
    ACTION_PREDICT,
    ACTION_QUIT,
    ACTION_THICKNESS,
    ACTION_UNDO,
    MODE_DRAW,
    MODE_IDLE,
    MODE_PEN_UP,
    SOURCE_CNN,
    SOURCE_GEMINI,
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

logger = logging.getLogger(__name__)

MSG_NOTHING_DRAWN = "Nothing drawn"
MSG_NO_MODEL = "CNN model not loaded (run scripts/train.py)"
MSG_LLM_UNAVAILABLE = "LLM fallback unavailable"
MSG_ASKING_LLM = "Low confidence, asking Gemini..."
CONTROLS_HINT = (
    "Index: draw | Index+Middle: pen up | Open hand: clear | Fist: predict | "
    "Hover a button 1s to click | "
    f"Keys: {config.KEY_CLEAR} clear, {config.KEY_PREDICT} predict, {config.KEY_UNDO} undo, "
    f"{config.KEY_MUTE} mute, {config.KEY_QUIT} quit"
)


@dataclass
class PredictionState:
    """What the HUD shows about the last prediction."""

    cnn: list[Prediction] = field(default_factory=list)
    llm: LLMResult | None = None
    source: str | None = None
    message: str | None = None

    def card(self) -> CardContent:
        """Convert to the prediction card's content (Gemini label wins as the title)."""
        if self.llm is not None:
            title: str | None = self.llm.label
        else:
            title = self.cnn[0].label if self.cnn else None
        bars = [(p.label, p.confidence) for p in self.cnn]
        return CardContent(title=title, bars=bars, source=self.source, message=self.message)


class AirDrawingApp:
    """Ties tracking, gestures, drawing, recognition, speech and the toolbar together."""

    def __init__(
        self,
        tracker: HandTracker,
        classifier: Classifier | None,
        fallback: GeminiFallback | None,
        threshold: float = config.DEFAULT_CONFIDENCE_THRESHOLD,
        speaker: Speaker | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Store collaborators; classifier/fallback/speaker may be None if unavailable."""
        self._tracker = tracker
        self._classifier = classifier
        self._fallback = fallback
        self._threshold = threshold
        self._speaker = speaker
        self.canvas = Canvas()
        self._debouncer = GestureDebouncer()
        self._dwell = DwellClicker(clock=clock)
        self.state = PredictionState()
        self._pending: Future[LLMResult | None] | None = None
        self._toolbar: Toolbar | None = None
        self._hover_id: str | None = None
        self._mouse_hover_id: str | None = None
        self._fingertip: tuple[int, int] | None = None
        self.over_toolbar = False
        self.color_index = 0
        self.thickness_index = config.DEFAULT_THICKNESS_INDEX
        self.quit_requested = False
        self._apply_pen()

    # --- actions --------------------------------------------------------------

    @property
    def muted(self) -> bool:
        """True when voice output is off (or unavailable)."""
        return self._speaker is None or self._speaker.muted

    def clear(self) -> None:
        """Erase the drawing and the last prediction."""
        self.canvas.clear()
        self.state = PredictionState()
        self._pending = None  # any in-flight LLM answer is now stale and ignored
        logger.info("Canvas cleared")

    def undo(self) -> None:
        """Remove the last stroke and the now-outdated prediction."""
        if self.canvas.undo():
            self.state = PredictionState()
            self._pending = None
            logger.info("Removed last stroke")

    def toggle_mute(self) -> None:
        """Turn voice output on or off."""
        if self._speaker is None:
            logger.warning("Voice output unavailable")
            return
        self._speaker.toggle_mute()

    def select_color(self, index: int) -> None:
        """Choose the pen colour (display only)."""
        self.color_index = index % len(config.PALETTE)
        self._apply_pen()

    def select_thickness(self, index: int) -> None:
        """Choose the pen thickness (display only)."""
        self.thickness_index = index % len(config.PEN_THICKNESSES)
        self._apply_pen()

    def predict(self) -> None:
        """Classify the drawing with the CNN, asking Gemini if confidence is low."""
        self.canvas.pen_up()
        self._pending = None
        strokes = self.canvas.strokes
        tensor = preprocess_strokes(strokes)
        if tensor is None:
            self.state = PredictionState(message=MSG_NOTHING_DRAWN)
            return
        if self._classifier is None:
            self.state = PredictionState(message=MSG_NO_MODEL)
            return
        predictions = self._classifier.predict(tensor)
        self.state = PredictionState(cnn=predictions, source=SOURCE_CNN)
        top = predictions[0]
        logger.info("CNN: %s (%.0f%%)", top.label, top.confidence * config.PERCENT)
        if top.confidence >= self._threshold:
            self._speak(top.label)
            return
        image = render_for_llm(strokes)
        if self._fallback is None or not self._fallback.available or image is None:
            self.state.message = MSG_LLM_UNAVAILABLE
            self._speak(top.label)
            return
        # Speak only once Gemini has answered (or failed), so the user hears one label.
        self._pending = self._fallback.submit(image)
        self.state.message = MSG_ASKING_LLM

    def poll_llm(self) -> None:
        """Apply the Gemini answer once the background call has finished."""
        if self._pending is None or not self._pending.done():
            return
        result = self._pending.result()
        self._pending = None
        if result is None:
            self.state.message = MSG_LLM_UNAVAILABLE
            if self.state.cnn:
                self._speak(self.state.cnn[0].label)
        else:
            self.state.llm = result
            self.state.source = SOURCE_GEMINI
            self.state.message = None
            self._speak(result.label)

    def press(self, button: Button) -> None:
        """Run the action of a toolbar button."""
        logger.info("Button: %s", button.id)
        if button.action == ACTION_CLEAR:
            self.clear()
        elif button.action == ACTION_PREDICT:
            self.predict()
        elif button.action == ACTION_UNDO:
            self.undo()
        elif button.action == ACTION_MUTE:
            self.toggle_mute()
        elif button.action == ACTION_QUIT:
            self.quit_requested = True
        elif button.action == ACTION_COLOR and button.index is not None:
            self.select_color(button.index)
        elif button.action == ACTION_THICKNESS and button.index is not None:
            self.select_thickness(button.index)

    def handle_key(self, key: int) -> bool:
        """React to a key press; return False when the app should quit."""
        if key == ord(config.KEY_QUIT):
            return False
        if key == ord(config.KEY_CLEAR):
            self.clear()
        elif key == ord(config.KEY_PREDICT):
            self.predict()
        elif key == ord(config.KEY_UNDO):
            self.undo()
        elif key == ord(config.KEY_MUTE):
            self.toggle_mute()
        return not self.quit_requested

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """OpenCV mouse callback: hover highlights, left click presses a button."""
        if self._toolbar is None:
            return
        button = self._toolbar.hit_test((x, y))
        self._mouse_hover_id = button.id if button else None
        if event == cv2.EVENT_LBUTTONDOWN and button is not None:
            self.press(button)

    # --- per-frame --------------------------------------------------------------

    def update(self, hand: HandResult | None, toolbar: Toolbar | None = None) -> Gesture:
        """Advance gestures, air clicks and the canvas for one frame; return the stable gesture."""
        raw = classify_gesture(hand.landmarks if hand is not None else None)
        stable = self._debouncer.update(raw)
        self._fingertip = hand.pen if hand is not None else None
        self.over_toolbar = bool(toolbar and hand and toolbar.contains(hand.pen))
        if self.over_toolbar and toolbar is not None and hand is not None:
            # Never draw over the toolbar; hovering a button long enough clicks it.
            self.canvas.pen_up()
            button = toolbar.hit_test(hand.pen)
            self._hover_id = button.id if button else None
            fired = self._dwell.update(self._hover_id)
            if fired and button is not None:
                self.press(button)
        else:
            self._hover_id = None
            self._dwell.update(None)
            if stable == Gesture.DRAW and hand is not None:
                # Skip single-frame flickers without breaking the stroke.
                if raw == Gesture.DRAW:
                    self.canvas.add_point(hand.pen)
            else:
                self.canvas.pen_up()
        if self._debouncer.changed:
            if stable == Gesture.CLEAR:
                self.clear()
            elif stable == Gesture.PREDICT:
                self.predict()
        self.poll_llm()
        return stable

    def mode(self, gesture: Gesture) -> str:
        """Badge text for the current mode: Draw, Pen up or Idle."""
        if self.over_toolbar:
            return MODE_PEN_UP
        if gesture == Gesture.DRAW:
            return MODE_DRAW
        if gesture == Gesture.PEN_UP:
            return MODE_PEN_UP
        return MODE_IDLE

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Mirror the frame, run one update and draw the overlay."""
        frame = cv2.flip(frame, 1)
        width = frame.shape[1]
        if self._toolbar is None or self._toolbar.frame_width != width:
            self._toolbar = Toolbar(width)
        hand = self._tracker.detect(frame)
        gesture = self.update(hand, self._toolbar)
        self.canvas.render(frame)
        if hand is not None:
            draw_landmarks(frame, hand)
        self.draw_overlay(frame, gesture)
        return frame

    def draw_overlay(self, frame: np.ndarray, gesture: Gesture) -> None:
        """Draw toolbar, prediction card, mode badge, dwell ring and controls hint."""
        if self._toolbar is None:
            return
        toolbar_state = ToolbarState(
            hover_id=self._hover_id or self._mouse_hover_id,
            color_index=self.color_index,
            thickness_index=self.thickness_index,
            muted=self.muted,
        )
        draw_toolbar(frame, self._toolbar, toolbar_state)
        top = self._toolbar.height + config.CARD_Y_OFFSET
        draw_prediction_card(frame, self.state.card(), top)
        draw_mode_badges(frame, self.mode(gesture), self.muted, top)
        if self._fingertip is not None:
            draw_progress_ring(frame, self._fingertip, self._dwell.progress)
        bottom = frame.shape[0] - config.HUD_MARGIN
        _put_text(frame, CONTROLS_HINT, (config.HUD_MARGIN, bottom), config.HUD_TEXT_COLOR)

    def run(self) -> None:
        """Open the camera and loop until the user quits."""
        capture = cv2.VideoCapture(config.CAMERA_INDEX)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {config.CAMERA_INDEX}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            cv2.namedWindow(config.WINDOW_NAME)
            cv2.setMouseCallback(config.WINDOW_NAME, self.on_mouse)
            logger.info("Camera opened. %s", CONTROLS_HINT)
            while not self.quit_requested:
                ok, frame = capture.read()
                if not ok:
                    logger.warning("Failed to read frame from camera")
                    break
                cv2.imshow(config.WINDOW_NAME, self.process_frame(frame))
                key = cv2.waitKey(config.KEY_WAIT_MS) & config.KEY_MASK
                if not self.handle_key(key):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
            self._tracker.close()
            if self._fallback is not None:
                self._fallback.close()
            if self._speaker is not None:
                self._speaker.close()

    def _apply_pen(self) -> None:
        """Push the selected colour and thickness to the canvas."""
        self.canvas.set_style(
            color=config.PALETTE[self.color_index][1],
            thickness=config.PEN_THICKNESSES[self.thickness_index],
        )

    def _speak(self, label: str) -> None:
        """Say the result once per drawing (the canvas version identifies the drawing)."""
        if self._speaker is not None:
            self._speaker.say(make_sentence(label), key=self.canvas.version)


def _put_text(
    frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    """Draw text with a dark outline so it is readable on any background."""
    for col, extra in ((config.HUD_SHADOW_COLOR, config.HUD_THICKNESS * 2), (color, 0)):
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            config.HUD_HINT_FONT_SCALE,
            col,
            config.HUD_THICKNESS + extra,
            cv2.LINE_AA,
        )


def run() -> int:
    """Build the app with real collaborators and run it; return an exit code."""
    try:
        tracker = HandTracker()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    try:
        classifier: Classifier | None = Classifier.from_checkpoint()
    except FileNotFoundError as exc:
        logger.warning("%s Predictions are disabled.", exc)
        classifier = None
    fallback = GeminiFallback()
    if not fallback.available:
        logger.warning(
            "%s: set %s and %s in .env",
            MSG_LLM_UNAVAILABLE,
            config.GEMINI_API_KEY_ENV,
            config.GEMINI_MODEL_ENV,
        )
    threshold = config.get_confidence_threshold()
    logger.info("Confidence threshold: %.2f", threshold)
    speaker = Speaker()
    try:
        AirDrawingApp(tracker, classifier, fallback, threshold, speaker).run()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0
