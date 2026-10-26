"""Main loop: webcam -> hand tracking -> gestures -> canvas -> CNN (+ Gemini fallback)."""

import logging
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from air_drawing import config
from air_drawing.canvas import Canvas
from air_drawing.classifier import Classifier, Prediction
from air_drawing.gestures import Gesture, GestureDebouncer, HoldTimer, classify_gesture
from air_drawing.hand_tracker import HandResult, HandTracker, draw_landmarks
from air_drawing.llm_fallback import GeminiFallback, LLMResult, TextResult
from air_drawing.preprocess import (
    bitmaps_to_tensor,
    render_for_llm,
    render_text_for_llm,
    save_debug_bitmap,
    strokes_to_bitmap,
)
from air_drawing.speech import PhrasePicker, Speaker, text_sentence
from air_drawing.ui import (
    ACTION_CLEAR,
    ACTION_COLOR,
    ACTION_MODE,
    ACTION_MUTE,
    ACTION_PREDICT,
    ACTION_QUIT,
    ACTION_THICKNESS,
    ACTION_UNDO,
    MODE_DRAW,
    MODE_IDLE,
    MODE_PEN_UP,
    RECOGNITION_SHAPE,
    RECOGNITION_TEXT,
    SOURCE_CNN,
    SOURCE_GEMINI,
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
)

logger = logging.getLogger(__name__)

MSG_NOTHING_DRAWN = "Nothing drawn"
MSG_NO_MODEL = "CNN model not loaded (run scripts/train.py)"
MSG_LLM_UNAVAILABLE = "LLM fallback unavailable"
MSG_ASKING_LLM = "Low confidence, asking Gemini..."
MSG_TEXT_NEEDS_GEMINI = "Text mode needs Gemini"
MSG_READING_TEXT = "Reading text with Gemini..."
MSG_NO_TEXT = "No text recognised"
TEXT_CONFIDENCE_LABEL = "confidence"
CONTROLS_HINT = (
    "Index: draw | Index+Middle: pen up | Open hand: clear | Fist: predict | "
    "Hover a button 1s to click | "
    f"Keys: {config.KEY_CLEAR} clear, {config.KEY_PREDICT} predict, {config.KEY_UNDO} undo, "
    f"{config.KEY_MUTE} mute, {config.KEY_TEXT_MODE} shape/text, {config.KEY_DEBUG} debug, "
    f"{config.KEY_QUIT} quit"
)


@dataclass
class PredictionState:
    """What the HUD shows about the last prediction."""

    cnn: list[Prediction] = field(default_factory=list)
    llm: LLMResult | None = None
    text: TextResult | None = None
    source: str | None = None
    message: str | None = None

    def card(self) -> CardContent:
        """Convert to the prediction card's content (Gemini label wins as the title)."""
        if self.text is not None:
            conf = self.text.confidence
            bars = [(TEXT_CONFIDENCE_LABEL, conf)] if conf is not None else []
            return CardContent(
                title=self.text.text or None, bars=bars, source=self.source, message=self.message
            )
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
        debug_dir: Path = config.DEBUG_DIR,
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
        self._clear_hold = HoldTimer(clock=clock)
        self._palm: tuple[int, int] | None = None
        self.debug = False
        self._debug_dir = debug_dir
        self.debug_bitmap: np.ndarray | None = None  # last exact CNN input (uint8 28x28)
        self.last_debug_path: Path | None = None
        self.state = PredictionState()
        self._pending: Future[LLMResult | None] | Future[TextResult | None] | None = None
        self._pending_text = False  # True when the pending call reads handwriting
        self.recognition = RECOGNITION_SHAPE
        self._phrases = PhrasePicker()
        self._last_spoken: tuple[int, str, str] | None = None
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

    @property
    def text_mode(self) -> bool:
        """True when drawings are read as handwritten text by Gemini."""
        return self.recognition == RECOGNITION_TEXT

    def toggle_recognition(self) -> None:
        """Switch between Shape mode (CNN + fallback) and Text mode (Gemini reads text)."""
        self.recognition = RECOGNITION_SHAPE if self.text_mode else RECOGNITION_TEXT
        self.state = PredictionState()  # the old result belongs to the other mode
        self._pending = None
        if self.text_mode and (self._fallback is None or not self._fallback.available):
            self.state.message = MSG_TEXT_NEEDS_GEMINI
        logger.info("Recognition mode: %s", self.recognition)

    def toggle_debug(self) -> None:
        """Show/hide the CNN input image; while on, each prediction also saves it."""
        self.debug = not self.debug
        logger.info("Debug view %s", "on" if self.debug else "off")

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
        """Recognise the drawing according to the current mode."""
        if self.text_mode:
            self.predict_text()
        else:
            self.predict_shape()

    def predict_text(self) -> None:
        """Send the drawing straight to Gemini to read handwritten text (no CNN)."""
        self.canvas.pen_up()
        self._pending = None
        image = render_text_for_llm(self.canvas.strokes)
        if image is None:
            self.state = PredictionState(message=MSG_NOTHING_DRAWN)
            return
        if self._fallback is None or not self._fallback.available:
            self.state = PredictionState(message=MSG_TEXT_NEEDS_GEMINI)
            return
        self._pending = self._fallback.submit_text(image)
        self._pending_text = True
        self.state = PredictionState(message=MSG_READING_TEXT)

    def predict_shape(self) -> None:
        """Classify the drawing with the CNN, asking Gemini if confidence is low."""
        self.canvas.pen_up()
        self._pending = None
        strokes = self.canvas.strokes
        bitmap = strokes_to_bitmap(strokes)
        if bitmap is None:
            self.state = PredictionState(message=MSG_NOTHING_DRAWN)
            return
        if self.debug:
            self._save_debug(bitmap)
        tensor = bitmaps_to_tensor(bitmap)  # exactly the image shown in the debug view
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
        # The CNN top-3 go along as hints; Gemini may still pick any category.
        hints = [(p.label, p.confidence) for p in predictions]
        self._pending = self._fallback.submit(image, hints)
        self._pending_text = False
        self.state.message = MSG_ASKING_LLM

    def poll_llm(self) -> None:
        """Apply the Gemini answer once the background call has finished."""
        if self._pending is None or not self._pending.done():
            return
        result = self._pending.result()
        self._pending = None
        if self._pending_text:
            self._apply_text(result if isinstance(result, TextResult) else None)
            return
        if not isinstance(result, LLMResult):
            self.state.message = MSG_LLM_UNAVAILABLE
            if self.state.cnn:
                self._speak(self.state.cnn[0].label)
        else:
            self.state.llm = result
            self.state.source = SOURCE_GEMINI
            self.state.message = None
            self._speak(result.label)

    def _apply_text(self, result: TextResult | None) -> None:
        """Show (and speak) the handwriting read by Gemini."""
        if result is None:
            self.state = PredictionState(message=MSG_TEXT_NEEDS_GEMINI)
            return
        self.state = PredictionState(text=result, source=SOURCE_GEMINI)
        if not result.text:
            self.state.message = MSG_NO_TEXT
        self._say(result.text, text_sentence(result.text))

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
        elif button.action == ACTION_MODE:
            self.toggle_recognition()
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
        elif key == ord(config.KEY_TEXT_MODE):
            self.toggle_recognition()
        elif key == ord(config.KEY_DEBUG):
            self.toggle_debug()
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
        self._palm = _point(hand.landmarks[config.PALM_LANDMARK]) if hand is not None else None
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
        # The open hand must be held for CLEAR_HOLD_SECONDS, so a brief open hand
        # while drawing does not wipe the canvas.
        if self._clear_hold.update(stable == Gesture.CLEAR and hand is not None):
            self.clear()
        if self._debouncer.changed and stable == Gesture.PREDICT:
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
            text_mode=self.text_mode,
        )
        draw_toolbar(frame, self._toolbar, toolbar_state)
        top = self._toolbar.height + config.CARD_Y_OFFSET
        draw_prediction_card(frame, self.state.card(), top)
        draw_mode_badges(frame, self.mode(gesture), self.muted, top, self.recognition)
        if self._fingertip is not None:
            draw_progress_ring(frame, self._fingertip, self._dwell.progress)
        if self._palm is not None:
            draw_progress_ring(
                frame,
                self._palm,
                self._clear_hold.progress,
                config.CLEAR_RING_COLOR,
                config.CLEAR_RING_RADIUS,
            )
        if self.debug:
            blank = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
            draw_debug_view(frame, self.debug_bitmap if self.debug_bitmap is not None else blank)
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

    @property
    def clear_progress(self) -> float:
        """Progress (0..1) of the open-hand clear hold."""
        return self._clear_hold.progress

    def _save_debug(self, bitmap: np.ndarray) -> None:
        """Keep the CNN input for the debug view and save it as debug_<time>.png."""
        self.debug_bitmap = bitmap
        try:
            self.last_debug_path = save_debug_bitmap(bitmap, self._debug_dir)
        except (OSError, ValueError) as exc:
            logger.warning("Could not save debug image: %s", exc)
            return
        logger.info("Saved CNN input to %s", self.last_debug_path)

    def _apply_pen(self) -> None:
        """Push the selected colour and thickness to the canvas."""
        self.canvas.set_style(
            color=config.PALETTE[self.color_index][1],
            thickness=config.PEN_THICKNESSES[self.thickness_index],
        )

    def _speak(self, label: str) -> None:
        """Say a shape label with a varied phrase, once per drawing."""
        self._say(label, None)

    def _say(self, result: str, sentence: str | None) -> None:
        """Speak a result once per (drawing, mode, result).

        The canvas version identifies the drawing. The check happens before a phrase
        is picked, so predicting the same drawing again stays silent even though the
        phrases are random. `sentence` None means "pick a shape phrase".
        """
        key = (self.canvas.version, self.recognition, result)
        if self._speaker is None or key == self._last_spoken:
            return
        self._last_spoken = key
        self._speaker.say(sentence or self._phrases.sentence(result), key=key)


def _point(xy: np.ndarray) -> tuple[int, int]:
    """Round a landmark to integer pixel coordinates."""
    return int(round(float(xy[0]))), int(round(float(xy[1])))


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
