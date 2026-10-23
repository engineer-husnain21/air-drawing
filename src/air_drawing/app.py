"""Main loop: webcam -> hand tracking -> gestures -> canvas -> CNN prediction."""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from air_drawing import config
from air_drawing.canvas import Canvas
from air_drawing.classifier import Classifier, Prediction
from air_drawing.gestures import Gesture, GestureDebouncer, classify_gesture
from air_drawing.hand_tracker import HandResult, HandTracker, draw_landmarks
from air_drawing.preprocess import preprocess_strokes

logger = logging.getLogger(__name__)

MSG_NOTHING_DRAWN = "Nothing drawn"
MSG_NO_MODEL = "CNN model not loaded (run scripts/train.py)"
SOURCE_CNN = "CNN"
CONTROLS_HINT = (
    "Index: draw | Index+Middle: pen up | Open hand: clear | Fist: predict | "
    f"Keys: {config.KEY_CLEAR} clear, {config.KEY_PREDICT} predict, {config.KEY_QUIT} quit"
)


@dataclass
class PredictionState:
    """What the HUD shows about the last prediction."""

    cnn: list[Prediction] = field(default_factory=list)
    source: str | None = None
    message: str | None = None


class AirDrawingApp:
    """Ties tracking, gestures, drawing and recognition together."""

    def __init__(self, tracker: HandTracker, classifier: Classifier | None) -> None:
        """Store collaborators; classifier may be None if no weights are available."""
        self._tracker = tracker
        self._classifier = classifier
        self.canvas = Canvas()
        self._debouncer = GestureDebouncer()
        self.state = PredictionState()

    # --- actions --------------------------------------------------------------

    def clear(self) -> None:
        """Erase the drawing and the last prediction."""
        self.canvas.clear()
        self.state = PredictionState()
        logger.info("Canvas cleared")

    def predict(self) -> None:
        """Classify the drawing with the CNN."""
        self.canvas.pen_up()
        tensor = preprocess_strokes(self.canvas.strokes)
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

    def handle_key(self, key: int) -> bool:
        """React to a key press; return False when the app should quit."""
        if key == ord(config.KEY_QUIT):
            return False
        if key == ord(config.KEY_CLEAR):
            self.clear()
        elif key == ord(config.KEY_PREDICT):
            self.predict()
        return True

    # --- per-frame --------------------------------------------------------------

    def update(self, hand: HandResult | None) -> Gesture:
        """Advance gesture state and the canvas for one frame; return the stable gesture."""
        raw = classify_gesture(hand.landmarks if hand is not None else None)
        stable = self._debouncer.update(raw)
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
        return stable

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Mirror the frame, run one update and draw the overlay."""
        frame = cv2.flip(frame, 1)
        hand = self._tracker.detect(frame)
        gesture = self.update(hand)
        self.canvas.render(frame)
        if hand is not None:
            draw_landmarks(frame, hand)
        draw_hud(frame, gesture, self.state)
        return frame

    def run(self) -> None:
        """Open the camera and loop until the user quits."""
        capture = cv2.VideoCapture(config.CAMERA_INDEX)
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open camera {config.CAMERA_INDEX}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            logger.info("Camera opened. %s", CONTROLS_HINT)
            while True:
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
            config.HUD_FONT_SCALE,
            col,
            config.HUD_THICKNESS + extra,
            cv2.LINE_AA,
        )


def hud_lines(gesture: Gesture, state: PredictionState) -> list[tuple[str, tuple[int, int, int]]]:
    """Return the HUD text lines (top-left) with their colours."""
    lines = [(f"Mode: {gesture.value.upper()}", config.HUD_ACCENT_COLOR)]
    for rank, pred in enumerate(state.cnn, start=1):
        lines.append(
            (f"{rank}. {pred.label} {pred.confidence * config.PERCENT:.0f}%", config.HUD_TEXT_COLOR)
        )
    if state.source is not None:
        lines.append((f"via {state.source}", config.HUD_ACCENT_COLOR))
    if state.message is not None:
        lines.append((state.message, config.HUD_WARNING_COLOR))
    return lines


def draw_hud(frame: np.ndarray, gesture: Gesture, state: PredictionState) -> None:
    """Draw mode, predictions, source label, messages and controls onto the frame."""
    y = config.HUD_MARGIN + config.HUD_LINE_HEIGHT
    for text, color in hud_lines(gesture, state):
        _put_text(frame, text, (config.HUD_MARGIN, y), color)
        y += config.HUD_LINE_HEIGHT
    bottom = frame.shape[0] - config.HUD_MARGIN
    _put_text(frame, CONTROLS_HINT, (config.HUD_MARGIN, bottom), config.HUD_TEXT_COLOR)


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
    try:
        AirDrawingApp(tracker, classifier).run()
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0
