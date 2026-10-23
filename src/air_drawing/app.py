"""Main loop: webcam -> hand tracking -> gestures -> canvas."""

import logging

import cv2
import numpy as np

from air_drawing import config
from air_drawing.canvas import Canvas
from air_drawing.gestures import Gesture, GestureDebouncer, classify_gesture
from air_drawing.hand_tracker import HandResult, HandTracker, draw_landmarks

logger = logging.getLogger(__name__)

CONTROLS_HINT = (
    "Index: draw | Index+Middle: pen up | Open hand: clear | Fist: predict | "
    f"Keys: {config.KEY_CLEAR} clear, {config.KEY_PREDICT} predict, {config.KEY_QUIT} quit"
)


class AirDrawingApp:
    """Ties tracking, gestures and drawing together."""

    def __init__(self, tracker: HandTracker) -> None:
        """Store collaborators."""
        self._tracker = tracker
        self.canvas = Canvas()
        self._debouncer = GestureDebouncer()

    # --- actions --------------------------------------------------------------

    def clear(self) -> None:
        """Erase the drawing."""
        self.canvas.clear()
        logger.info("Canvas cleared")

    def predict(self) -> None:
        """End the current stroke (recognition is added in a later phase)."""
        self.canvas.pen_up()
        logger.info("Predict requested (%d strokes)", len(self.canvas.strokes))

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
        draw_hud(frame, gesture)
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


def draw_hud(frame: np.ndarray, gesture: Gesture) -> None:
    """Draw the current mode and the controls onto the frame."""
    _put_text(
        frame,
        f"Mode: {gesture.value.upper()}",
        (config.HUD_MARGIN, config.HUD_MARGIN + config.HUD_LINE_HEIGHT),
        config.HUD_ACCENT_COLOR,
    )
    bottom = frame.shape[0] - config.HUD_MARGIN
    _put_text(frame, CONTROLS_HINT, (config.HUD_MARGIN, bottom), config.HUD_TEXT_COLOR)


def run() -> int:
    """Build the app with real collaborators and run it; return an exit code."""
    try:
        app = AirDrawingApp(tracker=HandTracker())
        app.run()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1
    return 0
