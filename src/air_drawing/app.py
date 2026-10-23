"""Main loop: webcam -> hand tracking -> on-screen landmarks."""

import logging

import cv2
import numpy as np

from air_drawing import config
from air_drawing.hand_tracker import HandResult, HandTracker, draw_landmarks

logger = logging.getLogger(__name__)


class AirDrawingApp:
    """Runs the live webcam loop."""

    def __init__(self, tracker: HandTracker) -> None:
        """Store collaborators."""
        self._tracker = tracker

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Mirror the frame, track the hand and draw the overlay."""
        frame = cv2.flip(frame, 1)
        hand: HandResult | None = self._tracker.detect(frame)
        if hand is not None:
            draw_landmarks(frame, hand)
        return frame

    def run(self) -> None:
        """Open the camera and loop until the user quits."""
        capture = cv2.VideoCapture(config.CAMERA_INDEX)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera {config.CAMERA_INDEX}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        logger.info("Camera opened. Press '%s' to quit.", config.KEY_QUIT)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    logger.warning("Failed to read frame from camera")
                    break
                cv2.imshow(config.WINDOW_NAME, self.process_frame(frame))
                key = cv2.waitKey(config.KEY_WAIT_MS) & config.KEY_MASK
                if key == ord(config.KEY_QUIT):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
            self._tracker.close()


def run() -> None:
    """Build the app with real collaborators and run it."""
    AirDrawingApp(tracker=HandTracker()).run()
