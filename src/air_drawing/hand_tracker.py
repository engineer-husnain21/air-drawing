"""MediaPipe Hand Landmarker wrapper returning 21 landmarks in pixel coordinates."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from air_drawing import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HandResult:
    """A detected hand: (21, 2) pixel landmarks plus handedness label."""

    landmarks: np.ndarray
    handedness: str

    @property
    def pen(self) -> tuple[int, int]:
        """Pixel position of the pen landmark (index fingertip)."""
        x, y = self.landmarks[config.PEN_LANDMARK]
        return int(round(x)), int(round(y))


class HandTracker:
    """Detects one hand per frame using the MediaPipe Tasks Hand Landmarker (VIDEO mode)."""

    def __init__(self, model_path: Path = config.HAND_LANDMARKER_PATH) -> None:
        """Load the hand landmarker model; raises FileNotFoundError if it is missing."""
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found at {model_path}. "
                "Run `python scripts/download_data.py` first."
            )
        # Loading from a buffer avoids MediaPipe path issues on Windows (spaces, non-ASCII).
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1
        logger.info("Hand landmarker loaded from %s", model_path)

    def detect(self, frame_bgr: np.ndarray) -> HandResult | None:
        """Detect a hand in a BGR frame; return pixel landmarks or None."""
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._next_timestamp_ms())
        if not result.hand_landmarks:
            return None
        points = np.array(
            [(lm.x * width, lm.y * height) for lm in result.hand_landmarks[0]],
            dtype=np.float32,
        )
        handedness = result.handedness[0][0].category_name if result.handedness else "Unknown"
        return HandResult(landmarks=points, handedness=handedness)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def _next_timestamp_ms(self) -> int:
        """Return a strictly increasing timestamp, as required by VIDEO mode."""
        now = int(time.monotonic() * 1000)
        self._last_timestamp_ms = max(now, self._last_timestamp_ms + 1)
        return self._last_timestamp_ms


def draw_landmarks(frame: np.ndarray, hand: HandResult) -> None:
    """Draw hand connections, landmarks and the pen tip onto the frame in place."""
    points = [(int(round(x)), int(round(y))) for x, y in hand.landmarks]
    for start, end in config.HAND_CONNECTIONS:
        cv2.line(
            frame, points[start], points[end], config.CONNECTION_COLOR, config.CONNECTION_THICKNESS
        )
    for point in points:
        cv2.circle(frame, point, config.LANDMARK_RADIUS, config.LANDMARK_COLOR, cv2.FILLED)
    cv2.circle(frame, hand.pen, config.PEN_RADIUS, config.PEN_COLOR, cv2.FILLED)
