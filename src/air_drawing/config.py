"""Central configuration: categories, thresholds, paths and tunable constants."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SAMPLES_DIR: Path = DATA_DIR / "samples"
MODELS_DIR: Path = PROJECT_ROOT / "models"
HAND_LANDMARKER_PATH: Path = MODELS_DIR / "hand_landmarker.task"
CNN_WEIGHTS_PATH: Path = MODELS_DIR / "cnn_quickdraw.pt"
ENV_FILE: Path = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

# --- Categories (single source of truth) -------------------------------------
CATEGORIES: tuple[str, ...] = (
    "circle",
    "triangle",
    "square",
    "star",
    "house",
    "car",
    "tree",
    "fish",
    "sun",
    "cloud",
    "apple",
    "umbrella",
    "cat",
    "clock",
    "bicycle",
)
OTHER_LABEL: str = "other"

# --- Downloads ---------------------------------------------------------------
QUICKDRAW_BASE_URL: str = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
HAND_LANDMARKER_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
BYTES_PER_MB: int = 1 << 20
DOWNLOAD_CHUNK_BYTES: int = BYTES_PER_MB
DOWNLOAD_TIMEOUT_S: float = 60.0

# --- Camera / window ---------------------------------------------------------
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
WINDOW_NAME: str = "Air Drawing Recognition"
KEY_WAIT_MS: int = 1
KEY_MASK: int = 0xFF
KEY_CLEAR: str = "c"
KEY_PREDICT: str = "p"
KEY_QUIT: str = "q"

# --- Hand tracking -----------------------------------------------------------
NUM_HANDS: int = 1
MIN_HAND_DETECTION_CONFIDENCE: float = 0.6
MIN_HAND_PRESENCE_CONFIDENCE: float = 0.6
MIN_TRACKING_CONFIDENCE: float = 0.5
NUM_LANDMARKS: int = 21

# Landmark indices (MediaPipe hand model)
WRIST: int = 0
THUMB_IP: int = 3
THUMB_TIP: int = 4
INDEX_PIP: int = 6
INDEX_TIP: int = 8
MIDDLE_PIP: int = 10
MIDDLE_TIP: int = 12
RING_PIP: int = 14
RING_TIP: int = 16
PINKY_MCP: int = 17
PINKY_PIP: int = 18
PINKY_TIP: int = 20
PEN_LANDMARK: int = INDEX_TIP

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)  # fmt: skip

# --- Gestures ----------------------------------------------------------------
GESTURE_DEBOUNCE_FRAMES: int = 5
# A finger is "up" when its tip is this much farther from the wrist than its PIP joint.
FINGER_EXTENDED_RATIO: float = 1.1
# The thumb is "up" when its tip is this much farther from the pinky MCP than its IP joint.
THUMB_EXTENDED_RATIO: float = 1.1

# --- Canvas / smoothing ------------------------------------------------------
SMOOTHING_ALPHA: float = 0.5  # weight of the newest point (1.0 = no smoothing)
MIN_POINT_DISTANCE_PX: float = 2.0
CANVAS_STROKE_THICKNESS: int = 8
CANVAS_STROKE_COLOR: tuple[int, int, int] = (255, 0, 255)  # BGR

# --- Preprocessing (Quick, Draw! format) -------------------------------------
IMAGE_SIZE: int = 28
RENDER_SIZE: int = 256  # draw at high resolution first, then downsample
RENDER_PADDING_RATIO: float = 0.1  # padding on each side; matches Quick, Draw! framing
RENDER_STROKE_THICKNESS: int = 13  # at RENDER_SIZE; matches Quick, Draw! ink density
PIXEL_MAX: float = 255.0
LLM_IMAGE_SIZE: int = 256

# --- Model -------------------------------------------------------------------
CONV_CHANNELS: tuple[int, ...] = (32, 64, 128)
CONV_KERNEL_SIZE: int = 3
POOL_SIZE: int = 2
HIDDEN_UNITS: int = 256
DROPOUT: float = 0.3
TOP_K: int = 3

# --- Training ----------------------------------------------------------------
DEFAULT_SAMPLES_PER_CLASS: int = 15000
DEFAULT_EPOCHS: int = 8
TRAIN_SPLIT: float = 0.8
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
RANDOM_SEED: int = 42

# --- LLM fallback ------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.6
GEMINI_API_KEY_ENV: str = "GEMINI_API_KEY"
GEMINI_MODEL_ENV: str = "GEMINI_MODEL"
CONFIDENCE_THRESHOLD_ENV: str = "CONFIDENCE_THRESHOLD"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_WORKERS: int = 1

# --- HUD ---------------------------------------------------------------------
HUD_FONT_SCALE: float = 0.8
HUD_THICKNESS: int = 2
HUD_LINE_HEIGHT: int = 32
HUD_MARGIN: int = 20
HUD_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
HUD_ACCENT_COLOR: tuple[int, int, int] = (0, 220, 255)
HUD_WARNING_COLOR: tuple[int, int, int] = (0, 0, 255)
HUD_SHADOW_COLOR: tuple[int, int, int] = (0, 0, 0)
LANDMARK_COLOR: tuple[int, int, int] = (0, 255, 0)
CONNECTION_COLOR: tuple[int, int, int] = (255, 255, 255)
PEN_COLOR: tuple[int, int, int] = (0, 0, 255)
LANDMARK_RADIUS: int = 4
PEN_RADIUS: int = 8
CONNECTION_THICKNESS: int = 2
PERCENT: float = 100.0


def get_confidence_threshold() -> float:
    """Return CONFIDENCE_THRESHOLD from the environment, falling back to the default."""
    raw = os.getenv(CONFIDENCE_THRESHOLD_ENV)
    if not raw:
        return DEFAULT_CONFIDENCE_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "Invalid %s=%r, using %s", CONFIDENCE_THRESHOLD_ENV, raw, DEFAULT_CONFIDENCE_THRESHOLD
        )
        return DEFAULT_CONFIDENCE_THRESHOLD
    return min(max(value, 0.0), 1.0)


def get_gemini_api_key() -> str | None:
    """Return the Gemini API key from the environment, or None if unset."""
    return os.getenv(GEMINI_API_KEY_ENV) or None


def get_gemini_model() -> str | None:
    """Return the Gemini model name from the environment, or None if unset."""
    return os.getenv(GEMINI_MODEL_ENV) or None
