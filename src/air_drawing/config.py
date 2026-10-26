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
KEY_MUTE: str = "m"
KEY_UNDO: str = "u"
KEY_TEXT_MODE: str = "t"
KEY_DEBUG: str = "d"

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
MIDDLE_MCP: int = 9
PALM_LANDMARK: int = MIDDLE_MCP  # centre of the clear-hold progress ring
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
HINT_DECIMALS: int = 0  # CNN hint percentages sent to Gemini

# --- Training ----------------------------------------------------------------
DEFAULT_SAMPLES_PER_CLASS: int = 15000
DEFAULT_EPOCHS: int = 12

# --- Training augmentation (mimics air drawing; applied per batch on CPU) ----
AUG_ROTATION_DEG: float = 15.0  # random rotation in [-15, +15] degrees
AUG_SCALE_RANGE: tuple[float, float] = (0.8, 1.1)  # drawing size factor
AUG_SHIFT: float = 0.1  # max shift as a fraction of half the image (~1.4 px)
AUG_ELASTIC_GRID: int = 4  # coarse noise grid upsampled to a smooth displacement field
AUG_ELASTIC_STRENGTH: float = 0.04  # max displacement as a fraction of half the image
AUG_THICKNESS_PROB: float = 0.5  # chance to thicken or thin the strokes
AUG_THICKEN_SHARE: float = 0.5  # of those, the share that is thickened (rest thinned)
AUG_THICKNESS_MAX_BLEND: float = 0.7  # 1.0 = full 3x3 dilation / erosion
AUG_MORPH_KERNEL: int = 3
TRAIN_SPLIT: float = 0.8
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
RANDOM_SEED: int = 42

# --- LLM fallback ------------------------------------------------------------
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.85
GEMINI_API_KEY_ENV: str = "GEMINI_API_KEY"
GEMINI_MODEL_ENV: str = "GEMINI_MODEL"
CONFIDENCE_THRESHOLD_ENV: str = "CONFIDENCE_THRESHOLD"
LLM_TEMPERATURE: float = 0.0
LLM_MAX_WORKERS: int = 1

# --- HUD ---------------------------------------------------------------------
HUD_HINT_FONT_SCALE: float = 0.5
HUD_THICKNESS: int = 2
HUD_MARGIN: int = 20
HUD_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
HUD_WARNING_COLOR: tuple[int, int, int] = (0, 0, 255)
HUD_SHADOW_COLOR: tuple[int, int, int] = (0, 0, 0)
LANDMARK_COLOR: tuple[int, int, int] = (0, 255, 0)
CONNECTION_COLOR: tuple[int, int, int] = (255, 255, 255)
PEN_COLOR: tuple[int, int, int] = (0, 0, 255)
LANDMARK_RADIUS: int = 4
PEN_RADIUS: int = 8
CONNECTION_THICKNESS: int = 2
PERCENT: float = 100.0

# --- Voice output (pyttsx3) --------------------------------------------------
VOICE_ENABLED: bool = True
VOICE_RATE: int = 170  # words per minute
VOICE_VOLUME: float = 0.9  # 0.0 - 1.0
SPEECH_QUEUE_SIZE: int = 4  # newest sentences win when the queue is full
SPEECH_JOIN_TIMEOUT_S: float = 2.0
SPEECH_TEMPLATE: str = "I think it's {article} {label}"  # default for make_sentence()
# Shape mode picks one at random, never the same one twice in a row.
SHAPE_PHRASES: tuple[str, ...] = (
    "That's {article} {label}",
    "Looks like {article} {label}",
    "I think it's {article} {label}",
    "{label}!",
)
TEXT_PHRASE: str = "You wrote {text}"
SPEECH_UNKNOWN: str = "I'm not sure what that is"
VOWELS: str = "aeiou"

# --- Toolbar / palette -------------------------------------------------------
UI_DESIGN_WIDTH: int = 1280  # layout is designed for this width and scaled to fit
TOOLBAR_HEIGHT: int = 84
TOOLBAR_PADDING: int = 14
TOOLBAR_GAP: int = 10
TOOLBAR_GROUP_GAP: int = 28
TOOLBAR_BG_COLOR: tuple[int, int, int] = (40, 30, 30)
TOOLBAR_BG_ALPHA: float = 0.75
BUTTON_WIDTH: int = 108
BUTTON_HEIGHT: int = 56
BUTTON_RADIUS: int = 16
BUTTON_FONT_SCALE: float = 0.62
BUTTON_HOVER_LIGHTEN: float = 0.35  # blend towards white when hovered
BUTTON_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
BUTTON_DARK_TEXT_COLOR: tuple[int, int, int] = (30, 30, 30)
BUTTON_SELECTED_COLOR: tuple[int, int, int] = (255, 255, 255)
BUTTON_SELECTED_THICKNESS: int = 3
LUMA_WEIGHTS_BGR: tuple[float, float, float] = (0.114, 0.587, 0.299)
BUTTON_COLORS: dict[str, tuple[int, int, int]] = {  # BGR
    "clear": (80, 80, 235),
    "predict": (90, 190, 40),
    "undo": (0, 160, 255),
    "mute": (200, 120, 60),
    "quit": (110, 110, 110),
    "mode": (160, 140, 20),  # shape mode
    "mode_text": (170, 60, 150),  # text mode
}
SWATCH_SIZE: int = 44
THICKNESS_BUTTON_WIDTH: int = 56
THICKNESS_BUTTON_COLOR: tuple[int, int, int] = (90, 75, 75)
PALETTE: tuple[tuple[str, tuple[int, int, int]], ...] = (  # (name, BGR)
    ("magenta", (255, 0, 255)),
    ("cyan", (255, 255, 0)),
    ("yellow", (0, 255, 255)),
    ("lime", (0, 255, 0)),
    ("orange", (0, 140, 255)),
    ("red", (60, 40, 255)),
    ("blue", (255, 110, 0)),
    ("white", (255, 255, 255)),
)
PEN_THICKNESSES: tuple[int, ...] = (4, 8, 14)
DEFAULT_THICKNESS_INDEX: int = 1

# --- Air click (dwell) -------------------------------------------------------
DWELL_SECONDS: float = 1.0
PROGRESS_RING_RADIUS: int = 26
PROGRESS_RING_THICKNESS: int = 5
PROGRESS_RING_COLOR: tuple[int, int, int] = (0, 230, 255)
PROGRESS_RING_TRACK_COLOR: tuple[int, int, int] = (80, 80, 80)
FULL_CIRCLE_DEG: float = 360.0
RING_START_DEG: float = -90.0  # start at 12 o'clock

# --- Prediction card / mode badge --------------------------------------------
CARD_X: int = 20
CARD_Y_OFFSET: int = 16  # below the toolbar
CARD_WIDTH: int = 330
CARD_RADIUS: int = 18
CARD_PADDING: int = 16
CARD_BG_COLOR: tuple[int, int, int] = (45, 35, 35)
CARD_BG_ALPHA: float = 0.8
CARD_TITLE_SCALE: float = 1.4
CARD_TITLE_THICKNESS: int = 3
CARD_TITLE_HEIGHT: int = 52
CARD_TEXT_SCALE: float = 0.6
CARD_TEXT_THICKNESS: int = 1
CARD_ROW_HEIGHT: int = 34
CARD_BAR_HEIGHT: int = 10
CARD_BAR_TRACK_COLOR: tuple[int, int, int] = (90, 80, 80)
CARD_BAR_COLORS: tuple[tuple[int, int, int], ...] = ((90, 200, 40), (0, 170, 255), (200, 110, 220))
CARD_CNN_COLOR: tuple[int, int, int] = (90, 200, 40)
CARD_GEMINI_COLOR: tuple[int, int, int] = (255, 150, 60)
BADGE_RADIUS: int = 18
BADGE_HEIGHT: int = 40
BADGE_PADDING_X: int = 18
BADGE_FONT_SCALE: float = 0.7
BADGE_COLORS: dict[str, tuple[int, int, int]] = {  # BGR
    "Draw": (90, 190, 40),
    "Pen up": (0, 160, 255),
    "Idle": (120, 120, 120),
}
MUTED_BADGE_COLOR: tuple[int, int, int] = (60, 60, 200)
RECOGNITION_BADGE_COLORS: dict[str, tuple[int, int, int]] = {  # BGR
    "Shape": (160, 140, 20),
    "Text": (170, 60, 150),
}
MODE_BUTTON_WIDTH: int = 150

# --- Text mode (Gemini reads handwriting) ------------------------------------
# Aspect ratio is kept; the stroke width is fixed in output pixels so letters stay
# readable no matter how large or small the word was written.
TEXT_IMAGE_LONG_SIDE: int = 1024
TEXT_IMAGE_PADDING: int = 48
TEXT_STROKE_THICKNESS: int = 8
CARD_MAX_TITLE_CHARS: int = 40  # longer recognised text is shortened with "..."
CARD_MIN_TITLE_SCALE: float = 0.6

# --- Clear gesture hold ------------------------------------------------------
CLEAR_HOLD_SECONDS: float = 1.0  # open hand must be held this long before clearing
CLEAR_RING_COLOR: tuple[int, int, int] = (60, 60, 255)  # BGR, red ring around the palm
CLEAR_RING_RADIUS: int = 40

# --- Debug view --------------------------------------------------------------
DEBUG_VIEW_SCALE: int = 6  # 28 px -> 168 px, nearest-neighbour so pixels stay sharp
DEBUG_VIEW_MARGIN: int = 20
DEBUG_VIEW_LABEL_HEIGHT: int = 26
DEBUG_VIEW_BORDER: int = 2
DEBUG_VIEW_BORDER_COLOR: tuple[int, int, int] = (0, 230, 255)
DEBUG_VIEW_LABEL: str = "CNN input 28x28"
DEBUG_DIR: Path = SAMPLES_DIR
DEBUG_FILE_PREFIX: str = "debug_"
DEBUG_TIME_FORMAT: str = "%Y%m%d-%H%M%S-%f"  # microseconds avoid name clashes


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
