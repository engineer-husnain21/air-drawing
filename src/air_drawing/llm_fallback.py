"""Gemini vision: fallback for low-confidence CNN predictions and handwriting reading.

Calls run in a background thread and every error is swallowed and logged,
so the webcam loop never blocks or crashes because of the LLM.
"""

import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from air_drawing import config

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "This image is a rough sketch drawn in the air with a finger, so the lines may be "
    "shaky. Which ONE of these categories does it show: {categories}? "
    'If none fits, use "{other}". '
    'Respond only with JSON: {{"label": "<one label>", "confidence": <number 0 to 1>}}.'
)
TEXT_PROMPT = (
    "This image shows handwriting written in the air with a finger, so the strokes are "
    "shaky and letters may touch. Read the handwritten text exactly as written (letters, "
    "digits or words). "
    'Respond only with JSON: {"text": "<the text>", "confidence": <number 0 to 1>}. '
    'If nothing is readable, use an empty text "".'
)


class LLMUnavailableError(Exception):
    """Raised when the LLM is not configured or the call fails."""


@dataclass(frozen=True)
class LLMResult:
    """A label chosen by the LLM, with its self-reported confidence if any."""

    label: str
    confidence: float | None


@dataclass(frozen=True)
class TextResult:
    """Handwritten text read by the LLM, with its self-reported confidence if any."""

    text: str
    confidence: float | None


def build_prompt(categories: tuple[str, ...] | list[str]) -> str:
    """Return the classification prompt for the given categories."""
    return PROMPT_TEMPLATE.format(categories=", ".join(categories), other=config.OTHER_LABEL)


def _load_json(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating Markdown code fences around it."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError(f"Unparseable LLM response: {text!r}") from exc
    if not isinstance(data, dict):
        raise LLMUnavailableError(f"Expected a JSON object, got: {text!r}")
    return data


def _confidence(data: dict[str, Any]) -> float | None:
    """Return the "confidence" field clamped to [0, 1], or None if missing/invalid."""
    try:
        return min(max(float(data["confidence"]), 0.0), 1.0)
    except (KeyError, TypeError, ValueError):
        return None


def parse_response(text: str, categories: tuple[str, ...] | list[str]) -> LLMResult:
    """Parse the model's JSON reply; unknown labels become "other"."""
    data = _load_json(text)
    try:
        label = str(data["label"]).strip().lower()
    except KeyError as exc:
        raise LLMUnavailableError(f"Missing label in LLM response: {text!r}") from exc
    if label not in categories:
        if label != config.OTHER_LABEL:
            logger.warning("LLM returned unknown label %r, using %r", label, config.OTHER_LABEL)
        label = config.OTHER_LABEL
    return LLMResult(label=label, confidence=_confidence(data))


def parse_text_response(text: str) -> TextResult:
    """Parse the model's JSON reply for handwriting; whitespace is normalised."""
    data = _load_json(text)
    if "text" not in data:
        raise LLMUnavailableError(f"Missing text in LLM response: {text!r}")
    value = data["text"]
    words = "" if value is None else " ".join(str(value).split())
    return TextResult(text=words, confidence=_confidence(data))


def encode_png(image: np.ndarray) -> bytes:
    """Encode an image as PNG bytes."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise LLMUnavailableError("Could not encode image as PNG")
    return buffer.tobytes()


class GeminiFallback:
    """Classifies drawings or reads handwriting with Gemini.

    Configured from GEMINI_API_KEY / GEMINI_MODEL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        categories: tuple[str, ...] = config.CATEGORIES,
    ) -> None:
        """Create the fallback; `client` can be injected (e.g. a fake in tests)."""
        self._api_key = api_key if api_key is not None else config.get_gemini_api_key()
        self._model = model if model is not None else config.get_gemini_model()
        self._client = client
        self._categories = categories
        self._executor = ThreadPoolExecutor(
            max_workers=config.LLM_MAX_WORKERS, thread_name_prefix="gemini"
        )

    @property
    def available(self) -> bool:
        """True when an API key and a model name are configured."""
        return bool(self._model) and (self._client is not None or bool(self._api_key))

    def classify(self, image: np.ndarray) -> LLMResult:
        """Blocking shape classification; raises LLMUnavailableError on any failure."""
        return parse_response(
            self._generate(image, build_prompt(self._categories)), self._categories
        )

    def read_text(self, image: np.ndarray) -> TextResult:
        """Blocking handwriting recognition; raises LLMUnavailableError on any failure."""
        return parse_text_response(self._generate(image, TEXT_PROMPT))

    def _generate(self, image: np.ndarray, prompt: str) -> str:
        """Send the image and prompt to Gemini and return the raw text reply."""
        if not self.available:
            raise LLMUnavailableError(
                f"Set {config.GEMINI_API_KEY_ENV} and {config.GEMINI_MODEL_ENV} in .env"
            )
        try:
            from google.genai import types

            response = self._get_client().models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=encode_png(image), mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=config.LLM_TEMPERATURE,
                    response_mime_type="application/json",
                ),
            )
            text = response.text
        except LLMUnavailableError:
            raise
        except Exception as exc:  # network, auth, quota, SDK errors...
            raise LLMUnavailableError(f"Gemini call failed: {exc}") from exc
        if not text:
            raise LLMUnavailableError("Empty Gemini response")
        return text

    def submit(self, image: np.ndarray) -> "Future[LLMResult | None]":
        """Classify in a background thread; the future yields None on failure."""
        return self._executor.submit(self._safe_classify, image)

    def submit_text(self, image: np.ndarray) -> "Future[TextResult | None]":
        """Read handwriting in a background thread; the future yields None on failure."""
        return self._executor.submit(self._safe_read_text, image)

    def close(self) -> None:
        """Stop the background worker without waiting for pending calls."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _safe_classify(self, image: np.ndarray) -> LLMResult | None:
        """Run classify(), logging and swallowing every error."""
        try:
            result = self.classify(image)
        except Exception as exc:
            logger.warning("LLM fallback unavailable: %s", exc)
            return None
        logger.info("Gemini says %s (confidence %s)", result.label, result.confidence)
        return result

    def _safe_read_text(self, image: np.ndarray) -> TextResult | None:
        """Run read_text(), logging and swallowing every error."""
        try:
            result = self.read_text(image)
        except Exception as exc:
            logger.warning("Gemini text reading unavailable: %s", exc)
            return None
        logger.info("Gemini read %r (confidence %s)", result.text, result.confidence)
        return result

    def _get_client(self) -> Any:
        """Create the Gemini client lazily."""
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client
