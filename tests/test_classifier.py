"""Tests for the CNN, the classifier wrapper, checkpoint loading and the Gemini fallback."""

from pathlib import Path

import numpy as np
import pytest
import torch

from air_drawing import config
from air_drawing.classifier import Classifier
from air_drawing.llm_fallback import (
    GeminiFallback,
    LLMResult,
    LLMUnavailableError,
    build_prompt,
    parse_response,
)
from air_drawing.model import QuickDrawCNN

CATEGORIES = ["circle", "square", "star", "sun"]


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = QuickDrawCNN(num_classes=len(CATEGORIES))
    path = tmp_path / "cnn.pt"
    torch.save({"state_dict": model.state_dict(), "categories": CATEGORIES}, path)
    return path


def blank() -> torch.Tensor:
    return torch.zeros(1, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)


def test_model_output_shape() -> None:
    model = QuickDrawCNN(num_classes=len(config.CATEGORIES))
    assert model(torch.zeros(2, 1, config.IMAGE_SIZE, config.IMAGE_SIZE)).shape == (
        2,
        len(config.CATEGORIES),
    )


def test_predict_returns_sorted_top_k(checkpoint: Path) -> None:
    preds = Classifier.from_checkpoint(checkpoint).predict(blank())
    assert len(preds) == config.TOP_K
    confidences = [p.confidence for p in preds]
    assert confidences == sorted(confidences, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confidences)
    assert sum(confidences) <= 1.0 + 1e-6
    assert {p.label for p in preds} <= set(CATEGORIES)


def test_top_k_is_capped_by_number_of_classes(checkpoint: Path) -> None:
    preds = Classifier.from_checkpoint(checkpoint).predict(blank(), k=10)
    assert len(preds) == len(CATEGORIES)
    assert sum(p.confidence for p in preds) == pytest.approx(1.0, abs=1e-5)


def test_prediction_is_deterministic(checkpoint: Path) -> None:
    classifier = Classifier.from_checkpoint(checkpoint)
    assert classifier.predict(blank()) == classifier.predict(blank())


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Classifier.from_checkpoint(tmp_path / "missing.pt")


# --- Gemini fallback (fake client, never touches the network) -----------------


class FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class FakeModels:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self._text = text
        self._error = error
        self.calls: list[dict] = []

    def generate_content(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return FakeResponse(self._text)


class FakeClient:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.models = FakeModels(text, error)


def drawing() -> np.ndarray:
    return np.full((config.LLM_IMAGE_SIZE, config.LLM_IMAGE_SIZE), 255, dtype=np.uint8)


def test_parse_response_valid() -> None:
    assert parse_response('{"label": "Star", "confidence": 0.8}', CATEGORIES) == LLMResult(
        "star", 0.8
    )


def test_parse_response_strips_code_fence_and_clamps() -> None:
    result = parse_response('```json\n{"label": "sun", "confidence": 3}\n```', CATEGORIES)
    assert result == LLMResult("sun", 1.0)


def test_parse_response_unknown_label_becomes_other() -> None:
    assert parse_response('{"label": "dragon"}', CATEGORIES) == LLMResult(config.OTHER_LABEL, None)


def test_parse_response_invalid_json_raises() -> None:
    with pytest.raises(LLMUnavailableError):
        parse_response("not json", CATEGORIES)


def test_prompt_lists_every_category() -> None:
    prompt = build_prompt(config.CATEGORIES)
    assert all(c in prompt for c in config.CATEGORIES)
    assert config.OTHER_LABEL in prompt


def test_fallback_unavailable_without_key_or_model() -> None:
    assert not GeminiFallback(api_key="", model="").available
    assert not GeminiFallback(api_key="key", model="").available
    assert not GeminiFallback(api_key="", model="some-model").available


def test_fallback_uses_configured_model() -> None:
    client = FakeClient('{"label": "star", "confidence": 0.9}')
    fallback = GeminiFallback(api_key="k", model="test-model", client=client)
    try:
        assert fallback.classify(drawing()) == LLMResult("star", 0.9)
        assert client.models.calls[0]["model"] == "test-model"
    finally:
        fallback.close()


def test_submit_runs_in_background_and_returns_result() -> None:
    fallback = GeminiFallback(model="m", client=FakeClient('{"label": "sun"}'), categories=("sun",))
    try:
        assert fallback.submit(drawing()).result(timeout=5) == LLMResult("sun", None)
    finally:
        fallback.close()


def test_submit_swallows_errors() -> None:
    fallback = GeminiFallback(model="m", client=FakeClient(error=RuntimeError("quota")))
    try:
        assert fallback.submit(drawing()).result(timeout=5) is None
    finally:
        fallback.close()


def test_submit_without_config_returns_none() -> None:
    fallback = GeminiFallback(api_key="", model="")
    try:
        assert fallback.submit(drawing()).result(timeout=5) is None
    finally:
        fallback.close()
