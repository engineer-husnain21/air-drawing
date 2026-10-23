"""Tests for the CNN, the classifier wrapper and checkpoint loading."""

from pathlib import Path

import pytest
import torch

from air_drawing import config
from air_drawing.classifier import Classifier
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
