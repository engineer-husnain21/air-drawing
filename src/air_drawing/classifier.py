"""Loads the trained CNN and returns top-k predictions."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from air_drawing import config
from air_drawing.model import QuickDrawCNN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """A label with its probability in [0, 1]."""

    label: str
    confidence: float


class Classifier:
    """Wraps a QuickDrawCNN and its category list."""

    def __init__(self, model: torch.nn.Module, categories: list[str]) -> None:
        """Use an already-built model; it is switched to eval mode."""
        self._model = model.eval()
        self.categories = list(categories)

    @classmethod
    def from_checkpoint(cls, path: Path = config.CNN_WEIGHTS_PATH) -> "Classifier":
        """Load weights saved by scripts/train.py; raises FileNotFoundError if missing."""
        if not path.is_file():
            raise FileNotFoundError(
                f"CNN weights not found at {path}. Run `python scripts/train.py` first."
            )
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        categories = list(checkpoint["categories"])
        model = QuickDrawCNN(num_classes=len(categories))
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded CNN with %d categories from %s", len(categories), path)
        return cls(model, categories)

    @torch.no_grad()
    def predict(self, image: torch.Tensor, k: int = config.TOP_K) -> list[Prediction]:
        """Return the top-k predictions for a (1, 1, 28, 28) tensor, best first."""
        probs = torch.softmax(self._model(image), dim=1)[0]
        values, indices = probs.topk(min(k, len(self.categories)))
        return [
            Prediction(self.categories[i], float(v))
            for v, i in zip(values.tolist(), indices.tolist(), strict=True)
        ]
