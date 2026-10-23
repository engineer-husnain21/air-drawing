"""Train the Quick, Draw! CNN on CPU and save the best weights to models/."""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from air_drawing import config  # noqa: E402
from air_drawing.model import QuickDrawCNN  # noqa: E402
from air_drawing.preprocess import bitmaps_to_tensor  # noqa: E402

logger = logging.getLogger("train")


def load_dataset(
    categories: list[str], samples_per_class: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Load up to `samples_per_class` random bitmaps per category from data/raw/."""
    images, labels = [], []
    for label, category in enumerate(categories):
        path = config.RAW_DATA_DIR / f"{category}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py`.")
        data = np.load(path, mmap_mode="r")
        count = min(samples_per_class, len(data))
        idx = np.sort(rng.choice(len(data), size=count, replace=False))
        images.append(np.asarray(data[idx], dtype=np.uint8))
        labels.append(np.full(count, label, dtype=np.int64))
        logger.info("Loaded %6d samples for %s", count, category)
    return np.concatenate(images), np.concatenate(labels)


def split(
    images: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> tuple[TensorDataset, TensorDataset]:
    """Shuffle and split into train/validation datasets."""
    order = rng.permutation(len(images))
    cut = int(len(order) * config.TRAIN_SPLIT)
    train_idx, val_idx = order[:cut], order[cut:]

    def to_dataset(idx: np.ndarray) -> TensorDataset:
        return TensorDataset(bitmaps_to_tensor(images[idx]), torch.from_numpy(labels[idx]))

    return to_dataset(train_idx), to_dataset(val_idx)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one pass; trains if an optimizer is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)
    return total_loss / max(seen, 1), correct / max(seen, 1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=config.DEFAULT_SAMPLES_PER_CLASS)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(config.CATEGORIES),
        choices=config.CATEGORIES,
        metavar="CATEGORY",
        help="Categories to train on (default: all).",
    )
    parser.add_argument("--output", type=Path, default=config.CNN_WEIGHTS_PATH)
    args = parser.parse_args(argv)
    if args.samples_per_class < 1 or args.epochs < 1:
        parser.error("--samples-per-class and --epochs must be >= 1")
    if len(set(args.categories)) < 2:
        parser.error("need at least 2 distinct categories")
    return args


def main(argv: list[str] | None = None) -> int:
    """Train the CNN; return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args(argv)
    categories = list(dict.fromkeys(args.categories))
    torch.manual_seed(config.RANDOM_SEED)
    rng = np.random.default_rng(config.RANDOM_SEED)

    try:
        images, labels = load_dataset(categories, args.samples_per_class, rng)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    train_ds, val_ds = split(images, labels, rng)
    logger.info("Train: %d  Val: %d  Classes: %d", len(train_ds), len(val_ds), len(categories))

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE)
    model = QuickDrawCNN(num_classes=len(categories))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f train_acc=%.4f  val_loss=%.4f val_acc=%.4f  (%.1fs)",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            time.perf_counter() - start,
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "categories": categories,
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                },
                args.output,
            )
            logger.info("Saved best model (val_acc=%.4f) to %s", val_acc, args.output)
    logger.info("Done. Best val_acc=%.4f", best_acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
