"""Tests for the training augmentation and CLI (tiny synthetic data, no downloads)."""

import math
import time

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from train import augment_batch, elastic_field, parse_args, random_thickness, run_epoch

from air_drawing import config
from air_drawing.model import QuickDrawCNN
from air_drawing.preprocess import strokes_to_bitmap

SIZE = config.IMAGE_SIZE


def drawings(n: int = 8) -> torch.Tensor:
    """A batch of realistic Quick, Draw!-style images (circles and squares) in [0, 1]."""
    images = []
    for i in range(n):
        if i % 2:
            pts = [(50 * math.cos(t / 10), 50 * math.sin(t / 10)) for t in range(64)]
        else:
            pts = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]
        bitmap = strokes_to_bitmap([pts])
        assert bitmap is not None
        images.append(torch.from_numpy(bitmap).float() / config.PIXEL_MAX)
    return torch.stack(images).unsqueeze(1)


def gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# --- augment_batch ------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 8, 64])
def test_augment_keeps_shape_dtype_and_range(n: int) -> None:
    x = drawings(n)
    out = augment_batch(x, gen())
    assert out.shape == (n, 1, SIZE, SIZE)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_augment_range_holds_for_random_input() -> None:
    out = augment_batch(torch.rand(32, 1, SIZE, SIZE), gen())
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_augment_changes_images_but_keeps_the_drawing() -> None:
    x = drawings(16)
    out = augment_batch(x, gen())
    assert not torch.allclose(out, x)
    ink_before, ink_after = x.sum(dim=(1, 2, 3)), out.sum(dim=(1, 2, 3))
    ratio = ink_after / ink_before
    assert bool(((ratio > 0.3) & (ratio < 2.5)).all())  # distorted, not destroyed


def test_augment_is_reproducible_with_a_seed() -> None:
    x = drawings(4)
    assert torch.equal(augment_batch(x, gen(1)), augment_batch(x, gen(1)))
    assert not torch.equal(augment_batch(x, gen(1)), augment_batch(x, gen(2)))


def test_augment_empty_image_stays_empty() -> None:
    out = augment_batch(torch.zeros(4, 1, SIZE, SIZE), gen())
    assert out.abs().max() == 0


def test_augment_does_not_modify_input() -> None:
    x = drawings(4)
    before = x.clone()
    augment_batch(x, gen())
    assert torch.equal(x, before)


def test_augment_is_cpu_friendly() -> None:
    x = torch.rand(config.BATCH_SIZE, 1, SIZE, SIZE)
    start = time.perf_counter()
    augment_batch(x, gen())
    assert time.perf_counter() - start < 1.0  # typically ~15 ms per batch


# --- Individual distortions ---------------------------------------------------------


def test_random_thickness_dilates_and_erodes(monkeypatch: pytest.MonkeyPatch) -> None:
    x = drawings(8)
    monkeypatch.setattr(config, "AUG_THICKNESS_PROB", 1.0)
    monkeypatch.setattr(config, "AUG_THICKEN_SHARE", 1.0)
    thick = random_thickness(x, gen())
    monkeypatch.setattr(config, "AUG_THICKEN_SHARE", 0.0)
    thin = random_thickness(x, gen())
    assert thick.sum() > x.sum() > thin.sum()
    for out in (thick, thin):
        assert out.shape == x.shape and out.min() >= 0.0 and out.max() <= 1.0


def test_random_thickness_off_is_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "AUG_THICKNESS_PROB", 0.0)
    x = drawings(4)
    assert torch.equal(random_thickness(x, gen()), x)


def test_elastic_field_is_small_and_smooth() -> None:
    field = elastic_field(5, gen())
    assert field.shape == (5, SIZE, SIZE, 2)
    assert field.abs().max() <= config.AUG_ELASTIC_STRENGTH + 1e-6
    neighbour_step = (field[:, 1:] - field[:, :-1]).abs().max()
    assert neighbour_step < config.AUG_ELASTIC_STRENGTH / 2  # smooth, not pixel noise


# --- CLI and training loop ----------------------------------------------------------


def test_cli_defaults_enable_augmentation_and_12_epochs() -> None:
    args = parse_args([])
    assert args.augment is True
    assert args.epochs == config.DEFAULT_EPOCHS == 12


def test_cli_no_augment_flag() -> None:
    assert parse_args(["--no-augment"]).augment is False


def test_run_epoch_with_augmentation_trains() -> None:
    torch.manual_seed(0)
    x, y = drawings(16), torch.tensor([0, 1] * 8)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = QuickDrawCNN(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss, acc = run_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, True, gen())
    assert math.isfinite(loss) and 0.0 <= acc <= 1.0


def test_validation_is_never_augmented() -> None:
    torch.manual_seed(0)
    x, y = drawings(8), torch.tensor([0, 1] * 4)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = QuickDrawCNN(num_classes=2)
    loss_fn = nn.CrossEntropyLoss()
    plain = run_epoch(model, loader, loss_fn)
    with_flag = run_epoch(model, loader, loss_fn, augment=True, generator=gen())
    assert plain == with_flag
