"""Small CNN for 28x28 Quick, Draw! bitmaps."""

import torch
from torch import nn

from air_drawing import config


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> MaxPool."""
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, config.CONV_KERNEL_SIZE, padding=config.CONV_KERNEL_SIZE // 2
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(config.POOL_SIZE),
    )


class QuickDrawCNN(nn.Module):
    """Three conv blocks, dropout and a two-layer linear head."""

    def __init__(
        self,
        num_classes: int,
        channels: tuple[int, ...] = config.CONV_CHANNELS,
        hidden_units: int = config.HIDDEN_UNITS,
        dropout: float = config.DROPOUT,
    ) -> None:
        """Build the network for `num_classes` outputs."""
        super().__init__()
        blocks = []
        in_channels = 1
        spatial = config.IMAGE_SIZE
        for out_channels in channels:
            blocks.append(_conv_block(in_channels, out_channels))
            in_channels = out_channels
            spatial //= config.POOL_SIZE
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * spatial * spatial, hidden_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, 1, 28, 28) images in [0, 1] to (N, num_classes) logits."""
        return self.head(self.features(x))
